#!/usr/bin/env python3
"""
Unit-test style harness: validate and time aiter flash attention vs eager attention.

Focus:
  - Correctness for q_len == k_len and q_len != k_len (KV-cache style)
  - Correct bottom-right causal masking behavior when q_len != k_len
  - Timing comparison (aiter.flash_attn_func vs eager matmul+softmax+matmul)

Usage:
  python scripts/unit_test_aiter_attention.py
  python scripts/unit_test_aiter_attention.py --q-len 11 --k-len 799 --causal
"""

from __future__ import annotations

import argparse
import math
import time
from typing import Callable

import torch


def _bench_ms(fn: Callable[[], torch.Tensor], *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / iters


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    # x: [B, S, H_kv, D] -> [B, S, H_q, D] where H_q = H_kv * n_rep
    if n_rep == 1:
        return x
    b, s, h, d = x.shape
    return x[:, :, :, None, :].expand(b, s, h, n_rep, d).reshape(b, s, h * n_rep, d)


def _eager_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool,
    softmax_scale: float,
) -> torch.Tensor:
    """
    Eager attention reference matching flash_attn_func semantics:
      - q,k,v are [B, S, H, D]
      - If causal=True and q_len != k_len, causal mask is bottom-right aligned.
    Returns: out [B, q_len, Hq, Dv]
    """
    # Compute attention scores in fp32 for stability, then cast back.
    qf = q.float()
    kf = k.float()
    vf = v.float()

    b, sq, hq, dq = qf.shape
    _, sk, hk, dk = kf.shape
    _, _, hv, dv = vf.shape
    assert hk == hv, "k/v head count mismatch"

    # GQA/MQA: repeat kv heads if needed
    assert hq % hk == 0, "Hq must be divisible by Hkv"
    n_rep = hq // hk
    kf = _repeat_kv(kf, n_rep)
    vf = _repeat_kv(vf, n_rep)

    # [B, H, Sq, D] @ [B, H, D, Sk] -> [B, H, Sq, Sk]
    q_t = qf.permute(0, 2, 1, 3)  # B,H,Sq,D
    k_t = kf.permute(0, 2, 3, 1)  # B,H,D,Sk
    scores = torch.matmul(q_t, k_t) * float(softmax_scale)

    if causal:
        # bottom-right aligned causal mask
        # allow attending up to key index (sk - sq + i) for each query i.
        # mask positions where j > sk - sq + i
        # Build [Sq, Sk] mask once (bool)
        i = torch.arange(sq, device=scores.device)[:, None]
        j = torch.arange(sk, device=scores.device)[None, :]
        allowed = j <= (sk - sq + i)
        scores = scores.masked_fill(~allowed[None, None, :, :], float("-inf"))

    probs = torch.softmax(scores, dim=-1)
    # [B, H, Sq, Sk] @ [B, H, Sk, Dv] -> [B, H, Sq, Dv]
    v_t = vf.permute(0, 2, 1, 3)  # B,H,Sk,Dv
    out = torch.matmul(probs, v_t)
    out = out.permute(0, 2, 1, 3).contiguous()  # B,Sq,H,Dv
    return out.to(dtype=q.dtype)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--q-len", type=int, default=11)
    p.add_argument("--k-len", type=int, default=799)
    p.add_argument("--heads-q", type=int, default=8)
    p.add_argument("--heads-kv", type=int, default=8)
    p.add_argument("--head-dim-qk", type=int, default=128)
    p.add_argument("--head-dim-v", type=int, default=128)
    p.add_argument("--causal", action="store_true")
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--rtol", type=float, default=1e-1)
    p.add_argument("--atol", type=float, default=1.5)
    args = p.parse_args()

    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    print(f"PyTorch: {torch.__version__}")
    print(f"Device:  {device}")
    if device.type == "cuda":
        print(f"GPU:     {torch.cuda.get_device_name(device)}")
        print(f"ROCm:    {getattr(torch.version, 'hip', None)}")
    print(f"dtype:   {dtype}")
    print(
        f"shape:   B={args.batch} q_len={args.q_len} k_len={args.k_len} "
        f"Hq={args.heads_q} Hkv={args.heads_kv} Dqk={args.head_dim_qk} Dv={args.head_dim_v}"
    )
    print(f"causal:  {args.causal}")
    print(f"warmup:  {args.warmup}  iters: {args.iters}")

    # Import aiter
    try:
        import aiter  # type: ignore
        from aiter.ops.mha import flash_attn_func  # type: ignore
    except Exception as e:
        raise SystemExit(f"aiter not available: {e}") from e

    b = args.batch
    sq = args.q_len
    sk = args.k_len
    hq = args.heads_q
    hk = args.heads_kv
    dqk = args.head_dim_qk
    dv = args.head_dim_v

    assert hq % hk == 0, "heads-q must be divisible by heads-kv (GQA/MQA)"

    q = torch.randn((b, sq, hq, dqk), device=device, dtype=dtype)
    k = torch.randn((b, sk, hk, dqk), device=device, dtype=dtype)
    v = torch.randn((b, sk, hk, dv), device=device, dtype=dtype)

    softmax_scale = 1.0 / math.sqrt(dqk)

    eager_fn = lambda: _eager_attention(q, k, v, causal=args.causal, softmax_scale=softmax_scale)
    aiter_fn = lambda: flash_attn_func(
        q,
        k,
        v,
        dropout_p=0.0,
        softmax_scale=softmax_scale,
        causal=args.causal,
        window_size=(-1, -1),
        bias=None,
        alibi_slopes=None,
        deterministic=True,
        return_lse=False,
        return_attn_probs=False,
        how_v3_bf16_cvt=0,  # gfx950: keep it deterministic/compatible
    )

    # Correctness vs fp32 eager reference (more stable for comparing different kernels)
    ref = eager_fn().float()
    out = aiter_fn().float()
    torch.testing.assert_close(out, ref, rtol=args.rtol, atol=args.atol)
    print(f"correctness: OK (rtol={args.rtol} atol={args.atol})")

    # Timing (note: eager is not fused; this is a baseline reference cost)
    eager_ms = _bench_ms(eager_fn, warmup=args.warmup, iters=args.iters)
    aiter_ms = _bench_ms(aiter_fn, warmup=args.warmup, iters=args.iters)
    print(f"eager: {eager_ms:.4f} ms/iter")
    print(f"aiter: {aiter_ms:.4f} ms/iter")


if __name__ == "__main__":
    main()

