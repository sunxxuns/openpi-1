#!/usr/bin/env python3
"""
Analyze a PyTorch profiler chrome trace (Perfetto/Chrome JSON) for:
  - GPU kernel totals (cat == "kernel")
  - HIP runtime API totals (cat == "cuda_runtime")
  - Kernel count + top-by-time + top-by-count
  - Attention-related rough totals (attn/fmha/softmax)

Usage:
  python scripts/analyze_policy_trace.py traces/policy_inference_compiled_default_replay.json
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python scripts/analyze_policy_trace.py <trace.json>")

    path = sys.argv[1]
    with open(path, "r") as f:
        data = json.load(f)

    events = data.get("traceEvents", [])

    # Aggregate kernel events
    k_dur = defaultdict(float)
    k_cnt = Counter()
    k_ts_min = None
    k_ts_max = None
    for ev in events:
        if ev.get("ph") != "X":
            continue
        if str(ev.get("cat", "")) != "kernel":
            continue
        name = ev.get("name")
        dur = ev.get("dur")
        ts = ev.get("ts")
        if not isinstance(name, str) or dur is None:
            continue
        dur = float(dur)
        k_dur[name] += dur
        k_cnt[name] += 1
        if ts is not None:
            ts = float(ts)
            k_ts_min = ts if k_ts_min is None else min(k_ts_min, ts)
            k_ts_max = (ts + dur) if k_ts_max is None else max(k_ts_max, ts + dur)

    # Aggregate runtime events
    r_dur = defaultdict(float)
    r_cnt = Counter()
    for ev in events:
        if ev.get("ph") != "X":
            continue
        if str(ev.get("cat", "")) != "cuda_runtime":
            continue
        name = ev.get("name")
        dur = ev.get("dur")
        if not isinstance(name, str) or dur is None:
            continue
        dur = float(dur)
        r_dur[name] += dur
        r_cnt[name] += 1

    kernel_total_ms = sum(k_dur.values()) / 1000.0
    runtime_total_ms = sum(r_dur.values()) / 1000.0
    kernel_span_ms = (
        (k_ts_max - k_ts_min) / 1000.0 if (k_ts_min is not None and k_ts_max is not None) else 0.0
    )
    util = (kernel_total_ms / kernel_span_ms) if kernel_span_ms > 0 else 0.0

    def is_attn(name: str) -> bool:
        nl = name.lower()
        return ("attn" in nl) or ("fmha" in nl) or ("flash" in nl) or ("softmax" in nl)

    attn_ms = sum(d for n, d in k_dur.items() if is_attn(n)) / 1000.0
    cijk_ms = sum(d for n, d in k_dur.items() if n.startswith("Cijk_")) / 1000.0
    aiter_gemm_ms = sum(d for n, d in k_dur.items() if "aiter::bf16gemm" in n) / 1000.0

    print(f"trace: {path}")
    print(f"kernel_total_ms:   {kernel_total_ms:.3f}")
    print(f"kernel_span_ms:    {kernel_span_ms:.3f}")
    print(f"kernel_util:       {util:.3f}")
    print(f"kernel_count:      {sum(k_cnt.values())}")
    print(f"runtime_total_ms:  {runtime_total_ms:.3f}")
    print("")
    print(f"attn_related_ms:   {attn_ms:.3f}")
    print(f"Cijk_total_ms:     {cijk_ms:.3f}")
    print(f"aiter_gemm_ms:     {aiter_gemm_ms:.3f}")

    print("\nTop kernels by total time:")
    for name, dur in sorted(k_dur.items(), key=lambda x: x[1], reverse=True)[:25]:
        print(f"{dur/1000.0:8.3f} ms  calls={k_cnt[name]:5d}  {name[:160]}")

    print("\nTop kernels by call count:")
    for name, cnt in k_cnt.most_common(25):
        print(f"calls={cnt:5d}  total={k_dur[name]/1000.0:8.3f} ms  avg={(k_dur[name]/cnt)/1000.0:7.4f} ms  {name[:120]}")

    print("\nTop cuda_runtime ops by total time:")
    for name, dur in sorted(r_dur.items(), key=lambda x: x[1], reverse=True)[:25]:
        print(f"{dur/1000.0:8.3f} ms  calls={r_cnt[name]:5d}  {name}")


if __name__ == "__main__":
    main()

