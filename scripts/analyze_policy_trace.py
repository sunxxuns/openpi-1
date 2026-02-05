#!/usr/bin/env python3
"""
Analyze a PyTorch profiler chrome trace (Perfetto/Chrome JSON) for:
  - GPU kernel totals (cat == "kernel")
  - HIP runtime API totals (cat == "cuda_runtime")
  - Kernel count + top-by-time + top-by-count
  - Attention-related rough totals (attn/fmha/softmax)
  - (New) Attribute GPU kernels back to CPU ops via "External id"

Usage:
  python scripts/analyze_policy_trace.py traces/policy_inference_compiled_default_replay.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a PyTorch profiler chrome trace JSON.")
    parser.add_argument("trace", help="Path to trace.json (Chrome/Perfetto format)")
    parser.add_argument("--top-kernels", type=int, default=25, help="Top kernels by total time to show")
    parser.add_argument("--top-count", type=int, default=25, help="Top kernels by call count to show")
    parser.add_argument(
        "--top-runtime",
        type=int,
        default=25,
        help='Top runtime API ops (cat=="cuda_runtime") by total time to show',
    )
    parser.add_argument(
        "--top-ops",
        type=int,
        default=25,
        help='Top CPU ops (cat=="cpu_op") by attributed GPU kernel time to show',
    )
    parser.add_argument(
        "--kernels-per-op",
        type=int,
        default=6,
        help="For each top op, show this many top kernels",
    )
    args = parser.parse_args()

    path = args.trace
    with open(path, "r") as f:
        data = json.load(f)

    events = data.get("traceEvents", [])

    # Build a mapping from External id -> CPU op event (name + shapes)
    cpu_name_by_ext: dict[int, str] = {}
    cpu_dims_by_ext: dict[int, object] = {}
    cpu_concrete_by_ext: dict[int, object] = {}
    for ev in events:
        if ev.get("ph") != "X":
            continue
        if str(ev.get("cat", "")) != "cpu_op":
            continue
        args_ev = ev.get("args") or {}
        ext = args_ev.get("External id")
        if isinstance(ext, int):
            name = ev.get("name")
            if isinstance(name, str):
                cpu_name_by_ext[ext] = name
                if "Input Dims" in args_ev:
                    cpu_dims_by_ext[ext] = args_ev.get("Input Dims")
                if "Concrete Inputs" in args_ev:
                    cpu_concrete_by_ext[ext] = args_ev.get("Concrete Inputs")

    # Aggregate kernel events
    k_dur = defaultdict(float)
    k_cnt = Counter()
    # kernel_name -> op_name -> total dur
    k_op_dur: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    # op_name -> kernel_name -> total dur
    op_k_dur: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    # op_name -> total attributed dur
    op_dur = defaultdict(float)
    op_cnt = Counter()
    op_shapes: dict[str, Counter[str]] = defaultdict(Counter)

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

        # Attribute kernel to CPU op via External id when available.
        args_ev = ev.get("args") or {}
        ext = args_ev.get("External id")
        op_name = "<no_cpu_op>"
        if isinstance(ext, int):
            op_name = cpu_name_by_ext.get(ext, "<no_cpu_op>")
            op_dur[op_name] += dur
            op_cnt[op_name] += 1
            k_op_dur[name][op_name] += dur
            op_k_dur[op_name][name] += dur

            # Keep a small "shape signature" counter for the op (best effort).
            dims = cpu_dims_by_ext.get(ext)
            if dims is not None:
                op_shapes[op_name][str(dims)] += 1
            else:
                concrete = cpu_concrete_by_ext.get(ext)
                if concrete is not None:
                    op_shapes[op_name][f"Concrete Inputs: {concrete}"] += 1

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
    for name, dur in sorted(k_dur.items(), key=lambda x: x[1], reverse=True)[: args.top_kernels]:
        op_hint = ""
        if name in k_op_dur:
            # Most-attributed CPU op for this kernel.
            op_best, op_best_dur = max(k_op_dur[name].items(), key=lambda x: x[1])
            op_hint = f"  op≈{op_best} ({op_best_dur/1000.0:.3f} ms)"
        print(f"{dur/1000.0:8.3f} ms  calls={k_cnt[name]:5d}  {name[:160]}{op_hint}")

    print("\nTop kernels by call count:")
    for name, cnt in k_cnt.most_common(args.top_count):
        print(f"calls={cnt:5d}  total={k_dur[name]/1000.0:8.3f} ms  avg={(k_dur[name]/cnt)/1000.0:7.4f} ms  {name[:120]}")

    print("\nTop cuda_runtime ops by total time:")
    for name, dur in sorted(r_dur.items(), key=lambda x: x[1], reverse=True)[: args.top_runtime]:
        print(f"{dur/1000.0:8.3f} ms  calls={r_cnt[name]:5d}  {name}")

    # Attribute GPU time to CPU ops (best effort via External id).
    if op_dur:
        print("\nTop cpu_op by attributed GPU kernel time (via External id):")
        for op_name, dur in sorted(op_dur.items(), key=lambda x: x[1], reverse=True)[: args.top_ops]:
            calls = op_cnt[op_name]
            avg_ms = (dur / calls) / 1000.0 if calls else 0.0
            # Most common shape signature for this op.
            shape_sig = ""
            if op_name in op_shapes and op_shapes[op_name]:
                shape_sig = op_shapes[op_name].most_common(1)[0][0]
                # Keep it readable.
                if len(shape_sig) > 160:
                    shape_sig = shape_sig[:160] + "..."
                shape_sig = f"  shapes≈{shape_sig}"
            print(f"{dur/1000.0:8.3f} ms  calls={calls:5d}  avg={avg_ms:7.4f} ms  {op_name}{shape_sig}")

            # Top kernels within this op.
            if args.kernels_per_op > 0:
                top_k = sorted(op_k_dur[op_name].items(), key=lambda x: x[1], reverse=True)[: args.kernels_per_op]
                for kname, kdur in top_k:
                    print(f"    {kdur/1000.0:8.3f} ms  {kname[:180]}")


if __name__ == "__main__":
    main()

