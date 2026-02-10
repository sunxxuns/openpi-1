#!/usr/bin/env python3
"""
Extract the last full CUDAGraph replay from a rocprofv3 SQLite DB
and export to chrome://tracing JSON.

Usage:
  python scripts/extract_rocprof_trace.py traces/rocprof/mi300x_pi0_graph_results.db \
    -o traces/mi300x_pi0_bsz1_cudagraph_rocprof_26ms.json

rocprofv3's kernel_dispatch table has hardware-measured start/end timestamps
for every GPU kernel, including those inside HIP graph replays.
"""

import argparse
import json
import sqlite3
import sys


def find_tables(cur):
    """Find the actual table names (they have UUID suffixes)."""
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {r[0] for r in cur.fetchall()}
    dispatch_table = None
    string_table = None
    symbol_table = None
    for t in tables:
        if "kernel_dispatch" in t:
            dispatch_table = t
        if "rocpd_string" in t:
            string_table = t
        if "kernel_symbol" in t:
            symbol_table = t
    return dispatch_table, string_table, symbol_table


def main():
    p = argparse.ArgumentParser(description="Extract CUDAGraph replay from rocprofv3 DB")
    p.add_argument("db", help="Path to rocprofv3 results DB")
    p.add_argument("-o", "--output", default=None, help="Output JSON path")
    p.add_argument("--gap-threshold-ms", type=float, default=1.0,
                   help="Gap threshold (ms) to separate graph replay blocks")
    args = p.parse_args()

    db = sqlite3.connect(args.db)
    cur = db.cursor()

    dispatch_table, string_table, symbol_table = find_tables(cur)
    if not dispatch_table:
        print("ERROR: No kernel_dispatch table found in DB.")
        print("Available tables:")
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        for r in cur.fetchall():
            print(f"  {r[0]}")
        sys.exit(1)

    # Check row count
    cur.execute(f"SELECT COUNT(*) FROM {dispatch_table}")
    total = cur.fetchone()[0]
    print(f"Table: {dispatch_table}")
    print(f"Total kernel dispatches: {total}")

    if total == 0:
        print("ERROR: No kernel dispatches found. rocprofv3 may not have captured graph internals.")
        sys.exit(1)

    # Build kernel name lookup from symbol table
    kernel_names = {}
    if symbol_table:
        cur.execute(f"SELECT id, display_name FROM {symbol_table}")
        for row in cur.fetchall():
            kernel_names[row[0]] = row[1]

    # Fetch all dispatches sorted by start time
    cur.execute(f"""
        SELECT kernel_id, start, end
        FROM {dispatch_table}
        ORDER BY start
    """)
    dispatches = cur.fetchall()
    print(f"Fetched {len(dispatches)} dispatches")

    # Find blocks separated by gaps > threshold
    gap_ns = args.gap_threshold_ms * 1e6  # ms -> ns
    blocks = []
    current_block = [dispatches[0]]
    for i in range(1, len(dispatches)):
        prev_end = dispatches[i - 1][2]
        curr_start = dispatches[i][1]
        gap = curr_start - prev_end
        if gap > gap_ns:
            blocks.append(current_block)
            current_block = [dispatches[i]]
        else:
            current_block.append(dispatches[i])
    blocks.append(current_block)

    print(f"\nFound {len(blocks)} kernel blocks (gap > {args.gap_threshold_ms} ms):")
    for i, block in enumerate(blocks):
        start = block[0][1]
        end = block[-1][2]
        span_ms = (end - start) / 1e6
        kernel_sum_ms = sum((d[2] - d[1]) for d in block) / 1e6
        print(f"  Block {i}: {len(block):5d} kernels, span={span_ms:8.1f} ms, "
              f"kernel_sum={kernel_sum_ms:8.1f} ms")

    # Find the last block with roughly the right kernel count (~1500-3000)
    # Graph replays should have consistent kernel counts
    replay_blocks = [b for b in blocks if len(b) > 500]
    if not replay_blocks:
        print("\nWARNING: No blocks with >500 kernels found. Using last block.")
        target_block = blocks[-1]
    else:
        # Use the last full replay block
        target_block = replay_blocks[-1]
        print(f"\nUsing last replay block: {len(target_block)} kernels")

    # Calculate stats
    block_start = target_block[0][1]
    block_end = target_block[-1][2]
    span_ms = (block_end - block_start) / 1e6
    kernel_sum_ms = sum((d[2] - d[1]) for d in target_block) / 1e6
    gap_sum_ms = span_ms - kernel_sum_ms

    print(f"\nTrace stats:")
    print(f"  Kernels:          {len(target_block)}")
    print(f"  Wall time (trace):{span_ms:.1f} ms (includes profiler gaps)")
    print(f"  Kernel sum:       {kernel_sum_ms:.1f} ms (actual GPU compute)")
    print(f"  Profiler gaps:    {gap_sum_ms:.1f} ms ({gap_sum_ms/span_ms*100:.0f}% overhead)")

    # Export to Chrome trace JSON
    trace_events = []
    for kid, start_ns, end_ns in target_block:
        dur_ns = end_ns - start_ns
        name = kernel_names.get(kid, f"kernel_{kid}")
        # Truncate long kernel names for readability
        if len(name) > 120:
            name = name[:117] + "..."
        trace_events.append({
            "name": name,
            "ph": "X",
            "ts": (start_ns - block_start) / 1000,  # relative, in microseconds
            "dur": dur_ns / 1000,
            "pid": 0,
            "tid": 0,
            "cat": "kernel",
        })

    trace = {"traceEvents": trace_events}

    output = args.output
    if not output:
        output = args.db.replace("_results.db", "_trace.json")
    with open(output, "w") as f:
        json.dump(trace, f, indent=None)
    size_mb = len(json.dumps(trace)) / 1e6
    print(f"\nExported: {output} ({size_mb:.1f} MB, {len(trace_events)} events)")


if __name__ == "__main__":
    main()
