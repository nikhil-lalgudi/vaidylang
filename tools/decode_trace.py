"""Summarize kernels within GPU benchmark decode ranges from an Nsight SQLite export."""

import argparse
from bisect import bisect_left
from collections import Counter
import json
from pathlib import Path
import sqlite3
from statistics import median


def summarize(trace):
    with sqlite3.connect(trace.resolve().as_uri() + "?mode=ro", uri=True) as db:
        ranges = db.execute(
            "SELECT start, end FROM NVTX_EVENTS WHERE text = 'decode' AND end > start ORDER BY start"
        ).fetchall()
        kernels = db.execute(
            """SELECT k.start, k.end, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k
               JOIN StringIds s ON k.shortName = s.id ORDER BY k.start"""
        ).fetchall()
    if not ranges or not kernels:
        raise ValueError("Export an Nsight capture from gpu_bench.py --profile with CUDA and NVTX enabled")

    starts = [start for start, _, _ in kernels]
    fractions, wall_ms, gpu_ms, launch_counts = [], [], [], []
    kernel_totals = Counter()
    for start, end in ranges:
        first, last = bisect_left(starts, start), bisect_left(starts, end)
        busy, frontier = 0, start
        for kernel_start, kernel_end, name in kernels[first:last]:
            kernel_end = min(kernel_end, end)
            busy += max(0, kernel_end - max(frontier, kernel_start))
            frontier = max(frontier, kernel_end)
            kernel_totals[name] += kernel_end - kernel_start
        fractions.append(busy / (end - start))
        wall_ms.append((end - start) / 1e6)
        gpu_ms.append(busy / 1e6)
        launch_counts.append(last - first)

    return {
        "decode_steps": len(ranges),
        "wall_ms_p50": median(wall_ms),
        "gpu_kernel_ms_p50": median(gpu_ms),
        "kernel_busy_fraction_p50": median(fractions),
        "kernels_per_step_p50": median(launch_counts),
        "top_kernels": [
            {"name": name, "total_ms": elapsed / 1e6}
            for name, elapsed in kernel_totals.most_common(8)
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.trace), indent=2))
