#!/usr/bin/env python3
"""Parameter split of an RWKV-7 checkpoint and cost of the output head.

Usage:
    python bench_head.py <checkpoint.pth> [threads]

Compares the full 65,536-way output head (batch 1 and 8) with a head
restricted to 64 rows, approximating one syllable's homophone set in a
factorised phonetic-semantic head.
"""
import os
import sys
import time

import torch


def median_time(fn, repeats: int) -> float:
    runs = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        runs.append(time.perf_counter() - t)
    runs.sort()
    return runs[len(runs) // 2]


def main() -> None:
    path = sys.argv[1]
    torch.set_num_threads(int(sys.argv[2]) if len(sys.argv) > 2 else os.cpu_count())
    sd = torch.load(path, map_location="cpu")
    total = sum(v.numel() for v in sd.values())
    head = sd["head.weight"].float()
    emb = sd["emb.weight"]
    print(f"total params {total:,}  emb {tuple(emb.shape)}  head {tuple(head.shape)}")
    print(f"body params (excl. emb+head): {total - emb.numel() - head.numel():,}")

    d = head.shape[1]
    with torch.no_grad():
        for batch in (1, 8):
            x = torch.randn(batch, d)
            ms = median_time(lambda: x @ head.T, 50) * 1e3
            print(f"full head, batch {batch}: {ms:.2f} ms")
        x = torch.randn(d)
        sub = head[:64].contiguous()
        us = median_time(lambda: x @ sub.T, 200) * 1e6
        print(f"64-row homophone head, batch 1: {us:.0f} us")


if __name__ == "__main__":
    main()
