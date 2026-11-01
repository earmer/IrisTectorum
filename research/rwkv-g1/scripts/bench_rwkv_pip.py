#!/usr/bin/env python3
"""CPU latency of an RWKV-7 checkpoint through the official `rwkv` pip package.

Usage:
    python bench_rwkv_pip.py <checkpoint path without .pth> [threads]

Measures: Chinese tokenisation round trip, prefill of 16/64/256 tokens,
single-step decoding (p50/p99 over 200 steps), state snapshot copy time and
peak resident memory. Requires `torch` and `rwkv` (tested with rwkv 0.8.32).
"""
import os
import resource
import statistics
import sys
import time

os.environ["RWKV_V7_ON"] = "1"
os.environ["RWKV_JIT_ON"] = "1"
os.environ["RWKV_CUDA_ON"] = "0"

import torch  # noqa: E402
from rwkv.model import RWKV  # noqa: E402
from rwkv.utils import PIPELINE  # noqa: E402


def main() -> None:
    ckpt = os.path.abspath(sys.argv[1])
    threads = int(sys.argv[2]) if len(sys.argv) > 2 else os.cpu_count()
    torch.set_num_threads(threads)

    model = RWKV(model=ckpt, strategy="cpu fp32")
    pipe = PIPELINE(model, "rwkv_vocab_v20230424")

    text = "今天天气很好，我们一起去公园散步吧。"
    ids = pipe.encode(text)
    print(f"chars {len(text)}  tokens {len(ids)}  round trip {pipe.decode(ids) == text}")

    with torch.no_grad():
        for n in (16, 64, 256):
            runs = []
            for _ in range(5):
                t = time.perf_counter()
                model.forward([1] * n, None)
                runs.append(time.perf_counter() - t)
            print(f"prefill {n:3d} tokens: {statistics.median(runs) * 1e3:.1f} ms")

        _, state = model.forward(ids, None)
        steps = []
        for i in range(200):
            t = time.perf_counter()
            _, state = model.forward([ids[i % len(ids)]], state)
            steps.append(time.perf_counter() - t)
        steps.sort()
        print(f"single step: p50 {steps[100] * 1e3:.2f} ms  p99 {steps[197] * 1e3:.2f} ms")

        t = time.perf_counter()
        _ = [s.clone() for s in state]
        n_values = sum(s.numel() for s in state)
        print(f"state snapshot copy: {(time.perf_counter() - t) * 1e6:.0f} us, {n_values} values")

    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"peak RSS: {rss_mb:.0f} MB")


if __name__ == "__main__":
    main()
