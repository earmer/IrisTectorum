# RWKV-7 G1 as the IME student backbone: first findings

Date: 2026-11-01. Status: tokenizer audit, pip-runtime CPU benchmark and
llama.cpp (F16 / Q8_0) CPU benchmark done.

## Context

The IME is end-to-end: a streaming transducer whose prediction network is a
character-level language model. The plan is a large teacher distilled into a
small on-device student. Qwen-family models use byte-level BPE with
multi-character Chinese tokens, which would need vocabulary surgery plus
continued pretraining. RWKV-7 G1 was examined as an alternative student
(and, at 1.5B+, as a same-tokenizer teacher).

## Model facts (from the model card, https://huggingface.co/BlinkDL/rwkv7-g1)

- Licence: Apache-2.0. Base models (web, code, synthetic, instruction/chat/reasoning data).
- Shapes: 0.1B = L12-D768, 0.4B = L24-D1024, 1.5B = L24-D2048,
  2.9B = L32-D2560, 7.2B = L32-D4096, 13.3B = L61-D4096. Vocab 65,536, head size 64.
- Latest checkpoints in the repo on 2026-11-01: 0.1B and 0.4B at `g1d`
  (2026-01-29, 2026-02-10); 1.5B to 13.3B at `g1l` (2026-10-31).
  The small models are on older data than the large ones.
- Training-data Chinese share: **not published**. World v3 is described as
  roughly 80% English, 10% multilingual, 10% code
  (https://arxiv.org/abs/2503.14456). No C-Eval/CMMLU results for G1 were found.
- Recurrent state: `L * (2D + 64D)` fp32 values → 0.1B: 608,256 values (≈2.4 MB);
  0.4B: 1,622,016 (≈6.5 MB).
- Runtimes: `rwkv` pip package, llama.cpp (`rwkv7` arch), rwkv.cpp, web-rwkv,
  rwkv-mobile (llama.cpp / ncnn / QNN / MLX / CoreML). ONNX export is community-only.

## Tokenizer audit

Vocabulary: `rwkv_vocab_v20230424.txt` from
https://raw.githubusercontent.com/BlinkDL/ChatRWKV/main/tokenizer/rwkv_vocab_v20230424.txt
(sha256 `e6dee3d4e31b4d5c40ac99508ac6c701ceef4bed681bf2167ce9a908552bca89`).
Encoding is greedy longest-match over bytes.

Character table: 通用规范汉字表, 8,105 characters in table order, from
https://github.com/iDvel/The-Table-of-General-Standard-Chinese-Characters
(local copy sha256 `d307c7ceddbbdffe3fc97c81d330ae46816f84eff1de7c9d435f6afe2d74f6f0`).

Script: `scripts/vocab_coverage.py`.

| Item | Result |
|---|---|
| Vocabulary entries | 65,529 (+ id 0 end-of-text, padded to 65,536) |
| Single-CJK-character tokens | 8,247 |
| Multi-CJK-character tokens | **0** |
| U+4E00–9FFF covered | 8,244 / 20,992 |
| 通用规范汉字表 covered | 5,982 / 8,105 (73.8%) |
| Level 1 (1–3500) | 3,500 / 3,500 |
| Level 2 (3501–6500) | 2,271 / 3,000 |
| Level 3 (6501–8105) | 211 / 1,605 |
| Extension A / supplementary-plane characters | 0 |

Consequences:

1. Chinese is already tokenised one character per token, so no merge removal
   is needed. The only vocabulary surgery is **adding** missing characters
   (2,123 for the 8,105 table, more for full Unihan coverage). These are
   currently encoded as 3–4 UTF-8 byte tokens, which can seed the new
   embeddings.
2. The model body never saw word-level Chinese tokens, so continued
   pretraining should only need to settle the new rows. This is a much
   smaller shift than converting a BPE model to character level.
3. G1 teacher (1.5B+) and student (0.1B/0.4B) share the tokenizer after
   identical additions, so token-level KL distillation is straightforward.

## CPU benchmark: official `rwkv` pip package

Machine: 4-core Intel Xeon @ 2.8 GHz (Cascade Lake class; AVX2, AVX-512, VNNI),
15 GiB RAM, no GPU. Software: PyTorch 2.15.0, `rwkv` 0.8.32, fp32, 4 threads.
Checkpoint: `rwkv7-g1d-0.1b-20260129-ctx8192.pth`
(sha256 `e10d7b1930c2644c5c6b194444774d6d82ec8212a78763493149de09aac7d83f`).

Scripts: `scripts/bench_rwkv_pip.py`, `scripts/bench_head.py`. The ranges below
cover two separate runs.

| Measurement | Result |
|---|---|
| Total parameters | 191,084,544 |
| Embedding / output head | 50.3M each (65,536 × 768) |
| Body (12 layers) | 90,421,248 |
| Chinese round trip (18 chars) | 18 tokens, exact |
| Prefill 16 / 64 / 256 tokens | 79–89 / 263–295 / 894–909 ms |
| Single step, batch 1 | p50 25.9–27.5 ms, p99 44.6–51.5 ms |
| Full 65,536-way head alone, batch 1 / 8 | 5.3–6.7 / 16.3–22.0 ms |
| 64-row homophone head, batch 1 | 10–11 µs |
| State snapshot copy | 0.6–0.7 ms |
| Peak RSS | 1.7–2.0 GB |

Interpretation:

- More than half of the "0.1B" model (≈100M of 191M) is vocabulary. Most of
  the 65,536 tokens are irrelevant to an IME.
- The full output head is about a quarter of each decoding step. A factorised
  syllable → homophone head reduces that part by roughly 500×.
- Multi-token prefill is much cheaper per token than single steps
  (16 tokens ≈ 3 steps), which supports verifying the previous keystroke's
  best candidate in one pass instead of decoding step by step.
- Part of the ~26 ms per step is the pip package's per-operation
  Python/PyTorch overhead; see the llama.cpp results below for a native runtime.
- State snapshots for Backspace rollback are cheap.

## CPU benchmark: llama.cpp

Same machine. llama.cpp commit `dde879b` (2026-11-01), CPU build with
`GGML_NATIVE=ON`, 4 threads, `llama-bench -p 16,64,256 -n 64 -r 5`.

Weights: the F16 GGUF from
https://huggingface.co/shoumenchougou/RWKV7-G1d-0.1B-GGUF (linked from the
official model card), file `rwkv7-g1d-0.1b-20260129-ctx8192-FP16.gguf`,
sha256 `6d41c88ecbd6e84dc46f146eace701c5981f7bd1691bf6de4d7cbfcb5a9f6afc`
(matches the published LFS hash). Q8_0 was produced locally with
`llama-quantize ... Q8_0`.

Correctness check (`scripts/greedy_ref.py` against `llama-simple`, greedy, 40
tokens, prompt `User: 请用一句话介绍北京。\n\nAssistant:`): the F16 GGUF output is
identical to the original `.pth` run through the `rwkv` package. Q8_0 agrees
for the first 21 characters, then diverges, which is ordinary quantisation
noise.

| Format | Size | Prefill 16 | Prefill 64 | Prefill 256 | Decode, per token |
|---|---|---|---|---|---|
| F16 | 367 MiB | 502 t/s (≈32 ms) | 699 t/s (≈92 ms) | 799 t/s (≈320 ms) | 69.2 t/s (**≈14.4 ms**) |
| Q8_0 | 201 MiB | 368 t/s (≈44 ms) | 532 t/s (≈120 ms) | 657 t/s (≈390 ms) | 93.4 t/s (**≈10.7 ms**) |

Interpretation:

- Native decoding is about 2.4× faster than the pip package (10.7 vs 25.9 ms
  per token), less than hoped. At batch 1, Q8_0 reads its ~201 MiB of weights
  per token; 201 MiB per 10.7 ms is ≈20 GB/s, which is consistent with decoding
  being memory-bandwidth-bound on this machine (inferred, not profiled).
- On that reading, the full 65,536-way head (≈26% of the parameters) is a
  matching share of each step's cost, which again favours the factorised head.
- Q8_0 is faster for decoding but slower for prefill than F16 on this CPU.
  An IME would want F16 or a better-tuned kernel for draft verification and Q8_0
  (or lower) for step-by-step decoding; measure before choosing.
- Verifying a 16-character draft costs ≈32 ms (F16). That is acceptable but
  not comfortable for one keystroke. The 0.4B model (≈3× the body
  parameters) would land around the 100 ms level on this CPU class unless the
  head is factorised and weights go to 4–5 bits. This is an estimate, not a
  measurement.

## Next steps

1. Benchmark Q4/Q5 variants and the 0.4B model in llama.cpp; profile to
   confirm whether decoding is bandwidth-bound.
2. Vocabulary extension tool: add the missing characters, initialise their
   embeddings from their UTF-8 byte tokens, and round-trip test on mixed
   Chinese, English, code and rare-character text.
3. Chinese bits-per-character evaluation of G1d 0.1B / 0.4B against a
   Qwen-family model on the same held-out text.

## Reproducing

```bash
pip install torch rwkv==0.8.32
python3 -I scripts/vocab_coverage.py rwkv_vocab_v20230424.txt general_standard_8105.txt
python scripts/bench_head.py rwkv7-g1d-0.1b-20260129-ctx8192.pth 4
python scripts/bench_rwkv_pip.py rwkv7-g1d-0.1b-20260129-ctx8192 4
python scripts/greedy_ref.py rwkv7-g1d-0.1b-20260129-ctx8192
```

llama.cpp:

```bash
cmake -S llama.cpp -B llama.cpp/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DLLAMA_CURL=OFF
```

```bash
cmake --build llama.cpp/build -j4 --target llama-bench llama-quantize llama-simple
```

```bash
llama.cpp/build/bin/llama-quantize g1d-0.1b-f16.gguf g1d-0.1b-q8_0.gguf Q8_0 4
```

```bash
llama.cpp/build/bin/llama-bench -m g1d-0.1b-f16.gguf,g1d-0.1b-q8_0.gguf -p 16,64,256 -n 64 -t 4 -r 5
```

Data files and checkpoints are not committed; fetch them from the URLs above
and check the hashes.
