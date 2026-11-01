# RWKV-7 G1 as the IME student backbone: first findings

Date: 2026-11-01. Status: tokenizer audit and pip-runtime CPU benchmark done;
native-runtime (llama.cpp / rwkv.cpp, int8) benchmark **not yet run**.

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
- The ~26 ms per step mostly comes from the pip package's per-operation
  Python/PyTorch overhead; the body is only ~90M multiply-accumulates per
  token. A native int8 runtime is expected to be far faster, **but this has
  not been measured yet**.
- State snapshots for Backspace rollback are cheap.

## Next steps

1. Benchmark the same checkpoint in a native runtime (llama.cpp `rwkv7` GGUF,
   int8) for single-step and 16-token-prefill latency. That result decides
   whether the 0.4B (≈3× the body) is affordable.
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
```

Data files and checkpoints are not committed; fetch them from the URLs above
and check the hashes.
