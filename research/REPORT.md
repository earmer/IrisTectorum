# IrisTectorum: research report to date

Period covered: 2026-11-01 to 2026-11-02.

Scope:

- review of an external IME architecture proposal
- the decisions taken since
- the proposed end-to-end design
- the first measurements on a candidate base model (RWKV-7 G1)
- the character inventory
- synthetic data
- corpus strategy and licensing stance

Labels used throughout:

- **Decision**: agreed direction for the project.
- **Recommendation**: proposed, not yet agreed or tested.
- **Measured**: produced on this project's own runs; scripts in `research/rwkv-g1/scripts/`.
- **Unverified**: taken from secondary sources or estimated.

---

## 1. Summary

1. The project builds an **end-to-end neural Chinese IME** as a research-led
   product (**Decision**).
2. Privacy and threat-model work is **deferred until after the first release**
   (**Decision**).
3. The proposed model is a **streaming transducer**: a key encoder, a
   character-level language model as the prediction network, and a joint
   network with a syllable → homophone factorised head (**Recommendation**).
4. Training uses a **large teacher distilled into a small on-device student**,
   both on a character-level vocabulary (**Recommendation**). Existing
   pretrained models are adapted rather than pretraining a new one; the
   approach is to make the tokenizer character-level and continue training
   (**Decision**).
5. **RWKV-7 G1 already tokenises Chinese one character per token** (no
   multi-character Chinese tokens at all). Only missing characters need adding:
   2,123 of the 8,105 in 通用规范汉字表 (**Measured**).
6. On a 4-core Xeon CPU, G1d 0.1B decodes at **≈10.7 ms per token (llama.cpp,
   Q8_0)** and verifies a 16-token draft in **≈32 ms (F16)**. The full
   65,536-way output head is ≈26% of parameters; a homophone-restricted head
   costs ≈10 µs instead of ≈5–7 ms in PyTorch (**Measured**). The 0.4B student is
   likely too slow on this CPU class without a factorised head and 4–5-bit
   weights (**Unverified**, estimate).
7. A dictionary-grounded character inventory (现代汉语词典 7th ed. + 中華語文大辭典)
   is **≈15.2k characters**: 11.2k on the simplified side, 12.1k on the
   traditional side. RWKV lacks ≈7.3k of them (**Measured**, §8).
8. Gemini-generated sentences are **not** a high-quality starting corpus. Their
   errors concentrate in rare characters (non-words, misuse, mixed or Japanese
   forms), and their register is ornate prose. They are admissible only as a
   capped, checked supplement (**Decision** on policy, **Measured** evidence, §9).
9. The product ships **code and model weights only**. Licence review is
   deferred to before release, but every data source is recorded in
   `research/data_manifest.md` from now on (**Decision**, §11).

---

## 2. Review of the external proposal

The starting point was a "Unified Neural-Symbolic Chinese IME" specification
written by Gemini. Its high-level split is sound: a deterministic
micro-backend, a hot-swappable neural core, and personal data kept outside the
weights. Several of its central claims do not hold.

### 2.1 Internal contradictions

| Claim | Problem |
|---|---|
| "100% recall of rare characters (龘, 翯)" | The output space is fixed to the 8,105 characters of 通用规范汉字表. 龘 is not in it, so the model can never emit it. A factorised head does not extend the vocabulary. |
| Permanent, never-changing ABI | The spec's own later features (app-register vector, fast-weight layer) have no slot in the input schema. A versioned ABI with a manifest is the workable form. |
| 8,105-character output tensor | No full-width punctuation, digits, Latin text or Traditional characters, despite the code-switching training data. |
| Personalisation "survives model updates" | Only the SymSpell hints do. Fast weights and LoRA are tied to one set of base weights and must be retrained after an update. |
| "Ambient KV-cache" on an SSM | SSMs have no KV cache; the naming is inconsistent. |
| Quanpin vs Shuangpin | Shuangpin is never specified, and the `;` auxiliary-code delimiter clashes with Microsoft Shuangpin, where `;` is *-ing*. |

### 2.2 Technical weaknesses

- **Hints at the front of the sequence.** Hints depend on the current keys and
  change every keystroke. With a recurrent model, changing the prefix forces
  re-encoding of everything after it, which loses the O(1) advantage.
- **SSM motivation overstated.** At 30–60M parameters and a few hundred tokens
  of context, a Transformer's KV cache is small. SSMs are also weaker at exact
  copying from context (Jelassi et al., 2024), which is what hint injection
  relies on.
- **No key–character alignment.** `P_phonetic(s | h_t^key)` assumes a known
  alignment, but alignment is the hard part with Jianpin and typos.
- **Two overlapping typo mechanisms** (DL automaton with hand-set penalties,
  plus trained-in corruption) with incompatible penalty scales.
- **Entropy spikes as word boundaries.** High entropy signals uncertainty, not
  boundaries. Candidate diversity in beam search is not addressed.
- **No latency budget or evaluation plan.**

### 2.3 UX features judged risky

- **Silent retroactive soft commit** through `replacementRange`: client support
  varies across apps; it can break undo and alter text after it has been
  read or sent.
- **Scraping window text and the clipboard.**
- **Spoken auxiliary codes** need training data the pipeline never produces.
- **Independent slot toggling** ignores dependencies between neighbouring characters.

### 2.4 Data concerns

- g2pW gives canonical readings, which is not what people type, and it was
  trained mainly on Taiwanese data (**Unverified**).
- DPO pairs such as 登录/登陆 and 权力/权利 involve valid words, so they only
  work when conditioned on context.
- The spec does not address scraping terms for Tieba, Zhihu, Weibo and
  Bilibili.

### 2.5 Framing of the baseline

The spec says traditional IMEs rely on "multi-gigabyte dictionaries". This is
overstated: librime-class dictionaries are tens of megabytes. The spec also
never compares against the strongest practical alternative, a dictionary
lattice with neural re-ranking. Gboard's FST-plus-neural-LM decoder
(Ouyang et al., 2017) is an example of that design.

---

## 3. Decisions taken

| # | Decision | Notes |
|---|---|---|
| D1 | End-to-end neural IME | Research-led product. A lattice-plus-reranker hybrid was proposed and set aside. |
| D2 | Privacy and threat model deferred to after the first release | Covers: local keystroke stores, ambient-context scraping, learned data resurfacing, retroactive edits, update-channel integrity. |
| D3 | Adapt a pretrained model instead of pretraining from scratch | Make the tokenizer character-level for Chinese, then continue training. |
| D4 | Ship code and model weights only | No dictionary tables or corpora are distributed. Runtime tables must come from clearly redistributable sources. |
| D5 | Licence review deferred to before release; provenance recorded from now on | `research/data_manifest.md` |
| D6 | Synthetic (LLM-generated) text only as a capped, checked, tagged supplement | Never as the base corpus and never in evaluation sets |

---

## 4. Proposed architecture (Recommendation)

### 4.1 Model: streaming transducer (RNN-T style)

| Component | Role | Size |
|---|---|---|
| Key encoder | Bidirectional over the current key buffer (rarely > 60 keys). Re-encoded every keystroke, which lets later keys fix earlier segmentation (`xian` vs `xi'an`). | 2–5M |
| Prediction network | Character LM conditioned on committed text. Recurrent backbone (RWKV-7) gives O(1) updates and cheap snapshots. | student 0.1–0.4B nominal |
| Joint network | Syllable head, then a character-within-syllable head | small |

Why a transducer rather than the spec's decoder-only prompt:

- Alignment comes with every candidate, so partial selection is exact and no
  separate aligner is needed.
- It streams by construction.
- It structurally discourages characters with no key evidence.

Training:

- Pruned RNN-T loss (k2) to keep memory manageable with a large vocabulary.
- Auxiliary syllable-level CTC loss on the key encoder.
- An attention-based encoder-decoder model of equal size as a comparison baseline.

### 4.2 Vocabulary and output

- All Unihan characters with readings (≈30k, including Traditional) instead of 8,105.
- The factorised head keeps cost per step small.
- A copy/passthrough token for Latin spans.
- Full-width punctuation and digits as ordinary tokens.

### 4.3 Personal vocabulary

- Hints placed next to the keys, not at the start of the sequence.
- A small attention block over a hint memory, even on a recurrent backbone.
- Trained with correct, near-miss, distractor and absent hints.
- Measure both the correct-use rate and the false-use rate.

### 4.4 Data

1. Text → pinyin keeping all acceptable polyphone readings.
2. Mixed-abbreviation simulation per word (`zhgguo`, `womenyiq`, `wxhn`).
3. Training on key-buffer prefixes, so mid-syllable input such as `zhongg` behaves sensibly.
4. Deterministic Shuangpin generation per scheme, conditioned on a scheme token.
5. Synthetic typos as a stopgap. Real keystroke traces are the long-term
   source; define the logging format early.

### 4.5 Sequence-level fine-tuning

MWER or DPO-style training with context-conditioned homophone negatives.

### 4.6 Evaluation

| Track | Metric |
|---|---|
| Full pinyin, Jianpin, mixed, typos, Shuangpin | Top-1 / top-5 sentence accuracy, character accuracy |
| Prefixes | Accuracy versus fraction of keys typed |
| Rare characters | Top-k recall with exact pinyin |
| User hints | Correct-use and false-use rates |
| Unsupported output | Rate of characters with no supporting key alignment |
| Latency | p50 / p99 per keystroke on the weakest target hardware |

Baselines:

- Rime with librime-octagram
- a PinyinGPT-style decoder-only model
- the attention-based encoder-decoder baseline

Ablations: factorised head; CTC auxiliary loss; RWKV vs Mamba vs a small
Transformer as the prediction network.

---

## 5. Model size, teacher–student, and the draft-model proposal

### 5.1 The 0.6B proposal

0.6B (e.g. Qwen3-0.6B) is reasonable for research but heavy to ship:

- ≈350–400 MB at 4 bits, permanently resident.
- Memory-bound decoding is too slow on typical laptops and phones.
- About 150M of its parameters are a 152k-entry token embedding that a
  character vocabulary would replace.

**Recommendation:**

- The teacher need not be small. Use the strongest model affordable to train
  (1.5B+).
- Distil into a 50–150M-class student.
- Apply the same character-level vocabulary to both, so token-level KL
  distillation works directly.

### 5.2 Draft model

- **Speculative decoding** fits poorly with the beam search a candidate bar needs.
- **A small model shown first and replaced later** makes candidates change
  under the user's fingers.
- **Recommendation:** use the previous keystroke's best candidate as the draft
  and verify it in one parallel pass. This needs no second model. The measured
  prefill speed in §7 supports it.

### 5.3 Making a BPE model character-level

These are the steps planned for Qwen-type models. Most are unnecessary for
RWKV G1 (§6.2).

1. Add a CJK pre-tokeniser split.
2. Drop the now-unreachable multi-character tokens.
3. Add missing characters, initialised from their byte-piece embeddings.
4. Train embeddings only, then the full model at a low learning rate with
   10–20% English/code replay.
5. Evaluate in bits per character, not per-token perplexity.

---

## 6. Base-model candidates and RWKV-7 G1 findings

### 6.1 Candidates considered

| Model | Architecture | Notes |
|---|---|---|
| RWKV-7 G1 (0.1B, 0.4B; 1.5B–13.3B) | Fully recurrent | Fixed-size state, cheap snapshots, Apache-2.0. Chinese share of training data unpublished. |
| LFM2 (350M, 700M) | 10 conv + 6 attention layers | Built for on-device use, Chinese listed as supported. Own licence (terms to be checked). Tokenizer not yet inspected. |
| Qwen2.5-0.5B / Qwen3-0.6B | Transformer | Strongest Chinese of these, heaviest; better as teacher. |
| From scratch, 30–60M | Any | Floor in the bake-off. |

English-centric small models (e.g. SmolLM2) were excluded.

### 6.2 RWKV-7 G1: model facts

Source: https://huggingface.co/BlinkDL/rwkv7-g1

- **Licence:** Apache-2.0.
- **Training data:** base models trained on web, code, synthetic and
  instruction/chat/reasoning data.
- **Shapes:**

  | Size | Shape |
  |---|---|
  | 0.1B | L12-D768 |
  | 0.4B | L24-D1024 |
  | 1.5B | L24-D2048 |
  | 2.9B | L32-D2560 |
  | 7.2B | L32-D4096 |
  | 13.3B | L61-D4096 |

  Vocabulary 65,536; head size 64.
- **Checkpoint data versions on 2026-11-01:** 0.1B and 0.4B at `g1d`
  (2026-01-29 / 2026-02-10); 1.5B and larger at `g1l` (2026-10-31). The small
  models lag by about nine months of data.
- **Recurrent state** is `L·(2D + 64D)` fp32 values:

  | Size | Values | fp32 |
  |---|---|---|
  | 0.1B | 608,256 | ≈2.4 MB |
  | 0.4B | ≈1.62M | ≈6.5 MB |

- **Runtimes:** `rwkv` pip package, llama.cpp (`rwkv7`), rwkv.cpp, web-rwkv,
  rwkv-mobile. ONNX export is community-only.
- **Training-data Chinese share:** not published. World v3 is described as
  ≈80% English, 10% multilingual, 10% code (arXiv 2503.14456). No C-Eval or
  CMMLU results found for G1 (**Unverified** beyond that).

### 6.3 Tokenizer audit (Measured)

Script: `research/rwkv-g1/scripts/vocab_coverage.py`.

| Item | Result |
|---|---|
| Vocabulary entries | 65,529 (+ id 0, padded to 65,536) |
| Single-CJK-character tokens | 8,247 |
| Multi-CJK-character tokens | **0** |
| U+4E00–9FFF covered | 8,244 / 20,992 |
| 通用规范汉字表 covered | 5,982 / 8,105 (73.8%) |
| Level 1 / 2 / 3 | 3,500/3,500 · 2,271/3,000 · 211/1,605 |
| Extension A / supplementary-plane characters | 0 |
| Round trip of an 18-character sentence | 18 tokens, exact |

Implications:

- No merge removal is needed. Vocabulary work reduces to **adding**
  characters, seeded from their 3–4 UTF-8 byte tokens.
- The body never saw word-level Chinese tokens, so adaptation should be mild.
- A G1 teacher (1.5B+) and G1 student share the tokenizer after identical
  additions.

---

## 7. CPU measurements (Measured)

Machine:

- 4-core Intel Xeon @ 2.8 GHz (Cascade Lake class; AVX2, AVX-512, VNNI)
- 15 GiB RAM, no GPU

Checkpoint: `rwkv7-g1d-0.1b-20260129-ctx8192`.

### 7.1 Parameter split

| Part | Parameters |
|---|---|
| Total | 191,084,544 |
| Input embedding | 50.3M |
| Output head | 50.3M |
| Body (12 layers) | 90.4M |

### 7.2 Official `rwkv` pip package (fp32, PyTorch 2.15.0, rwkv 0.8.32)

Ranges cover two runs.

| Measurement | Result |
|---|---|
| Prefill 16 / 64 / 256 tokens | 79–89 / 263–295 / 894–909 ms |
| Single step | p50 25.9–27.5 ms, p99 44.6–51.5 ms |
| Full head alone, batch 1 / 8 | 5.3–6.7 / 16.3–22.0 ms |
| 64-row homophone head | 10–11 µs |
| State snapshot copy | 0.6–0.7 ms |
| Peak RSS | 1.7–2.0 GB |

### 7.3 llama.cpp (commit `dde879b`, CPU, 4 threads)

F16 GGUF from https://huggingface.co/shoumenchougou/RWKV7-G1d-0.1B-GGUF
(linked from the official card; sha256 matches the published LFS hash). Q8_0
was quantised locally.

| Format | Size | Prefill 16 | Prefill 64 | Prefill 256 | Decode per token |
|---|---|---|---|---|---|
| F16 | 367 MiB | ≈32 ms | ≈92 ms | ≈320 ms | ≈14.4 ms |
| Q8_0 | 201 MiB | ≈44 ms | ≈120 ms | ≈390 ms | ≈10.7 ms |

Correctness, checked with greedy decoding of 40 tokens on a Chinese prompt:

- The F16 GGUF is **identical** to the original `.pth`.
- Q8_0 matches for 21 characters, then diverges, which is normal
  quantisation noise.

### 7.4 Reading of the results

- **Native decoding is ≈2.4× faster** than the pip package, less than first
  expected.
- **Decoding looks memory-bandwidth-bound.** At ≈20 GB/s implied (201 MiB per
  10.7 ms), cost scales with weight bytes read (inferred, not profiled).
- **The output head is ≈26% of parameters**, and so roughly that share of
  per-step cost. The factorised head is therefore the clearest single saving.
- **Multi-token prefill is far cheaper per token than single steps.** This
  supports previous-candidate draft verification.
- **Q8_0 decodes faster but prefills slower than F16** on this CPU. Formats may
  need to differ per phase.
- **The 0.4B student** (≈3× the body) is estimated near 100 ms per keystroke on
  this CPU class without a factorised head and 4–5-bit weights
  (**Unverified**).

---

## 8. Character inventory: simplified and traditional (Measured)

Details and scripts: `research/data-analysis/`.

### 8.1 Via OpenCC

| Set | Characters | Single tokens in RWKV | Missing |
|---|---|---|---|
| 通用规范汉字表 | 8,105 | 5,982 | 2,123 |
| Traditional / variant forms not in the table | 2,919 | 1,624 | 1,295 |
| Union | 11,024 | 7,606 | 3,418 |
| Level 1 + traditional forms | 4,928 | 4,823 (97.9%) | 105 |

The common core is already covered; the gap is in the rare tail.

### 8.2 Via the two dictionaries

| Set | Characters | Single tokens in RWKV |
|---|---|---|
| Simplified side (现汉 + CSLD simplified) | 11,182 | 6,154 |
| Traditional side (CSLD traditional + 现汉 traditional forms) | 12,124 | 6,125 |
| Union | 15,177 | 7,854 |

Further results from the dictionaries:

- **Homophone groups** (现汉, toneless, before cleaning): median 15, p90 44,
  max 165 (*yi*). This bounds the factorised head's per-step choice.
- **Polyphones:** 654 single-character headwords have several readings.
- **Taiwan vs mainland readings:** 2,311 of the 7,976 CSLD entries that give both
  pinyins differ even without tones. The input simulation must accept both.
- **Traditional ↔ simplified:** 274 simplified characters map to more than one
  traditional form. CSLD's word-level pairs resolve most of these better than
  OpenCC's character table.

### 8.3 Implications for teacher and student

- Teacher and student receive the **same vocabulary extension** (same IDs).
- The teacher's new rows must be **trained before distillation**, otherwise
  the student learns noise.
- The student's loss combines ground-truth cross-entropy with teacher KL. The
  KL weight is reduced where the gold character is rare or newly added.
- The teacher is checked per register (simplified, traditional) and on
  new-character positions, not only on averages.

### 8.4 Traditional characters: open product question

- Native traditional output needs genuine traditional text; converted
  simplified text inherits conversion errors.
- Most Taiwan users type Zhuyin, not pinyin; Hong Kong users are split across
  Cangjie, Quick and others. Supporting them changes the key encoder and its
  input simulation, not just the vocabulary.

---

## 9. Synthetic (Gemini) text (Measured evidence, Decision D6)

Two Gemini-generated lists of 100 graded sentences were analysed
(`research/data-analysis/README.md` §3).

- **Narrow coverage.** List A has 1,028 distinct characters and List B 1,264,
  with only 2–3 level-3 and 5 out-of-table characters each. They cannot test an
  11–15k inventory.
- **Dictionary attestation.** Of non-level-1 occurrences, 17% (A) and 12% (B)
  do not sit inside any dictionary word. Manual review splits these into real
  errors and classical terms missing from modern dictionaries.
  - Real errors in A: 黓暗, 龫发, 龆龀好学, 繾绻 (mixed script), and an ASCII full stop.
  - Real errors in B: 渓 (Japanese form) and 鸊鷉 (traditional).
- **Semantic misuse slips through.** 蹀躞起舞 passes the dictionary check
  although the sense is wrong.
- **Register.** The prose is ornate and repetitive (斑驳, 璀璨夺目, 宛如, 展现出),
  far from what people type.

Why not as a base corpus:

- Errors concentrate where the teacher's prior is weakest, so the IME would
  learn to propose non-words.
- The register pulls the model towards essay prose.
- Repeated training on model output erodes the rare tail (Shumailov et al.,
  Nature 2024; cited from memory).
- Google's terms may restrict training use (unverified).

Policy (D6):

1. Real attestations come first.
2. Generation is used only for characters below a minimum count of real
   examples.
3. Generated sentences must pass four filters: inventory, script consistency,
   dictionary attestation, then semantic review.
4. Synthetic text is capped at a few per cent, tagged, and never used for
   evaluation.

Of the two lists, List B is the better probe once 渓 → 溪 and 鸊鷉 → 䴙䴘 are
fixed. Even then it is a smoke test, not an evaluation set.

---

## 10. Corpus strategy

### 10.1 "Junk" has two kinds for an IME

Standard web filters prefer textbook-like prose: C4-style heuristics,
educational-value classifiers (FineWeb-Edu style), CCI3.0-HQ's classifier. An
IME needs chat, comments, fragments, slang and code-switching, which those
filters rank lowest.

| Kind | Examples | Treatment |
|---|---|---|
| Junk for both | Gambling and adult SEO spam, boilerplate, garbled encodings, machine translation, keyword stuffing, near-duplicates | Remove |
| Junk for an LLM, signal for an IME | Casual chat, forum replies, comments, fragments, slang, code-switching | Keep, with its own quota |
| LLM-generated text in recent crawls | Fluent but unlike typing | Separate category, kept low |

### 10.2 Sources by role

Availability and licences of the open corpora below were not checked
(**Unverified**).

- **Teacher continued training:** open Chinese pretraining corpora such as
  CCI3.0-HQ (≈500 GB), MAP-CC (≈800B tokens) and the OpenCSG
  Fineweb-edu-chinese series. Good for knowledge and formal registers;
  educationally biased.
- **Rare characters:** classical and literary text.
  - A candidate is the `bookraw` corpus in a separate project (329 authors,
    ≈4,268 works in its catalogue). Pre-modern authors spotted include
    左传, 李清照, 宋之问, 宋祁, 洪应明 (菜根谭), 高明, 周德清 and 唐玄宗.
  - Which works are actually present in its cleaned data, and how much text is
    pre-modern, has not been confirmed.
  - To check there: script per copy, deduplication across editions, and
    copyright of modern authors.
- **Traditional:** genuine traditional-script sources (Taiwan and Hong Kong
  text), not conversions.
- **IME register:** conversational text; licensing is the main constraint.

### 10.3 Dictionaries as filters

1. **Per-character targets.** Count corpus occurrences of each inventory
   character and set a minimum (e.g. 50) for new characters. Mine real text
   first and generate only for the remainder.
2. **Garbled text and spam.** Use the share of a document's characters that
   fall outside the 15,177-character inventory or into private-use or
   compatibility ranges.
3. **Keyword stuffing.** After segmentation, combine the out-of-dictionary
   word rate with the repetition rate.

### 10.4 Evaluation

- Use real held-out text per register: formal, conversational, classical and
  traditional, deduplicated against training data.
- Report bits per character and homophone ranking per register, plus a
  rare-character breakdown.

---

## 11. Licensing stance (Decision D4, D5)

- **Output risk is low.** The model only completes short spans in response to
  typing.
- **Obligations remain on use and distribution.** These are:
  - shipping dictionary-derived tables, which D4 rules out;
  - contractual terms on model outputs used for training;
  - provenance questions at publication or weight release.
- **Two habits from now on:**
  1. Record every source in `research/data_manifest.md`.
  2. Build any runtime table the code needs from clearly redistributable
     sources.
- **Unverified points:**
  - China's 生成式人工智能服务管理暂行办法 requires lawful training-data
    sources. Whether it applies to an IME is unclear.
  - CSLD is believed to be CC BY-NC-ND 3.0 TW.
  - 现代汉语词典 is proprietary and used here for research only.

---

## 12. Environment constraints

- The cloud container is CPU-only.
- Pretraining or distillation here is infeasible. For example, 0.1B on 1B
  tokens at ≈100 GFLOPS is about 70 days.
- Suitable work here: tokenizer surgery, data-pipeline prototypes on
  samples, small evaluations, CPU latency benchmarks.
- Training needs rented GPUs.
- `huggingface.co` access was enabled for this environment. Downloads are
  served via `us.aws.cdn.hf.co`.

---

## 13. Open questions and risks

1. **Chinese strength of RWKV G1.** The Chinese share of training data is
   unpublished, so it needs a bits-per-character measurement against a Qwen
   model on the same held-out text. This is the main risk to choosing G1.
2. **Small-model staleness.** The 0.1B/0.4B checkpoints are on older data
   versions than the 1.5B+.
3. **Latency headroom.** Not yet measured: 4–5-bit formats, the 0.4B model,
   a profile confirming bandwidth-bound decoding.
4. **The LFM2 alternative** is untested; its tokenizer and licence terms are
   unchecked.
5. **Training compute and corpora.** Availability is not yet known; it sets
   realistic model sizes and pipeline scope.
6. **First target platform** (macOS IMK, Windows TSF, Linux fcitx5/IBus,
   mobile) is undecided.
7. **Traditional-character scope.** Pinyin users only, or Zhuyin (Taiwan) and
   Cangjie/Quick (Hong Kong) users too? This changes the key encoder, not just
   the vocabulary (§8.4).
8. **Real-text coverage of the ≈7.3k new characters.** Unknown until
   per-character counts are run on the chosen corpora, including `bookraw`.
9. **Licences** of CSLD, the open corpora and Gemini output are unverified
   (deferred, D5).

---

## 14. Next steps

1. Run llama.cpp benchmarks of Q4/Q5 formats and the 0.4B model, and profile
   decoding.
2. Build a vocabulary-extension tool for G1: add the missing characters, seed
   them from byte tokens, and round-trip test on mixed Chinese/English/code
   and rare-character text.
3. Run a Chinese bits-per-character evaluation: G1d 0.1B / 0.4B against a
   Qwen-family model on identical held-out text.
4. Run the LFM2-350M arm of the bake-off on the same metrics.
5. Prototype the data pipeline: polyphone-aware pinyin, mixed-abbreviation and
   prefix simulation, Shuangpin generation.
6. Clean the dictionary pinyin and build the target inventory and homophone
   groups. Fix `ɡ`, split multi-reading fields, strip homograph indices, and
   handle the sandhi marks.
7. Write a standalone per-character coverage counter to run on `bookraw`
   (`books_cleaned.parquet`) and other corpora, split by 文言/白话 and by script,
   with deduplication by work.
8. Build the synthetic-text gate (inventory, script, attestation, review) for
   characters still short of real examples.

---

## Sources

**RWKV-7 G1**

- RWKV-7 G1 model card: https://huggingface.co/BlinkDL/rwkv7-g1
- RWKV-7 "Goose" paper: https://arxiv.org/abs/2503.14456
- RWKV World tokenizer vocabulary: https://raw.githubusercontent.com/BlinkDL/ChatRWKV/main/tokenizer/rwkv_vocab_v20230424.txt
- G1d 0.1B GGUF: https://huggingface.co/shoumenchougou/RWKV7-G1d-0.1B-GGUF
- llama.cpp: https://github.com/ggml-org/llama.cpp

**Character table**

- 通用规范汉字表 list: https://github.com/iDvel/The-Table-of-General-Standard-Chinese-Characters

**Other candidates**

- LFM2-700M: https://huggingface.co/LiquidAI/LFM2-700M
- LFM2-350M ONNX: https://huggingface.co/onnx-community/LFM2-350M-ONNX

**Character and dictionary data** (details and hashes in `research/data_manifest.md`)

- OpenCC: https://github.com/BYVoid/OpenCC
- 中華語文大辭典 data: https://github.com/g0v/moedict-data-csld
- 现代汉语词典 7th ed. word list: supplied by the project, not public

**Chinese corpora** (availability and licences unverified)

- CCI3.0-HQ: https://arxiv.org/pdf/2410.18505
- OpenCSG Chinese Corpus: https://arxiv.org/pdf/2501.08197
- MAP-CC: https://app.alphaneural.io/datasets/m-a-p/MAP-CC

**Literature** (cited from memory, not re-checked during this work)

- Shumailov et al., Nature 2024, on training with model-generated data

- Jelassi et al., 2024, *Repeat After Me*
- Ouyang et al., 2017, *Mobile keyboard input decoding with finite-state transducers*
- Tan et al., ACL 2022, PinyinGPT
- Kuang et al., 2022, pruned RNN-T

Detailed tables, hashes and reproduction commands for the measurements are in
`research/rwkv-g1/README.md`.
