# Data analysis: character inventory, dictionaries and synthetic probe sets

Date: 2026-11-02. All figures are produced by the scripts in `scripts/`; input
files are listed in `../data_manifest.md` and are not committed.

## 1. Simplified + traditional coverage (via OpenCC)

Script: `scripts/trad_coverage.py`.

| Set | Characters | Single tokens in RWKV | Missing |
|---|---|---|---|
| 通用规范汉字表 (simplified) | 8,105 | 5,982 | 2,123 |
| Traditional / variant forms not in the table | 2,919 | 1,624 | 1,295 |
| Union | 11,024 | 7,606 | 3,418 |
| Level 1 (3,500) + their traditional forms | 4,928 | 4,823 (97.9%) | 105 |

274 simplified characters map to more than one traditional form (e.g. 发 → 發/髮,
后 → 後/后). Character-level conversion is therefore ambiguous; native
traditional output needs traditional training text.

## 2. Dictionary-based inventory

Script: `scripts/lexicon_stats.py`. Sources: 现代汉语词典 7th ed. word list (XH7)
and 中華語文大辭典 (CSLD).

| Set | Characters | Single tokens in RWKV | In 8105 table |
|---|---|---|---|
| XH7, all characters | 9,180 | 6,058 | 8,105 |
| Simplified side (XH7 + CSLD simplified) | 11,182 | 6,154 | 8,105 |
| Traditional side (CSLD traditional + XH7 traditional forms) | 12,124 | 6,125 | 5,615 |
| Union | 15,177 | 7,854 | 8,105 |

- A dictionary-grounded target inventory is ≈15k characters; RWKV lacks ≈7.3k of them.
- XH7 homophone groups (toneless, before cleaning): median 15, p90 44, max 165 (*yi*).
  The factorised head chooses among at most ≈165 candidates per step.
- 654 XH7 single-character headwords have more than one reading.
- CSLD entries giving both Taiwan and mainland pinyin: 7,976. Of these, 2,311 differ
  even without tones, spaces and hyphens. The input simulation must accept both.

Cleaning needed before use:

- CSLD uses `ɡ` (U+0261) for *g*.
- The Taiwan column marks 一/不 tone sandhi.
- The mainland column is mostly empty; it appears to be filled only where it
  differs (unverified).
- XH7 headwords carry homograph indices (埃1/埃2) and parenthesised traditional
  forms. Some pinyin fields pack several readings, which creates ≈90 spurious
  syllable strings; "487 distinct pinyin strings" is not a syllable count.

## 3. Gemini-generated probe sentences

Two lists of 100 sentences graded from common to rare characters. Scripts:
`scripts/probe_stats.py`, `scripts/attestation_check.py`.

| | List A (Gemini 3.5 Flash Light) | List B (Gemini 4 Argon) |
|---|---|---|
| Distinct characters | 1,028 | 1,264 |
| Level 2 / level 3 / outside the table | 72 / 3 / 5 | 121 / 2 / 5 |
| Not single RWKV tokens | 25 | 13 |
| Rare share by quintile | 0.9 → 1.9 → 4.7 → 6.4 → 4.5% | 0.7 → 5.3 → 6.2 → 5.1 → 6.9% |
| Non-level-1 occurrences / not inside any dictionary word | 107 / 18 (17%) | 154 / 19 (12%) |

Manual review of the flagged items:

| | Real errors | Correct but not in modern dictionaries |
|---|---|---|
| List A | 黓暗 and 龫发 (not attested as words); 龆龀好学 (misuse); 繾绻 (mixed script); ASCII "." in sentence 56 | 饫甘餍肥, 黥面, 窳陋 |
| List B | 渓 (Japanese form; Chinese 溪); 鸊鷉 (traditional; simplified 䴙䴘) | 簠簋, 黍稷, 玉琮, 敕勒川, 漶漫 |

The dictionary check also misses errors. For example, 蹀躞起舞 passes because 蹀躞
is a dictionary word, although it does not suit a butterfly.

Conclusions:

- List B is the better probe, after fixing 渓 → 溪 and 鸊鷉 → 䴙䴘.
- Neither list is an evaluation set. They are synthetic, cover only ≈1.3k
  characters with a handful of rare ones, and are written in an ornate literary
  register.
- Synthetic text is admissible only as a capped, tagged supplement for
  characters short of real examples. It must pass inventory, script and
  attestation checks and a semantic review, and it is never used in evaluation.

## Reproducing

Any Python 3; no third-party packages.

```bash
python3 scripts/trad_coverage.py rwkv_vocab_v20230424.txt table_8105.txt STCharacters.txt TWVariants.txt HKVariants.txt
```

```bash
python3 scripts/lexicon_stats.py rwkv_vocab_v20230424.txt table_8105.txt xh7.txt csld.txt
```

```bash
python3 scripts/probe_stats.py rwkv_vocab_v20230424.txt table_8105.txt STCharacters.txt list_a.txt list_b.txt
```

```bash
python3 scripts/attestation_check.py table_8105.txt xh7.txt csld.txt list_a.txt list_b.txt
```
