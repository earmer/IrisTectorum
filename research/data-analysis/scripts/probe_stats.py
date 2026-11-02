#!/usr/bin/env python3
"""Character-level profile of a sentence list (levels, coverage, script, rarity by quintile).

Usage: python3 probe_stats.py <rwkv_vocab.txt> <table_8105.txt> <STCharacters.txt> <sentences.txt>...
"""
import sys
from collections import Counter

from common import is_cjk, rwkv_single_chars, general_table, levels, opencc_map

vocab, table, st, *files = sys.argv[1:]
single = rwkv_single_chars(vocab)
lvl = levels(general_table(table))
trad = {t for k, vs in opencc_map(st).items() for t in vs if t != k and t not in lvl}
for fn in files:
    with open(fn, encoding="utf-8") as f:
        L = [l.rstrip("\n") for l in f]
    U = {c for l in L for c in l if is_cjk(c)}
    out = sorted(c for c in U if c not in lvl)
    print(f"== {fn}: sentences {len(L)}, distinct CJK {len(U)}, by level {dict(Counter(lvl.get(c, 'outside') for c in U))}")
    print(f"  outside table: {''.join(out)} (OpenCC traditional forms: {''.join(c for c in out if c in trad)})")
    print(f"  not single RWKV tokens: {''.join(sorted(c for c in U if c not in single))}")
    print(f"  ASCII characters: {[(i + 1, ch) for i, l in enumerate(L) for ch in l if ord(ch) < 128]}")
    q = len(L) // 5
    for k in range(5):
        cs = [c for l in L[k * q:(k + 1) * q] for c in l if is_cjk(c)]
        print(f"  sentences {k * q + 1}-{(k + 1) * q}: level>=2/outside share {sum(lvl.get(c, 4) >= 2 for c in cs) / len(cs):.1%}")
