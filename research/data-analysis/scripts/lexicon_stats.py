#!/usr/bin/env python3
"""Character inventory, RWKV coverage and pinyin statistics from two dictionaries.

Usage: python3 lexicon_stats.py <rwkv_vocab.txt> <table_8105.txt> <xh7_wordlist.txt> <csld.txt>
"""
import re
import statistics
import sys
import unicodedata
from collections import defaultdict

from common import is_cjk, rwkv_single_chars, general_table, xh7_rows, csld_rows

vocab, table, xh, csld = sys.argv[1:5]
single = rwkv_single_chars(vocab)
tab = set(general_table(table))


def toneless(p):
    p = p.replace("ɡ", "g").replace("ü", "v").replace("Ü", "v")
    p = "".join(ch for ch in unicodedata.normalize("NFD", p) if not unicodedata.combining(ch))
    return re.sub(r"[^a-zv]", "", p.lower())


xh_chars, xh_trad, heads, readings = set(), set(), set(), defaultdict(set)
for row in xh7_rows(xh):
    if row[0] == "word":
        _, w, full, py = row
        xh_chars.update(c for c in w if is_cjk(c))
        if len(w) == 1 and is_cjk(w):
            heads.add(w)
            readings[w].add(toneless(py))
            xh_trad.update(c for c in full[1:] if is_cjk(c))
    else:
        xh_trad.update(c for c in row[1] if is_cjk(c))

tw, cn = set(), set()
n = both = differ = 0
for t, s, pt, pc in csld_rows(csld):
    n += 1
    tw.update(c for c in t if is_cjk(c))
    cn.update(c for c in s if is_cjk(c))
    if pt and pc:
        both += 1
        differ += toneless(pt) != toneless(pc)

print(f"XH7 single-char headwords {len(heads)}, chars in all words {len(xh_chars)}, traditional/variant forms {len(xh_trad)}")
print(f"CSLD entries {n}, both pinyins {both}, toneless differ {differ}")
sets = {
    "XH7 chars": xh_chars,
    "simplified side (XH7 + CSLD simp)": xh_chars | cn,
    "traditional side (CSLD trad + XH7 trad)": tw | xh_trad,
    "union": xh_chars | xh_trad | tw | cn,
}
for k, v in sets.items():
    print(f"{k:42s} {len(v):6d}  RWKV single {sum(c in single for c in v):6d}  in 8105 {sum(c in tab for c in v):5d}")

syl = defaultdict(set)
for c, rs in readings.items():
    for r in rs:
        if r:
            syl[r].add(c)
sizes = sorted(len(v) for v in syl.values())
poly = sum(1 for rs in readings.values() if len({r for r in rs if r}) > 1)
print(f"homophone groups (uncleaned): median {statistics.median(sizes)}, p90 {sizes[int(.9 * len(sizes))]}, "
      f"max {sizes[-1]} ({max(syl, key=lambda k: len(syl[k]))}); polyphonic headwords {poly}; "
      f"distinct pinyin strings {len(syl)} (includes multi-reading artefacts)")
