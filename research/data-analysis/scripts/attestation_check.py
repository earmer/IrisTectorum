#!/usr/bin/env python3
"""Flag non-level-1 characters that do not sit inside any dictionary word in their sentence.

A flag is a prompt for manual review, not proof of misuse: classical terms absent
from modern dictionaries are flagged too, and semantic misuse of a real word is not.

Usage: python3 attestation_check.py <table_8105.txt> <xh7_wordlist.txt> <csld.txt> <sentences.txt>...
"""
import re
import sys
from collections import defaultdict

from common import is_cjk, general_table, xh7_rows, csld_rows, strip_homograph_index

table, xh, csld, *files = sys.argv[1:]
lvl1 = set(general_table(table)[:3500])
words = set()
for row in xh7_rows(xh):
    if row[0] == "word":
        words.add(strip_homograph_index(row[1]))
for t, s, _, _ in csld_rows(csld):
    words.update(re.split(r"[，、；。]", t))
    words.update(re.split(r"[，、；。]", s))
by_char = defaultdict(list)
for w in words:
    if len(w) >= 2:
        for c in set(w):
            by_char[c].append(w)

for fn in files:
    with open(fn, encoding="utf-8") as f:
        lines = [l.strip() for l in f]
    rare, flagged = 0, []
    for i, s in enumerate(lines, 1):
        for j, c in enumerate(s):
            if is_cjk(c) and c not in lvl1:
                rare += 1
                if not any(w in s for w in by_char.get(c, [])):
                    flagged.append(f"{i}:{c}「{s[max(0, j - 2):j + 3]}」")
    print(f"{fn}: non-level-1 occurrences {rare}, flagged {len(flagged)} ({len(flagged) / rare:.0%})")
    print("  " + " ".join(flagged))
