#!/usr/bin/env python3
"""RWKV vocabulary coverage of 通用规范汉字表 plus its traditional forms (via OpenCC).

Usage: python3 trad_coverage.py <rwkv_vocab.txt> <table_8105.txt> <STCharacters.txt> <TWVariants.txt> <HKVariants.txt>
"""
import sys
from common import rwkv_single_chars, general_table, opencc_map

vocab, table, st, tw, hk = sys.argv[1:6]
single = rwkv_single_chars(vocab)
simp = general_table(table)
S = set(simp)
stm, twm, hkm = opencc_map(st), opencc_map(tw), opencc_map(hk)
trad = set()
for c in simp:
    for t in stm.get(c, [c]):
        trad.add(t)
        trad.update(twm.get(t, []) + hkm.get(t, []))
cov = lambda X: sum(c in single for c in X)
tonly = trad - S
U = S | trad
print(f"simplified table: {len(S)}, single tokens {cov(S)}")
print(f"traditional/variant forms not in the table: {len(tonly)}, single tokens {cov(tonly)}")
print(f"union: {len(U)}, single tokens {cov(U)}, missing {len(U) - cov(U)}")
onemany = [c for c in simp if len(stm.get(c, [])) > 1]
print(f"simplified chars with >1 traditional form: {len(onemany)}, e.g. {''.join(onemany[:20])}")
l1 = set(simp[:3500])
for c in simp[:3500]:
    l1.update(stm.get(c, [c]))
print(f"level 1 + traditional forms: {len(l1)}, single tokens {cov(l1)}")
