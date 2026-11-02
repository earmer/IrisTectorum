#!/usr/bin/env python3
"""Measure how the RWKV World tokenizer vocabulary covers Chinese characters.

Usage:
    python3 -I vocab_coverage.py <rwkv_vocab_v20230424.txt> <general_standard_8105.txt>

The second file must list the 8,105 characters of 通用规范汉字表 in table
order (one per line), so the level split (1-3500, 3501-6500, 6501-8105) is
correct.
"""
import ast
import sys


def is_cjk(ch: str) -> bool:
    o = ord(ch)
    return (
        0x4E00 <= o <= 0x9FFF        # CJK Unified Ideographs
        or 0x3400 <= o <= 0x4DBF     # Extension A
        or 0x20000 <= o <= 0x3FFFF   # Extensions B and later (SIP/TIP)
        or 0xF900 <= o <= 0xFAFF     # Compatibility Ideographs
    )


def load_vocab(path: str):
    """Return (entry_count, set of single-CJK tokens, count of multi-CJK tokens)."""
    entries = 0
    single = set()
    multi = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            # Format: "<id> <python literal> <byte length>"
            literal = line[line.index(" "):line.rindex(" ")]
            tok = ast.literal_eval(literal)
            entries += 1
            if isinstance(tok, bytes):
                try:
                    tok = tok.decode("utf-8")
                except UnicodeDecodeError:
                    continue  # raw byte / partial character token
            n_cjk = sum(map(is_cjk, tok))
            if n_cjk == 1 and len(tok) == 1:
                single.add(tok)
            elif n_cjk >= 1:
                multi += 1
    return entries, single, multi


def main() -> None:
    vocab_path, table_path = sys.argv[1], sys.argv[2]
    entries, single, multi = load_vocab(vocab_path)
    with open(table_path, encoding="utf-8") as f:
        table = [line.strip() for line in f if line.strip()]

    print(f"vocab entries:              {entries}")
    print(f"single-CJK-char tokens:     {len(single)}")
    print(f"multi-CJK-char tokens:      {multi}")
    basic = sum(0x4E00 <= ord(c) <= 0x9FFF for c in single)
    print(f"U+4E00-9FFF covered:        {basic} / 20992")
    print(f"8105 table present:         {sum(c in single for c in table)} / {len(table)}")
    for name, a, b in (("level 1 (1-3500)", 0, 3500),
                       ("level 2 (3501-6500)", 3500, 6500),
                       ("level 3 (6501-8105)", 6500, 8105)):
        seg = table[a:b]
        missing = [c for c in seg if c not in single]
        print(f"  {name:20s} present {len(seg) - len(missing)} / {len(seg)}"
              f"  e.g. missing: {''.join(missing[:20])}")


if __name__ == "__main__":
    main()
