"""Shared loaders for the data-analysis scripts."""
import ast
import re


def is_cjk(ch: str) -> bool:
    o = ord(ch)
    return (0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF
            or 0x20000 <= o <= 0x3FFFF or 0xF900 <= o <= 0xFAFF)


def rwkv_single_chars(vocab_path):
    """Characters that are a single token in rwkv_vocab_v20230424.txt."""
    single = set()
    with open(vocab_path, encoding="utf-8") as f:
        for line in f:
            tok = ast.literal_eval(line[line.index(" "):line.rindex(" ")])
            if isinstance(tok, bytes):
                try:
                    tok = tok.decode("utf-8")
                except UnicodeDecodeError:
                    continue
            if len(tok) == 1:
                single.add(tok)
    return single


def general_table(path):
    """通用规范汉字表 in table order (one character per line)."""
    with open(path, encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


def levels(table):
    return {c: (1 if i < 3500 else 2 if i < 6500 else 3) for i, c in enumerate(table)}


def opencc_map(path):
    """OpenCC dictionary file: key -> list of values."""
    m = {}
    with open(path, encoding="utf-8") as f:
        for l in f:
            if l.startswith("#") or "\t" not in l:
                continue
            k, v = l.rstrip("\n").split("\t")
            m[k] = v.split()
    return m


def xh7_rows(path):
    """现代汉语词典 7th-ed. word list: yields ('word', w, full, pinyin) or ('variant', form)."""
    with open(path, encoding="utf-8") as f:
        for l in f:
            if l.startswith("#") or not l.strip():
                continue
            fld = l.rstrip("\n").split("\t")
            if len(fld) == 4:
                yield ("word", fld[0], fld[2], fld[3])
            elif len(fld) == 2:
                yield ("variant", fld[0])


def csld_rows(path):
    """中華語文大辭典 CSV export: yields (trad, simp_or_trad, tw_pinyin, cn_pinyin)."""
    with open(path, encoding="utf-8") as f:
        for l in f:
            if l.startswith("#") or not l.strip():
                continue
            fld = l.rstrip("\n").split("\t")
            if len(fld) == 8:
                yield fld[0], fld[1] or fld[0], fld[6], fld[7]


def strip_homograph_index(word):
    return re.sub(r"\d+$", "", word)
