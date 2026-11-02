#!/usr/bin/env python3
"""Greedy decoding with the original .pth via the rwkv pip package.

Usage: python greedy_ref.py <checkpoint path without .pth>
Used to check that a GGUF conversion produces the same output in llama.cpp.
"""
import os, sys
os.environ["RWKV_V7_ON"]="1"; os.environ["RWKV_JIT_ON"]="1"; os.environ["RWKV_CUDA_ON"]="0"
import torch
from rwkv.model import RWKV
from rwkv.utils import PIPELINE
m=RWKV(model=os.path.abspath(sys.argv[1]),strategy="cpu fp32"); p=PIPELINE(m,"rwkv_vocab_v20230424")
prompt="User: 请用一句话介绍北京。\n\nAssistant:"
out,st=m.forward(p.encode(prompt),None); toks=[]
for _ in range(40):
    t=int(torch.argmax(out)); 
    if t==0: break
    toks.append(t); out,st=m.forward([t],st)
print(repr(p.decode(toks)))
