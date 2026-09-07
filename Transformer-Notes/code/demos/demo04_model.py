# -*- coding: utf-8 -*-
"""演示 4：完整模型的数据流——每一步张量形状怎么变。
运行：python demo04_model.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformer_part1 import make_model, subsequent_mask  # noqa: E402


if __name__ == "__main__":
    torch.manual_seed(42)
    # 小号模型：词表 11，2 层，d_model=32，4 个头（demo 用，跑得快）
    model = make_model(src_vocab=11, tgt_vocab=11, N=2, d_model=32, d_ff=64, h=4)
    nparams = sum(p.numel() for p in model.parameters())
    print("模型总参数: %.1fk" % (nparams / 1000))
    print()

    src = torch.randint(1, 11, (2, 6))   # (batch=2, src_len=6)
    tgt = torch.randint(1, 11, (2, 5))   # (batch=2, tgt_len=5)
    src_mask = torch.ones(2, 1, 6)       # 没有 padding，全 1
    tgt_mask = subsequent_mask(5).expand(2, 5, 5)  # 只看不未来
    print("输入张量：src", tuple(src.shape), "tgt", tuple(tgt.shape))
    print()

    emb_src = model.src_embed(src)       # 词嵌入 + 位置编码
    print("src 嵌入后 (batch, len, d_model):", tuple(emb_src.shape))
    memory = model.encode(src, src_mask)
    print("编码器输出 memory:", tuple(memory.shape))
    emb_tgt = model.tgt_embed(tgt)
    print("tgt 嵌入后:", tuple(emb_tgt.shape))
    dec_out = model.decode(memory, src_mask, tgt, tgt_mask)
    print("解码器输出:", tuple(dec_out.shape))
    logits = model.generator(dec_out)
    print("输出层 log-softmax:", tuple(logits.shape), "（覆盖整个词表 11）")
    out = model(src, tgt, src_mask, tgt_mask)
    print("model(...) 一步到位的结果形状:", tuple(out.shape))
    print()
    print("输出层里每个位置概率最大（argmax）的词：")
    print(logits.argmax(-1).tolist())
