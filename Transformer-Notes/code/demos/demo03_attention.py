# -*- coding: utf-8 -*-
"""演示 3：缩放点积注意力 + 多头注意力的形状与权重。
运行：python demo03_attention.py
"""

import os
import sys

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformer_part1 import attention, MultiHeadedAttention, subsequent_mask  # noqa: E402


if __name__ == "__main__":
    torch.manual_seed(42)

    # ---------- 单头：手算一遍 ----------
    d_k = 4
    q = torch.tensor([[[1.0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]])  # (1,3,4)
    k = q.clone()
    v = torch.tensor([[[10.0, 0], [0, 20], [0, 0]]])  # (1,3,2)，value 可以比 key 短
    out, attn = attention(q, k, v)
    print("单头示例：q=k=单位向量，v 是 3 个互不相同的向量")
    print("注意力权重矩阵 (每行和为 1)：")
    print(attn[0].numpy().round(3))
    print("加权结果 = 每行权重 x 对应 v：")
    print(out[0].numpy().round(2))
    print("行和检查:", attn[0].sum(-1).numpy().round(6))
    print()

    # ---------- 掩码如何把未来位置清零 ----------
    q2 = torch.randn(1, 3, d_k)
    _, attn_m = attention(q2, q2, q2, mask=subsequent_mask(3))
    print("带 subsequent_mask(3) 的注意力权重（左下严格为 0，即看不到未来）：")
    print(attn_m[0].numpy().round(3))
    print()

    # ---------- 多头：形状变化 + 可视化 ----------
    mha = MultiHeadedAttention(h=4, d_model=16, dropout=0.0)
    x = torch.randn(2, 6, 16)  # (batch=2, len=6, d_model=16)
    src_mask = torch.ones(2, 1, 6)  # 无 padding，全部允许
    y = mha(x, x, x, src_mask)
    print("多头注意力输入 x:", tuple(x.shape), "-> 输出 y:", tuple(y.shape))
    print("内部注意力权重 self.attn 形状:", tuple(mha.attn.shape), "(batch, heads, q_len, k_len)")
    print("第 0 个头对第 0 个 batch 的权重（行和为 1）：")
    print(mha.attn[0, 0].detach().numpy().round(3))
    print("行和检查:", mha.attn[0, 0].detach().sum(-1).numpy().round(6))

    img_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "images")
    os.makedirs(img_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].imshow(attn[0].detach().numpy(), cmap="Blues")
    axes[0].set_title("single-head attention weights")
    axes[0].set_xlabel("key position")
    axes[0].set_ylabel("query position")
    axes[1].imshow(mha.attn[0, 0].detach().numpy(), cmap="Blues")
    axes[1].set_title("MHA head 0 weights")
    axes[1].set_xlabel("key position")
    axes[1].set_ylabel("query position")
    fig.tight_layout()
    out = os.path.join(img_dir, "fig_attn_weights.png")
    fig.savefig(out, dpi=120)
    print("图片已保存:", out)
