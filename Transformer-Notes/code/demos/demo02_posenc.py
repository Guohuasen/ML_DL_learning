# -*- coding: utf-8 -*-
"""演示 2：位置编码长什么样。
运行：python demo02_posenc.py
"""

import os
import sys

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformer_part1 import PositionalEncoding  # noqa: E402


if __name__ == "__main__":
    torch.manual_seed(42)
    pe = PositionalEncoding(d_model=16, dropout=0.0, max_len=50)
    table = pe.pe[0]  # (50, 16)
    print("位置编码表形状 (max_len, d_model):", tuple(table.shape))
    print()
    print("前 5 个位置、前 4 个维度的值：")
    print("        dim0(sin)   dim1(cos)   dim2(sin)   dim3(cos)")
    for pos in range(5):
        row = table[pos, :4]
        print("pos %d: %10.4f %10.4f %10.4f %10.4f" % (pos, row[0], row[1], row[2], row[3]))
    print()
    print("同一维度 0 随位置变化（sin 波）：")
    print("  ", " ".join("%7.4f" % v for v in table[:12, 0].tolist()))
    print("同一维度 1 随位置变化（cos 波）：")
    print("  ", " ".join("%7.4f" % v for v in table[:12, 1].tolist()))

    img_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "images")
    os.makedirs(img_dir, exist_ok=True)

    # 图 1：论文经典画法——几条 sin/cos 曲线
    pe2 = PositionalEncoding(d_model=20, dropout=0.0, max_len=100)
    y = pe2.pe[0].numpy()  # (100, 20)
    fig, ax = plt.subplots(figsize=(8, 4))
    for dim in [4, 5, 6, 7]:
        ax.plot(range(100), y[:, dim], label="dim %d" % dim)
    ax.set_xlabel("position")
    ax.set_ylabel("value")
    ax.set_title("Positional Encoding curves (dim 4/5/6/7)")
    ax.legend()
    fig.tight_layout()
    out1 = os.path.join(img_dir, "fig_posenc_curves.png")
    fig.savefig(out1, dpi=120)

    # 图 2：位置 x 维度的热力图，直观看到"竖条纹"
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    im = ax2.imshow(y[:50, :16].T, aspect="auto", cmap="RdBu_r")
    ax2.set_xlabel("position")
    ax2.set_ylabel("dimension")
    ax2.set_title("PE heatmap (50 positions x 16 dims)")
    fig2.colorbar(im, ax=ax2)
    fig2.tight_layout()
    out2 = os.path.join(img_dir, "fig_posenc_heatmap.png")
    fig2.savefig(out2, dpi=120)
    print("图片已保存:", out1, "|", out2)
