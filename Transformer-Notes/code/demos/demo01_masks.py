# -*- coding: utf-8 -*-
"""演示 1：掩码到底是什么样子。
运行：python demo01_masks.py（在 demos 目录下）
"""

import os
import sys

import torch
import matplotlib
matplotlib.use("Agg")  # 无界面环境绘图
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformer_part1 import subsequent_mask  # noqa: E402


def make_std_mask(tgt, pad):
    """目标序列掩码 = 非 padding 掩码 & 不看未来掩码（与源码 Batch 相同）。"""
    tgt_mask = (tgt != pad).unsqueeze(-2)
    tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
    return tgt_mask


if __name__ == "__main__":
    print("① subsequent_mask(5)：能看的位置是 True，禁看的位置是 False")
    print(subsequent_mask(5).squeeze(0).numpy())
    print()

    tgt = torch.tensor([[1, 2, 3, 0, 0]])  # 0 是 padding 占位符
    print("② 目标序列 tgt =", tgt.tolist(), "（0 表示 padding），pad=0")
    print("make_std_mask 结果：行=当前词位置，列=它能看的位置")
    print(make_std_mask(tgt, 0).squeeze(0).numpy())
    print()

    img_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "images")
    os.makedirs(img_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4))
    axes[0].imshow(subsequent_mask(10).squeeze(0).numpy(), cmap="Greys")
    axes[0].set_title("subsequent_mask(10)\n(white = allowed to see)")
    axes[0].set_xlabel("key position (to look at)")
    axes[0].set_ylabel("query position")
    m = make_std_mask(torch.tensor([[1, 2, 3, 4, 0, 0]]), 0)
    axes[1].imshow(m.squeeze(0).numpy(), cmap="Greys")
    axes[1].set_title("std mask for tgt=[1,2,3,4,0,0]\n(blocked: padding + future)")
    axes[1].set_xlabel("key position (to look at)")
    axes[1].set_ylabel("query position")
    fig.tight_layout()
    out = os.path.join(img_dir, "fig_masks.png")
    fig.savefig(out, dpi=120)
    print("图片已保存:", out)
