# -*- coding: utf-8 -*-
"""演示 5：小规模"抄写任务"训练 + 贪心解码（真实运行效果）。
训练目标：模型把输入序列原样"抄"出来。
运行：python demo05_train.py（CPU 上约几十秒）
"""

import os
import sys

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformer_part1 import make_model, subsequent_mask  # noqa: E402


class Batch:
    """把一个 batch 的 src/tgt 整理成模型要的形状，并生成掩码（与源码等价）。"""

    def __init__(self, src, tgt=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if tgt is not None:
            self.tgt = tgt[:, :-1]        # 输入解码器用：去掉最后一个词
            self.tgt_y = tgt[:, 1:]       # 预测目标用：去掉起始词（对齐错一位）
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
        return tgt_mask


class LabelSmoothing(nn.Module):
    """标签平滑损失（源码原版）；smoothing=0 时退化成普通 KL 损失。"""

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size

    def forward(self, x, target):
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        return self.criterion(x, true_dist.clone().detach())


class SimpleLossCompute:
    """把生成器的输出和损失拼在一起：先过 Generator，再算损失并归一。"""

    def __init__(self, generator, criterion):
        self.generator = generator
        self.criterion = criterion

    def __call__(self, x, y, norm):
        x = self.generator(x)
        sloss = (
            self.criterion(
                x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)
            )
            / norm
        )
        return sloss.data * norm, sloss


def rate(step, model_size, factor, warmup):
    """学习率预热 + 衰减公式（源码原版），防止训练初期不稳定。"""
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    """贪心解码：每次挑概率最大的词，接在末尾继续生成。"""
    memory = model.encode(src, src_mask)
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
    for _ in range(max_len - 1):
        out = model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )
    return ys


def data_gen(V, batch_size, nbatches):
    """生成随机"抄写任务"数据：src=tgt，内容随机数字，开头固定为 1。"""
    for _ in range(nbatches):
        data = torch.randint(1, V, size=(batch_size, 10))
        data[:, 0] = 1
        src = data.clone().detach()
        tgt = data.clone().detach()
        yield Batch(src, tgt, 0)


if __name__ == "__main__":
    torch.manual_seed(42)
    V = 11
    model = make_model(V, V, N=2, d_model=64, d_ff=128, h=4)
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.5, betas=(0.9, 0.98), eps=1e-9
    )
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step, model_size=model.src_embed[0].d_model, factor=1.0, warmup=100
        ),
    )

    losses = []
    for epoch in range(25):
        model.train()
        total_loss = 0.0
        total_tokens = 0
        for batch in data_gen(V, batch_size=40, nbatches=10):
            out = model.forward(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
            loss, loss_node = SimpleLossCompute(model.generator, criterion)(
                out, batch.tgt_y, batch.ntokens
            )
            loss_node.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            lr_scheduler.step()
            total_loss += loss
            total_tokens += batch.ntokens
        avg = total_loss / total_tokens
        losses.append(avg)
        print("epoch %2d | 平均 loss %.4f | lr %.5f" % (epoch + 1, avg, optimizer.param_groups[0]["lr"]))

    model.eval()
    src = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    src_mask = torch.ones(1, 1, src.shape[1])
    pred = greedy_decode(model, src, src_mask, max_len=src.shape[1], start_symbol=0)
    print()
    print("输入序列: ", src.tolist())
    print("预测输出:", pred.tolist())
    print("（任务目标是原样抄写，所以理想输出 = 输入序列）")

    img_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "images")
    os.makedirs(img_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(losses) + 1), losses, marker="o")
    ax.set_xlabel("epoch")
    ax.set_ylabel("avg loss")
    ax.set_title("copy task: loss curve")
    fig.tight_layout()
    out = os.path.join(img_dir, "fig_loss_curve.png")
    fig.savefig(out, dpi=120)
    print("图片已保存:", out)
