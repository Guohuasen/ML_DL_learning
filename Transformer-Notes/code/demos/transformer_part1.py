# -*- coding: utf-8 -*-
"""模型核心（Part 1）：与 the_annotated_transformer.py 的模型部分一一对应。

整理自 harvardnlp/annotated-transformer（MIT License, Alexander Rush 等）。
为便于独立运行，仅依赖 torch；原源码行号见 code/the_annotated_transformer.py。
"""

import math
import copy

import torch
import torch.nn as nn
from torch.nn.functional import log_softmax


def clones(module, N):
    """复制出 N 个相同模块（各自独立参数），用于堆叠 N 层。"""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class EncoderDecoder(nn.Module):
    """总装类：编码器 + 解码器 + 输入嵌入 + 输出头。"""

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        """完整流程：编码 -> 用编码结果解码。"""
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        """编码：源词嵌入 -> N 层编码器。"""
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        """解码：目标词嵌入 -> N 层解码器（读 memory）。"""
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):
    """输出头：线性层把 d_model 映射回词表大小，再取 log-softmax。"""

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return log_softmax(self.proj(x), dim=-1)


class Encoder(nn.Module):
    """编码器 = N 个完全相同的层顺序堆叠 + 末尾一层 LayerNorm。"""

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class LayerNorm(nn.Module):
    """层归一化：对每个 token 的 d_model 维做标准化（减均值除标准差）。
    a_2/b_2 是可学习的缩放(scale)与偏移(shift)。"""

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """残差 + 归一化的"包装盒"：
    注意顺序（源码注释特别说明）：先 norm 再子层，最后加回原输入 x。"""

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    """单层编码器：自注意力 + 前馈网络，各配一个残差包装盒。"""

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        # 第一个包装盒装"自注意力"，q=k=v=同一个 x
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    """解码器 = N 个相同层堆叠 + 末尾 LayerNorm。"""

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    """单层解码器：掩码自注意力 + 交叉注意力（读编码器记忆）+ 前馈。"""

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        # ① 自注意力：只能看已生成的词（tgt_mask 防作弊）
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        # ② 交叉注意力：query 来自解码端，key/value 来自编码器记忆
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        # ③ 前馈网络
        return self.sublayer[2](x, self.feed_forward)


def subsequent_mask(size):
    """生成"禁止看未来"的掩码：允许看到的位置为 True。
    返回形状 (1, size, size)，第 i 行代表第 i 个词能看到哪些位置。"""
    attn_shape = (1, size, size)
    # triu(..., diagonal=1)：严格上三角为 1（未来位置）
    mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return mask == 0


def attention(query, key, value, mask=None, dropout=None):
    """缩放点积注意力（论文核心公式）。
    返回：(加权后的 value, 注意力权重矩阵)。"""
    d_k = query.size(-1)
    # Q·K^T / sqrt(d_k)：每个 query 和所有 key 打分
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        # 被掩码的位置塞一个非常大的负数，softmax 后趋近 0
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    # 按权重把 value 加权求和
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    """多头注意力：把 d_model 切成 h 份分别做注意力，再拼回。"""

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0  # 切分的前提：能整除
        self.d_k = d_model // h
        self.h = h
        # 4 个线性层：Q/K/V 的投影 + 输出投影（W_Q W_K W_V W_O）
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None  # 保存最近一次的注意力权重，方便可视化
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # 掩码加一维，好和 (b, h, q_len, k_len) 的分数广播对齐
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 对 Q/K/V 分别线性投影，切成 h 头：
        # (b, len, d_model) -> (b, len, h, d_k) -> (b, h, len, d_k)
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 多头并行做缩放点积注意力
        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        # 把 h 个头拼回一个 d_model 维
        x = (
            x.transpose(1, 2)
            .contiguous()  # view 前必须 contiguous，否则报错
            .view(nbatches, -1, self.h * self.d_k)
        )
        # 最后一个线性层做输出投影
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    """逐位置前馈网络：d_model -> d_ff -> d_model（中间放大，论文 d_ff=2048）。"""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))


class Embeddings(nn.Module):
    """词嵌入：词序号 -> d_model 维向量，并乘 sqrt(d_model) 放大（论文做法）。"""

    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """位置编码：固定 sin/cos 公式给每个位置生成 d_model 维"坐标"。
    偶数维用 sin、奇数维用 cos，频率随维数升高而降低。"""

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)  # (max_len, 1)
        # 频率因子：论文公式里的 1/10000^(2i/d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维 sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维 cos
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)，与 batch 广播
        self.register_buffer("pe", pe)  # 随模型保存/搬移，不参与梯度

    def forward(self, x):
        # x: (batch, len, d_model)，按当前序列长度截取 pe 加上去
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


def make_model(
    src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1
):
    """总装厂：从超参数一次性构建完整模型。"""
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab),
    )

    # 论文用 Xavier 均匀初始化（只初始化矩阵类参数）
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
