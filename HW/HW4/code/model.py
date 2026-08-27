# -*- coding: utf-8 -*-
"""
model.py —— 模型（Transformer 编码器做说话人分类）
来源：leedl-tutorial HW4_Self-Attention 笔记本，整理并补充注释。

shape 流向（务必跟一遍，这是模型的地图）：
  (B, L, 40) --pre_net--> (B, L, 80) --TransformerEncoderLayer--> (B, L, 80)
  --mean(dim=1)--> (B, 80) --pred_layer--> (B, 600)
  B=batch大小, L=补齐后的帧数, 40=mel维, 80=d_model, 600=说话人数
"""
import torch.nn as nn


class Classifier(nn.Module):
    """一个"从头训练"的说话人分类器：升维 -> 自注意力编码 -> 时间池化 -> 分类头"""

    def __init__(self, input_dim=40, d_model=80, n_spks=600, dropout=0.1):
        """
        参数：
          input_dim (int): 输入特征维度（mel 维数，本项目 40）
          d_model (int):   自注意力隐层维度（帧向量的"加工宽度"）
          n_spks (int):    说话人总数（输出类别数，决定最后一层宽度）
          dropout (float): 分类头 Dropout 概率
        """
        super(Classifier, self).__init__()
        # pre_net = pre(前置) + net(网络)：把 40 维帧向量升维到 80 维
        self.pre_net = nn.Linear(input_dim, d_model)

        # 核心：一层"自注意力编码器"（TransformerEncoderLayer 已经把
        #       多头注意力 + 前馈 + 残差 + 层归一化封装成一层了）
        # TODO（Strong baseline）：把这里换成 Conformer（Transformer 的一种变体）
        #    参考论文 https://arxiv.org/abs/2005.08100
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,        # 帧向量维度 80（注意力 Q/K/V 也在这个维度上做）
            dim_feedforward=256,    # 内部前馈层先放大到 256 再缩回 80
            nhead=2,                # 多头注意力头数（必须能整除 d_model：80/2=40）
            batch_first=True,       # 输入顺序 (B, L, D) 而不是默认的 (L, B, D)
            activation="gelu",      # 前馈层激活函数：GELU（ReLU 的平滑版，Transformer 主流）
        )

        # 分类头：把 (B, 80) 的"说话人特征向量"映射成 600 类得分
        self.pred_layer = nn.Sequential(
            nn.Linear(d_model, d_model),   # 80 -> 80（先过渡一层，学到更多非线性）
            nn.ReLU(),
            nn.Dropout(dropout),           # 随机丢 10% 神经元，防过拟合
            nn.Linear(d_model, n_spks),    # 80 -> 600：每类一个得分
        )

    def forward(self, mels):
        """
        输入 mels: (B, L, 40)
        输出 logits: (B, 600) —— 每个样本对 600 位说话人各有一个得分
        """
        out = self.pre_net(mels)           # (B, L, 40) -> (B, L, 80)
        out = self.encoder_layer(out)      # 自注意力：每一帧"看"整段所有帧后更新自己
        #   因为开了 batch_first=True，这里不需要手动 permute/transpose
        #   （等价写法：out.permute(1,0,2) -> encoder -> out.transpose(0,1)）
        # 时间维平均池化：把 L 帧浓缩成 1 个"整段音频"的特征向量
        stats = out.mean(dim=1)            # (B, L, 80) -> (B, 80)
        #   坑：mean 会把 padding 补的 -20 也算进去（短音频被拉低），
        #   这是教程简化版；正规做法是 masked mean（带掩码的平均）
        return self.pred_layer(stats)      # (B, 600)
