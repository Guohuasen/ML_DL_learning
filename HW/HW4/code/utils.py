# -*- coding: utf-8 -*-
"""
utils.py —— 工具函数（随机种子 / 学习率调度 / 模型结构图）
来源：leedl-tutorial HW4_Self-Attention 笔记本，整理并补充注释。
"""
import os
import math
import random

import numpy as np
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def set_seed(seed=87):
    """
    固定所有"随机数发生器"（保证每次运行结果可复现）。

    为什么有这么多 seed？
    神经网络里有 4 个"独立掷骰子"的地方，各自用自己的随机数发生器：
      1. Python random     -> 数据打乱 / 随机切帧（本项目的 start=random.randint 依赖它）
      2. numpy np.random   -> 预处理中可能的随机操作
      3. torch CPU / GPU   -> 权重初始化、dropout 等
      4. cuDNN (GPU 加速库) -> 卷积/注意力底层算法选择
    只有 4 个骰子都固定，两次运行才完全一致。

    参数：
      seed (int): 随机种子数字
    返回值：无（只做"播种"副作用）
    """
    np.random.seed(seed)                     # numpy 的随机数发生器
    random.seed(seed)                        # Python 内置 random 模块
    torch.manual_seed(seed)                  # PyTorch CPU 的随机数发生器
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)     # GPU 上所有设备
        torch.cuda.manual_seed(seed)         # 当前 GPU 设备（与上行重复，保留无妨）
    # python 全局：固定字符串哈希随机化（影响 dict/set 遍历顺序）
    os.environ["PYTHONHASHSEED"] = str(seed)
    # cuDNN 三件套：确定性算法 + 关闭自动选最快算法
    #   deterministic=True  -> 同一输入一定产生同一输出（可复现）
    #   benchmark=False     -> 不要"试跑各算法挑最快的"（试跑自带随机性）
    #   enabled=False       -> 直接关闭 cuDNN（最彻底、也最慢；教程为可复现选择关闭）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    print(f"Set env random_seed = {seed}")


def get_cosine_schedule_with_warmup(opt, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    """
    生成"warmup + 余弦衰减"的学习率调度器（Transformer 训练的标准配方）。

    为什么 Transformer 需要 warmup？
      CNN 训练多数直接用小学习率就稳；但 Transformer 层数深、每个 block 内有
      LayerNorm 和残差连接，训练初期注意力权重很乱，一上来大学习率容易震荡/发散。
      标准做法：前 num_warmup_steps 步，学习率从 0 线性升到优化器设定值；
      之后按余弦曲线从 1 平滑衰减到接近 0。

    参数：
      opt (Optimizer): 优化器（调度器绑定在它上面改学习率）
      num_warmup_steps (int): 预热多少步（本项目 1000）
      num_training_steps (int): 总共训练多少步（余弦衰减到哪）
      num_cycles (float): 余弦周期数，默认 0.5
      last_epoch (int): 从第几步开始，默认 -1（从头开始）
    返回值：一个 LambdaLR 调度器对象

    原理：LambdaLR 每走一步调用 lr_lambda(step)，
          真实学习率 = 优化器设定的 lr x lr_lambda(step)。
    """
    def lr_lambda(current_step):
        # 预热阶段：倍数从 0 线性涨到 1（即学习率 0 -> 设定值）
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # 衰减阶段：progress 从 0 走到 1，cos 曲线让倍数从 1 平滑降到 0
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(opt, lr_lambda, last_epoch)


# 可选依赖：torchviz 没有也能跑其它功能，这里做个安全导入
try:
    from torchviz import make_dot
except ImportError:  # 没装 torchviz 时 model_plot 不可用，但不影响训练
    def make_dot(*args, **kwargs):
        raise RuntimeError("未安装 torchviz，请先执行 !pip install torchviz")


def model_plot(model_class, input_sample):
    """
    把模型画成一张"结构图"（依赖 torchviz，安装：!pip install torchviz）。
    功能：可视化网络每一层输入输出的 shape 流向，适合核对模型结构。

    用法（notebook 中）：
        x = torch.randn(1, 100, 40)
        model_plot(Classifier, x)   # 传入的是"类"不是实例

    参数：
      model_class: 模型类（不传实例）
      input_sample: 示例输入张量（形状要合法）
    返回值：graphviz 图形对象（notebook 里直接显示出来）
    """
    clf = model_class()                        # 实例化模型（用默认参数）
    y = clf(input_sample)                      # 前向传播一次（顺带得到各层输出）
    clf_view = make_dot(y, params=dict(list(clf.named_parameters()) + [("x", input_sample)]))
    return clf_view
