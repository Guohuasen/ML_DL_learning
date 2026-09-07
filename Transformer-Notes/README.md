# Transformer 源码精读（Attention Is All You Need）

讲解《Attention Is All You Need》的 Transformer PyTorch 源码（基于 harvardnlp/annotated-transformer，MIT 协议）。

## 内容

| 文件 | 说明 |
|---|---|
| `transformer-full-notes.md` | **完整版精读**（宏观 → 逐层 → 细节，含真实运行效果），建议按 第 1 章 → demo04 → 第 4 章 → demo05 的顺序读 |
| `code/the_annotated_transformer.py` | 原始源码（文档中标注的行号以它为准） |
| `code/demos/` | 5 个可运行演示脚本（掩码 / 位置编码 / 注意力 / 形状 / 训练） |
| `images/` | 演示生成的效果图（文档内直接引用） |

## 快速开始

```bash
cd code/demos
python demo01_masks.py     # 掩码长什么样
python demo02_posenc.py    # 位置编码数值与曲线
python demo03_attention.py # 注意力权重
python demo04_model.py     # 全模型形状流水线
python demo05_train.py     # 训练 25 轮 + 推理（约 30-60 秒）
```

需要：Python 3.11+、PyTorch >= 2.0、matplotlib。

## 阅读顺序建议

1. 先读 `transformer-full-notes.md` 第 1 章（宏观总览）；
2. 跑 `demo04_model.py` 亲眼看张量形状；
3. 精读第 4 章（注意力核心）+ 跑 `demo03`；
4. 跑 `demo05_train.py` 看训练日志与损失曲线；
5. 回头补齐第 3/5/6/7 章细节，最后用第 9 章自测。

来源：[harvardnlp/annotated-transformer](https://github.com/harvardnlp/annotated-transformer) | 论文 [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
