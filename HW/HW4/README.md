# HW4 说话人识别（Self-Attention）— leedl 教程版

> 对应课程：李宏毅机器学习（2023 春）HW4 + Datawhale《李宏毅深度学习教程》(leedl-tutorial) HW4_Self-Attention
> 任务：根据 40 维 log-mel 音频特征序列，用 Transformer 编码器做 **600 人说话人分类**（数据集 VoxCeleb2 采样）。

## 目录结构

```
HW/HW4/
├── README.md                                  # 本说明
├── 讲解_HW4_自注意力_leedl教程.md             # 配套讲解（宏观->逐层->细节，含一页纸总结）
└── code/
    ├── utils.py       # 随机种子 set_seed / warmup+余弦学习率 / 模型结构图
    ├── dataset.py     # myDataset、InferenceDataset、collate_batch（变长序列对齐）
    ├── model.py       # Classifier（pre_net -> TransformerEncoderLayer -> 时间池化 -> 分类头）
    └── train.py       # 主程序：训练 + 验证 + 早停 + 预测 + submission.csv
```

## 运行方法

```bash
# 1. 依赖
pip install torch numpy tqdm tensorboard

# 2. 准备数据（二选一）
#    a) Kaggle 下载 ml2022spring-hw4 官方数据并解压成一个 Dataset 文件夹
#    b) 复用你自己已有的 ml2022spring-hw4.zip，解压后得到 Dataset/（内含
#       metadata.json / mapping.json / testdata.json / uttr-*.pt）

# 3. 改路径（重点坑）
#    打开 code/train.py，把 config['dataset_dir'] 改成你的 Dataset 绝对路径
#    （默认值是 Kaggle 的 ../input/...，本地不换一定报错）

# 4. 训练 + 预测 + 提交
cd code
python train.py
# 产出：models/model.ckpt（最佳权重） + submission.csv
```

## 提交格式（防静默错，重要）

```csv
Id,Category
uttr-b52ddeaacf1b42ff9c947eadce3e1966.pt,id03074
...
```

- `Id` = 测试文件名（`uttr-xxx.pt`），不是数字下标；
- `Category` = 说话人 id 字符串（如 `id00464`），不是整数类别；
- 用 `mapping.json` 的 `id2speaker` 把模型输出的整数下标转回字符串。

## Baseline 路线（在 code 上做小改动即可）

| 档位 | 做什么 | 改动位置 |
|---|---|---|
| Easy | 跑通代码，会用 Transformer | 不用改 |
| Medium | 调超参：d_model / nhead / n_layers / lr / warmup_steps | model.py + train.py 的 config |
| Strong | 把 TransformerEncoderLayer 换成 Conformer | model.py 的 TODO 处 |
| Boss | Self-Attention Pooling + Additive Margin Softmax | model.py 的 mean(dim=1) 与损失部分 |

> 详细讲解见同目录下的 `讲解_HW4_自注意力_leedl教程.md`。
