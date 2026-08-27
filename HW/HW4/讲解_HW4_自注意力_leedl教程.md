# 讲解：HW4 说话人识别 · Self-Attention（leedl 教程版）

> 配套代码：`code/` 目录（utils.py / dataset.py / model.py / train.py）
> 来源：Datawhale《李宏毅深度学习教程》`HW4_Self-Attention.ipynb`
> 讲解方式：宏观 → 逐层 → 细节；重要内容已用 🔴 标出。

## 📖 时间有限导读

如果只有 1 小时，按这个顺序看：
1. **看数据**（cell 9）→ **Dataset**（cell 10）→ **模型**（cell 12）→ **collate**（cell 22）→ **学习率**（cell 15/16）→ **训练**（cell 18，和 HW3 相同扫一眼）。
2. 其余（导入、config、提交）快速浏览即可。

---

## 一、宏观（先建立地图）

### 1.1 这份 notebook 在干什么

```
数据流（和 HW3 同一副骨架，只换了"输入"和"模型"）：
音频特征 .pt(帧,40) -> Dataset 切/补帧 -> collate 对齐成长度 -> (B,帧,40)
  -> pre_net 升维 -> Transformer 编码器(自注意力) -> 时间维平均 -> 分类头 -> 600 人得分
  -> CrossEntropyLoss -> 训练五连（和 HW3 一字不差）
```

一句话：**HW3 是"图 + CNN"，这份是"音频帧序列 + Transformer"，骨架完全一样**。

### 1.2 和 HW3 的对比

| 环节 | HW3（图像分类） | HW4（音频分类） |
|---|---|---|
| 输入 | 图片 `(B,3,128,128)` | 音频特征序列 `(B,L,40)` |
| 输入形状 | 定长（Resize 统一） | **变长**（91~7193 帧不等） |
| 特征提取 | CNN（卷积） | **TransformerEncoderLayer（自注意力）** |
| 池化 | MaxPool2d | `mean(dim=1)` 时间维平均 |
| 输出 | 11 类食物 | 600 位说话人 |
| 训练 | 五连 + 早停 + 存 best | 完全相同，**多了 warmup 学习率** |

### 1.3 三个"新学费"（重点）

| 新东西 | 在哪 | 解决什么问题 |
|---|---|---|
| **变长序列怎么喂模型** | dataset.py（segment + `pad_sequence`） | 每条音频帧数不一样，得对齐成一个 batch |
| **Transformer 编码器怎么用** | model.py（`TransformerEncoderLayer`） | 自注意力：序列里每一帧"和所有帧互相看" |
| **Transformer 专属学习率预热 warmup** | utils.py | CNN 直接大学习率没事，Transformer 初期容易崩，要从小往大升 |

### 1.4 各 cell 的"票价"（重要度标记）

| Cell | 内容 | 重要度 |
|---|---|---|
| 0 | 任务说明（Easy/Medium/Strong/Boss） | 🟢 扫一眼 |
| 2~3 | 装包、import | ⚪ 知道作用即可 |
| 4 | 下载数据 | ⚪ 你已有数据 |
| 6~7 | `set_seed` 随机种子 | 🔴 必看 |
| 9 | 查看数据 | 🔴 必看（上手先看数据） |
| 10 | `myDataset` / `InferenceDataset` | 🔴 必看 |
| 12 | `Classifier`（Transformer） | 🔴 必看中的必看 |
| 15~16 | warmup 学习率 | 🔴 必看 |
| 18 | `trainer` 训练 | 🟢 和 HW3 相同，只讲差异 |
| 20 | config 参数 | 🟢 参数表 |
| 22 | `collate_batch` + `random_split` | 🔴 必看 |
| 24、26~27 | 训练/预测/提交 | 🟢 提交格式注意 1 处 |

---

## 二、逐层讲解

### 2.1 ⚪ 加载包（cell 2~3）—— 只认新面孔

```python
import torch.nn as nn            # 神经网络积木库
from torch.utils.data import Dataset, random_split, DataLoader  # 模板/随机切分/装载器
from torch.nn.utils.rnn import pad_sequence   # ⚪新：把不同长度序列"补齐成一样长"
from torch.optim.lr_scheduler import LambdaLR # ⚪新：按步自动改学习率的调度器
from torch.utils.tensorboard import SummaryWriter  # ⚪新：TensorBoard 指标记录
from torchviz import make_dot    # ⚪新：把模型画成结构图
from rich.console import Console # ⚪新：彩色打印
import warnings; warnings.filterwarnings('ignore')  # 忽略警告（学习期可开，正式别开）
```

> 💡 英文拆义：`pad_sequence` = pad(填充)+sequence(序列)；`random_split` = 随机切分；`SummaryWriter` = summary(统计)+writer(写入者)。
> ⚠️ 坑：`from torch import functional as F` 是笔记本里的残留垃圾行（从没用过），可删。

### 2.2 🔴 `set_seed`（cell 6~7）—— 从原理推出"该设哪些种子"

```python
def set_seed(seed=87):
    np.random.seed(seed)                     # numpy 的随机数发生器
    random.seed(seed)                        # Python 内置 random 模块
    torch.manual_seed(seed)                  # PyTorch CPU 的随机数发生器
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)     # GPU 上所有设备
        torch.cuda.manual_seed(seed)         # 当前 GPU 设备（与上行重复）
    os.environ["PYTHONHASHSEED"] = str(seed) # 固定字符串哈希随机化
    torch.backends.cudnn.deterministic = True  # 确定性算法
    torch.backends.cudnn.benchmark = False     # 不自动挑最快算法
    torch.backends.cudnn.enabled = False       # 直接关 cuDNN（最彻底也最慢）
    print(f"Set env random_seed = {seed}")
```

**为什么需要这么多 seed？（不要再死记，从原理推）**

> 神经网络里有 4 个"独立掷骰子"的地方，各用各的随机数发生器：
> 1. **Python `random`** → 数据打乱 / 随机切帧（`start = random.randint(...)` 依赖它）；
> 2. **numpy `np.random`** → 预处理中的随机操作；
> 3. **torch CPU / GPU** → 权重初始化、dropout 等；
> 4. **cuDNN（GPU 加速库）** → 卷积/注意力底层算法选取。
>
> 只有 4 个骰子都固定，两次运行才完全一致。`PYTHONHASHSEED` 固定字符串哈希（影响 dict 遍历顺序）；`deterministic=True` 保证 cuDNN 确定性，`benchmark=False` 防止"试跑选最快算法"（自带随机性），`enabled=False` 是关到最彻底（学习期可接受，代价是慢一点）。

屏幕效果：运行打印 `Set env random_seed = 87`。

### 2.3 🔴 查看数据（cell 9）—— 动手前第一步永远是看数据

```python
data_dir = '../input/ml2022spring-hw4/Dataset'   # 坑：Kaggle 路径，本地要改
map_js = json.load((Path(data_dir) / 'mapping.json').open())   # Path 用 / 拼路径
matedata_js = json.load((Path(data_dir) / 'metadata.json').open())
mel = torch.load(Path(data_dir) / 'uttr-xxx.pt')              # 读回 (L,40) 张量
```

会在屏幕看到：

```
mapping.json | keys =  dict_keys(['speaker2id', 'id2speaker'])
metadata.json | keys =  dict_keys(['n_mels', 'speakers'])
n_mels = 40
speakers['id00559'][:5] = [{'feature_path': 'uttr-...pt', 'mel_len': 435}, ...]
mel.shape = torch.Size([435, 40])
```

> 🎯 这就是"动手前先打印 shape"技能：`(435, 40)` = 435 帧 × 40 维，和 json 里 `mel_len=435` 对上了——数据闭环建立。

### 2.4 🔴 `myDataset` / `InferenceDataset`（cell 10）—— 核心中的核心

**三件套**（PyTorch `Dataset` 模板的契约）：
- `__init__`：读 json、把嵌套结构"拍平"成一张样本表 `[[特征文件名, 标签], ...]`；
- `__len__`：`len(dataset)` 自动调用，返回样本总数；
- `__getitem__(index)`：`dataset[i]` 自动调用，返回第 i 条 `(特征, 标签)`。

**逐点拆**：
- **"拍平"嵌套 json**：metadata 是 `{'说话人': [{feature_path, mel_len}, ...]}`，训练要的是顺序可索引的列表——双循环 `for speaker, utt in metadata.items(): for utt_i in utt: self.data.append([...])`；
- **标签从哪来**：`self.speaker2id[speaker]` 查表（说话人 id 字符串 → 整数编号），和 HW3"从文件名解析标签"同理；
- **`feat_path, speaker = ...` 解包赋值**：一行把列表拆成两个变量；
- **`random.randint(0, len(mel) - segment_len)` 随机切 128 帧**：这就是"时序增强"——每次取样本随机换一段，等于一段音频产生多个样本（和图像裁剪同理）；⚠️ 注意验证集也用了同一个随机的 Dataset（教程的小瑕疵，官方版验证固定从 0 开始），不影响大方向；
- **标签转 `torch.FloatTensor([speaker]).long()`**：CrossEntropyLoss 要求 long 类型；
- **`InferenceDataset`**：测试集无标签，且要保留文件名（提交 Id 用）；
- **`zip(*batch)`**：把 `[(名1,张1),(名2,张2)]` 还原成两组；`torch.stack` 沿新维度叠成 `(B,L,40)`（要求形状一致，所以测试 batch_size=1）。

### 2.5 🔴 `collate_batch` + `random_split`（cell 22）—— 变长序列怎么对齐

```python
def collate_batch(batch):
    mel, speaker = zip(*batch)
    # 关键：批内长度不一，用 pad_sequence 补到最长；batch 维放最前
    mel = pad_sequence(mel, batch_first=True, padding_value=-20)
    return mel, torch.FloatTensor(speaker).long()
```

**`pad_sequence`**：功能=把一批长度不等的张量按最长补齐再叠成 3 维；参数 `batch_first=True`（batch 维在前）、`padding_value=-20`（填充值）；返回 `(B, 最长L, 40)`。

**为什么填充 `-20`？** 特征是 log-mel（取了对数），值域约 `[-20.7, 7.5]`——`-20` 相当于"静音"，比补 0 更合理。

肉眼验证（notebook cell 21 的例子）：

```python
from torch.nn.utils.rnn import pad_sequence
a = torch.ones(25, 40); b = torch.ones(22, 40); c = torch.ones(15, 40)
pad_sequence([a, b, c], batch_first=True).size()
# 输出：torch.Size([3, 25, 40])   （三条都补到 25 帧）
```

**`random_split`**：按比例把整个 dataset 随机切一段，返回两个"只含下标的 Subset"；`trainset, validset = random_split(dataset, [trainlen, len(dataset)-trainlen])`。
⚠️ 坑：随机切分可能把同一说话人的音频同时分进训练/验证（严谨做法按说话人划），教程简化了，学习无妨。

**DataLoader 多出的参数**：`drop_last=True`（丢弃最后一个不满批的）、`num_workers=8`（多进程读数据）、`collate_fn=collate_batch`（自定义装车方式——变长数据必须指定）。

### 2.6 🔴 `Classifier`（cell 12）—— 最重要的一页

**shape 流向（跟着走一遍）**：

| 步骤 | 输入 | 操作 | 输出 |
|---|---|---|---|
| 输入 | `(B,L,40)` | — | — |
| pre_net | `(B,L,40)` | `nn.Linear(40,80)` | `(B,L,80)` |
| encoder_layer | `(B,L,80)` | 自注意力 | `(B,L,80)` |
| mean(dim=1) | `(B,L,80)` | 时间维平均 | `(B,80)` |
| pred_layer | `(B,80)` | 两层 Linear | `(B,600)` |

**必懂点**：
1. **自注意力在干嘛**：想象开会时每个人发言前都看看全屋所有人的脸再反应。第 t 帧先和所有帧（含自己）算相似度（Q·K），再用相似度做权重汇总别人信息（乘 V）——参数 Q/K/V 是模型自己学出来的；
2. **`batch_first=True`**：老接口默认输入 `(L,B,D)`；开了它直接喂 `(B,L,D)`。和手动 `permute(1,0,2) → encoder → transpose(0,1)` 等价（内部封装）——你之前的 hw4_my_zip.py 就是手动版，这份是开关版；
3. **`activation="gelu"`**：GELU 是 ReLU 的"平滑版"，Transformer 家族主流默认；
4. **`mean(dim=1)` 时间池化**：L 帧浓缩成 1 个"整段音频"向量。注意它会把 padding 的 -20 也平均进去（短音频被拉低）——教程简化，正规做法是 masked mean；
5. **`nhead` 必须能整除 `d_model`**（80/2=40 刚好）。

### 2.7 🔴 学习率 warmup（cell 15~16）—— Transformer 专属技能

**为什么需要 warmup？** CNN 用小学习率就稳；Transformer 层数深、初期注意力权重乱，上来就大学习率容易震荡/发散。标准做法：前 `warmup_steps` 步 lr 从 0 线性升到设定值，之后余弦曲线平滑衰减到接近 0（BERT 时代传下来的配方）。

**`LambdaLR` 原理**：`真实学习率 = 优化器设定的 lr × lr_lambda(step)`：
- warmup 期：`lr_lambda = step / warmup_steps`（0→1）；
- 衰减期：`0.5 * (1 + cos(π·progress))`（1→0 的余弦曲线）。

**两个必知**：
1. `scheduler.step()` 放在**每个 batch 后**（不是每个 epoch 后）——warmup 按"步"调度，和 `ReduceLROnPlateau`（按 epoch 看验证集）完全不同；
2. **总步数 = `len(train_loader) * n_epochs`**——余弦衰减终点依赖它，算错会导致学习率衰减过快/过慢。

屏幕效果（cell 15 运行后）：一张曲线图——先陡峭爬升（0→最大），再平滑衰减到接近贴地。

### 2.8 🟢 `trainer`（cell 18）—— 和 HW3 几乎一样，只看差异

| 差异 | 说明 |
|---|---|
| `scheduler.step()` 在每个 batch 后 | warmup 按"步"调度 |
| 早停看的是 **valid loss** 而非 acc | `mean_valid_loss < best_loss`（指标习惯不同，本质相同） |
| `set_description` / `set_postfix` | 进度条左侧显示 `Epoch [1/35]`，右侧显示 loss/acc |
| `SummaryWriter` 写 TensorBoard | 训练完 `tensorboard --logdir runs` 浏览器看曲线（工程习惯，非必须） |

**`@torch.no_grad()` 装饰器 vs `with torch.no_grad():` 上下文**（同一功能的两种写法）：

| | `@torch.no_grad()` | `with torch.no_grad():` |
|---|---|---|
| 作用范围 | **整个函数** | **只包住的代码块** |
| 适用 | 整个函数只做推理/验证 | 函数前半段要梯度、后半段不要 |
| 本质 | 都是进入 no_grad 推断模式，跳过 autograd 图 | 同左 |

别用错：**`model.eval()` + `no_grad` 是一对"关训练开关"**——eval 关 Dropout/BatchNorm 统计，no_grad 关梯度记录，两个都要做。

### 2.9 🟢 config（cell 20）—— 一张表看完

```python
config = {
    'seed': 87, 'dataset_dir': '../input/ml2022spring-hw4/Dataset',  # 坑：本地必须改
    'n_epochs': 35, 'batch_size': 64,
    'scheduler_flag': True, 'warmup_steps': 1000,      # warmup 开关 + 预热步数
    'learning_rate': 1e-3, 'early_stop': 300,          # 学习率 + 早停轮数(实为epoch)
    'n_workers': 8, 'save_path': './models/model.ckpt'
    # 注意：'valid_steps': 2000 是残留配置，trainer 根本没用到
}
```

> ⚠️ 最大的坑：`dataset_dir` 是 Kaggle 的 `/input/...` 路径，本机跑必须改成你的数据目录，否则第一步就 `FileNotFoundError`。

### 2.10 🟢 训练 + 提交（cell 24、26~27）—— 只有 1 个重点

```python
model_best = Classifier().to(device)                          # 重建"结构完全相同"的模型
model_best.load_state_dict(torch.load(config['save_path']))
...
test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)  # 600 分里取最大下标
pred_final_cls += [mapping["id2speaker"][str(test_label[0])]] # 整数 -> 说话人 id 字符串
```

**提交格式（防静默错，和 Hoper-J 0.833 notebook 一致）**：
- `Id` = 测试文件名（`uttr-xxx.pt`），**不是数字下标**；
- `Category` = 说话人 id 字符串（`id00464` 之类），**不是整数类别**；
- 用 `mapping.json` 的 `id2speaker` 把整数下标转回字符串。

⚠️ 小坑：`torch.load` 没写 `map_location` 时，GPU 训练的权重在 CPU 加载会报错；整理版代码已加 `map_location=device`。

---

## 三、细节：语法小抄 + 易混点 + 常见坑

### 3.1 本 notebook 出现的高级语法（一页小抄）

| 语法 | 出处 | 一句话解释 |
|---|---|---|
| 解包赋值 `a, b = [x, y]` | `feat_path, speaker = ...` | 右侧列表拆开依次给左侧两个变量 |
| 星号解包 `zip(*batch)` | collate 两处 | `*` 把列表展开成多个参数传给 zip |
| `Path / "x"` | `Path(data_dir) / 'mapping.json'` | pathlib 重载了 `/` 运算符 = 拼接路径 |
| `json.load(f)` | 多处 | 从**文件对象**读 JSON 成字典 |
| `nn.Sequential(...)` | pred_layer | 把多个层"排队"顺序执行 |
| 嵌套循环展平 | dataset `__init__` | 把嵌套字典转成一张扁平的样本表 |
| `with torch.no_grad():` | trainer / 推理 | 上下文管理器：这一段不记梯度 |
| `@torch.no_grad()` | （你自己脚本里） | 装饰器：整个函数不记梯度 |
| 魔法方法 | `__init__/__len__/__getitem__` | 让类支持 `len()` 和 `[]` |
| `pad_sequence` | collate | 变长序列补到一样长再叠 |
| `random_split` | cell 22 | 按长度比例随机切 dataset |
| `LambdaLR` | utils.py | 按"步"改学习率的调度器（配 lambda 函数） |

### 3.2 易混概念对照表

| 概念 | 区分点 |
|---|---|
| `model.train()` vs `model.eval()` | 开关 Dropout / BatchNorm 统计模式 |
| `no_grad` vs `zero_grad()` | no_grad=不建计算图（省钱）；zero_grad=清掉上一次的梯度 |
| `torch.stack` vs `pad_sequence` | 等长→stack；不等长→pad 补齐再叠 |
| 按 epoch 调 lr vs 按 step 调 | ReduceLROnPlateau=看验证集；LambdaLR/warmup=每 batch 无脑调 |
| `shuffle=True` vs `False` | 训练打乱防顺序记忆；验证/测试必须 False（保顺序对齐 Id） |
| `@torch.no_grad()` vs `with ...` | 整个函数 vs 局部代码块（见 2.8） |

### 3.3 常见坑清单

1. **路径坑**：`../input/...` 是 Kaggle 专属；本机跑必改 `dataset_dir`；
2. **`torch.load` 默认设备**：GPU 训的权重 CPU 加载会报错 → 加 `map_location=device`；
3. **变长批次忘传 `collate_fn`**：报 `stack expected each tensor to be equal size`；
4. **早停看 loss vs acc**：换判据记得连变量一起换（`best_loss`/`best_acc` 两套）；
5. **`mean(dim=1)` 把 padding 也算进去**：短音频被 `-20` 拉低，正规做法 masked mean；
6. **`nhead` 必须整除 `d_model`**（80/2、80/4 都可以，改成奇数会报错）；
7. **`scheduler.step()` 位置**：warmup 调度器放每个 batch 后，别放 epoch 后；
8. **Windows 上 `num_workers>0` 可能报共享内存错** → 改 0。

---

## 四、一页纸总结

| 模块 | 一句话 | 关键概念 | 重要度 |
|---|---|---|---|
| set_seed | 每个随机源单独设种 + cudnn 确定性 | 4 类发生器 + PYTHONHASHSEED | 🔴 |
| 看数据 | 动手前先打印 json 结构和 .pt shape | 数据闭环 | 🔴 |
| myDataset | json 拍平成样本表 + 随机切 128 帧 | 魔法方法/解包/时序增强 | 🔴 |
| InferenceDataset | 只读测试特征、保留文件名 | 预测顺序 = Id 顺序 | 🔴 |
| collate_batch | `pad_sequence` 补到等长（-20 静音值） | 变长序列进 batch | 🔴 |
| Classifier | pre_net→TransformerEncoder→mean→分类头 | 自注意力/batch_first/时间池化 | 🔴 |
| warmup lr | 先 0→lr 线性升，再余弦衰减，每 batch 步进 | LambdaLR / warmup 原理 | 🔴 |
| trainer | 和 HW3 相同的五连+早停+存 best | 只多了 scheduler.step 位置 | 🟢 |
| 提交 | Id=文件名，Category=id2speaker 字符串 | 防静默错 | 🟢 |
| config | 所有参数集中一处 | 路径是最大坑 | 🟢 |

## 💬 一分钟回顾（合上眼睛说一遍）

> "HW4 是把**变长的音频特征序列**送进 **Transformer 编码器**做 600 人分类。数据端：`myDataset` 每次从音频里随机切 128 帧（时序增强），`collate_batch` 用 `pad_sequence` 把一批评到一样长；模型端：`pre_net` 升维 → 一层 `TransformerEncoderLayer`（自注意力）→ `mean(dim=1)` 时间池化 → 分类头；训练端：和 HW3 完全一样的五连+早停，唯一新东西是 **warmup+余弦学习率**（前 1000 步从 0 升到 1e-3，之后余弦衰减，每 batch 走一步）；提交端：Id 用文件名、Category 用说话人 id 字符串。三个新学费 = 变长对齐、自注意力、warmup。"

---

## 附：notebook cell 与整理后代码的对应

| 原 notebook cell | 整理后文件 |
|---|---|
| cell 6~7（all_seed） | `code/utils.py` 的 `set_seed` |
| cell 15~16（学习率） | `code/utils.py` 的 `get_cosine_schedule_with_warmup` |
| cell 10（Dataset） | `code/dataset.py` |
| cell 22（collate / loaders） | `code/dataset.py` + `code/train.py` 的 main |
| cell 12（模型） | `code/model.py` |
| cell 18（trainer） | `code/train.py` 的 `trainer` |
| cell 20（config） | `code/train.py` 的 `config` |
| cell 24、26~27（训练+提交） | `code/train.py` 的 `main` + `predict_and_submit` |
