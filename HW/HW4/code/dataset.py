# -*- coding: utf-8 -*-
"""
dataset.py —— 数据集与批量打包（变长音频特征怎么进模型）
来源：leedl-tutorial HW4_Self-Attention 笔记本，整理并补充注释。

数据格式（VoxCeleb2 预处理后）：
  metadata.json -> {"n_mels": 40, "speakers": {"id00559": [{"feature_path": "uttr-xxx.pt", "mel_len": 435}, ...]}}
  mapping.json  -> {"speaker2id": {"id00559": 1, ...}, "id2speaker": {"1": "id00559", ...}}
  testdata.json -> {"n_mels": 40, "utterances": [{"feature_path": "uttr-xxx.pt", "mel_len": 813}, ...]}
  每个 uttr-xxx.pt 是一个 (帧数, 40) 的 float32 张量（log-mel 声学特征）
"""
import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class myDataset(Dataset):
    """
    训练/验证数据集。Dataset 是 PyTorch 的"数据集模板"，
    只要实现三个魔法方法，DataLoader 就知道怎么把数据一批批喂给模型：
      __init__    初始化：读 json、把所有音频拍平成一张"样本表"
      __len__     len(dataset) 返回样本总数
      __getitem__ dataset[i]   返回第 i 条 (特征, 标签)
    """

    def __init__(self, data_dir, segment_len=128):
        super(myDataset, self).__init__()
        self.data_dir = data_dir
        self.segment_len = segment_len          # 每段音频切多少帧（默认 128）

        # 1) 读 mapping.json：说话人 id 字符串 -> 整数编号（做标签用）
        mapping = json.load((Path(data_dir) / "mapping.json").open())
        self.speaker2id = mapping["speaker2id"]

        # 2) 读 metadata.json 里的 speakers 字典
        metadata = json.load((Path(data_dir) / "metadata.json").open())["speakers"]
        self.speaker_num = len(metadata.keys())  # 说话人总数（600）

        # 3) 把嵌套结构"拍平"成样本表：[[特征文件名, 标签编号], ...]
        #    json 是嵌套字典，训练要的是顺序可索引的列表，所以展平
        self.data = []
        for speaker, utt in metadata.items():    # speaker='id00559', utt=[{feat,mel_len},...]
            for utt_i in utt:
                self.data.append([utt_i["feature_path"], self.speaker2id[speaker]])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # 解包赋值：从 [文件名, 标签] 里一行拆出两个变量
        feat_path, speaker = self.data[index]
        mel = torch.load(Path(self.data_dir) / feat_path)   # (L, 40)
        # 随机切 128 帧：这就是"时序增强"——每次取样本都随机换一段，
        # 相当于同一段音频产生多个训练样本（和图像裁剪/翻转同理）
        if len(mel) > self.segment_len:
            start = random.randint(0, len(mel) - self.segment_len)  # 随机起点
            mel = torch.FloatTensor(mel[start: start + self.segment_len])
        else:
            mel = torch.FloatTensor(mel)         # 不够长就不切（collate 会补齐）
        # 标签转成 int64 张量：CrossEntropyLoss 要求标签是 long 类型
        speaker = torch.FloatTensor([speaker]).long()
        return mel, speaker

    def get_speaker_number(self):
        """工具方法：告诉外面说话人总数"""
        return self.speaker_num


class InferenceDataset(Dataset):
    """测试集：只有特征、没有标签；并且要保留文件名（提交时 Id 用）"""

    def __init__(self, data_dir):
        super(InferenceDataset, self).__init__()
        metadata = json.load((Path(data_dir) / "testdata.json").open())
        self.data_dir = data_dir
        self.data = metadata["utterances"]       # 测试样本列表

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        utt = self.data[index]
        return utt["feature_path"], torch.load(Path(self.data_dir) / utt["feature_path"])


def collate_batch(batch):
    """
    DataLoader 的"装车方式"（collate_fn 参数指定的函数）。
    功能：把一批 (特征, 标签) 合并成一个 batch 张量。
    为什么需要自定义？每条音频长度不一样，默认打包会直接报错，
    必须用 pad_sequence 把一批评到"最长的那条"的长度。
    """
    mel, speaker = zip(*batch)                   # 星号解包：拆成特征列表 + 标签列表
    # pad_sequence = pad(填充)+sequence(序列)：
    #   把 (batch) 个长度不一的张量补齐后叠成 3 维；
    #   batch_first=True -> 形状 (B, 最长L, 40)；padding_value=-20 是填充值
    #   填充 -20 的原因：特征是 log-mel（取了对数），值域约 [-20.7, 7.5]，
    #   -20 相当于"静音"，比补 0 更合理
    mel = pad_sequence(mel, batch_first=True, padding_value=-20)
    return mel, torch.FloatTensor(speaker).long()


def inference_collate_batch(batch):
    """
    测试集专用打包：测试时 batch_size=1，每条长度本来就可能不同，
    这里只做 zip 拆包 + torch.stack。
    注意：torch.stack 要求形状完全一致，所以测试集保持 batch_size=1；
    以后想加大测试 batch，就要用 pad_sequence 而不是 stack。
    """
    feat_path, mels = zip(*batch)                # 还原成 (文件名列表) 和 (张量列表)
    return feat_path, torch.stack(mels)          # (B, L, 40)
