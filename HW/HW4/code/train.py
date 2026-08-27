# -*- coding: utf-8 -*-
"""
train.py —— 主程序：训练 + 验证 + 预测 + 生成提交文件
来源：leedl-tutorial HW4_Self-Attention 笔记本，整理并补充注释。

运行前必读：
  1. 依赖安装：pip install torch numpy tqdm tensorboard
  2. 数据：把 ml2022spring-hw4 的数据解压成一个 Dataset 文件夹（内含
     metadata.json / mapping.json / testdata.json / uttr-*.pt），
     然后把下面 config 里的 dataset_dir 改成它的绝对路径。
  3. 训练完会自动用 best 模型预测测试集并写出 submission.csv：
        Id      = 测试文件名（如 uttr-xxx.pt）
        Category= 说话人 id 字符串（如 id00464）
     这是本作业 Kaggle 的标准提交格式（不要写成数字下标/整数类别）。
"""
import os
import json
import csv
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# 本地模块
from utils import set_seed, get_cosine_schedule_with_warmup
from dataset import myDataset, InferenceDataset, collate_batch, inference_collate_batch
from model import Classifier

# ==================== 配置区（所有参数集中在这里） ====================
config = {
    'seed': 87,
    'dataset_dir': '../input/ml2022spring-hw4/Dataset',  # 坑：Kaggle 路径，本地跑必须改成你的目录
    'n_epochs': 35,
    'batch_size': 64,
    'scheduler_flag': True,      # 是否用 warmup+余弦 学习率调度
    'warmup_steps': 1000,        # 前 1000 步预热
    'learning_rate': 1e-3,
    'early_stop': 300,           # 验证集连续多少轮不改善就停
    'n_workers': 8,              # DataLoader 子进程数（Windows 上遇到共享内存报错就改 0）
    'save_path': './models/model.ckpt',
    'submission_path': 'submission.csv',
}


def trainer(train_loader, valid_loader, model, config, device):
    """训练 + 验证 + 早停 + 保存最佳模型（和 HW3 骨架相同，新东西只有 scheduler）"""
    criterion = nn.CrossEntropyLoss()                          # 分类损失（内部=softmax+NLL）
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    if config['scheduler_flag']:
        # 总步数 = 每轮批数 x 轮数（warmup/余弦曲线按"步"走，不是按"轮"走）
        total_steps = len(train_loader) * config['n_epochs']
        scheduler = get_cosine_schedule_with_warmup(optimizer, config['warmup_steps'], total_steps)

    writer = SummaryWriter()                                   # TensorBoard 记录器（可选）
    os.makedirs(os.path.dirname(config['save_path']) or '.', exist_ok=True)

    n_epochs, best_loss, step, early_stop_count = config['n_epochs'], math.inf, 0, 0
    for epoch in range(n_epochs):
        model.train()
        loss_record, train_accs = [], []
        train_pbar = tqdm(train_loader, position=0, leave=True)

        for x, y in train_pbar:
            optimizer.zero_grad()                              # 五连 1
            x, y = x.to(device), y.to(device)
            pred = model(x)                                    # 五连 2
            loss = criterion(pred, y)                          # 五连 3
            loss.backward()                                    # 五连 4
            optimizer.step()                                   # 五连 5
            if config['scheduler_flag']:
                scheduler.step()                               # 每个 batch 后走一步学习率

            step += 1
            acc = (pred.argmax(dim=-1) == y).float().mean()    # 本批准确率
            l_ = loss.detach().item()
            loss_record.append(l_)
            train_accs.append(acc.detach().item())
            # 进度条左侧显示 第几轮，右侧显示 平均loss/acc（肉眼实时看到训练）
            train_pbar.set_description(f'Epoch [{epoch+1}/{n_epochs}]')
            train_pbar.set_postfix({'loss': f'{l_:.5f}', 'acc': f'{acc:.5f}'})

        mean_train_acc = sum(train_accs) / len(train_accs)
        mean_train_loss = sum(loss_record) / len(loss_record)
        writer.add_scalar('Loss/train', mean_train_loss, step)
        writer.add_scalar('ACC/train', mean_train_acc, step)

        # ------- 验证：eval 关 Dropout，no_grad 关梯度记录，两者缺一不可 -------
        model.eval()
        loss_record, valid_accs = [], []
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():                              # 上下文管理器版（只包这段）
                pred = model(x)
                loss = criterion(pred, y)
                acc = (pred.argmax(dim=-1) == y).float().mean()
            loss_record.append(loss.item())
            valid_accs.append(acc.detach().item())

        mean_valid_acc = sum(valid_accs) / len(valid_accs)
        mean_valid_loss = sum(loss_record) / len(loss_record)
        print(f'Epoch [{epoch+1}/{n_epochs}]: Train loss: {mean_train_loss:.4f}, acc: {mean_train_acc:.4f} '
              f'| Valid loss: {mean_valid_loss:.4f}, acc: {mean_valid_acc:.4f}')
        writer.add_scalar('Loss/valid', mean_valid_loss, step)
        writer.add_scalar('ACC/valid', mean_valid_acc, step)

        # 早停：只看验证 loss（换指标记得连变量一起换）
        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(model.state_dict(), config['save_path'])  # 只存权重字典
            print(f'Saving model with loss {best_loss:.3f}...')
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count >= config['early_stop']:
            print('\nModel is not improving, so we halt the training session.')
            return


def predict_and_submit(test_loader, data_dir, config, device):
    """加载最佳模型 -> 预测测试集 -> 写 submission.csv"""
    mapping = json.load(open(os.path.join(data_dir, 'mapping.json'), encoding='utf-8'))

    model_best = Classifier().to(device)
    # map_location=device：防止"GPU 上训的权重在 CPU 上加载报错"
    model_best.load_state_dict(torch.load(config['save_path'], map_location=device))
    model_best.eval()

    pred_id, pred_final_cls = [], []
    with torch.no_grad():                                     # 推理也不需要梯度
        for name, data in tqdm(test_loader, desc='predict'):
            test_pred = model_best(data.to(device))
            test_label = torch.argmax(test_pred.cpu(), dim=1).numpy()  # 600 分里取最大下标
            pred_id += name
            pred_final_cls += [mapping['id2speaker'][str(test_label[0])]]  # 整数 -> 说话人 id

    with open(config['submission_path'], 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Id', 'Category'])                        # 表头
        for i, c in zip(pred_id, pred_final_cls):
            w.writerow([i, c])
    print(f'已生成 {config["submission_path"]}，共 {len(pred_id)} 行')


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_seed(config['seed'])
    print('device:', device)

    data_dir = config['dataset_dir']
    dataset = myDataset(data_dir)                             # 全部训练数据（含标签）
    speaker_num = dataset.get_speaker_number()

    # 90% 训练 / 10% 验证（random_split 返回两个 Subset，只含下标）
    trainlen = int(0.9 * len(dataset))
    trainset, validset = random_split(dataset, [trainlen, len(dataset) - trainlen])
    testset = InferenceDataset(data_dir)                      # 测试集（无标签）

    train_loader = DataLoader(trainset, batch_size=config['batch_size'], shuffle=True,
                              drop_last=True, num_workers=config['n_workers'],
                              pin_memory=True, collate_fn=collate_batch)
    valid_loader = DataLoader(validset, batch_size=config['batch_size'],
                              drop_last=True, num_workers=config['n_workers'],
                              pin_memory=True, collate_fn=collate_batch)
    test_loader = DataLoader(testset, batch_size=1, shuffle=False,
                             drop_last=False, num_workers=config['n_workers'],
                             pin_memory=True, collate_fn=inference_collate_batch)

    model = Classifier(input_dim=40, d_model=80, n_spks=speaker_num, dropout=0.1).to(device)
    trainer(train_loader, valid_loader, model, config, device)
    predict_and_submit(test_loader, data_dir, config, device)


if __name__ == '__main__':
    main()
