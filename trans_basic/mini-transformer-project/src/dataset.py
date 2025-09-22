import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split

class ToyDataset(Dataset):
    """一个简单的随机序列分类数据集"""
    def __init__(self, X, y):
        # 转成张量（Embedding 和 CrossEntropy 都要求 long 类型）
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_dataloaders(batch_size=32, seq_len=10, vocab_size=20, n_samples=2000):
    """
    生成玩具数据并返回 DataLoader
    - seq_len: 序列长度
    - vocab_size: 词表大小（假设每个位置是 0~vocab_size-1 的整数）
    - n_samples: 样本总数
    """
    # 1. 随机生成序列
    X = np.random.randint(0, vocab_size, size=(n_samples, seq_len))
    # 2. 简单规则生成标签：偶数token数 > 一半 => 标签1，否则0
    y = (np.sum(X % 2 == 0, axis=1) > (seq_len // 2)).astype(np.int64)

    # 3. 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. 构建 Dataset 和 DataLoader
    train_dataset = ToyDataset(X_train, y_train)
    val_dataset = ToyDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader
