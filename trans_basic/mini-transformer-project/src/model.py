# src/model.py
import torch.nn as nn

class MiniTransformer(nn.Module):
    def __init__(self, vocab_size=20, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        # 1) embedding：把整数token变成向量
        self.embed = nn.Embedding(vocab_size, d_model)

        # 2) transformer encoder：捕捉序列依赖
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True  # 保证输入输出形状是 (batch, seq_len, d_model)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 3) 池化：把整个序列的信息聚合成一个向量
        self.pool = nn.AdaptiveAvgPool1d(1)

        # 4) 分类器：输出类别概率
        self.fc = nn.Linear(d_model, 2)  # 假设二分类

    def forward(self, x):
        """
        x: (batch, seq_len) 的整数索引
        """
        x = self.embed(x)  # (B, L, d_model)
        x = self.encoder(x)  # (B, L, d_model)
        x = x.transpose(1, 2)  # (B, d_model, L)，为池化准备
        x = self.pool(x).squeeze(-1)  # (B, d_model)
        return self.fc(x)  # (B, num_classes)
