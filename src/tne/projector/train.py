"""
TNE-Core 阶段二 · Projector 训练脚本（torch，需 GPU）

在 GPU 环境运行：用真实标注的 (hidden_state, 九维标签) 对训练 MLP。

注意：
  - 真实 hidden state 来自 vLLM/TGI 抽取（见 hidden_state_protocol.py）
  - 真实标签需要人工标注（5000-10000 条）
  - 本脚本在无 GPU 环境不运行，仅作为 GPU 环境的训练入口
"""
from __future__ import annotations
import sys, os, json, argparse
import torch
import torch.nn as nn
import torch.optim as optim
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


class TNEProjectorModel(nn.Module):
    """真正的 torch 版 Projector：共享表示层 + 9 个三分类头。"""

    def __init__(self, input_dim: int = 4096, hidden_dim: int = 512,
                 num_dims: int = 9):
        super().__init__()
        self.num_dims = num_dims
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        # 9 个三分类头（每维 -1/0/+1），共享表示层建模维度关联
        self.heads = nn.ModuleList(
            [nn.Linear(hidden_dim, 3) for _ in range(num_dims)]
        )

    def forward(self, x: torch.Tensor):
        """
        x: (batch, input_dim)
        Returns: (batch, num_dims, 3) logits
        """
        h = self.shared(x)
        logits = torch.stack([head(h) for head in self.heads], dim=1)
        return logits


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # 数据加载（由 data_gen / 真实标注提供）
    # 这里只定义训练循环，数据由外部准备
    model = TNEProjectorModel(
        input_dim=args.input_dim, hidden_dim=args.hidden_dim, num_dims=9
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    print(f"模型: {sum(p.numel() for p in model.parameters())/1e6:.2f}M 参数")
    print("训练循环就绪。需 GPU 环境 + 真实标注数据。")
    # 完整训练循环由 GPU 环境按数据加载器补全


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dim", type=int, default=4096)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
