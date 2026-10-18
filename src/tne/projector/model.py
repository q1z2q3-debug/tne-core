"""
TNE-Core 阶段二 · Projector 模型定义

轻量 MLP：hidden state (dim) → 9×3 logits（每个维度一个三分类头）。

设计决策：
  - 9 个三分类头（每维 -1/0/+1），共享一个表示层
  - 共享表示层建模"维度间关联"（v3.2 教训：九个独立分类器相乘 = 学不到关联）
  - torch 实现，供 GPU 环境训练；verify_arch.py 提供 numpy 等价验证

决策依据（v3.2 教训）：
  - 0.858⁹≈25% 吻合独立概率乘积 → 必须共享表示层建模维度关联
  - 输出硬三态而非连续回归 → 三态是认知判断
"""
from __future__ import annotations
from typing import List, Optional


class MLPProjector:
    """
    三元九维 Projector。
    forward: hidden states → 9×3 logits
    训练：9 个三分类头，交叉熵损失。
    """

    def __init__(self, input_dim: int = 4096, hidden_dim: int = 512,
                 num_dims: int = 9):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_dims = num_dims
        # 占位：真实实现用 torch.nn.Module
        # self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(),
        #                          nn.Linear(hidden_dim, num_dims * 3))
        self._weights = None  # 由 train.py 加载

    def forward(self, hidden_states: List[float]) -> List[List[float]]:
        """
        前向：返回 9×3 logits 矩阵。
        当前为 numpy 兼容实现（verify_arch.py 使用），torch 版见 train.py。
        """
        import numpy as np
        x = np.array(hidden_states, dtype=float)
        if self._weights is None:
            # 未加载权重时用零矩阵（仅占位）
            return [[0.0, 0.0, 0.0] for _ in range(self.num_dims)]
        W1, b1, W2, b2 = self._weights
        h = np.maximum(0.0, W1 @ x + b1)
        logits = W2 @ h + b2
        return logits.reshape(self.num_dims, 3).tolist()

    def load_weights(self, W1, b1, W2, b2):
        """加载权重（由 train.py 保存）。"""
        self._weights = (W1, b1, W2, b2)
