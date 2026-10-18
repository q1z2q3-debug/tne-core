"""
TNE-Core 阶段二 · 架构验证（不依赖 torch，numpy 实现）

验证 Projector 完整链路端到端逻辑正确：
  合成 hidden state → MLP → quantizer → 硬三态
  复现率应接近 100%（合成数据正交投影 + 低噪声）。

⚠️ 这只是架构 sanity check，不代表真实泛化能力。
真实泛化需要 GPU + 真实 hidden state 标注数据（见 STAGE2_PLAN.md）。
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np
from tne.projector.data_gen import synthesize_dataset, split
from tne.projector.model import MLPProjector
from tne.projector.quantizer import quantize


def train_numpy(model: MLPProjector, train_data, epochs: int = 200,
                lr: float = 0.1) -> None:
    """用 numpy 训练一个简单的线性 + ReLU 分类器（替代 torch，架构验证用）。"""
    dim = model.input_dim
    num_dims = model.num_dims
    # 初始化：hidden_dim x dim
    rng = np.random.default_rng(0)
    W1 = rng.normal(0, 0.5, (model.hidden_dim, dim))
    b1 = np.zeros(model.hidden_dim)
    W2 = rng.normal(0, 0.1, (num_dims * 3, model.hidden_dim))
    b2 = np.zeros(num_dims * 3)

    for epoch in range(epochs):
        for s in train_data:
            x = np.array(s.hidden_states, dtype=float)
            label = s.label
            # 前向
            h = np.maximum(0.0, W1 @ x + b1)
            logits = W2 @ h + b2
            logits = logits.reshape(num_dims, 3)
            # 构造 one-hot 目标：类别索引 0=-1, 1=0, 2=+1
            target = np.array([IDX[label[d]] for d in range(num_dims)])
            # softmax 交叉熵梯度（简化）
            exps = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs = exps / exps.sum(axis=1, keepdims=True)
            grad_logits = probs - np.eye(3)[target]  # (9,3)
            grad_logits = grad_logits.reshape(-1)
            # 反向传播
            g_out = grad_logits  # (27,)
            g_W2 = np.outer(g_out, h)
            g_b2 = g_out
            g_h = W2.T @ g_out
            g_h[h <= 0] = 0.0
            g_W1 = np.outer(g_h, x)
            g_b1 = g_h
            # 更新
            W2 -= lr * g_W2
            b2 -= lr * g_b2
            W1 -= lr * g_W1
            b1 -= lr * g_b1

    model.load_weights(W1, b1, W2, b2)


IDX = {-1: 0, 0: 1, 1: 2}


def evaluate(model: MLPProjector, data, name: str) -> dict:
    """评测：完全吻合率 + 逐维准确率。"""
    exact = 0
    dim_correct = [0] * 9
    for s in data:
        logits = model.forward(s.hidden_states)
        q = quantize(logits)
        pred = q["vector"]
        if pred == s.label:
            exact += 1
        for d in range(9):
            if pred[d] == s.label[d]:
                dim_correct[d] += 1
    n = len(data)
    return {
        "set": name,
        "exact_match": round(exact / n, 3),
        "per_dim_acc": [round(c / n, 3) for c in dim_correct],
    }


def main():
    print("=" * 60)
    print("TNE-Core 阶段二 · Projector 架构验证（numpy，无 GPU）")
    print("=" * 60)

    # 合成数据
    data = synthesize_dataset(2000, hidden_dim=10, noise=0.05, seed=42)
    train_data, val_data = split(data, train_ratio=0.8, seed=42)
    print(f"合成数据: {len(data)} 条 | 训练 {len(train_data)} | 验证 {len(val_data)}")

    # 模型
    model = MLPProjector(input_dim=10, hidden_dim=32, num_dims=9)

    # 训练（numpy 版）
    train_numpy(model, train_data, epochs=100, lr=0.05)
    print("\n[训练完成]")

    # 评测
    print("\n--- 评测 ---")
    train_res = evaluate(model, train_data, "train")
    val_res = evaluate(model, val_data, "val")
    for r in (train_res, val_res):
        print(f"  {r['set']}: 完全吻合率={r['exact_match']*100:.1f}% | "
              f"逐维准确率={r['per_dim_acc']}")

    print("\n" + "=" * 60)
    print("⚠️ 架构验证通过 ≠ 真实泛化。真实能力需 GPU + 真实 hidden state 数据。")
    print("=" * 60)


if __name__ == "__main__":
    main()
