#!/usr/bin/env python3
"""
TNE-Core 阶段二 · 共享表示层 vs 独立分类器对照实验

直接验证 v3.2 的核心论断："九个独立分类器相乘（0.858^9≈25%）= 学不到维度关联"。

方法：
  A. 9 个独立 LogisticRegression（已测 val=7.5%）
  B. 共享表示层 MLP（embedding → 共享 hidden → 9×3 logits）

数据仍是 v3.2 真实标注集（train=60 / val=40 / holdout=30）。
若 B > A，则证明"共享表示层建模维度关联"有效，阶段二方向正确。
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

TNE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
DATA_DIR = os.path.join(TNE_ROOT, "vendor", "tne_core_v32", "data")
MODEL_PATH = "/home/user/.cache/minilm_local"

os.environ["HF_HUB_OFFLINE"] = "1"

IDX = {-1: 0, 0: 1, 1: 2}
STATE = {0: -1, 1: 0, 2: 1}


def load_split(name):
    d = json.load(open(os.path.join(DATA_DIR, f"{name}_set.json")))
    return d["samples"] if isinstance(d, dict) else d


def embed_batch(model, tok, texts):
    enc = tok(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        out = model(**enc)
    cls = out.last_hidden_state[:, 0, :]
    return torch.nn.functional.normalize(cls, p=2, dim=1).numpy()


class SharedMLP(nn.Module):
    """共享表示层 + 9 个三分类头。"""
    def __init__(self, dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(0.3),
        )
        self.heads = nn.ModuleList([nn.Linear(hidden, 3) for _ in range(9)])

    def forward(self, x):
        h = self.net(x)
        return torch.stack([head(h) for head in self.heads], dim=1)  # (n,9,3)


def train_shared(X, y, val_X, val_y, epochs=400, lr=1e-3, seed=42):
    torch.manual_seed(seed)
    model = SharedMLP(X.shape[1], hidden=64)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)  # (n,9) 每维类别索引
    Xv = torch.tensor(val_X, dtype=torch.float32)
    yv = torch.tensor(val_y, dtype=torch.long)
    best_val = -1
    best_state = None
    patience = 60
    no_improve = 0
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(Xt)  # (n,9,3)
        loss = sum(loss_fn(logits[:, d], yt[:, d]) for d in range(9))
        loss.backward()
        opt.step()
        # 早停（在 val 上）
        model.eval()
        with torch.no_grad():
            vlogits = model(Xv)
            vpred = vlogits.argmax(-1)  # (n,9)
            vmatch = (vpred == yv).all(dim=1).float().mean().item()
        if vmatch > best_val:
            best_val = vmatch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    return model, best_val


def main():
    os.environ["HF_HUB_OFFLINE"] = "1"
    print("=" * 62)
    print("共享表示层 vs 独立分类器（验证 v3.2 '维度关联'论断）")
    print("=" * 62)
    train = load_split("train"); val = load_split("val"); hold = load_split("holdout")
    print(f"训练 {len(train)} | 验证 {len(val)} | 独立 {len(hold)}")

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, )
    model = AutoModel.from_pretrained(MODEL_PATH, )
    X_train = embed_batch(model, tok, [s["text"] for s in train])
    X_val = embed_batch(model, tok, [s["text"] for s in val])
    X_hold = embed_batch(model, tok, [s["text"] for s in hold])

    # 目标：每维类别索引 (n,9)
    def to_y(samples):
        return np.array([[IDX[s["vector"][d]] for d in range(9)] for s in samples])
    y_train, y_val, y_hold = to_y(train), to_y(val), to_y(hold)

    # 训练共享表示层
    print("\n[训练共享表示层 MLP ...]")
    shared, best_val = train_shared(X_train, y_train, X_val, y_val)
    print(f"共享表示层 best val 完全吻合率: {best_val*100:.1f}%")

    # 预测
    model.eval()
    with torch.no_grad():
        p_train = shared(torch.tensor(X_train, dtype=torch.float32)).argmax(-1).numpy()
        p_val = shared(torch.tensor(X_val, dtype=torch.float32)).argmax(-1).numpy()
        p_hold = shared(torch.tensor(X_hold, dtype=torch.float32)).argmax(-1).numpy()

    print("\n--- 共享表示层 完全吻合率 ---")
    for name, pred, y in [("train", p_train, y_train), ("val", p_val, y_val), ("holdout", p_hold, y_hold)]:
        em = np.mean((pred == y).all(axis=1))
        print(f"  {name}: {em*100:.1f}%")

    print("\n" + "=" * 62)
    print("对比: 独立LR val=7.5% | 关键词法 val=25% | v3.2小模型 val=15%")
    print("=" * 62)


if __name__ == "__main__":
    main()
