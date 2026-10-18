#!/usr/bin/env python3
"""
TNE-Core 阶段二 · 离线小模型路线实测
（用户要求：不复读 v3.2 的"走不通"，亲自验证）

方法：MiniLM 语义 embedding (384d) → LogisticRegression 三分类头
数据：v3.2 真实标注集（天地人九维）train=60 / val=40 / holdout=30

对比 v3.2 结论：
  - 关键词法 val=25% / holdout=36.7%
  - v3.2 小模型（char n-gram）val=15%
本实验验证：环境解除后（能装 torch + 能下 MiniLM），语义 embedding 路线是否可行。
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "vendor", "tne_core_v32"))

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression

TNE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
DATA_DIR = os.path.join(TNE_ROOT, "vendor", "tne_core_v32", "data")
MODEL_PATH = "/home/user/.cache/minilm_local"

os.environ["HF_HUB_OFFLINE"] = "1"

IDX = {-1: 0, 0: 1, 1: 2}
STATE = {0: -1, 1: 0, 2: 1}


def load_split(name):
    d = json.load(open(os.path.join(DATA_DIR, f"{name}_set.json")))
    samples = d["samples"] if isinstance(d, dict) else d
    return samples


def embed_batch(model, tok, texts):
    enc = tok(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        out = model(**enc)
    cls = out.last_hidden_state[:, 0, :]
    return torch.nn.functional.normalize(cls, p=2, dim=1).numpy()


def main():
    os.environ["HF_HUB_OFFLINE"] = "1"
    print("=" * 62)
    print("TNE-Core 阶段二 · 离线小模型路线实测（MiniLM + LR）")
    print("=" * 62)

    # 加载数据
    train = load_split("train")
    val = load_split("val")
    holdout = load_split("holdout")
    print(f"训练 {len(train)} | 验证 {len(val)} | 独立 {len(holdout)}")

    # 加载 MiniLM
    print("\n[加载 MiniLM ...]")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, )
    model = AutoModel.from_pretrained(MODEL_PATH, )

    # embedding
    X_train = embed_batch(model, tok, [s["text"] for s in train])
    X_val = embed_batch(model, tok, [s["text"] for s in val])
    X_hold = embed_batch(model, tok, [s["text"] for s in holdout])
    print(f"embedding 维度: {X_train.shape[1]}")

    # 9 个三分类头（LogisticRegression，L2 正则）
    heads = []
    for d in range(9):
        y = np.array([IDX[s["vector"][d]] for s in train])
        clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
        clf.fit(X_train, y)
        heads.append(clf)

    def predict_all(X):
        """九维联合预测。"""
        preds = []
        for d in range(9):
            p = heads[d].predict(X)
            preds.append([STATE[int(i)] for i in p])
        return np.array(preds).T  # (n, 9)

    def exact_match(X, samples):
        pred = predict_all(X)
        return np.mean([list(pred[i]) == samples[i]["vector"] for i in range(len(samples))])

    # 评测
    print("\n--- 评测（完全吻合率）---")
    for name, X, S in [("train", X_train, train),
                       ("val", X_val, val),
                       ("holdout", X_hold, holdout)]:
        em = exact_match(X, S)
        print(f"  {name}: {em*100:.1f}%")

    print("\n" + "=" * 62)
    print("对比 v3.2: 关键词法 val=25% | v3.2小模型(ngram) val=15%")
    print("=" * 62)


if __name__ == "__main__":
    main()
