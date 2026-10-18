#!/usr/bin/env python3
"""
TNE-Core · 自训 encode 模型（知识蒸馏，教师=Doubao）

用教师（Doubao）标注的九维种子数据微调 MiniLM，固化成本地 encode 模型。

验证分层（诚实标注）：
  层1：模型学会教师的判断规则（train/val 划分，可验证）
  层2：对真实新任务的合理性（人工 sanity check）

注意：爻位九维 ≠ v3.2 天地人九维，不能用 v3.2 的 val/holdout 直接验证。
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.environ["HF_HUB_OFFLINE"] = "1"

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

TNE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
DATA_PATH = os.path.join(TNE_ROOT, "data", "teacher_seeds.json")
MODEL_PATH = "/home/user/.cache/minilm_local"
OUT_PATH = os.path.join(TNE_ROOT, "data", "trained_encoder.pt")

IDX = {-1: 0, 0: 1, 1: 2}
STATE = {0: -1, 1: 0, 2: 1}


class Encoder(nn.Module):
    """MiniLM + 共享表示层 + 9 三分类头。"""
    def __init__(self, base):
        super().__init__()
        self.base = base  # MiniLM
        hidden = base.config.hidden_size  # 384
        self.shared = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.1),
        )
        self.heads = nn.ModuleList([nn.Linear(hidden, 3) for _ in range(9)])

    def forward(self, input_ids, attention_mask):
        out = self.base(input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]  # (n,384)
        h = self.shared(cls)
        return torch.stack([head(h) for head in self.heads], dim=1)  # (n,9,3)


def embed_cls(tok, base, texts, max_len=64):
    """用 MiniLM 输出 CLS 作为句子表示（供推理时手工拼 head）。"""
    enc = tok(texts, padding=True, truncation=True, max_length=max_len,
              return_tensors="pt")
    with torch.no_grad():
        out = base(**enc)
    return out.last_hidden_state[:, 0, :]


def train(args):
    torch.manual_seed(args.seed)
    data = json.load(open(DATA_PATH, encoding="utf-8"))["samples"]
    n = len(data)
    print(f"教师种子数据: {n} 条")

    # 留出划分
    idx = list(range(n))
    np.random.seed(args.seed); np.random.shuffle(idx)
    n_val = max(1, int(n * args.val_ratio))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    train_d, val_d = [data[i] for i in train_idx], [data[i] for i in val_idx]
    print(f"训练 {len(train_d)} | 验证 {len(val_d)}")

    # 加载模型
    print("[加载 MiniLM ...]")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    base = AutoModel.from_pretrained(MODEL_PATH, local_files_only=True)
    model = Encoder(base)
    # 冻结 base（CPU 高效），只训练 shared + heads
    for p in base.parameters():
        p.requires_grad = False
    for p in model.shared.parameters():
        p.requires_grad = True
    for head in model.heads:
        for p in head.parameters():
            p.requires_grad = True

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    # 预计算 embedding（一次性），避免每 epoch 重复跑 MiniLM
    def embed_samples(samples):
        enc = tok([s["text"] for s in samples], padding=True, truncation=True,
                  max_length=64, return_tensors="pt")
        with torch.no_grad():
            out = base(**enc)
        return out.last_hidden_state[:, 0, :]  # (n,384)

    def to_batch_emb(samples, emb):
        y = torch.tensor([[IDX[s["vector"][d]] for d in range(9)] for s in samples])
        return emb, y

    train_emb = embed_samples(train_d)
    val_emb = embed_samples(val_d)
    test_emb = embed_samples([{"text": t, "vector": [0]*9} for t in
                              ["给公司做一份年度财务审计", "写一首关于春天的词",
                               "用户投诉产品有严重问题", "学习如何做数据分析",
                               "解决两个部门之间的长期矛盾"]])

    best_val = -1
    best_state = None
    print("\n[训练 head（base 冻结）...]")
    for ep in range(args.epochs):
        model.train()
        x, y = to_batch_emb(train_d, train_emb)
        opt.zero_grad()
        h = model.shared(x)
        logits = torch.stack([head(h) for head in model.heads], dim=1)
        loss = sum(loss_fn(logits[:, d], y[:, d]) for d in range(9))
        loss.backward()
        opt.step()

        # 验证
        model.eval()
        with torch.no_grad():
            hv = model.shared(val_emb)
            vlogits = torch.stack([head(hv) for head in model.heads], dim=1)
            vpred = vlogits.argmax(-1)
            _, y_val = to_batch_emb(val_d, val_emb)
            vmatch = (vpred == y_val).all(dim=1).float().mean().item()
            ht = model.shared(train_emb)
            tlogits = torch.stack([head(ht) for head in model.heads], dim=1)
            tpred = tlogits.argmax(-1)
            _, y_tr = to_batch_emb(train_d, train_emb)
            tmatch = (tpred == y_tr).all(dim=1).float().mean().item()
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"  ep{ep}: loss={loss.item():.3f} train={tmatch*100:.1f}% val={vmatch*100:.1f}%")
        if vmatch > best_val:
            best_val = vmatch
            best_state = {k: v.clone() for k, v in model.state_dict().items()
                          if not k.startswith("base")}

    # 载入最优（只加载 head，base 保持不变）
    model.load_state_dict(best_state, strict=False)
    print(f"\n最佳 val 完全吻合率: {best_val*100:.1f}%")

    # 保存（base 已在外部，只存 head 权重 + 配置）
    torch.save({
        "head": best_state,
        "base_config": base.config.to_dict(),
    }, OUT_PATH)
    print(f"模型已保存: {OUT_PATH}")

    # Sanity check：新任务（用预计算的 test_emb）
    print("\n[Sanity check 新任务]")
    model.eval()
    tests = ["给公司做一份年度财务审计", "写一首关于春天的词", "用户投诉产品有严重问题",
             "学习如何做数据分析", "解决两个部门之间的长期矛盾"]
    with torch.no_grad():
        ht = model.shared(test_emb)
        logits = torch.stack([head(ht) for head in model.heads], dim=1)
        pred = logits.argmax(-1).numpy()
    for text, vec in zip(tests, pred):
        vec9 = [STATE[int(i)] for i in vec]
        vecs = "".join("+" if v == 1 else "0" if v == 0 else "-" for v in vec9)
        print(f"  「{text[:24]}」→ {vecs}")

    return {"val": best_val}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
