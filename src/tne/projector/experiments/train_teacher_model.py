#!/usr/bin/env python3
"""
TNE-Core · 自训 encode 模型（知识蒸馏，教师=Doubao）

用教师（Doubao）标注的九维种子数据微调 MiniLM，固化成本地 encode 模型。

支持两种模式：
  --finetune_layers 0  : 冻结 base，只训 head（快，预计算 embedding）
  --finetune_layers N>0: 微调 base 最后 N 层 + head（慢，提升表示质量）

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


def make_opt(base, model, finetune_layers, lr):
    """按 finetune_layers 决定可训参数。"""
    for p in base.parameters():
        p.requires_grad = False
    if finetune_layers > 0:
        # 微调 base 最后 N 层（MiniLM 12 层，encoder.layer）
        layers = list(base.encoder.layer)
        for layer in layers[-finetune_layers:]:
            for p in layer.parameters():
                p.requires_grad = True
    for p in model.shared.parameters():
        p.requires_grad = True
    for head in model.heads:
        for p in head.parameters():
            p.requires_grad = True
    trainable = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)


def embed_cls(tok, base, texts, max_len=64):
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

    idx = list(range(n))
    np.random.seed(args.seed); np.random.shuffle(idx)
    n_val = max(1, int(n * args.val_ratio))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    train_d, val_d = [data[i] for i in train_idx], [data[i] for i in val_idx]
    print(f"训练 {len(train_d)} | 验证 {len(val_d)}")

    print("[加载 MiniLM ...]")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    base = AutoModel.from_pretrained(MODEL_PATH, local_files_only=True)
    model = Encoder(base)
    opt = make_opt(base, model, args.finetune_layers, args.lr)
    loss_fn = nn.CrossEntropyLoss()

    def labels(samples):
        return torch.tensor([[IDX[s["vector"][d]] for d in range(9)]
                             for s in samples])

    y_tr, y_val = labels(train_d), labels(val_d)

    # sanity 任务
    tests = ["给公司做一份年度财务审计", "写一首关于春天的词", "用户投诉产品有严重问题",
             "学习如何做数据分析", "解决两个部门之间的长期矛盾"]

    if args.finetune_layers == 0:
        # 模式1：冻结 base，预计算 embedding，只训 head（快）
        print(f"\n[训练 head（base 冻结）· epochs={args.epochs} lr={args.lr}]")
        train_emb = embed_cls(tok, base, [s["text"] for s in train_d])
        val_emb = embed_cls(tok, base, [s["text"] for s in val_d])
        test_emb = embed_cls(tok, base, tests)
        best_val, best_state = -1, None
        for ep in range(args.epochs):
            model.train()
            h = model.shared(train_emb)
            logits = torch.stack([head(h) for head in model.heads], dim=1)
            loss = sum(loss_fn(logits[:, d], y_tr[:, d]) for d in range(9))
            opt.zero_grad(); loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                hv = model.shared(val_emb)
                vp = torch.stack([head(hv) for head in model.heads], dim=1).argmax(-1)
                vmatch = (vp == y_val).all(dim=1).float().mean().item()
                ht = model.shared(train_emb)
                tp = torch.stack([head(ht) for head in model.heads], dim=1).argmax(-1)
                tmatch = (tp == y_tr).all(dim=1).float().mean().item()
            if ep % 5 == 0 or ep == args.epochs - 1:
                print(f"  ep{ep}: loss={loss.item():.3f} train={tmatch*100:.1f}% val={vmatch*100:.1f}%")
            if vmatch > best_val:
                best_val = vmatch
                best_state = {k: v.clone() for k, v in model.state_dict().items()
                              if not k.startswith("base")}
        model.load_state_dict(best_state, strict=False)
        torch.save({"head": best_state, "base_config": base.config.to_dict(),
                    "mode": "head_only"}, OUT_PATH)
        with torch.no_grad():
            hv = model.shared(val_emb)
            vp = torch.stack([head(hv) for head in model.heads], dim=1).argmax(-1)
            per_dim = [(vp[:, d] == y_val[:, d]).float().mean().item() * 100
                       for d in range(9)]
            ht = model.shared(test_emb)
            pred = torch.stack([head(ht) for head in model.heads], dim=1).argmax(-1).numpy()
        print("\n  每维准确率（g1..g9）:")
        print("  " + "  ".join(f"g{i+1}:{per_dim[i]:.0f}%" for i in range(9)))
        print(f"  每维平均: {np.mean(per_dim):.1f}%  (随机基线 33%)")
    else:
        # 模式2：微调 base 后 N 层 + head（慢，完整前向+反向）
        print(f"\n[微调 base 后 {args.finetune_layers} 层 + head · epochs={args.epochs} lr={args.lr}]")
        batch = args.batch
        best_val, best_state = -1, None
        for ep in range(args.epochs):
            model.train()
            perm = torch.randperm(len(train_d))
            tot_loss = 0
            for i in range(0, len(train_d), batch):
                bidx = perm[i:i+batch]
                texts = [train_d[j]["text"] for j in bidx]
                enc = tok(texts, padding=True, truncation=True, max_length=64,
                          return_tensors="pt")
                logits = model(enc.input_ids, enc.attention_mask)
                loss = sum(loss_fn(logits[:, d], y_tr[bidx, d]) for d in range(9))
                opt.zero_grad(); loss.backward(); opt.step()
                tot_loss += loss.item() * len(bidx)
            # 验证
            model.eval()
            with torch.no_grad():
                e_val = tok([s["text"] for s in val_d], padding=True, truncation=True,
                            max_length=64, return_tensors="pt")
                vp = model(e_val.input_ids, e_val.attention_mask).argmax(-1)
                vmatch = (vp == y_val).all(dim=1).float().mean().item()
                e_tr = tok([s["text"] for s in train_d], padding=True, truncation=True,
                           max_length=64, return_tensors="pt")
                tp = model(e_tr.input_ids, e_tr.attention_mask).argmax(-1)
                tmatch = (tp == y_tr).all(dim=1).float().mean().item()
            if ep % 2 == 0 or ep == args.epochs - 1:
                print(f"  ep{ep}: loss={tot_loss:.3f} train={tmatch*100:.1f}% val={vmatch*100:.1f}%")
            if vmatch > best_val:
                best_val = vmatch
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        torch.save({"head": best_state, "base_config": base.config.to_dict(),
                    "mode": f"finetune_{args.finetune_layers}"}, OUT_PATH)
        with torch.no_grad():
            e_t = tok(tests, padding=True, truncation=True, max_length=64,
                      return_tensors="pt")
            pred = model(e_t.input_ids, e_t.attention_mask).argmax(-1).numpy()

    print(f"\n最佳 val 完全吻合率: {best_val*100:.1f}%  |  已保存: {OUT_PATH}")

    print("\n[Sanity check 新任务]")
    for text, vec in zip(tests, pred):
        vec9 = [STATE[int(i)] for i in vec]
        vecs = "".join("+" if v == 1 else "0" if v == 0 else "-" for v in vec9)
        print(f"  「{text[:24]}」→ {vecs}")
    return {"val": best_val}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--finetune_layers", type=int, default=0,
                    help="微调 base 最后 N 层（0=只训 head）")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
