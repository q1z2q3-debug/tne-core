#!/usr/bin/env python3
"""
TNE-Core · 内化蒸馏 · 全量中文解码器训练
用 teacher_seeds 全量 560 条（train 504 / val 56）训练中文 GPT2，
产出可落地的"并行输出九维三态码"解码器。
关键：val 56 条为未见样本，每维准确率衡量真实泛化（非训练集过拟合）。
"""
import json, sys, os, torch, time, argparse
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from transformers import GPT2LMHeadModel, AutoTokenizer

BASE = "/home/user/Doubao/chats/38446456964142338/tne-core"
sys.path.insert(0, BASE + "/src")

INSTR = "判断以下情境在九维三态下的认知坐标\n任务："
OUT_TAG = "九维向量："
MODEL = "uer/gpt2-chinese-cluecorpussmall"

def vec_to_code(vec):
    return "".join("+" if g==1 else "0" if g==0 else "-" for g in vec)

def build(data):
    return [(INSTR + x["input"] + "\n" + OUT_TAG, vec_to_code(x["vector"]) + "\n") for x in data]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=8e-5)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--n_train", type=int, default=None, help="训练子集大小(CPU限速时用小集)")
    args = ap.parse_args()

    sft = json.load(open(f"{BASE}/data/distill_sft.json", encoding="utf-8"))
    train_data = sft["train"] if args.n_train is None else sft["train"][:args.n_train]
    train_pairs = build(train_data)   # 子集
    val_pairs = build(sft["val"])     # 56 未见样本（完整泛化评测）

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = GPT2LMHeadModel.from_pretrained(MODEL)
    model.train()
    print(f"全量训练 | train {len(train_pairs)} / val {len(val_pairs)} | {args.epochs}ep lr={args.lr} bs={args.bs}")

    def encode(src, tgt, max_len=96):
        src_ids = tok.encode(src, add_special_tokens=False)
        tgt_ids = tok.encode(tgt, add_special_tokens=False)
        ids = (src_ids + tgt_ids)[:max_len]
        ids_t = torch.tensor(ids)
        labels = ids_t.clone()
        labels[:len(src_ids)] = -100
        return ids_t, labels

    def make_batches(pairs, bs):
        batches = []
        for i in range(0, len(pairs), bs):
            chunk = pairs[i:i+bs]
            ids_l, lab_l = [], []
            for s, t in chunk:
                i_, l_ = encode(s, t); ids_l.append(i_); lab_l.append(l_)
            maxl = max(len(x) for x in ids_l)
            pad = torch.full((len(chunk), maxl), tok.pad_token_id, dtype=torch.long)
            labpad = torch.full((len(chunk), maxl), -100, dtype=torch.long)
            for k in range(len(chunk)):
                pad[k, :len(ids_l[k])] = ids_l[k]
                labpad[k, :len(lab_l[k])] = lab_l[k]
            att = (pad != tok.pad_token_id).long()
            batches.append((pad, att, labpad))
        return batches

    train_batches = make_batches(train_pairs, args.bs)
    val_batches = make_batches(val_pairs, 16)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    t0 = time.time()
    best_val = 0.0
    for ep in range(1, args.epochs + 1):
        model.train()
        tot = 0.0; nb = 0
        for ids, att, lab in train_batches:
            out = model(input_ids=ids, attention_mask=att, labels=lab)
            loss = out.loss
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item(); nb += 1
        if ep % 5 == 0 or ep == args.epochs:
            # val 每维准确率
            model.eval()
            dv = eval_val(model, tok, val_pairs)
            print(f"  ep {ep:>2} | train loss {tot/nb:.3f} | val 每维准确率 {dv:.1%} | {time.time()-t0:.0f}s", flush=True)
            if dv > best_val:
                best_val = dv
                torch.save(model.state_dict(), f"{BASE}/tmp/distill_zh_best.pt")
    print(f"最优 val 每维准确率 {best_val:.1%}")

def eval_val(model, tok, pairs):
    model.eval()
    dim_total = dim_correct = 0
    with torch.no_grad():
        for src, tgt in pairs:
            inp = tok.encode(src, return_tensors="pt")
            out = model.generate(input_ids=inp, max_new_tokens=16,
                                 do_sample=False, pad_token_id=tok.pad_token_id)
            gen = tok.decode(out[0][inp.shape[1]:], skip_special_tokens=True)
            code = "".join(c for c in gen if c in "+0-")[:9]
            gold = tgt.strip()
            if len(code) == 9:
                for k in range(9):
                    dim_total += 1
                    if code[k] == gold[k]:
                        dim_correct += 1
    return dim_correct / dim_total if dim_total else 0.0

if __name__ == "__main__":
    main()
