#!/usr/bin/env python3
"""
TNE-Core · 内化蒸馏 · 中文小数据过拟合冒烟测试
按指导修正：
1. 换中文模型 uer/gpt2-chinese-cluecorpussmall（102M，中文无 UNK）
2. 正确 -100 掩码：labels=ids，prompt 段置 -100（HF 内部 shift 已处理）
3. 小数据过拟合：10 条 × 100 epoch，loss 应降至极低且格式 100% 正确
"""
import json, sys, os, torch, time
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from transformers import GPT2LMHeadModel, AutoTokenizer

BASE = "/home/user/Doubao/chats/38446456964142338/tne-core"
sys.path.insert(0, BASE + "/src")

INSTR = "判断以下情境在九维三态下的认知坐标\n任务："
OUT_TAG = "九维向量："
MODEL = "uer/gpt2-chinese-cluecorpussmall"

def build(data, n):
    pairs = []
    for x in data[:n]:
        code = "".join("+" if g==1 else "0" if g==0 else "-" for g in x["vector"])
        pairs.append((INSTR + x["text"] + "\n" + OUT_TAG, code + "\n"))
    return pairs

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()
    seeds = json.load(open(f"{BASE}/data/teacher_seeds.json", encoding="utf-8"))["samples"]
    train_pairs = build(seeds, args.n)  # 可控数据量

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = GPT2LMHeadModel.from_pretrained(MODEL)
    model.train()
    print(f"中文模型 {MODEL.split('/')[-1]} | 参数 {model.num_parameters()/1e6:.0f}M | 小数据 {len(train_pairs)} 条")

    def encode(src, tgt, max_len=96):
        src_ids = tok.encode(src, add_special_tokens=False)
        tgt_ids = tok.encode(tgt, add_special_tokens=False)
        ids = (src_ids + tgt_ids)[:max_len]
        ids_t = torch.tensor(ids)
        labels = ids_t.clone()
        labels[:len(src_ids)] = -100  # 正确掩码：只监督输出段
        return ids_t, labels

    def make_batches(pairs, bs=10):
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

    batches = make_batches(train_pairs)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    epochs = args.epochs
    t0 = time.time()
    print(f"过拟合测试 | {epochs} epochs | lr=1e-4 | 只监督输出段(-100掩码)")
    for ep in range(1, epochs + 1):
        model.train()
        tot = 0.0; nb = 0
        for ids, att, lab in batches:
            out = model(input_ids=ids, attention_mask=att, labels=lab)
            loss = out.loss
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item(); nb += 1
        if ep % 10 == 0 or ep == epochs:
            print(f"  ep {ep:>3} | loss {tot/nb:.4f} | {time.time()-t0:.0f}s", flush=True)

    # 评测：格式正确率 + 全一致率 + 每维准确率（对标 encode 模型每维判准）
    model.eval()
    ok_fmt = ok_full = 0
    dim_total = dim_correct = 0
    with torch.no_grad():
        for src, tgt in train_pairs:
            inp = tok.encode(src, return_tensors="pt")
            out = model.generate(input_ids=inp, max_new_tokens=16,
                                 do_sample=False, pad_token_id=tok.pad_token_id)
            gen = tok.decode(out[0][inp.shape[1]:], skip_special_tokens=True)
            code = "".join(c for c in gen if c in "+0-")[:9]
            valid = len(code) == 9 and all(c in "+0-" for c in code)
            if valid:
                ok_fmt += 1
                if code == tgt.strip():
                    ok_full += 1
                # 每维准确率
                for k in range(9):
                    dim_total += 1
                    if k < len(code) and code[k] == tgt.strip()[k]:
                        dim_correct += 1
            print(f"  {src.split('：')[-1][:12]}... → 「{gen.strip()}」 期望「{tgt.strip()}」")
    per_dim = dim_correct / dim_total if dim_total else 0
    print(f"\n[结果] 格式正确率 {ok_fmt}/{len(train_pairs)} = {ok_fmt/len(train_pairs):.1%}"
          f" | 全一致率 {ok_full}/{len(train_pairs)} = {ok_full/len(train_pairs):.1%}"
          f" | 每维准确率 {dim_correct}/{dim_total} = {per_dim:.1%}")

if __name__ == "__main__":
    main()
