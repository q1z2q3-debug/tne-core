#!/usr/bin/env python3
"""
TNE-Core · 内化蒸馏 · 第二层：tiny-gpt2 最小格式验证（CPU，手写循环）
目标：证明"输入任务 → 并行输出九维三态码"蒸馏管线在 CPU 成立。
只用 torch，不依赖 datasets/pyarrow（规避 numpy 冲突）。
"""
import json, sys, os, torch, time
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from transformers import GPT2LMHeadModel, AutoTokenizer

BASE = "/home/user/Doubao/chats/38446456964142338/tne-core"
sys.path.insert(0, BASE + "/src")

INSTR = "判断以下情境在九维三态下的认知坐标\n任务："
OUT_TAG = "九维向量："

def build_supervised(data):
    return [(INSTR + x["input"] + "\n" + OUT_TAG, x["output"] + "\n") for x in data]

def main():
    sft = json.load(open(f"{BASE}/data/distill_sft.json", encoding="utf-8"))
    train_pairs = build_supervised(sft["train"])
    val_pairs = build_supervised(sft["val"])

    tok = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = GPT2LMHeadModel.from_pretrained("sshleifer/tiny-gpt2")
    model.train()

    def encode_pair(src, tgt, max_len=64):
        src_ids = tok.encode(src, add_special_tokens=False)
        tgt_ids = tok.encode(tgt, add_special_tokens=False)
        ids = (src_ids + tgt_ids)[:max_len]
        # 标准因果 LM：HF 内部处理 shift，labels=ids 即"每个位置预测下一 token"
        return torch.tensor(ids), torch.tensor(ids)

    def make_batches(pairs, bs=8):
        batches = []
        for i in range(0, len(pairs), bs):
            chunk = pairs[i:i+bs]
            ids_list, lab_list = [], []
            for s, t in chunk:
                i_, l_ = encode_pair(s, t)
                ids_list.append(i_); lab_list.append(l_)
            maxl = max(len(x) for x in ids_list)
            pad = torch.full((len(chunk), maxl), tok.pad_token_id, dtype=torch.long)
            labpad = torch.full((len(chunk), maxl), -100, dtype=torch.long)
            for k in range(len(chunk)):
                pad[k, :len(ids_list[k])] = ids_list[k]
                labpad[k, :len(lab_list[k])] = lab_list[k]
            att = (pad != tok.pad_token_id).long()
            batches.append((pad, att, labpad))
        return batches

    train_batches = make_batches(train_pairs)
    val_batches = make_batches(val_pairs)

    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
    epochs = 12
    print(f"tiny-gpt2 格式验证训练 | train {len(train_pairs)} / val {len(val_pairs)} | {epochs} epochs")
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        tot = 0.0; nb = 0
        for ids, att, lab in train_batches:
            out = model(input_ids=ids, attention_mask=att, labels=lab)
            loss = out.loss
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item(); nb += 1
        if (ep+1) % 5 == 0 or ep == epochs-1:
            model.eval()
            vloss = 0.0
            with torch.no_grad():
                for ids, att, lab in val_batches:
                    out = model(input_ids=ids, attention_mask=att, labels=lab)
                    vloss += out.loss.item()
            print(f"  ep {ep+1:>2} | train loss {tot/nb:.3f} | val loss {vloss/len(val_batches):.3f} | {time.time()-t0:.0f}s")
    torch.save(model.state_dict(), f"{BASE}/tmp/distill_tiny.pt")

    # 评测：格式正确率 + 全一致率
    model.eval()
    def eval_set(pairs, label):
        ok_fmt = ok_full = 0
        total = len(pairs)
        with torch.no_grad():
            for src, tgt in pairs:
                inp = tok.encode(src, return_tensors="pt")
                out = model.generate(input_ids=inp, max_new_tokens=24,
                                     do_sample=False, pad_token_id=tok.pad_token_id)
                gen = tok.decode(out[0][inp.shape[1]:], skip_special_tokens=True)
                code = "".join(c for c in gen if c in "+0-")[:9]
                valid = len(code) == 9 and all(c in "+0-" for c in code)
                gold = tgt.strip()
                if valid:
                    ok_fmt += 1
                    if code == gold:
                        ok_full += 1
        print(f"[{label}] 格式正确率 {ok_fmt}/{total} = {ok_fmt/total:.1%} | 全一致率 {ok_full}/{total} = {ok_full/total:.1%}")
        return ok_fmt/total, ok_full/total

    eval_set(train_pairs, "train")
    eval_set(val_pairs, "val")

if __name__ == "__main__":
    main()
