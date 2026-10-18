#!/usr/bin/env python3
"""
TNE-Core · 内化蒸馏 · 解码器真实调用演示
加载 distill_zh_best.pt，对全新任务并行输出九维三态码，
并与外挂 encode 模型（TrainedEncoderJudger）判定对比一致率。
"""
import json, sys, os, torch
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from transformers import GPT2LMHeadModel, AutoTokenizer
sys.path.insert(0, "/home/user/Doubao/chats/38446456964142338/tne-core/src")

BASE = "/home/user/Doubao/chats/38446456964142338/tne-core"
MODEL = "uer/gpt2-chinese-cluecorpussmall"
CKPT = f"{BASE}/tmp/distill_zh_best.pt"
INSTR = "判断以下情境在九维三态下的认知坐标\n任务："
OUT = "九维向量："

NEW_TASKS = [
    "组织一场跨部门的产品评审会议",
    "深夜反复失眠睡不好",
    "给初创公司制定股权激励方案",
    "把旧仓库的代码全面重构",
    "突然收到大客户的紧急投诉",
]

def code_to_vec(code):
    return [1 if c == "+" else -1 if c == "-" else 0 for c in code]

def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = GPT2LMHeadModel.from_pretrained(MODEL)
    model.load_state_dict(torch.load(CKPT, map_location="cpu"))
    model.eval()

    # 外挂 encode 模型对照
    try:
        from tne.projector.encoder_judger import TrainedEncoderJudger
        judger = TrainedEncoderJudger()
        has_judger = True
    except Exception as e:
        print("encode 对照模型加载失败:", str(e)[:80]); has_judger = False

    print("=" * 60)
    print("内化蒸馏解码器 · 全新任务并行九维输出")
    print("=" * 60)
    dim_total = dim_correct = 0
    with torch.no_grad():
        for task in NEW_TASKS:
            src = INSTR + task + "\n" + OUT
            inp = tok.encode(src, return_tensors="pt")
            out = model.generate(input_ids=inp, max_new_tokens=16,
                                 do_sample=False, pad_token_id=tok.pad_token_id)
            gen = tok.decode(out[0][inp.shape[1]:], skip_special_tokens=True)
            code = "".join(c for c in gen if c in "+0-")[:9]
            if len(code) == 9:
                vec = code_to_vec(code)
                # encode 对照
                if has_judger:
                    jr = judger.judge_with_confidence(task)
                    jvec = jr["vector"]
                    n_match = sum(1 for a, b in zip(vec, jvec) if a == b)
                    dim_total += 9; dim_correct += n_match
                    print(f"任务「{task}」")
                    print(f"  内化解码: {code}")
                    print(f"  encode对照: {''.join('+' if g==1 else '0' if g==0 else '-' for g in jvec)}  逐维一致 {n_match}/9")
                else:
                    print(f"任务「{task}」 → {code}")
            else:
                print(f"任务「{task}」 → 输出不合法: 「{gen}」")
    if has_judger and dim_total:
        print(f"\n内化解码 vs encode 判定 一致率: {dim_correct}/{dim_total} = {dim_correct/dim_total:.1%}")

if __name__ == "__main__":
    main()
