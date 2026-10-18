#!/usr/bin/env python3
"""
TNE-Core · 内化蒸馏 · 第一层：SFT 数据生成
从教师种子 + 轨迹库合成"输入任务 → 九维三态向量"的训练数据。
输出格式（可供 decoder 语言建模）：
  输入：判断以下情境在九维三态下的认知坐标\n任务：<text>
  输出：九维向量：<g1g2g3g4g5g6g7g8g9>
"""
import json, sys, random
sys.path.insert(0, "/home/user/Doubao/chats/38446456964142338/tne-core/src")

def vec_to_code(vec):
    # ASCII 三态码（降级验证：tiny-gpt2 为英文分词器，用 +/0/- 便于格式学习）
    return "".join("+" if g==1 else "0" if g==0 else "-" for g in vec)

def gen_sft(seeds_path, out_path, split_ratio=0.9):
    seeds = json.load(open(seeds_path, encoding="utf-8"))["samples"]
    random.seed(42)
    random.shuffle(seeds)
    n = len(seeds)
    k = int(n * split_ratio)

    def to_examples(items):
        ex = []
        for s in items:
            ex.append({
                "input": s["text"],
                "output": vec_to_code(s["vector"]),
                "vector": s["vector"],
            })
        return ex

    train = to_examples(seeds[:k])
    val = to_examples(seeds[k:])
    json.dump({"train": train, "val": val},
              open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"SFT 数据生成完成: train {len(train)} / val {len(val)} → {out_path}")
    print("输出格式样例:", json.dumps(train[0], ensure_ascii=False))
    return train, val

if __name__ == "__main__":
    gen_sft("/home/user/Doubao/chats/38446456964142338/tne-core/data/teacher_seeds.json",
            "/home/user/Doubao/chats/38446456964142338/tne-core/data/distill_sft.json")
