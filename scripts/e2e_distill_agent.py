#!/usr/bin/env python3
"""
TNE-Core · 内化蒸馏 · 接入 Agent 端到端演示
用 DistillDecoderJudger（内化解码器）作为 vector_judger 跑完整认知循环：
任务 → 九维定位 → 习惯偏置 → 张力检测 → 动力信号。
对照外挂 encode 模式，展示"内化"与"外挂"两条路线。
"""
import sys
sys.path.insert(0, "/home/user/Doubao/chats/38446456964142338/tne-core/src")

from tne.projector.distill_judger import DistillDecoderJudger, SEM
from tne.trajectory.store import TrajectoryStore
from tne.trajectory.habit import HabitQuery
from tne.agent.llm_agent import LLMAgent, make_encoder_agent
from tne.drive.tension import detect_and_generate
from tne.space.space import encode
from tne.space.equivalence import class_of
from tne.space.distance import nearest_neighbors

TASKS = [
    "深夜反复失眠睡不好",
    "给初创公司制定股权激励方案",
    "把旧仓库的代码全面重构",
    "突然收到大客户的紧急投诉",
    "团队内部互相推诿没人干活",
]

def symbols(vec):
    return "".join("+" if g == 1 else "0" if g == 0 else "-" for g in vec)

def main():
    store = TrajectoryStore("tne_trajectories.db")
    habit = HabitQuery(store)
    judger = DistillDecoderJudger()   # 内化解码器

    print("=" * 64)
    print("内化蒸馏 · Agent 完整认知循环（用解码器并行输出九维态）")
    print("=" * 64)
    for task in TASKS:
        vec = judger.judge(task)
        idx = encode(vec)
        rep = class_of(idx)
        hb = habit.query(vec)
        neighbors = [{"gua_idx": ni, "distance": round(d, 4), "gua_vector": ng,
                      "class_rep": class_of(ni)}
                     for ni, d, ng in nearest_neighbors(vec, top_n=3)]
        jrc = judger.judge_with_confidence(task)
        drive = detect_and_generate(hb, jrc["confidence"], neighbors)

        conf = sum(jrc["confidence"]) / 9
        print(f"\n任务「{task}」")
        print(f"  九维定位: {symbols(vec)}  卦位#{idx}  等价类#{rep}  置信度{conf:.2f}")
        print(f"  爻位语义: " + " ".join(f"{SEM[i]}={'+' if v==1 else '0' if v==0 else '-'}" for i, v in enumerate(vec)))
        mode = "习惯" if hb["mode"] == "habit" else "探索"
        print(f"  习惯偏置: {mode}  推荐「{hb['recommendation'].split('「')[-1].rstrip('。') if '「' in hb['recommendation'] else hb['recommendation'][:20]}」")
        if drive and drive["tensions"]:
            ts = " + ".join(t["type"] for t in drive["tensions"])
            ss = " + ".join(s["type"] for s in drive["signals"])
            print(f"  认知动力: 张力[{ts}] → 信号[{ss}]")
        else:
            print("  认知动力: 稳定习惯，无需额外动力注入")
    store.close()
    print("\n✓ 内化解码器接入 Agent，认知循环完整闭环")

if __name__ == "__main__":
    main()
