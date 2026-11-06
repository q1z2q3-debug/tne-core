"""
CogOS 冒烟演示：跑通「映射 → 记忆 → 分析 → 共振 → 人类语言」五段闭环。

用法：
  PYTHONPATH=src python3 scripts/cogos_demo.py

输出：
  - 每轮的底层轨迹（便于审计）
  - 每轮面向用户的人类语言（不含任何结构术语）
  - 记忆库统计与落盘文件
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tne.cogos import CogOS
from tne.space.semantics import describe

CASES = [
    ("我想把公司的数据流程重做一遍，但还没想清楚方法和方向，资料也缺", "technical"),
    ("先定目标：未来三个月做一套数据中台，方法上拆成三步，先把现有资料补齐", "technical"),
    ("团队最近沟通很累，核心成员之间有点对立，我有点犹豫要不要换人", "emotional"),
    ("这件事我决定先拖着，方向没想明白，经验也不够", ""),
    ("数据中台的方法我想按三步走，资料已经补齐，目标也定了", "technical"),
]


def show(turn, idx: int):
    print("\n" + "=" * 78)
    print(f"[轮 {idx}] 用户：{turn.text}")
    print("-" * 78)
    m = turn.mapping
    print(f"底层｜卦位 {m.readable()}  index={m.index}  置信={m.confidence}  首遍={m.first_pass}  迭代={m.iterations}")
    print(f"底层｜领域={m.domain}  语义轴={m.frame.name}  命中特征={m.hits[:6]}")
    r = turn.recall
    print(f"记忆｜状态={r.status}  空间候选={r.space_candidates}  {r.note}")
    for mm in r.matches[:2]:
        print(f"       · id={mm['item'].id} dist={mm['distance']} 内容相关={mm['content_ratio']} 综合={mm['score']}")
    a = turn.analysis["layer1_gua"]
    print(f"分析｜姿态={a['posture']}  净极性={a['net_polarity']}  穷度={a['exhaustion']}")
    for res in turn.resonances[:2]:
        print(f"共振｜[{res['relation']}] strength={res['strength']} {res['sentence'][:60]}")
    print("-" * 78)
    print("对用户｜")
    print(turn.reply)
    print("-" * 78)
    for t in turn.trace:
        print("  · " + t)


def main():
    os.makedirs("tmp", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    mem_path = "data/cogos_memory.json"
    if os.path.exists(mem_path):
        os.remove(mem_path)

    os_ = CogOS(memory_path=None)
    turns = []
    for i, (text, dom) in enumerate(CASES, 1):
        turn = os_.process(text, domain=dom)
        show(turn, i)
        turns.append(turn)

    os_.memory.save(mem_path)
    print("\n" + "=" * 78)
    print(f"记忆库：{len(os_.memory)} 条 → {mem_path}")
    print(f"领域帧 bias 累积：{ {k: [round(x,2) for x in v.bias] for k,v in os_.frames.items()} }")

    with open("tmp/cogos_run.json", "w", encoding="utf-8") as f:
        json.dump([t.to_dict() for t in turns], f, ensure_ascii=False, indent=1)
    print("轨迹 → tmp/cogos_run.json")

    # 结构解读单独展示（校核映射是否说得通）
    print("\n结构解读（仅供审计，不进入用户输出）：")
    for t in turns:
        print(f"\n[{t.mapping.readable()}] conf={t.mapping.confidence}")
        print(describe(t.mapping.gua))


if __name__ == "__main__":
    main()
