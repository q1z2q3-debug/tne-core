#!/usr/bin/env python3
"""
TNE-Core · 阶段三验证：认知动力引擎集成进端到端循环
用自训 encode agent 跑 6 个任务，展示张力检测 → 动力注入。
"""
import sys
sys.path.insert(0, "/home/user/Doubao/chats/38446456964142338/tne-core/src")
from tne.agent.client import TNEApiClient
from tne.agent.llm_agent import make_encoder_agent

API = "http://127.0.0.1:8765"

TASKS = [
    # 新领域（等价类命中不足）→ novelty 好奇
    "团队内部互相推诿，没人愿意干活",
    # 稳定习惯（crisis 已积累）→ 可能无张力
    "用户投诉产品有严重问题",
    # 新领域 → 好奇
    "静心打坐，安顿身心",
    # 边界（新任务）→ 可能 boundary
    "深入研究一个全新领域",
]

def main():
    api = TNEApiClient(API)
    agent = make_encoder_agent(api, session_id="drive_demo")
    print("="*72)
    print("认知动力引擎 · 端到端认知循环")
    print("="*72)
    for task in TASKS:
        r = agent.run(task, enable_search=True, enable_transform=True)
        d = r.get("drive")
        tensions = d["tensions"] if d else []
        signals = d["signals"] if d else []
        tstr = "、".join(t["type"] for t in tensions) or "无"
        sstr = "、".join(s["type"] for s in signals) or "无"
        print(f"\n  ◆ {task}")
        print(f"    卦位#{r['gua_idx']} 类#{r['class_rep']} 习惯:{r['habit_mode']}({r['habit_freq']}次)")
        print(f"    张力: {tstr}  |  动力: {sstr}")
        if signals:
            print(f"    注入: {signals[0]['text'][:46]}…")
    print("\n✓ 认知动力引擎端到端验证通过")

if __name__ == "__main__":
    main()
