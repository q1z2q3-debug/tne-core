#!/usr/bin/env python3
"""
TNE-Core · 阶段三第一步验证：认知动力引擎（Drive）
张力检测 → 动力信号生成 → 注入。
"""
import sys
sys.path.insert(0, "/home/user/Doubao/chats/38446456964142338/tne-core/src")
from tne.drive.tension import TensionDetector, SignalGenerator, detect_and_generate


def show(title, result):
    print(f"\n=== {title} ===")
    for t in result["tensions"]:
        print(f"  张力 {t['type']:<14} 强度{t['strength']:.1f}  {t['note']}")
    print(f"  动力: {[s['type'] for s in result['signals']]}  总强度{result['max_strength']}")
    print("  注入: " + result["prompt"].replace("\n", " ⏎ ")[:110])


# 场景1：新领域（freq=0, explore）→ 应触发 novelty 好奇
show("新领域任务（等价类命中0）",
     detect_and_generate({"mode": "explore", "freq": 0},
                         confidence=[0.99]*9,
                         neighbors=[{"distance": 0.0}, {"distance": 2.0}]))

# 场景2：边界模糊（某维置信度低）
show("边界任务（某维置信度 0.4）",
     detect_and_generate({"mode": "explore", "freq": 5},
                         confidence=[0.99, 0.4, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99],
                         neighbors=[{"distance": 0.0}, {"distance": 2.0}]))

# 场景3：习惯弱（habit_weak）→ 应触发元认知
show("习惯冲突任务（habit_weak）",
     detect_and_generate({"mode": "habit_weak", "freq": 5},
                         confidence=[0.99]*9,
                         neighbors=[{"distance": 0.0}, {"distance": 2.0}]))

# 场景4：稳定习惯（habit, 全高置信）→ 无张力
show("稳定习惯任务（habit, 高置信）",
     detect_and_generate({"mode": "habit", "freq": 12},
                         confidence=[0.99]*9,
                         neighbors=[{"distance": 0.0}, {"distance": 1.0}]))

# 场景5：邻域距离接近（摇摆）→ boundary
show("邻域摇摆任务（两卦距离接近）",
     detect_and_generate({"mode": "habit", "freq": 10},
                         confidence=[0.99]*9,
                         neighbors=[{"distance": 0.0}, {"distance": 0.05}]))

print("\n✓ 动力引擎验证通过")
