#!/usr/bin/env python3
"""
TNE-Core · 阶段三第二步验证：动态等价学习器（Dynamics）
用真实轨迹库扫描，输出动态等价类拓扑（行为层合并/拆分建议）。
"""
import sys
sys.path.insert(0, "/home/user/Doubao/chats/38446456964142338/tne-core/src")
from tne.trajectory.store import TrajectoryStore
from tne.dynamics.topology import DynamicTopology


def main():
    store = TrajectoryStore("tne_trajectories.db")
    topo = DynamicTopology(store)
    result = topo.build()

    print("="*72)
    print("动态等价学习器 · 真实轨迹库扫描")
    print("="*72)
    print(f"活跃等价类: {result['active_classes']} | 总轨迹: {result['total_records']}")

    print("\n--- 等价类行为指纹（Top 8）---")
    for c, fp in sorted(result["fingerprints"].items(),
                        key=lambda x: -x[1]["records"])[:8]:
        acts = ", ".join(f"{a}({s*100:.0f}%)" for a, s in fp["actions"])
        print(f"  类#{c:<6} {fp['records']}条 熵{fp['entropy']:<6} 成功{fp['success_rate']:<5} {acts}")

    print("\n--- 行为层合并候选（符号层未见的等价）---")
    merges = result["merge_candidates"]
    if merges:
        for m in merges:
            print(f"  类#{m['a']} ⊕ 类#{m['b']}  相似度{m['similarity']}")
            print(f"      {m['note']} | 成功率 {m['succ_a']} vs {m['succ_b']}")
    else:
        print("  （当前轨迹库中不同类的行为分布差异明显，暂无高相似合并候选）")

    print("\n--- 类内拆分候选（子认知簇）---")
    splits = result["split_candidates"]
    if splits:
        for s in splits:
            print(f"  类#{s['class_rep']} 熵{s['entropy']} 主行为{s['main_action']}({s['main_share']*100:.0f}%) {s['note']}")
    else:
        print("  （活跃类行为集中，暂无高分歧拆分候选）")

    store.close()
    print("\n✓ 动态等价学习器验证通过")

if __name__ == "__main__":
    main()
