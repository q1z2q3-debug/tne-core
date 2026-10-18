"""
TNE-Core · 动态等价学习器（Dynamics）

静态 5002 等价类是结构算子（符号层）定义的：错/综/错综只保证符号对称，
未必贴合真实认知。动态等价学习从"行为效果"发现新的等价：

  - merge_finder  行为分布相似的类 → 合并（行为层等价）
  - split_finder  类内行为分歧大 → 拆分（子行为簇）
  - topology      输出动态等价类映射表

核心思想：两个不同卦位/等价类，若在真实任务中始终采用相同行为且效果相同，
则它们在"行为认知层"等价——即使符号层没有错/综关系。
"""
from __future__ import annotations
from typing import List, Dict, Any, Optional
from collections import Counter
import math

# 系统内部元行为（测试/推理记录），不参与行为等价学习
META_ACTIONS = {"e2e_test", "llm_reasoning", "llm_tool", "demo_test",
                "sys", "test", "verify"}


def _is_meta(action: str) -> bool:
    a = (action or "").strip().lower()
    return a in META_ACTIONS or a.startswith(("e2e_", "llm_", "test_", "demo_"))


def _behavior_vector(records: List[Dict]) -> Dict[str, int]:
    """等价类 → 真实行为计数向量 {action: count}（过滤元行为）。"""
    return Counter(r["action"] for r in records
                   if not _is_meta(r.get("action", "")))


def _normalize(vec: Dict[str, int]) -> Dict[str, float]:
    """归一化为行为分布（概率）。"""
    total = sum(vec.values())
    if total == 0:
        return {}
    return {k: v / total for k, v in vec.items()}


def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    """两个行为分布的余弦相似度。"""
    keys = set(a) | set(b)
    dot = sum(a.get(k, 0) * b.get(k, 0) for k in keys)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _entropy(dist: Dict[str, float]) -> float:
    """行为分布熵（0 = 单一行为，大 = 分歧大）。"""
    return -sum(p * math.log2(p) for p in dist.values() if p > 0)


def _success_rate(records: List[Dict]) -> float:
    if not records:
        return 0.0
    return sum(1 for r in records if r["outcome"] == "success") / len(records)


def _avg_reward(records: List[Dict]) -> float:
    if not records:
        return 0.0
    return sum(r["reward"] or 0.0 for r in records) / len(records)


class MergeFinder:
    """行为分布相似的类 → 建议合并。"""

    def __init__(self, similarity_threshold: float = 0.8,
                 min_records: int = 2, min_success_gap: float = 0.0):
        self.sim_th = similarity_threshold
        self.min_rec = min_records
        self.succ_gap = min_success_gap

    def find(self, class_records: Dict[int, List[Dict]]) -> List[Dict[str, Any]]:
        """
        class_records: {class_rep: [轨迹记录...]}（仅活跃类）
        返回合并建议列表 [{a, b, similarity, note}]
        """
        classes = [c for c, recs in class_records.items()
                   if len(recs) >= self.min_rec]
        merged = []
        for i in range(len(classes)):
            for j in range(i + 1, len(classes)):
                ca, cb = classes[i], classes[j]
                da = _normalize(_behavior_vector(class_records[ca]))
                db = _normalize(_behavior_vector(class_records[cb]))
                sim = _cosine(da, db)
                if sim >= self.sim_th:
                    sa = _success_rate(class_records[ca])
                    sb = _success_rate(class_records[cb])
                    if abs(sa - sb) <= self.succ_gap:
                        merged.append({
                            "a": ca, "b": cb,
                            "similarity": round(sim, 3),
                            "actions_a": sorted(da.items(), key=lambda x: -x[1])[:3],
                            "actions_b": sorted(db.items(), key=lambda x: -x[1])[:3],
                            "succ_a": round(sa, 2), "succ_b": round(sb, 2),
                            "note": "行为分布高度相似且效果一致，行为层等价",
                        })
        return merged


class SplitFinder:
    """类内行为分歧大 → 建议拆分。"""

    def __init__(self, entropy_threshold: float = 1.0,
                 min_records: int = 4, main_share_threshold: float = 0.6):
        self.ent_th = entropy_threshold
        self.min_rec = min_records
        self.main_th = main_share_threshold

    def find(self, class_records: Dict[int, List[Dict]]) -> List[Dict[str, Any]]:
        """
        返回拆分建议 [{class_rep, entropy, main_action, main_share, note}]
        """
        splits = []
        for c, recs in class_records.items():
            if len(recs) < self.min_rec:
                continue
            dist = _normalize(_behavior_vector(recs))
            if not dist:
                continue
            ent = _entropy(dist)
            main_action, main_share = max(dist.items(), key=lambda x: x[1])
            if ent >= self.ent_th and main_share < self.main_th:
                splits.append({
                    "class_rep": c, "entropy": round(ent, 3),
                    "main_action": main_action, "main_share": round(main_share, 3),
                    "n_actions": len(dist), "records": len(recs),
                    "note": "类内行为分歧大，可能包含多个子认知簇，建议拆分",
                })
        return splits


class DynamicTopology:
    """汇总输出动态等价类映射表。"""

    def __init__(self, store):
        self.store = store
        self.merge_finder = MergeFinder()
        self.split_finder = SplitFinder()

    def build(self, limit: int = 500) -> Dict[str, Any]:
        """扫描轨迹库所有活跃类，输出动态等价拓扑。"""
        # 收集所有出现过的等价类
        rows = self.store.conn.execute(
            "SELECT DISTINCT class_rep FROM trajectories ORDER BY class_rep").fetchall()
        active = {}
        for r in rows:
            c = r["class_rep"]
            active[c] = self.store.query_by_class(c, limit=limit)

        # 每个类的行为指纹（供展示）
        fingerprints = {}
        for c, recs in active.items():
            dist = _normalize(_behavior_vector(recs))
            fingerprints[c] = {
                "records": len(recs),
                "actions": sorted(dist.items(), key=lambda x: -x[1])[:5],
                "entropy": round(_entropy(dist), 3),
                "success_rate": round(_success_rate(recs), 2),
                "avg_reward": round(_avg_reward(recs), 3),
            }

        merges = self.merge_finder.find(active)
        splits = self.split_finder.find(active)

        return {
            "active_classes": len(active),
            "total_records": sum(len(r) for r in active.values()),
            "fingerprints": fingerprints,
            "merge_candidates": merges,
            "split_candidates": splits,
        }
