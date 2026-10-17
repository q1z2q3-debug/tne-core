"""
TNE-Core · 习惯偏置查询（Habit Query）
查当前态的历史行为分布，返回习惯建议。
命中 → 习惯模式（推荐复用高频成功行为）
未命中 → 探索模式（建议主动尝试）
"""
from __future__ import annotations
from typing import List, Dict, Optional, Any
from .store import TrajectoryStore
from ..space.space import encode
from ..space.equivalence import class_of


class HabitQuery:
    def __init__(self, store: TrajectoryStore, min_freq: int = 3, success_threshold: float = 0.6):
        self.store = store
        self.min_freq = min_freq          # 最低命中次数才算"有习惯"
        self.success_threshold = success_threshold  # 成功率阈值

    def query(self, gua: List[int]) -> Dict[str, Any]:
        """查询当前卦位的习惯偏置。
        返回 {mode, class_rep, freq, top_actions, recommendation}
        mode: 'habit'（命中习惯）/ 'explore'（未命中，建议探索）
        """
        gua_idx = encode(gua)
        rep = class_of(gua_idx)
        records = self.store.query_by_class(rep, limit=500)
        freq = len(records)

        if freq < self.min_freq:
            return {
                "mode": "explore",
                "class_rep": rep,
                "gua_idx": gua_idx,
                "freq": freq,
                "recommendation": "等价类命中不足，进入探索模式：建议主动尝试多种行为并记录结果。",
                "top_actions": [],
            }

        # 统计行为分布
        action_stats: Dict[str, Dict] = {}
        for r in records:
            a = r["action"]
            if a not in action_stats:
                action_stats[a] = {"count": 0, "success": 0, "reward_sum": 0.0}
            action_stats[a]["count"] += 1
            if r["outcome"] == "success":
                action_stats[a]["success"] += 1
            action_stats[a]["reward_sum"] += r["reward"] or 0.0

        # 按 (成功率, 频次) 排序
        ranked = []
        for a, s in action_stats.items():
            rate = s["success"] / s["count"] if s["count"] > 0 else 0.0
            ranked.append({
                "action": a,
                "count": s["count"],
                "success_rate": round(rate, 3),
                "avg_reward": round(s["reward_sum"] / s["count"], 3),
            })
        ranked.sort(key=lambda x: (x["success_rate"], x["count"]), reverse=True)

        best = ranked[0] if ranked else None
        if best and best["success_rate"] >= self.success_threshold:
            rec = f"习惯模式：推荐复用「{best['action']}」（历史 {best['count']} 次，成功率 {best['success_rate']:.0%}）。"
            mode = "habit"
        else:
            rec = "历史行为成功率不足，建议谨慎复用，优先探索新行为。"
            mode = "habit_weak"

        return {
            "mode": mode,
            "class_rep": rep,
            "gua_idx": gua_idx,
            "freq": freq,
            "top_actions": ranked[:5],
            "recommendation": rec,
        }
