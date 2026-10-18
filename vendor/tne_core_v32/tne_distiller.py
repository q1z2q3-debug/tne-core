"""TNE-Core v3.2 - Distiller Module

v3.2 改动：
  - TrajectoryRecord 改由 tne_schema 统一定义（消除与 collector 的重复定义）
  - 新增 add_trajectories() 批量蒸馏接口
  - 修复 typical_state_signature 从未被写入的 bug —— 该键是 recommend 排序的
    唯一依据，此前恒走默认值，等于推荐完全没用上状态信息
  - export / load 补齐 trigger_conditions 与 last_used，往返不丢信息
"""

import json
import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from collections import defaultdict

try:  # 既支持作为包导入（tne_core.tne_distiller），也支持直接跑（sys.path 指向本目录）
    from .tne_schema import (TrajectoryRecord, state_signature, parse_signature,
                             signature_distance)
except ImportError:  # pragma: no cover
    from tne_schema import (TrajectoryRecord, state_signature, parse_signature,
                            signature_distance)

__all__ = ["TrajectoryRecord", "ReusablePattern", "TNE_Distiller"]


@dataclass
class ReusablePattern:
    pattern_id: str
    task_type: str
    trigger_conditions: Dict[str, Any]
    execution_chain: List[str]
    success_rate: float
    usage_count: int
    last_used: float
    source_tasks: List[str] = field(default_factory=list)

    def state_signature(self) -> str:
        return self.trigger_conditions.get("typical_state_signature", "p5z3n1")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_type": self.task_type,
            "execution_chain": self.execution_chain,
            "success_rate": self.success_rate,
            "usage_count": self.usage_count,
            "last_used": self.last_used,
            "source_tasks": self.source_tasks,
            "trigger_conditions": self.trigger_conditions,
        }


def _round_to_nine(p: float, z: float, n: float):
    """把三个非负浮点取整，保证和为 9（最大余数法）。

    分别 round 会出问题：p=2.0 / z=7.5 / n=-0.5 之类时
    round 后得到 2+8+0=10 甚至算出 n=-1 这种非法 signature。
    """
    vals = [max(0.0, p), max(0.0, z), max(0.0, n)]
    floors = [int(v) for v in vals]
    need = 9 - sum(floors)
    if need > 0:
        order = sorted(range(3), key=lambda i: vals[i] - floors[i], reverse=True)
        for i in order[:need]:
            floors[i] += 1
    elif need < 0:
        order = sorted(range(3), key=lambda i: vals[i] - floors[i])
        for i in order[:-need]:
            floors[i] = max(0, floors[i] - 1)
    return floors[0], floors[1], floors[2]


class TNE_Distiller:
    def __init__(self, min_samples: int = 2, min_success_rate: float = 0.85,
                 min_efficacy: float = 0.85):
        self.min_samples = min_samples
        self.min_success_rate = min_success_rate
        self.min_efficacy = min_efficacy
        self.patterns: Dict[str, ReusablePattern] = {}
        self.trajectories: List[TrajectoryRecord] = []

    # ---------- 单条 / 批量 ----------

    def add_trajectory(self, record: TrajectoryRecord) -> List[str]:
        self.trajectories.append(record)
        return self._distill(record)

    def add_trajectories(self, records: List[TrajectoryRecord]) -> Dict[str, Any]:
        """批量蒸馏。返回汇总报告，失败的单条不计入总数但不中断整批。"""
        report = {"total": len(records), "accepted": 0, "rejected": 0,
                  "new_patterns": [], "reject_reasons": defaultdict(int)}
        for rec in records:
            self.trajectories.append(rec)
            errs = rec.validate()
            if errs:
                report["rejected"] += 1
                report["reject_reasons"][errs[0]] += 1
                continue
            new = self._distill(rec)
            report["accepted"] += 1
            report["new_patterns"].extend(new)
        report["new_patterns"] = list(dict.fromkeys(report["new_patterns"]))
        report["reject_reasons"] = dict(report["reject_reasons"])
        return report

    # ---------- 蒸馏核心 ----------

    def _distill(self, record: TrajectoryRecord) -> List[str]:
        extracted = []
        if record.efficacy_score < self.min_efficacy:
            return extracted
        if len(record.habits_used) < 2:
            return extracted

        key = f"{record.task_type}::{'>'.join(record.habits_used)}"
        sig = state_signature(record.state_before)

        if key in self.patterns:
            p = self.patterns[key]
            p.usage_count += 1
            p.last_used = record.timestamp
            p.success_rate = p.success_rate * 0.8 + record.efficacy_score * 0.2
            p.source_tasks.append(record.task_id)
            # 增量平均更新典型触发状态 —— 这是 v3.1 缺失的一环
            p.trigger_conditions = self._merge_signature(p.trigger_conditions, sig)
        else:
            self.patterns[key] = ReusablePattern(
                pattern_id=key,
                task_type=record.task_type,
                trigger_conditions={
                    "min_efficacy": self.min_efficacy,
                    "habit_chain_length": len(record.habits_used),
                    "typical_state_signature": sig["sig"],
                    "_sig_sum_p": sig["p"],
                    "_sig_sum_z": sig["z"],
                    "_sig_sum_n": sig["n"],
                    "_sig_count": 1,
                },
                execution_chain=record.habits_used.copy(),
                success_rate=record.efficacy_score,
                usage_count=1,
                last_used=record.timestamp,
                source_tasks=[record.task_id]
            )
            extracted.append(key)

        record.patterns_extracted.extend(extracted)
        return extracted

    @staticmethod
    def _merge_signature(tc: Dict[str, Any], sig: Dict[str, Any]) -> Dict[str, Any]:
        """把新观测的状态并入模式的典型特征。

        注意：_sig_sum_* 存的是累积和而不是均值。v3.2 初版存了均值再除 cnt，
        均值会逐次衰减（第 2 次 0.5、第 3 次 0.17…），三次之后 signature 就
        失真成 p0z1n8 这种荒谬值。
        """
        cnt = tc.get("_sig_count", 0) + 1
        sp = tc.get("_sig_sum_p", tc.get("_sig_p", 0)) + sig["p"]
        sz = tc.get("_sig_sum_z", tc.get("_sig_z", 0)) + sig["z"]
        sn = tc.get("_sig_sum_n", tc.get("_sig_n", 0)) + sig["n"]
        p, z, n = sp / cnt, sz / cnt, sn / cnt
        pi, zi, ni = _round_to_nine(p, z, n)
        tc.update({
            "typical_state_signature": f"p{pi}z{zi}n{ni}",
            "_sig_sum_p": sp, "_sig_sum_z": sz, "_sig_sum_n": sn,
            "_sig_p": p, "_sig_z": z, "_sig_n": n,  # 均值，仅供外部读取
            "_sig_count": cnt,
        })
        return tc

    # ---------- 推荐 ----------

    def recommend_pattern(self, task_type: str, state: List[int]) -> Optional[ReusablePattern]:
        candidates = []
        for p in self.patterns.values():
            if p.task_type != task_type:
                continue
            if p.success_rate < self.min_success_rate:
                continue
            if p.usage_count < self.min_samples:
                continue
            candidates.append((self._compat(state, p), p))

        if not candidates:
            return None
        # 同分时按成功率再排一次，避免状态距离相同的模式随机胜出
        candidates.sort(key=lambda x: (x[0], x[1].success_rate, x[1].usage_count), reverse=True)
        return candidates[0][1]

    def recommend_with_scores(self, task_type: str, state: List[int],
                              top_k: int = 3) -> List[Dict[str, Any]]:
        """带分数的 Top-K 推荐，便于观察排序依据。"""
        scored = []
        for p in self.patterns.values():
            if p.task_type != task_type:
                continue
            if p.success_rate < self.min_success_rate:
                continue
            if p.usage_count < self.min_samples:
                continue
            scored.append({
                "pattern_id": p.pattern_id,
                "compat": round(self._compat(state, p), 3),
                "success_rate": round(p.success_rate, 3),
                "usage_count": p.usage_count,
                "signature": p.state_signature(),
                "execution_chain": p.execution_chain,
            })
        scored.sort(key=lambda x: (x["compat"], x["success_rate"], x["usage_count"]), reverse=True)
        return scored[:top_k]

    def _compat(self, state: List[int], pattern: ReusablePattern) -> float:
        """v3.1 只比较了正向维度数量，负向维被完全忽略 ——
        [-1,0,0,0,0,0,0,0,0] 与 [0,0,0,0,0,0,0,0,0] 会被判为同样匹配。
        这里改用完整 p/z/n 三维距离。"""
        dist = signature_distance(state, pattern.state_signature())
        return 1.0 - dist

    # ---------- 统计 / 持久化 ----------

    def get_stats(self) -> Dict[str, Any]:
        if not self.patterns:
            return {"total": 0}
        by_type = defaultdict(int)
        for p in self.patterns.values():
            by_type[p.task_type] += 1
        return {
            "total": len(self.patterns),
            "by_type": dict(by_type),
            "avg_success": round(
                sum(p.success_rate for p in self.patterns.values()) / len(self.patterns), 4),
            "total_usage": sum(p.usage_count for p in self.patterns.values()),
            "ready_to_recommend": sum(
                1 for p in self.patterns.values()
                if p.usage_count >= self.min_samples
                and p.success_rate >= self.min_success_rate),
        }

    def export_patterns(self, filepath: str) -> None:
        data = {pid: p.to_dict() for pid, p in self.patterns.items()}
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_patterns(self, filepath: str) -> None:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for pid, d in data.items():
            tc = d.get("trigger_conditions", {})
            # 老数据没有 signature 字段时补一个，保证 _compat 不退化
            if "typical_state_signature" not in tc:
                tc.setdefault("typical_state_signature", "p5z3n1")
                tc.setdefault("_sig_count", 0)
            self.patterns[pid] = ReusablePattern(
                pattern_id=pid,
                task_type=d["task_type"],
                trigger_conditions=tc,
                execution_chain=d["execution_chain"],
                success_rate=d["success_rate"],
                usage_count=d["usage_count"],
                last_used=d.get("last_used", time.time()),
                source_tasks=d.get("source_tasks", [])
            )


distiller = TNE_Distiller()
