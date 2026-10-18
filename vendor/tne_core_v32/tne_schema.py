"""TNE-Core v3.2 - 统一数据契约

v3.1 及之前，TrajectoryRecord 在 tne_distiller 与 tne_trajectory_collector 里
各定义了一份 —— 两份字段不一致（collector 版没有 description 默认值对齐），
main.py 用 distiller 版却按 collector 版传参，直接 TypeError。
这里定为唯一真源，两个模块从此处 import 并 re-export，保证序列化完全兼容。
"""

import time
from dataclasses import dataclass, field, asdict, fields
from typing import List, Dict, Any, Optional


@dataclass
class TrajectoryRecord:
    """一条完整轨迹。字段顺序即 jsonl 落盘顺序。"""
    task_id: str
    timestamp: float
    task_type: str
    state_before: List[int]
    state_after: List[int]
    habits_used: List[str]
    efficacy_score: float
    patterns_extracted: List[str] = field(default_factory=list)
    description: str = ""

    # --- 序列化 ---
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrajectoryRecord":
        known = {f.name for f in fields(cls)}
        extra = set(d) - known
        if extra:
            # 不静默吞掉未知字段，避免老数据里的信息无声丢失
            raise ValueError(f"轨迹含未知字段 {sorted(extra)}；已知字段 {sorted(known)}")
        missing = {f.name for f in fields(cls) if f.name not in d}
        if missing:
            raise ValueError(f"轨迹缺字段 {sorted(missing)}")
        return cls(**d)

    @classmethod
    def create(cls, task_id: str, task_type: str, state_before: List[int],
               habits_used: List[str], efficacy_score: float,
               state_after: List[int] = None, description: str = "",
               timestamp: float = None) -> "TrajectoryRecord":
        """便捷构造：state_after 缺省等于 state_before（状态未发生变化）。"""
        return cls(
            task_id=task_id,
            timestamp=timestamp if timestamp is not None else time.time(),
            task_type=task_type,
            state_before=list(state_before),
            state_after=list(state_after) if state_after is not None else list(state_before),
            habits_used=list(habits_used),
            efficacy_score=efficacy_score,
            patterns_extracted=[],
            description=description,
        )

    # --- 校验 ---
    def validate(self) -> List[str]:
        errs = []
        if len(self.state_before) != 9:
            errs.append(f"state_before 长度应为 9，实际 {len(self.state_before)}")
        if len(self.state_after) != 9:
            errs.append(f"state_after 长度应为 9，实际 {len(self.state_after)}")
        for name in ("state_before", "state_after"):
            v = getattr(self, name)
            if any(x not in (-1, 0, 1) for x in v):
                errs.append(f"{name} 含非 -1/0/1 值: {v}")
        if not 0.0 <= self.efficacy_score <= 1.0:
            errs.append(f"efficacy_score 应在 0~1，实际 {self.efficacy_score}")
        if len(self.habits_used) < 2:
            errs.append("habits_used 少于 2 项，永远无法蒸馏出模式")
        return errs


def state_signature(state: List[int]) -> Dict[str, Any]:
    """把九维向量压成触发特征串。

    v3.1 的 _compat() 读 trigger_conditions['typical_state_signature']，
    但那个键从来没被写入过 —— 所有模式恒走默认值 'p5z3n1'，
    等于推荐时完全没用上状态信息。这里提供真正的生成与解析。
    """
    p = sum(1 for x in state if x > 0)
    z = sum(1 for x in state if x == 0)
    n = sum(1 for x in state if x < 0)
    return {"sig": f"p{p}z{z}n{n}", "p": p, "z": z, "n": n}


def parse_signature(sig: str) -> Dict[str, int]:
    """'p5z3n1' -> {'p':5,'z':3,'n':1}；非法输入回落中位。"""
    import re
    m = re.match(r"^p(\d+)z(\d+)n(\d+)$", sig or "")
    if not m:
        return {"p": 5, "z": 3, "n": 1}
    return {"p": int(m.group(1)), "z": int(m.group(2)), "n": int(m.group(3))}


def signature_distance(state: List[int], sig: str) -> float:
    """状态与模式特征的距离，0=完全吻合，1=最远。最大距离为 9（九维全错）。"""
    a = state_signature(state)
    b = parse_signature(sig)
    d = abs(a["p"] - b["p"]) + abs(a["z"] - b["z"]) + abs(a["n"] - b["n"])
    return d / 2.0 / 9.0  # p/z/n 三者偏差总和最大 18，归一到 0~1
