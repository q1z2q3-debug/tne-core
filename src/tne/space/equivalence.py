"""
TNE-Core · 等价类（Equivalence）
遍历全集 19683 卦，按 {G, F_C(G), F_Z(G), F_CZ(G)} 合并重复卦，
统计全部独立等价组。预期结果：5002 组。
"""
from __future__ import annotations
from typing import Dict, List, Tuple
from .space import TOTAL_STATES, decode, encode
from .operators import equivalence_set


def build_equivalence_classes() -> Tuple[Dict[int, int], List[List[int]]]:
    """构建全集等价类映射。
    返回：
      class_of: {卦索引 → 等价类代表索引}
      groups:   等价类列表，每组为卦索引列表
    """
    class_of: Dict[int, int] = {}
    groups: List[List[int]] = []

    for idx in range(TOTAL_STATES):
        if idx in class_of:
            continue
        gua = decode(idx)
        members = equivalence_set(gua)
        member_indices = sorted(encode(g) for g in members)
        representative = member_indices[0]
        groups.append(member_indices)
        for mi in member_indices:
            class_of[mi] = representative

    return class_of, groups


def count_equivalence_classes() -> int:
    """等价组总数（应为 5002）。"""
    _, groups = build_equivalence_classes()
    return len(groups)


# 模块级缓存（首次调用后缓存，避免重复计算）
_CACHE = None


def get_equivalence() -> Tuple[Dict[int, int], List[List[int]]]:
    """获取缓存的等价类映射。"""
    global _CACHE
    if _CACHE is None:
        _CACHE = build_equivalence_classes()
    return _CACHE


def class_of(idx: int) -> int:
    """查询某卦所属等价类代表索引。"""
    class_of_map, _ = get_equivalence()
    return class_of_map[idx]


def class_members(idx: int) -> List[int]:
    """查询某卦所属等价类的全部成员索引。"""
    class_of_map, groups = get_equivalence()
    rep = class_of_map[idx]
    for g in groups:
        if g[0] == rep:
            return g
    return [rep]
