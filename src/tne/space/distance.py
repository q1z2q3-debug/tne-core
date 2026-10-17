"""
TNE-Core · 空间距离（Distance）
欧氏距离 + Top-N 邻域检索。

D(G_A, G_B) = sqrt(Σ (g_Ai - g_Bi)²)
用途：输入目标 9 维向量，在 19683 全集检索 Top-N 最邻近卦。
"""
from __future__ import annotations
import math
from typing import List, Tuple
from .space import TOTAL_STATES, decode, validate, DIM


def euclidean(a: List[int], b: List[int]) -> float:
    """两卦的欧氏距离。"""
    validate(a)
    validate(b)
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def squared_distance(a: List[int], b: List[int]) -> int:
    """平方距离（整数，用于排序时避免开方，更快）。"""
    validate(a)
    validate(b)
    return sum((x - y) ** 2 for x, y in zip(a, b))


def nearest_neighbors(target: List[int], top_n: int = 10) -> List[Tuple[int, float, List[int]]]:
    """在 19683 全集中检索 Top-N 最邻近卦。
    返回 [(索引, 距离, 卦向量), ...]，按距离升序。
    """
    validate(target)
    results = []
    for idx in range(TOTAL_STATES):
        gua = decode(idx)
        d = squared_distance(target, gua)
        results.append((d, idx, gua))
    results.sort(key=lambda x: x[0])
    return [(idx, math.sqrt(d), gua) for d, idx, gua in results[:top_n]]


def nearest_in_class(target: List[int], class_rep: int, top_n: int = 5,
                     class_members: List[int] = None) -> List[Tuple[int, float, List[int]]]:
    """在指定等价类成员中检索 Top-N 最邻近卦。"""
    if class_members is None:
        from .equivalence import class_members as get_members
        class_members = get_members(class_rep)
    results = []
    for idx in class_members:
        gua = decode(idx)
        d = squared_distance(target, gua)
        results.append((d, idx, gua))
    results.sort(key=lambda x: x[0])
    return [(idx, math.sqrt(d), gua) for d, idx, gua in results[:top_n]]
