"""
TNE-Core · 结构算子（Operators）
错卦 F_C / 综卦 F_Z / 错综 F_CZ / 互卦 F_H / 动爻变卦 F_B

公理锁定（不可修改）：
  F_C(+1)=-1, F_C(-1)=+1, F_C(0)=0
  F_Z: 序列反转
  F_CZ = F_C(F_Z(G))  （先综后错）
  F_H: 剔除 g1,g9，取中心连续 5 爻 [g3,g4,g5,g6,g7]
  F_B(+1)=0, F_B(-1)=0, F_B(0)=+1，支持 1~9 爻同动
"""
from __future__ import annotations
from typing import List, Set
from .space import DIM, YAO_STATES, validate


def f_cuoo(gua: List[int]) -> List[int]:
    """错卦 F_C：逐爻对偶。"""
    validate(gua)
    return [-g for g in gua]


def f_zong(gua: List[int]) -> List[int]:
    """综卦 F_Z：序列反转。"""
    validate(gua)
    return list(reversed(gua))


def f_cuoo_zong(gua: List[int]) -> List[int]:
    """错综 F_CZ：先综后错。"""
    return f_cuoo(f_zong(gua))


def f_hu(gua: List[int]) -> List[int]:
    """互卦 F_H：内核投影，输出 5 维中心连续爻 [g3,g4,g5,g6,g7]。
    注意：固定输出 5 维，不补齐为 9 维。"""
    validate(gua)
    # g1=索引0, g9=索引8; g3=索引2, g7=索引6
    return gua[2:7]


def f_bian(gua: List[int], moving_yaos: Set[int]) -> List[int]:
    """动爻变卦 F_B：指定爻位发动。
    moving_yaos: 爻位集合，1-based（1=上爻 g1, 9=下爻 g9）。
    F_B(+1)=0, F_B(-1)=0, F_B(0)=+1。"""
    validate(gua)
    assert all(1 <= y <= DIM for y in moving_yaos), f"爻位必须在 [1,{DIM}]"
    result = list(gua)
    for y in moving_yaos:
        i = y - 1  # 转 0-based
        g = result[i]
        result[i] = 0 if g != 0 else 1
    return result


def equivalence_set(gua: List[int]) -> List[List[int]]:
    """卦 G 的等价集合 {G, F_C(G), F_Z(G), F_CZ(G)}，去重后返回。"""
    validate(gua)
    seen = []
    seen_set = set()
    for g in [gua, f_cuoo(gua), f_zong(gua), f_cuoo_zong(gua)]:
        key = tuple(g)
        if key not in seen_set:
            seen_set.add(key)
            seen.append(g)
    return seen


# 别名（符合论文命名）
F_C = f_cuoo
F_Z = f_zong
F_CZ = f_cuoo_zong
F_H = f_hu
F_B = f_bian
