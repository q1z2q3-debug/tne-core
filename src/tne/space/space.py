"""
TNE-Core · 相空间内核（Space）
三元九维 19683 态空间 · 三态编码/解码 · 全集枚举

公理锁定（不可修改）：
  单爻三态：阳(+1) / 中和(0) / 阴(-1)
  卦体：9 维有序向量 G=[g1..g9]，g1=上爻，g9=下爻
  全集：3^9 = 19683 卦
"""
from __future__ import annotations
from itertools import product
from typing import List, Tuple

YAO_STATES = (-1, 0, 1)
YAO_NAMES = {-1: "阴", 0: "中和", 1: "阳"}
YAO_SYMBOLS = {-1: "━ ━", 0: "━━━", 1: "━━━"}  # 0/1 同为阳爻显示，中和用特殊标记
YAO_SHORT = {-1: "阴", 0: "中", 1: "阳"}

DIM = 9
TOTAL_STATES = 3 ** DIM  # 19683


def encode(gua: List[int]) -> int:
    """九维三态向量 → 整数索引（base-3，g1 为最高位）。
    取值范围 [0, 19682]。"""
    assert len(gua) == DIM, f"卦体必须为 {DIM} 维，当前 {len(gua)}"
    idx = 0
    for g in gua:
        assert g in YAO_STATES, f"爻态必须 ∈ {{-1,0,1}}，当前 {g}"
        idx = idx * 3 + (g + 1)  # -1→0, 0→1, +1→2
    return idx


def decode(idx: int) -> List[int]:
    """整数索引 → 九维三态向量。"""
    assert 0 <= idx < TOTAL_STATES, f"索引越界 [0,{TOTAL_STATES-1}]，当前 {idx}"
    gua = []
    for _ in range(DIM):
        gua.append((idx % 3) - 1)
        idx //= 3
    gua.reverse()
    return gua


def all_states() -> List[List[int]]:
    """全集 19683 卦向量枚举。"""
    return [list(p) for p in product(YAO_STATES, repeat=DIM)]


def all_indices() -> range:
    """全集索引范围。"""
    return range(TOTAL_STATES)


def gua_str(gua: List[int]) -> str:
    """卦向量的可读字符串，从上爻到下爻。"""
    return "[" + ",".join(f"{g:+d}" for g in gua) + "]"


def gua_yao_str(gua: List[int]) -> str:
    """用 阴/中/阳 表示的爻序列。"""
    return "".join(YAO_SHORT[g] for g in gua)


def validate(gua: List[int]) -> bool:
    """校验卦向量合法性。"""
    return len(gua) == DIM and all(g in YAO_STATES for g in gua)
