"""
CogOS · 三层分析

层级 1 卦层：整体姿态（净极性 / 充能 / 穷度 / 是否到穷点）
层级 2 爻层：逐爻语义（九维各自的状态与解读）
层级 3 索引层：由算子变换与空间邻域给出的关联线索（错/综/互/变 + 近邻）

分析产物同时用于共振创造：索引层即使与内容不相关，也可以指导内容层联想。
"""

from typing import Dict, List

from tne.space.bian import exhaustion, is_exhausted, exhaustion_positions
from tne.space.operators import f_cuoo, f_zong, f_cuoo_zong, f_hu, f_bian, equivalence_set
from tne.space.semantics import describe, YOURAO_SEMANTICS
from tne.space.distance import nearest_neighbors
from tne.space.space import gua_str

DIM = 9


def analyze_gua(gua: List[int]) -> Dict:
    """层级 1：卦层整体姿态。"""
    total = sum(gua)
    polar = round(total / DIM, 3)
    yang = sum(1 for g in gua if g == 1)
    yin = sum(1 for g in gua if g == -1)
    neutral = DIM - yang - yin
    ex = exhaustion(gua)
    return {
        "gua_str": gua_str(gua),
        "net_polarity": polar,
        "yang": yang,
        "yin": yin,
        "neutral": neutral,
        "balance": abs(polar),
        "exhaustion": {k: round(v, 3) if isinstance(v, float) else v for k, v in ex.items()},
        "at_turning_point": is_exhausted(gua),
        "turning_positions": exhaustion_positions(gua),
        "posture": _posture_name(polar, ex.get("total", 0.0) if isinstance(ex.get("total"), (int, float)) else ex.get("exhaustion", 0.0)),
    }


def _posture_name(polar: float, ex: float) -> str:
    if ex >= 0.42:
        return "穷极将变"
    if polar >= 0.3:
        return "进取显发"
    if polar <= -0.3:
        return "收敛内蓄"
    return "居中平衡"


def analyze_yaos(gua: List[int]) -> List[Dict]:
    """层级 2：爻层逐维解读。"""
    out = []
    for i in range(DIM):
        sem = YOURAO_SEMANTICS[i + 1]
        v = gua[i]
        out.append(
            {
                "g": i + 1,
                "dim": sem["name"],
                "pos": sem["pos"],
                "value": v,
                "state": {1: "阳·显生发", 0: "中和·居间", -1: "阴·收敛"}[v],
                "text": {1: sem["+1"], 0: sem["0"], -1: sem["-1"]}[v],
            }
        )
    return out


def analyze_index(gua: List[int], top_n: int = 6) -> Dict:
    """层级 3：索引层（算子变换 + 空间邻域），为联想提供结构线索。"""
    return {
        "cuoo": f_cuoo(gua),
        "zong": f_zong(gua),
        "cuoo_zong": f_cuoo_zong(gua),
        "hu": f_hu(gua),                       # 5维内核，不补齐
        "hu_str": gua_str(f_hu(gua)),
        "eq_class_size": len(equivalence_set(gua)),
        "neighbors": [
            {"gua": g, "gua_str": gua_str(g), "distance": round(d, 3)}
            for _, d, g in nearest_neighbors(gua, top_n=top_n)
        ],
        "moving_candidates": [i + 1 for i, v in enumerate(gua) if v == 0],  # 中和位最易动
    }


def analyze(gua: List[int], text: str = "") -> Dict:
    """三层合一。"""
    return {
        "layer1_gua": analyze_gua(gua),
        "layer2_yao": analyze_yaos(gua),
        "layer3_index": analyze_index(gua),
        "input": text[:120],
    }
