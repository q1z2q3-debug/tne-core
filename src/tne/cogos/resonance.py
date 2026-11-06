"""
CogOS · 共振创造（异质同构）

共振不是"找相似的原文"，而是"找结构相同、内容无关"的东西：
  索引层（卦位/算子/邻域）先建立联系，内容层再被结构牵引。

产出物是"联想"，不是"复制"。共振强度由结构关系决定：
  - 对偶（错卦）：完全反向的同一位置
  - 倒映（综卦）：同一序列的镜像读法
  - 内核（互卦）：取中段五维后与对方重合
  - 邻近（距离）：处在认知空间的相邻格
"""

from typing import Dict, List

from tne.space.operators import f_cuoo, f_zong, f_hu
from tne.space.space import gua_str
from tne.space.semantics import YOURAO_SEMANTICS

DIM = 9

REL_LABEL = {
    "dual": "对偶",
    "mirror": "倒映",
    "kernel": "内核",
    "near": "邻近",
}


def _structure_relation(cur: List[int], other: List[int]) -> str:
    if other == f_cuoo(cur):
        return "dual"
    if other == f_zong(cur):
        return "mirror"
    if len(other) == 5 and other == f_hu(cur):
        return "kernel"
    return "near"


def _layer(i: int) -> str:
    return "上" if i < 3 else ("中" if i < 6 else "根")


def _dominant_dims(gua: List[int], k: int = 2) -> List[int]:
    """取最"有话说"的维度：非中和且靠前的。"""
    idx = [i for i, v in enumerate(gua) if v != 0]
    return idx[:k] if idx else list(range(k))


def build(
    cur_gua: List[int],
    text: str,
    associations: List[Dict],
    index: Dict,
) -> List[Dict]:
    """从联想候选中生成共振条目（结构 → 语义 → 人话）。"""
    out: List[Dict] = []
    dims = _dominant_dims(cur_gua)

    for m in associations[:3]:
        item = m["item"]
        rel = _structure_relation(cur_gua, item.gua)
        # 结构对齐：同位置同态的维度
        shared = [
            i
            for i in range(min(DIM, len(item.gua)))
            if cur_gua[i] != 0 and cur_gua[i] == item.gua[i]
        ]
        anchor = shared[:2] if shared else dims

        anchors_text = "、".join(f"{YOURAO_SEMANTICS[i+1]['name']}（{_layer(i)}层）" for i in anchor)
        out.append(
            {
                "relation": REL_LABEL[rel],
                "gua_str": gua_str(item.gua),
                "content": item.content[:80],
                "anchors": anchor,
                "strength": round(1.0 / (1.0 + m["distance"]) + (0.3 if rel != "near" else 0.0), 3),
                "sentence": (
                    f"两者的关注点都落在 {anchors_text} 上（{REL_LABEL[rel]}关系），"
                    f"但内容讲的是另一件事：{item.content[:40]}"
                ),
            }
        )

    # 索引层的纯结构线索：即便没有内容候选，也给出可能的分支
    if not out:
        hu = index.get("hu") or []
        out.append(
            {
                "relation": "内核",
                "gua_str": gua_str(hu) if hu else "",
                "content": "无内容候选，仅有内核投影",
                "anchors": dims,
                "strength": 0.35,
                "sentence": "当前没有可对齐的旧内容，这一步只留下内核线索，供后续分析使用。",
            }
        )
    out.sort(key=lambda x: x["strength"], reverse=True)
    return out
