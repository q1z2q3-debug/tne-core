"""
CogOS · 语义轴与领域帧（四层调制的第①②层）

四层调制不是同时叠加，而是分时介入：
  ① 领域帧   → 映射前，给基准语义轴与长期偏移 bias
  ② 上下文   → 映射前，一次性调临时权重 weights
  ③ 共振反馈 → 映射后，用初步卦位找记忆共振，迭代调轴重映射
  ④ 演化累积 → 跨轮，把上轮校准写回 bias

本模块只负责 ①②④ 的数据结构与更新规则；③ 在 mapping.py 中完成。
"""

from dataclasses import dataclass, field
from typing import Dict, List

DIM = 9

# g1..g9 的维度名，与 space/semantics.py 的 YOURAO_SEMANTICS 对齐
DIM_NAMES = [
    "vision",      # g1 愿景定向
    "decision",    # g2 决断定夺
    "output",      # g3 产出输出
    "method",      # g4 方法路径
    "core",        # g5 核心主态
    "relation",    # g6 协作关联
    "input",       # g7 输入素材
    "accumulate",  # g8 积累沉淀
    "root",        # g9 根性立身
]


@dataclass
class SemanticFrame:
    """一套语义轴：基准权重 + 长期偏移。

    weights: 每个维度的话语权重（上下文/领域调制后）
    bias:    每个维度的长期偏移（演化累积产出，跨轮生效）
    """

    name: str = "general"
    weights: List[float] = field(default_factory=lambda: [1.0] * DIM)
    bias: List[float] = field(default_factory=lambda: [0.0] * DIM)

    def copy(self) -> "SemanticFrame":
        return SemanticFrame(self.name, list(self.weights), list(self.bias))

    def blended_with(self, other: "SemanticFrame", alpha: float = 0.5) -> "SemanticFrame":
        """与另一套轴融合（共振反馈用），alpha 为 other 的占比。"""
        w = [(1 - alpha) * a + alpha * b for a, b in zip(self.weights, other.weights)]
        b_ = [(1 - alpha) * a + alpha * b for a, b in zip(self.bias, other.bias)]
        return SemanticFrame(f"{self.name}+{other.name}", w, b_)

    def normalize(self) -> "SemanticFrame":
        """权重归一到均值 1，保持相对结构不变。"""
        mean = sum(self.weights) / DIM or 1.0
        self.weights = [max(0.05, w / mean) for w in self.weights]
        return self


# ── 预置领域帧（第①层：选基准） ───────────────────────────────
DOMAIN_FRAMES: Dict[str, SemanticFrame] = {
    # 通用：不偏科
    "general": SemanticFrame("general"),
    # 技术/工程：方法、输入、产出优先
    "technical": SemanticFrame(
        "technical",
        weights=[0.8, 0.9, 1.2, 1.8, 1.1, 0.9, 1.5, 1.0, 0.8],
    ),
    # 情绪/关系：核心态、协作、根性优先
    "emotional": SemanticFrame(
        "emotional",
        weights=[0.9, 0.9, 0.8, 0.7, 1.8, 1.7, 0.8, 1.1, 1.3],
    ),
    # 决策/战略：愿景、决断、方法优先
    "strategic": SemanticFrame(
        "strategic",
        weights=[1.8, 1.8, 1.0, 1.4, 1.1, 0.9, 1.0, 0.9, 1.0],
    ),
    # 学习/积累：输入、积累、方法优先
    "learning": SemanticFrame(
        "learning",
        weights=[0.9, 0.8, 0.9, 1.5, 1.0, 1.0, 1.7, 1.8, 1.1],
    ),
}


def get_domain_frame(name: str) -> SemanticFrame:
    """取领域帧副本；未知领域回落到 general。"""
    return DOMAIN_FRAMES.get(name, DOMAIN_FRAMES["general"]).copy()


# ── 第④层：演化累积（跨轮写回 bias） ──────────────────────────
def accumulate_bias(frame: SemanticFrame, residual: List[float], lr: float = 0.15, cap: float = 1.5) -> SemanticFrame:
    """把一轮的校准残差累积进基准偏移。residual 为 9 维，建议是
    (本轮稳定卦位 - 首遍卦位) 的加权差，表示"轴偏了多少"。"""
    new_bias = []
    for b, r in zip(frame.bias, residual):
        v = b + lr * r
        v = max(-cap, min(cap, v))
        new_bias.append(v)
    return SemanticFrame(frame.name, list(frame.weights), new_bias)
