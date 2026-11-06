"""
CogOS · 映射器（四层调制的完整时序）

一轮映射的生命周期：

  【跨轮】④ 演化累积：frame.bias 已含上轮校准
  【本轮】
    ① 领域帧        → 选基准语义轴
    ② 上下文调制    → 按主题调临时权重
    ③ 第一遍映射    → 内容 → 卦位（粗定位）
    ④ 共振反馈      → 用初步卦位找记忆共振 → 调轴 → 重映射 → 收敛
    ⑤ 稳定卦位 + 置信度

对外只暴露 map_text(text, ...) -> MappingResult。
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from tne.space.space import encode, gua_str
from tne.space.distance import nearest_neighbors

from .axes import DIM, SemanticFrame, get_domain_frame
from .lexicon import raw_scores

# 符号化阈值：低于此分视为"中和"（势能平衡点，不出方向）
SIGN_EPS = 0.5
# 单维置信度归一基准：命中 1~2 次即接近 1.0
CONF_MARGIN = 1.5


@dataclass
class MappingResult:
    gua: List[int]
    index: int
    confidence: float
    first_pass: List[int]
    frame: SemanticFrame
    domain: str
    iterations: int
    hits: List[str] = field(default_factory=list)
    trace: List[Dict] = field(default_factory=list)

    def readable(self) -> str:
        return gua_str(self.gua)

    def to_dict(self) -> Dict:
        return {
            "gua": self.gua,
            "gua_str": self.readable(),
            "index": self.index,
            "confidence": round(self.confidence, 4),
            "first_pass": self.first_pass,
            "domain": self.domain,
            "frame": self.frame.name,
            "iterations": self.iterations,
        }


def _symbolize(score: float, eps: float = SIGN_EPS) -> int:
    if score >= eps:
        return 1
    if score <= -eps:
        return -1
    return 0


def _vectorize(scores: List[float], frame: SemanticFrame, eps: float = SIGN_EPS):
    """把原始分按当前语义轴压成硬三态向量，并给出每维置信度。"""
    gua, confs = [], []
    for i in range(DIM):
        s = scores[i] * frame.weights[i] + frame.bias[i]
        gua.append(_symbolize(s, eps))
        confs.append(min(1.0, abs(s) / CONF_MARGIN))
    return gua, confs


def aggregate_confidence(confs: List[float]) -> float:
    """整体置信度 = 有信号维的平均强度 × sqrt(覆盖率)。

    只对"有方向"的维度平均，避免 6 个中和维把整体拉到很低；
    再用覆盖率开方惩罚"只命中一两维"的稀疏映射。
    """
    active = [c for c in confs if c > 0]
    if not active:
        return 0.0
    coverage = len(active) / DIM
    return round((sum(active) / len(active)) * (coverage ** 0.5), 4)


def detect_domain(text: str) -> str:
    """第②层的粗主题判定：按命中密度选领域帧。"""
    from .lexicon import LEXICON

    counts = {d: 0 for d in ("technical", "emotional", "strategic", "learning")}
    buckets = {
        "technical": [3, 2, 6],
        "emotional": [4, 5, 8],
        "strategic": [0, 1, 3],
        "learning": [6, 7, 3],
    }
    for dom, dims in buckets.items():
        for d in dims:
            counts[dom] += sum(text.count(w) for w in LEXICON[d]["pos"]) + sum(
                text.count(w) for w in LEXICON[d]["neg"]
            )
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else "general"


def context_modulate(frame: SemanticFrame, text: str, domain_hint: str = "") -> SemanticFrame:
    """第②层：上下文调制。主题与领域帧不一致时，向主题帧小幅融合。"""
    dom = domain_hint or detect_domain(text)
    if dom == frame.name or dom == "general":
        return frame
    topic_frame = get_domain_frame(dom)
    out = frame.blended_with(topic_frame, alpha=0.45).normalize()
    out.name = f"{frame.name}@{dom}"
    return out


def map_text(
    text: str,
    domain: str = "",
    frame: Optional[SemanticFrame] = None,
    resonance_provider: Optional[Callable[[List[int]], List[SemanticFrame]]] = None,
    max_iter: int = 3,
) -> MappingResult:
    """把一段自然语言映射到九维卦位。

    resonance_provider: 给定初步卦位，返回候选记忆自带的语义帧（第④层用）。
    没有 provider 时跳过共振反馈，直接输出第一遍结果。
    """
    # ① 领域帧
    base = frame.copy() if frame else get_domain_frame(domain or "general")
    # ② 上下文调制（映射前，一次性）
    ctx_frame = context_modulate(base, text, domain_hint=domain)

    scores, hits = raw_scores(text)
    trace: List[Dict] = []

    # ③ 第一遍映射
    first_gua, confs = _vectorize(scores, ctx_frame)
    cur_frame = ctx_frame
    cur_gua = first_gua
    iterations = 0

    # ④ 共振反馈（映射后介入）
    if resonance_provider is not None:
        stable = False
        while iterations < max_iter and not stable:
            cands = resonance_provider(cur_gua) or []
            if not cands:
                break
            # 取记忆共振中最接近的一套轴做融合
            fused = cur_frame
            for c in cands[:2]:
                fused = fused.blended_with(c, alpha=0.5)
            fused = fused.normalize()
            new_gua, new_confs = _vectorize(scores, fused)
            iterations += 1
            trace.append(
                {
                    "iter": iterations,
                    "frame": fused.name,
                    "gua": new_gua,
                    "changed": new_gua != cur_gua,
                }
            )
            if new_gua == cur_gua:
                stable = True
            cur_frame = fused
            cur_gua = new_gua
            confs = new_confs

    conf = aggregate_confidence(confs)
    return MappingResult(
        gua=cur_gua,
        index=encode(cur_gua),
        confidence=conf,
        first_pass=first_gua,
        frame=cur_frame,
        domain=domain or ctx_frame.name,
        iterations=iterations,
        hits=hits,
        trace=trace,
    )


def neighbors_of(gua: List[int], top_n: int = 8):
    """空间邻域（供记忆调动与联想使用）。"""
    return nearest_neighbors(gua, top_n=top_n)
