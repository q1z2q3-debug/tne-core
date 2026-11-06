"""
CogOS · 五段闭环（映射 → 记忆调动 → 三层分析 → 共振创造 → 人类语言输出）

设计原则：
  - 认知在底层运转，输出只出人类语言；结构（卦位/爻/算子/距离）不对外显露。
  - 映射方向不成立（置信度过低或全中和）时不写入记忆。
  - 空间无候选则不进入内容层；空间有候选但内容不相关，转为联想而非命中。
  - 每轮结束把校准残差累积进该领域的 bias，供下一轮使用（跨轮演化）。
"""

import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .axes import SemanticFrame, accumulate_bias, get_domain_frame
from .mapping import MappingResult, map_text
from .memory import MemorySpace, RecallResult
from .analysis import analyze
from .resonance import build as build_resonance

# 方向成立的最低置信度；低于此不记忆
DIRECTION_CONF = 0.35


def _now() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@dataclass
class Turn:
    text: str
    mapping: MappingResult
    recall: RecallResult
    analysis: Dict
    resonances: List[Dict]
    reply: str
    remembered: bool
    trace: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "mapping": self.mapping.to_dict(),
            "recall": {"status": self.recall.status, "space_candidates": self.recall.space_candidates, "note": self.recall.note},
            "resonances": [{k: r[k] for k in ("relation", "gua_str", "strength", "sentence")} for r in self.resonances],
            "reply": self.reply,
            "remembered": self.remembered,
            "trace": self.trace,
        }


class CogOS:
    def __init__(self, memory_path: Optional[str] = None):
        self.memory = MemorySpace(memory_path)
        self.frames: Dict[str, SemanticFrame] = {}

    def _frame_for(self, domain: str) -> SemanticFrame:
        if domain not in self.frames:
            self.frames[domain] = get_domain_frame(domain)
        return self.frames[domain]

    # ── 主流程 ────────────────────────────────────────────
    def process(self, text: str, domain: str = "", remember: bool = True, respond: bool = True) -> Turn:
        trace: List[str] = []

        # ① 映射（含四层调制与共振反馈）
        mapping = map_text(
            text,
            domain=domain,
            frame=self._frame_for(domain or "general") if domain else None,
            resonance_provider=self.memory.frames_near,
        )
        trace.append(
            f"映射：{mapping.readable()} conf={mapping.confidence} 首遍={mapping.first_pass} 迭代={mapping.iterations}"
        )

        # 方向判定（决定是否记忆）
        direction_ok = mapping.confidence >= DIRECTION_CONF and any(g != 0 for g in mapping.gua)

        # ② 记忆调动（先空间后内容）
        recall = self.memory.recall(text, mapping.gua)
        trace.append(f"记忆：status={recall.status} 空间候选={recall.space_candidates} {recall.note}")

        # ③ 三层分析
        ana = analyze(mapping.gua, text)

        # ④ 共振创造
        res = build_resonance(
            mapping.gua,
            text,
            recall.matches if recall.status in ("hit", "association") else [],
            ana["layer3_index"],
        )
        trace.append(f"共振：{len(res)} 条，最强={res[0]['relation'] if res else '无'}")

        # ⑤ 人类语言输出
        reply = self._compose_reply(mapping, recall, ana, res) if respond else ""
        trace.append("输出：人类语言（结构不外显）")

        # ── 写回 ──
        remembered = False
        if remember and direction_ok and recall.status == "miss":
            self.memory.add(
                gua=mapping.gua,
                key_terms=_key_terms(text, mapping.hits),
                content=text,
                frame=mapping.frame,
                created=_now(),
            )
            remembered = True
            trace.append("写回：新建记忆（该位置此前无内容）")
        elif recall.status == "hit" and recall.matches:
            trace.append(f"写回：命中记忆 id={recall.matches[0]['item'].id}，hits+1")
        elif not direction_ok:
            trace.append("写回：方向未立（置信度不足/全中和），不记忆")
        elif recall.status == "association":
            trace.append("写回：仅留联想，不覆盖原记忆")

        # 跨轮演化：把共振反馈造成的位移累积进该领域 bias
        residual = [float(a - b) for a, b in zip(mapping.gua, mapping.first_pass)]
        if any(r != 0.0 for r in residual):
            dom = domain or mapping.domain or "general"
            self.frames[dom] = accumulate_bias(self._frame_for(dom), residual)
            trace.append("演化：本轮轴位移已累积进领域帧 bias（下一轮生效）")

        return Turn(text, mapping, recall, ana, res, reply, remembered, trace)

    # ── 输出层：结构 → 人类语言 ───────────────────────────
    def _compose_reply(self, m: MappingResult, r: RecallResult, ana: Dict, res: List[Dict]) -> str:
        g1 = ana["layer1_gua"]
        posture = g1["posture"]

        # 主方向：最强非中维（取前两个）
        strong = [y for y in ana["layer2_yao"] if y["value"] != 0]
        strong.sort(key=lambda y: abs(y["value"]), reverse=True)
        focus = "、".join(f"{y['dim']}（{y['text']}）" for y in strong[:2]) if strong else "整体还在摇摆，没有定下重心"

        lines = [f"你现在整体处在「{posture}」的状态，主要发力点在{focus}。"]

        if g1["at_turning_point"]:
            lines.append("这个位置已经走到尽头，继续加码收益会递减，适合换一种做法，而不是再使劲。")

        if r.status == "hit" and r.matches:
            it = r.matches[0]["item"]
            lines.append(f"这个思路你之前走过一遍（记得当时是：{it.content[:36]}），可以直接接着上次的结论往下推。")
        elif r.status == "association" and res:
            lines.append(f"结构上有一件事和它很像，内容却完全不同：{res[0]['sentence']}")
        elif r.status == "miss":
            lines.append("这是一条新的路，之前没有留下任何相近的记录，先把这次想清楚的东西存下来。")

        return "\n".join(lines)


def _key_terms(text: str, hits: List[str]) -> List[str]:
    """从命中解释里抽出关键词，作为内容层的检索锚点。"""
    import re

    terms = []
    for h in hits:
        pieces = re.split(r"[+\-]", h)
        if len(pieces) >= 2 and pieces[1]:
            terms.append(pieces[1])
    # 补几个文本里出现的实词片段（2-4 字）
    for seg in re.findall(r"[\u4e00-\u9fa5]{2,4}", text)[:6]:
        terms.append(seg)
    seen, out = set(), []
    for t in terms:
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out[:10]
