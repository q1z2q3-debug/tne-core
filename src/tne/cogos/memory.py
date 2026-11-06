"""
CogOS · 记忆空间（先空间、后内容）

检索顺序严格按两层走：
  1) 空间层：等价类 + 欧氏邻域，找"同一认知位置附近"的记忆
  2) 内容层：仅对空间候选做关键词/内容相关度判定

空间无候选 → 直接 miss，不进入内容层（没有位置就没有该处的内容）。
空间有候选但内容不相关 → 记为 association（联想候选，交给共振层）。
空间有候选且内容相关 → hit（命中），可继续推进。
"""

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from tne.space.equivalence import class_of, class_members
from tne.space.space import encode, gua_str
from tne.space.distance import euclidean

from .axes import SemanticFrame

# 命中阈值：内容相关度
HIT_RATIO = 0.34
# 空间候选上限
SPACE_K = 8


@dataclass
class MemoryItem:
    id: int
    gua: List[int]
    key_terms: List[str]
    content: str
    frame: SemanticFrame
    created: str = ""
    hits: int = 0
    outcome: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "gua": self.gua,
            "key_terms": self.key_terms,
            "content": self.content,
            "frame": {"name": self.frame.name, "weights": self.frame.weights, "bias": self.frame.bias},
            "created": self.created,
            "hits": self.hits,
            "outcome": self.outcome,
        }

    @staticmethod
    def from_dict(d: Dict) -> "MemoryItem":
        fr = d.get("frame") or {}
        return MemoryItem(
            id=d["id"],
            gua=d["gua"],
            key_terms=d.get("key_terms", []),
            content=d.get("content", ""),
            frame=SemanticFrame(fr.get("name", "general"), fr.get("weights", [1.0] * 9), fr.get("bias", [0.0] * 9)),
            created=d.get("created", ""),
            hits=d.get("hits", 0),
            outcome=d.get("outcome"),
        )


@dataclass
class RecallResult:
    status: str                      # hit / association / miss
    space_candidates: int
    matches: List[Dict] = field(default_factory=list)
    note: str = ""


class MemorySpace:
    """以卦位为索引的认知记忆库，可落盘为 JSON。"""

    def __init__(self, path: Optional[str] = None):
        self.items: List[MemoryItem] = []
        self._by_class: Dict[int, List[int]] = {}
        self.path = path
        if path and os.path.exists(path):
            self.load(path)

    # ── 写入 ──────────────────────────────────────────────
    def add(self, gua: List[int], key_terms: List[str], content: str, frame: SemanticFrame,
            created: str = "", outcome: Optional[str] = None) -> MemoryItem:
        item = MemoryItem(
            id=len(self.items) + 1,
            gua=list(gua),
            key_terms=list(key_terms),
            content=content,
            frame=frame.copy(),
            created=created,
            outcome=outcome,
        )
        self.items.append(item)
        self._by_class.setdefault(class_of(encode(gua)), []).append(item.id)
        return item

    # ── 空间层 ────────────────────────────────────────────
    def spatial_candidates(self, gua: List[int], k: int = SPACE_K) -> List[Tuple[MemoryItem, float]]:
        """等价类命中 + 欧氏近邻，返回 (item, 距离)。"""
        if not self.items:
            return []
        cid = class_of(encode(gua))
        cand: List[Tuple[MemoryItem, float]] = []

        # 1) 同等价类（结构等价，最强空间命中）
        ids = self._by_class.get(cid, [])
        by_id = {it.id: it for it in self.items}
        for i in ids:
            it = by_id[i]
            cand.append((it, euclidean(gua, it.gua)))

        # 2) 等价类成员的全部卦位再补最近邻
        for member in class_members(encode(gua)):
            if member == encode(gua):
                continue
            for it in self.items:
                if it.id in [c[0].id for c in cand]:
                    continue
                cand.append((it, euclidean(gua, it.gua)))

        cand.sort(key=lambda x: x[1])
        # 去重保留最小距离
        seen, out = set(), []
        for it, d in cand:
            if it.id in seen:
                continue
            seen.add(it.id)
            out.append((it, d))
            if len(out) >= k:
                break
        return out

    # ── 内容层 ────────────────────────────────────────────
    @staticmethod
    def content_ratio(item: MemoryItem, text: str) -> float:
        """关键词覆盖度：记忆自带 key_terms 有多少出现在当前输入里。"""
        if not item.key_terms:
            return 0.0
        used = sum(1 for t in item.key_terms if t and t in text)
        ratio = used / len(item.key_terms)
        # 内容片段的重合加分（最长公共子串的粗略替代）
        if item.content:
            frag = item.content[:24]
            if frag and frag in text:
                ratio = min(1.0, ratio + 0.25)
        return round(ratio, 4)

    # ── 完整检索 ──────────────────────────────────────────
    def recall(self, text: str, gua: List[int], top_n: int = 3) -> RecallResult:
        spatial = self.spatial_candidates(gua)
        if not spatial:
            return RecallResult("miss", 0, [], "空间无候选：该认知位置没有任何记忆，直接新建")

        scored = []
        for it, dist in spatial:
            ratio = self.content_ratio(it, text)
            # 总分 = 内容相关为主，空间距离为辅
            total = 0.7 * ratio + 0.3 * (1.0 / (1.0 + dist))
            scored.append({"item": it, "distance": round(dist, 3), "content_ratio": ratio, "score": round(total, 4)})
        scored.sort(key=lambda x: x["score"], reverse=True)

        best = scored[0]
        if best["content_ratio"] >= HIT_RATIO:
            best["item"].hits += 1
            return RecallResult("hit", len(spatial), scored[:top_n], "空间有候选且内容相关，命中既有记忆")
        return RecallResult("association", len(spatial), scored[:top_n], "空间有候选但内容不相关，转为联想候选")

    # ── 共振轴提供器（给 mapping 第④层） ─────────────────
    def frames_near(self, gua: List[int], k: int = 4) -> List[SemanticFrame]:
        """只把"强结构关系"的记忆轴交给映射器。

        仅邻近平移（near）不足以调轴——否则无关内容的邻近记忆会污染当前语义轴。
        只有对偶（错卦）、倒映（综卦）、内核（互卦）才构成真正的结构共振。
        """
        from tne.space.operators import f_cuoo, f_zong, f_hu

        out: List[SemanticFrame] = []
        for it, _ in self.spatial_candidates(gua, k=k):
            if it.gua == f_cuoo(gua) or it.gua == f_zong(gua) or it.gua == f_hu(gua):
                out.append(it.frame)
        return out

    # ── 持久化 ────────────────────────────────────────────
    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump([it.to_dict() for it in self.items], f, ensure_ascii=False, indent=1)

    def load(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.items = [MemoryItem.from_dict(d) for d in data]
        self._by_class = {}
        for it in self.items:
            self._by_class.setdefault(class_of(encode(it.gua)), []).append(it.id)

    def __len__(self) -> int:
        return len(self.items)
