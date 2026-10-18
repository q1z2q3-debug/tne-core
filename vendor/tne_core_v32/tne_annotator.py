"""TNE-Core v3.4 - 人在回路标注缓存

背景：本地词表在独立集上只有 25%，且错误是分散的（encode 纯桶率 73.2%），
27% 的桶混入了不同标注的样本，会顺着 pipeline 污染模式库。

出路不是继续补词，而是换架构：
    运行时命中的句子 → 缓存直接返回「我（LLM）在对话里判定的向量」，绝对可靠
    运行时没见过的句子 → 进 pending 队列，攒够一批交给我批量标注，回填后永久生效

这样完全不需要任何 API —— 代价是需要一次人工（对话）介入，
但真实业务里任务描述高度重复，标注一轮就能覆盖大部分流量。
"""

import json
import os
import time
import hashlib
from typing import List, Dict, Any, Optional, Tuple

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CACHE_PATH = os.path.join(DATA_DIR, "annotation_cache.json")
PENDING_PATH = os.path.join(DATA_DIR, "pending_queue.jsonl")


def _norm(text: str) -> str:
    """归一化：去空白与全角空格，作为缓存键。"""
    return "".join((text or "").split())


def _key(text: str) -> str:
    return hashlib.sha1(_norm(text).encode("utf-8")).hexdigest()[:16]


class AnnotationCache:
    """精确匹配缓存。命中即返回 LLM 标注值，不经过词表。"""

    def __init__(self, path: str = CACHE_PATH, pending_path: str = PENDING_PATH):
        self.path = path
        self.pending_path = pending_path
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.stats = {"hit": 0, "miss": 0, "queued": 0}
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.cache = data.get("entries", {})

    def save(self):
        payload = {
            "_meta": {
                "count": len(self.cache),
                "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
                "source": "LLM 在对话中逐条判定并回填",
            },
            "entries": self.cache,
        }
        with open(self.path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def get(self, text: str) -> Optional[List[int]]:
        e = self.cache.get(_key(text))
        if e is None:
            self.stats["miss"] += 1
            return None
        self.stats["hit"] += 1
        return list(e["vector"])

    def put(self, text: str, vector: List[int], note: str = "",
            annotator: str = "llm") -> None:
        if len(vector) != 9 or any(v not in (-1, 0, 1) for v in vector):
            raise ValueError(f"向量非法: {vector}")
        self.cache[_key(text)] = {
            "text": text,
            "vector": list(vector),
            "note": note,
            "annotator": annotator,
            "ts": time.time(),
        }

    def put_many(self, items: List[Dict[str, Any]]) -> int:
        """items: [{"text":..., "vector":[...], "note":...}]"""
        n = 0
        for it in items:
            self.put(it["text"], it["vector"], it.get("note", ""),
                     it.get("annotator", "llm"))
            n += 1
        return n

    # ---------- 待标注队列 ----------

    def enqueue(self, text: str, task_type: str = "", context: str = "") -> bool:
        """未命中缓存的句子入队，等批量标注。已存在则跳过。"""
        k = _key(text)
        if k in self.cache:
            return False
        existing = {json.loads(l)["key"] for l in self._read_pending()}
        if k in existing:
            return False
        with open(self.pending_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({"key": k, "text": text,
                                "task_type": task_type, "context": context},
                               ensure_ascii=False) + "\n")
        self.stats["queued"] += 1
        return True

    def _read_pending(self) -> List[str]:
        if not os.path.exists(self.pending_path):
            return []
        with open(self.pending_path, 'r', encoding='utf-8') as f:
            return [l for l in f if l.strip()]

    def pending(self) -> List[Dict[str, Any]]:
        return [json.loads(l) for l in self._read_pending()]

    def export_pending_prompt(self, limit: int = 50) -> str:
        """导出成可直接贴给 LLM 的标注请求。这是整个回路的接口。"""
        items = self.pending()[:limit]
        if not items:
            return "（队列为空，无需标注）"
        lines = [
            "请对以下句子做三元九维判定，每维取 +1 / 0 / -1。",
            "",
            "0 天·性 +1创造生成 / 0维持 / -1破坏损耗",
            "1 天·习 +1习惯延续 / 0首次 / -1习惯中断",
            "2 天·运 +1外部驱动 / 0平稳 / -1外部阻力",
            "3 地·性 +1环境切换 / 0不变 / -1环境退化",
            "4 地·习 +1工具方法 / 0无 / -1缺失",
            "5 地·运 +1状态升级 / 0稳定 / -1降级",
            "6 人·性 +1关系协作 / 0独立 / -1孤立冲突",
            "7 人·习 +1自律内省 / 0无 / -1放纵失控",
            "8 人·运 +1内在驱动 / 0无 / -1耗竭",
            "",
            "注意：判断任务本身的态势，不是文本题材。",
            "",
            "只输出 JSON 数组，每项 {\"text\": 原文, \"vector\": [9个整数], \"note\": \"理由\"}：",
            "",
        ]
        for i, it in enumerate(items, 1):
            lines.append(f"{i}. {it['text']}")
        return "\n".join(lines)

    def clear_pending(self) -> None:
        if os.path.exists(self.pending_path):
            open(self.pending_path, 'w', encoding='utf-8').close()

    def stats_report(self) -> Dict[str, Any]:
        total = self.stats["hit"] + self.stats["miss"]
        return {
            "cache_size": len(self.cache),
            "hit": self.stats["hit"],
            "miss": self.stats["miss"],
            "hit_rate": round(self.stats["hit"] / total, 3) if total else 0.0,
            "pending": len(self.pending()),
        }


class HybridProjector:
    """缓存优先 → 词表兜底 → 未命中则入队等待标注。"""

    def __init__(self, local_projector=None, cache: AnnotationCache = None,
                 auto_enqueue: bool = True):
        from tne_projector_v5 import YuanbaoProjectorV4
        self.local = local_projector or YuanbaoProjectorV4(use_llm="local").local
        self.cache = cache or AnnotationCache()
        self.auto_enqueue = auto_enqueue
        self.stats = {"cache_hit": 0, "lexicon": 0, "queued": 0}

    def encode(self, text: str, task_type: str = "") -> Dict[str, Any]:
        v = self.cache.get(text)
        if v is not None:
            self.stats["cache_hit"] += 1
            return {"vector": v, "source": "cache", "confidence": 1.0}

        r = self.local.encode(text)
        out = {"vector": r.vector, "source": "lexicon",
               "confidence": r.confidence, "reasoning": r.reasoning}

        # 词表没把握（弱信号维度多）时入队，等我来标
        if self.auto_enqueue and r.ambiguity_flags:
            if self.cache.enqueue(text, task_type):
                self.stats["queued"] += 1
            out["queued"] = True

        self.stats["lexicon"] += 1
        return out

    def batch_encode(self, texts: List[str], task_type: str = "") -> List[Dict[str, Any]]:
        return [self.encode(t, task_type) for t in texts]
