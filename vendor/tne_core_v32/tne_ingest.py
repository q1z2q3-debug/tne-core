"""TNE-Core v3.2 - 批量入库

v3.1 的 run_pipeline 每跑一次就 new 一个 Distiller 再 load/export 一次 patterns.json：
  - 模式的状态（usage_count / success_rate / 典型触发特征）全靠文件往返累积，
    中途异常即丢；跑一百次就是一百次 IO 往返
  - 多条轨迹之间看不到彼此，无法做批内去重与整体校验

TNESession 把「一个 Projector + 一个 Distiller + 一个 Collector」的生命周期
拉长到整批，入库结束才落盘一次。
"""

import json
import os
import time
from typing import List, Dict, Any, Optional, Iterable

try:
    from .tne_projector_v5 import YuanbaoProjectorV4
    from .tne_distiller import TNE_Distiller
    from .tne_trajectory_collector import TrajectoryCollector
    from .tne_schema import TrajectoryRecord
except ImportError:  # pragma: no cover
    from tne_projector_v5 import YuanbaoProjectorV4
    from tne_distiller import TNE_Distiller
    from tne_trajectory_collector import TrajectoryCollector
    from tne_schema import TrajectoryRecord

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, "data")
PATTERNS_PATH = os.path.join(DATA_DIR, "patterns.json")
TRAJECTORY_PATH = os.path.join(DATA_DIR, "trajectories.jsonl")

DEFAULT_HABITS = ["以终为始", "统合综效", "不断更新"]


class TNESession:
    """一次批量入库的共享上下文。推荐用 with 语句，退出时自动落盘。"""

    def __init__(self, data_dir: str = None, use_llm: str = "auto",
                 min_samples: int = 2, min_success_rate: float = 0.85,
                 fresh: bool = False):
        self.data_dir = data_dir or DATA_DIR
        os.makedirs(self.data_dir, exist_ok=True)
        self.patterns_path = os.path.join(self.data_dir, "patterns.json")
        self.trajectory_path = os.path.join(self.data_dir, "trajectories.jsonl")

        self.projector = YuanbaoProjectorV4(use_llm=use_llm)
        self.distiller = TNE_Distiller(min_samples=min_samples,
                                       min_success_rate=min_success_rate)
        self.collector = TrajectoryCollector(storage_path=self.trajectory_path)

        if fresh:
            self.reset()
        else:
            try:
                self.distiller.load_patterns(self.patterns_path)
            except FileNotFoundError:
                pass

        self.stats = {"encoded": 0, "ingested": 0, "dupe_skipped": 0,
                      "invalid": 0, "new_patterns": [], "llm_fallback": 0}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def reset(self) -> None:
        """清空模式库与轨迹库，从零开始累积。"""
        self.distiller.patterns.clear()
        self.distiller.trajectories.clear()
        self.collector.clear()

    # ---------- 单条 ----------

    def ingest(self, text: str, task_type: str, efficacy: float,
               task_id: str = None, habits: List[str] = None,
               description: str = "") -> Dict[str, Any]:
        """encode 一条文本并入库。efficacy 必须显式给（不能拿 confidence 顶替）。"""
        result = self.projector.encode(text)
        self.stats["encoded"] += 1
        if result.backend == "llm_fallback":
            self.stats["llm_fallback"] += 1

        tid = task_id or f"{task_type}_{int(time.time()*1000)}"
        rec = TrajectoryRecord.create(
            task_id=tid, task_type=task_type,
            state_before=result.vector,
            habits_used=habits or DEFAULT_HABITS,
            efficacy_score=efficacy, description=description or text[:60],
        )

        errs = rec.validate()
        if errs:
            self.stats["invalid"] += 1
            return {"ok": False, "task_id": tid, "errors": errs, "vector": result.vector}

        written = self.collector.record(rec)
        if not written:
            self.stats["dupe_skipped"] += 1
            return {"ok": False, "task_id": tid, "errors": ["task_id 重复，已跳过"],
                    "vector": result.vector}

        self.distiller.add_trajectory(rec)
        self.stats["ingested"] += 1
        return {"ok": True, "task_id": tid, "vector": result.vector,
                "confidence": result.confidence, "backend": result.backend}

    # ---------- 批量 ----------

    def ingest_batch(self, items: Iterable[Dict[str, Any]],
                     show_progress: bool = False) -> Dict[str, Any]:
        """批量入库。

        items 每项：
          {
            "text":       "原始输入文本",        # 必填
            "task_type":  "创作叙事",            # 必填
            "efficacy":   0.92,                  # 必填，0~1，真实成效
            "task_id":    "t_001",               # 可选，不填自动生成
            "habits":     ["以终为始", ...],      # 可选，至少 2 项
            "description": "备注"                # 可选
          }
        """
        t0 = time.time()
        records: List[TrajectoryRecord] = []
        encoded_vectors = []

        for i, item in enumerate(items, 1):
            text = item.get("text", "")
            if not text:
                self.stats["invalid"] += 1
                continue

            result = self.projector.encode(text)
            self.stats["encoded"] += 1
            if result.backend == "llm_fallback":
                self.stats["llm_fallback"] += 1
            encoded_vectors.append(result.vector)

            tid = item.get("task_id") or f"{item.get('task_type','unknown')}_{i:04d}"
            rec = TrajectoryRecord.create(
                task_id=tid,
                task_type=item.get("task_type", "unknown"),
                state_before=result.vector,
                habits_used=item.get("habits") or DEFAULT_HABITS,
                efficacy_score=float(item.get("efficacy", 0.0)),
                description=item.get("description", text[:60]),
            )
            records.append(rec)
            if show_progress and i % 10 == 0:
                print(f"    encode {i} 条...")

        # 批内去重：同一批里出现相同 task_id 只保留第一条
        seen = set()
        deduped = []
        for r in records:
            if r.task_id in seen:
                self.stats["dupe_skipped"] += 1
                continue
            seen.add(r.task_id)
            deduped.append(r)
        records = deduped

        # 校验分流
        valid, invalid = [], []
        for r in records:
            errs = r.validate()
            (invalid if errs else valid).append((r, errs))
        self.stats["invalid"] += len(invalid)

        # 落盘（单次 open）
        w = self.collector.record_many([r for r, _ in valid])
        self.stats["ingested"] += w["written"]
        self.stats["dupe_skipped"] += w["skipped_dupe"]

        # 蒸馏
        report = self.distiller.add_trajectories([r for r, _ in valid])
        self.stats["new_patterns"].extend(report["new_patterns"])

        elapsed = time.time() - t0
        return {
            "input": len(list(items)) if not isinstance(items, list) else len(items),
            "encoded": self.stats["encoded"],
            "ingested": w["written"],
            "dupe_skipped": self.stats["dupe_skipped"],
            "invalid": self.stats["invalid"],
            "new_patterns": report["new_patterns"],
            "elapsed_sec": round(elapsed, 3),
            "invalid_detail": [{"task_id": r.task_id, "errors": e} for r, e in invalid][:5],
        }

    def ingest_file(self, path: str, show_progress: bool = False) -> Dict[str, Any]:
        """从 jsonl 批量导入。每行一个 item（字段同 ingest_batch）。"""
        items = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    items.append(json.loads(line))
        return self.ingest_batch(items, show_progress=show_progress)

    # ---------- 查询 ----------

    def recommend(self, task_type: str, state: List[int] = None,
                  text: str = None, top_k: int = 3) -> List[Dict[str, Any]]:
        """给新任务推荐模式。可直接给文本（自动 encode），也可给九维向量。"""
        if state is None:
            if text is None:
                raise ValueError("state 与 text 至少给一个")
            state = self.projector.encode(text).vector
        return self.distiller.recommend_with_scores(task_type, state, top_k=top_k)

    def report(self) -> Dict[str, Any]:
        return {
            "session": self.stats,
            "patterns": self.distiller.get_stats(),
            "trajectories": self.collector.stats(),
        }

    def close(self) -> None:
        self.distiller.export_patterns(self.patterns_path)


# ---------- 便捷函数 ----------

def build_batch_from_texts(texts: List[str], task_type: str,
                           efficacies: List[float] = None,
                           habits: List[str] = None) -> List[Dict[str, Any]]:
    """把纯文本列表组装成 ingest_batch 所需的 items。"""
    if efficacies and len(efficacies) != len(texts):
        raise ValueError("efficacies 长度必须与 texts 一致")
    return [
        {
            "text": t,
            "task_type": task_type,
            "efficacy": efficacies[i] if efficacies else 0.9,
            "task_id": f"{task_type}_{i+1:03d}",
            "habits": habits or DEFAULT_HABITS,
        }
        for i, t in enumerate(texts)
    ]


def load_trajectory_export(path: str = TRAJECTORY_PATH) -> List[TrajectoryRecord]:
    """把已落盘的轨迹导出为强类型对象，用于回放 / 重放 / 迁移。"""
    c = TrajectoryCollector(storage_path=path)
    return c.load_records()
