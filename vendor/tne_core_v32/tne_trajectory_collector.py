"""TNE-Core v3.2 - Trajectory Collector

v3.2 改动：
  - TrajectoryRecord 改由 tne_schema 统一定义
  - 新增 record_many()：批量轨迹单次 open 落盘，每条一次 IO 在跑几千条时是瓶颈
  - task_id 幂等：重复入库会被跳过并计数（此前会静默产生重复记录，
    导致 usage_count 虚高、success_rate 失真）
  - buffer 设上限并可显式 flush/clear，不再无界增长
"""

import json
import os
import time
from dataclasses import asdict, fields
from typing import List, Dict, Any, Optional, Iterable

try:
    from .tne_schema import TrajectoryRecord
except ImportError:  # pragma: no cover
    from tne_schema import TrajectoryRecord

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_STORAGE = os.path.join(_HERE, "data", "trajectories.jsonl")

__all__ = ["TrajectoryRecord", "TrajectoryCollector"]

BUFFER_LIMIT = 500  # 超过则自动落盘并清空，防止长跑时内存无界增长


class TrajectoryCollector:
    def __init__(self, storage_path: str = DEFAULT_STORAGE, buffer_limit: int = BUFFER_LIMIT):
        self.storage_path = storage_path
        self.buffer: List[TrajectoryRecord] = []
        self.buffer_limit = buffer_limit
        self._seen_ids: Optional[set] = None  # 惰性加载，首次去重时构建
        self._ensure()

    def _ensure(self):
        d = os.path.dirname(self.storage_path)
        if d:
            os.makedirs(d, exist_ok=True)
        if not os.path.exists(self.storage_path):
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                pass

    # ---------- 写入 ----------

    def record(self, record: TrajectoryRecord, dedupe: bool = True) -> bool:
        """返回 True 表示实际写入；False 表示因 task_id 重复被跳过。"""
        if dedupe and self._is_duplicate(record.task_id):
            return False
        self.buffer.append(record)
        self._flush([record])
        self._mark(record.task_id)  # 批内也要登记，否则同批重复 id 会全部写进去
        if len(self.buffer) >= self.buffer_limit:
            self.buffer.clear()
        return True

    def record_many(self, records: Iterable[TrajectoryRecord], dedupe: bool = True) -> Dict[str, Any]:
        """批量写入：单次 open，多行落盘。"""
        written = []
        skipped = 0
        for rec in records:
            if dedupe and self._is_duplicate(rec.task_id):
                skipped += 1
                continue
            # 边遍历边登记：否则同一批里的重复 id 要等整批写完才被发现
            if dedupe:
                self._mark(rec.task_id)
            self.buffer.append(rec)
            written.append(rec)

        if written:
            self._flush(written)
        if len(self.buffer) >= self.buffer_limit:
            self.buffer.clear()

        return {"written": len(written), "skipped_dupe": skipped, "total": len(written) + skipped}

    def _flush(self, records: List[TrajectoryRecord]) -> None:
        with open(self.storage_path, 'a', encoding='utf-8') as f:
            for rec in records:
                f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + '\n')

    def _is_duplicate(self, task_id: str) -> bool:
        if self._seen_ids is None:
            self._seen_ids = {r.get("task_id") for r in self.load_all()}
        return task_id in self._seen_ids

    def _mark(self, task_id: str) -> None:
        if self._seen_ids is None:
            self._seen_ids = {r.get("task_id") for r in self.load_all()}
        self._seen_ids.add(task_id)

    # ---------- 读取 ----------

    def load_all(self) -> List[Dict[str, Any]]:
        records = []
        try:
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))
        except FileNotFoundError:
            pass
        return records

    def load_records(self) -> List[TrajectoryRecord]:
        """读回强类型对象；损坏行单独跳过并计数，不让一条脏数据毁掉整批。"""
        out = []
        bad = 0
        for d in self.load_all():
            try:
                out.append(TrajectoryRecord.from_dict(d))
            except (ValueError, TypeError):
                bad += 1
        self._load_errors = bad
        return out

    def stats(self) -> Dict[str, Any]:
        records = self.load_all()
        if not records:
            return {"total": 0}
        by_type = {}
        for r in records:
            t = r.get("task_type", "unknown")
            by_type[t] = by_type.get(t, 0) + 1
        eff = [r.get("efficacy_score", 0) for r in records]
        return {
            "total": len(records),
            "by_type": by_type,
            "avg_efficacy": round(sum(eff) / len(eff), 4),
            "min_efficacy": round(min(eff), 4),
            "max_efficacy": round(max(eff), 4),
        }

    # ---------- 维护 ----------

    def clear(self) -> None:
        """清空存储与内存索引（供测试与重建使用）。"""
        with open(self.storage_path, 'w', encoding='utf-8') as f:
            pass
        self.buffer.clear()
        self._seen_ids = set()


collector = TrajectoryCollector()
