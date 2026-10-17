"""
TNE-Core · 轨迹库（Trajectory Store）
存 (态, 行为, 结果, 时间戳)，每轮 Agent 交互自动落盘。
使用 SQLite，零依赖，MVP 阶段足够。
"""
from __future__ import annotations
import json
import os
import sqlite3
from datetime import datetime
from typing import List, Dict, Optional, Any
from ..space.space import encode, decode, gua_str
from ..space.equivalence import class_of

SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "schema.sql")


class TrajectoryStore:
    def __init__(self, db_path: str = "tne_trajectories.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        with open(SCHEMA_PATH, encoding="utf-8") as f:
            self.conn.executescript(f.read())
        self.conn.commit()

    def record(self, session_id: str, gua: List[int], action: str,
               action_detail: Optional[Dict] = None,
               outcome: Optional[str] = None, reward: float = 0.0) -> int:
        """记录一轮轨迹。返回记录 ID。"""
        gua_idx = encode(gua)
        rep = class_of(gua_idx)
        cur = self.conn.execute(
            """INSERT INTO trajectories
               (session_id, gua_idx, gua_vector, class_rep, action, action_detail, outcome, reward)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (session_id, gua_idx, json.dumps(gua), rep, action,
             json.dumps(action_detail, ensure_ascii=False) if action_detail else None,
             outcome, reward),
        )
        self.conn.commit()
        return cur.lastrowid

    def query_by_gua(self, gua_idx: int, limit: int = 100) -> List[Dict]:
        """按卦位查询历史轨迹。"""
        rows = self.conn.execute(
            "SELECT * FROM trajectories WHERE gua_idx=? ORDER BY timestamp DESC LIMIT ?",
            (gua_idx, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def query_by_class(self, class_rep: int, limit: int = 100) -> List[Dict]:
        """按等价类查询历史轨迹。"""
        rows = self.conn.execute(
            "SELECT * FROM trajectories WHERE class_rep=? ORDER BY timestamp DESC LIMIT ?",
            (class_rep, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def class_coverage(self) -> Dict[str, Any]:
        """等价类覆盖率统计。"""
        total_classes = 5002
        covered = self.conn.execute(
            "SELECT COUNT(DISTINCT class_rep) FROM trajectories"
        ).fetchone()[0]
        total_records = self.conn.execute("SELECT COUNT(*) FROM trajectories").fetchone()[0]
        return {
            "total_classes": total_classes,
            "covered_classes": covered,
            "coverage_rate": round(covered / total_classes, 4),
            "total_records": total_records,
        }

    def close(self):
        self.conn.close()
