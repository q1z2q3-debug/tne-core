"""
TNE-Core · 坐标服务 API（FastAPI）
LLM 通过 function calling 调用的认知层工具。

端点：
  POST /encode    文本/向量 → 卦位索引 + 等价类
  POST /search    目标向量 → Top-N 邻域卦
  POST /transform 卦位 → 错/综/错综/互/变全套推演
  POST /habit     当前卦位 → 习惯偏置查询
  POST /trajectory 记录一轮轨迹
  GET  /stats     轨迹库覆盖率统计
  GET  /health    健康检查
"""
from __future__ import annotations
import os
import sys
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# 确保 src 在路径中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from tne.space.space import encode, decode, gua_str, gua_yao_str, TOTAL_STATES, DIM
from tne.space.operators import f_cuoo, f_zong, f_cuoo_zong, f_hu, f_bian
from tne.space.equivalence import class_of, class_members, count_equivalence_classes
from tne.space.distance import nearest_neighbors, euclidean
from tne.trajectory.store import TrajectoryStore
from tne.trajectory.habit import HabitQuery

app = FastAPI(title="TNE-Core 坐标服务", version="0.1.0",
              description="三元九维 19683 认知体系 · 可定位可泛化的符号认知层")

DB_PATH = os.environ.get("TNE_DB", "tne_trajectories.db")
store = TrajectoryStore(DB_PATH)
habit = HabitQuery(store)


# ---------- 请求模型 ----------
class EncodeRequest(BaseModel):
    vector: List[int] = Field(..., description="九维三态向量，每维 ∈ {-1,0,1}")

class SearchRequest(BaseModel):
    vector: List[int]
    top_n: int = 10

class TransformRequest(BaseModel):
    vector: List[int]
    moving_yaos: Optional[List[int]] = Field(None, description="动爻位 1-9，不传则只算错综互")

class HabitRequest(BaseModel):
    vector: List[int]

class TrajectoryRequest(BaseModel):
    session_id: str
    vector: List[int]
    action: str
    action_detail: Optional[dict] = None
    outcome: Optional[str] = None
    reward: float = 0.0


# ---------- 端点 ----------
@app.get("/health")
def health():
    return {"status": "ok", "total_states": TOTAL_STATES,
            "equivalence_classes": count_equivalence_classes()}

@app.post("/encode")
def encode_endpoint(req: EncodeRequest):
    if len(req.vector) != DIM or any(g not in (-1, 0, 1) for g in req.vector):
        raise HTTPException(400, "必须是 9 维三态向量")
    idx = encode(req.vector)
    rep = class_of(idx)
    return {
        "gua_idx": idx,
        "gua_vector": req.vector,
        "gua_str": gua_str(req.vector),
        "yao_str": gua_yao_str(req.vector),
        "class_rep": rep,
        "class_members": class_members(idx),
    }

@app.post("/search")
def search_endpoint(req: SearchRequest):
    if len(req.vector) != DIM or any(g not in (-1, 0, 1) for g in req.vector):
        raise HTTPException(400, "必须是 9 维三态向量")
    results = nearest_neighbors(req.vector, top_n=req.top_n)
    return {
        "target": req.vector,
        "neighbors": [
            {"gua_idx": idx, "distance": round(d, 4), "gua_vector": g,
             "class_rep": class_of(idx)}
            for idx, d, g in results
        ],
    }

@app.post("/transform")
def transform_endpoint(req: TransformRequest):
    if len(req.vector) != DIM or any(g not in (-1, 0, 1) for g in req.vector):
        raise HTTPException(400, "必须是 9 维三态向量")
    g = req.vector
    out = {
        "original": g,
        "cuoo_FC": f_cuoo(g),
        "zong_FZ": f_zong(g),
        "cuoo_zong_FCZ": f_cuoo_zong(g),
        "hu_FH_5dim": f_hu(g),
    }
    if req.moving_yaos:
        out["bian_FB"] = f_bian(g, set(req.moving_yaos))
    return out

@app.post("/habit")
def habit_endpoint(req: HabitRequest):
    if len(req.vector) != DIM or any(g not in (-1, 0, 1) for g in req.vector):
        raise HTTPException(400, "必须是 9 维三态向量")
    return habit.query(req.vector)

@app.post("/trajectory")
def trajectory_endpoint(req: TrajectoryRequest):
    if len(req.vector) != DIM or any(g not in (-1, 0, 1) for g in req.vector):
        raise HTTPException(400, "必须是 9 维三态向量")
    rid = store.record(req.session_id, req.vector, req.action,
                       req.action_detail, req.outcome, req.reward)
    return {"record_id": rid, "status": "saved"}

@app.get("/stats")
def stats_endpoint():
    return store.class_coverage()
