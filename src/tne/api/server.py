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
import json
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

class CognitionRequest(BaseModel):
    text: str = Field(..., description="任务/状态的自然语言描述")
    moving_yaos: Optional[List[int]] = Field(None, description="动爻位 1-9")
    top_n: int = 5


# ---------- 自训 encode 判断器（懒加载）----------
_judger = None

def get_judger():
    global _judger
    if _judger is None:
        from tne.projector.encoder_judger import TrainedEncoderJudger
        _judger = TrainedEncoderJudger()
    return _judger


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


# ---------- 认知全景端点（文本 → 完整认知定位）----------
@app.post("/cognition")
def cognition_endpoint(req: CognitionRequest):
    """
    输入任务/状态的自然语言描述 → 自训 encode 模型判断九维向量，
    返回完整认知全景：向量+置信度 / 卦位 / 等价类 / 错综互变推演 / 习惯偏置 / 邻域。
    """
    if not req.text.strip():
        raise HTTPException(400, "text 不能为空")
    judger = get_judger()
    jr = judger.judge_with_confidence(req.text)
    vector = jr["vector"]
    if len(vector) != DIM or any(g not in (-1, 0, 1) for g in vector):
        raise HTTPException(500, "encode 模型输出非法向量")

    idx = encode(vector)
    rep = class_of(idx)

    # 结构变换（错/综/错综/互，可选变爻）
    gua = vector
    transform = {
        "cuoo_FC": f_cuoo(gua),
        "zong_FZ": f_zong(gua),
        "cuoo_zong_FCZ": f_cuoo_zong(gua),
        "hu_FH_5dim": f_hu(gua),
    }
    if req.moving_yaos:
        transform["bian_FB"] = f_bian(gua, set(req.moving_yaos))

    # 习惯偏置
    hb = habit.query(vector)

    # 邻域泛化
    neighbors = [
        {"gua_idx": ni, "distance": round(d, 4), "gua_vector": ng,
         "class_rep": class_of(ni)}
        for ni, d, ng in nearest_neighbors(vector, top_n=req.top_n)
    ]

    # 认知动力引擎：张力检测 → 动力信号
    drive = None
    try:
        from tne.drive.tension import detect_and_generate
        drive = detect_and_generate(hb, jr["confidence"], neighbors)
    except Exception:
        drive = None

    return {
        "text": req.text,
        "vector": vector,
        "confidence": jr["confidence"],
        "gua_idx": idx,
        "gua_str": gua_str(vector),
        "yao_str": gua_yao_str(vector),
        "class_rep": rep,
        "class_members": class_members(idx),
        "transform": transform,
        "habit": hb,
        "neighbors": neighbors,
        "drive": drive,
    }


# ============ 九维金融认知 /finance ============
FIN_BASE = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "finance")


def _load_finance_data():
    import glob
    files = sorted(glob.glob(os.path.join(FIN_BASE, "*.json")))
    out = {}
    for f in files:
        try:
            d = json.load(open(f, encoding="utf-8"))
            out[d["code"]] = d
        except Exception:
            pass
    return out


def _fin_mapper():
    # 复用 finance_mapper 的 map_nine/diagnose
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts"))
    try:
        import finance_mapper
        return finance_mapper
    except Exception:
        return None


@app.get("/finance")
def finance_endpoint(code: Optional[str] = None):
    """
    九维金融认知：读 data/finance 下个股数据 → 九维画像/卦位/认知主态。
    ?code=600519.SH → 单股诊断; 无code → 全部股票摘要。
    """
    fm = _fin_mapper()
    data = _load_finance_data()
    if not data:
        raise HTTPException(404, "未找到金融数据 (data/finance/*.json)")
    if code:
        d = data.get(code)
        if not d:
            raise HTTPException(404, f"未找到股票 {code}")
        g = fm.map_nine(d)
        r = fm.diagnose(g, d)
        return {"name": d["name"], "code": d["code"], "industry": d["industry"],
                "vector": r["vector"], "code_str": r["code_str"], "gua_idx": r["gua_idx"],
                "class_rep": r["class_rep"], "mode": r["mode"], "rec": r["rec"],
                "transform": r["transform"], "neighbors": r["neighbors"],
                "tensions": r["tensions"], "signals": r["signals"],
                "quote": {k: d["quote"][k] for k in ("close", "chg_pct", "pe_ttm", "pb", "turnover_pct")}}
    summary = []
    for code_key, d in data.items():
        g = fm.map_nine(d)
        r = fm.diagnose(g, d)
        summary.append({"name": d["name"], "code": d["code"], "industry": d["industry"],
                        "code_str": r["code_str"], "gua_idx": r["gua_idx"], "mode": r["mode"]})
    return {"count": len(summary), "stocks": summary}


# ============ 周期传导 /finance/cycle ============
# K线时间维度=认知链: 日(动能±1%)→周(波段)→月(趋势)→年(根性,±2%)。快慢背离=转折张力。
_CYCLE_LAYERS = [("day", "quote", "chg_pct", 1.0), ("week", "trend", "two_week_pct", 2.0),
                 ("month", "trend", "month_pct", 2.0), ("year", "trend", "year_pct", 2.0)]
_STATE_NAME = {-1: "阴", 0: "衡", 1: "阳"}


def _cycle_tri(v, thr):
    return 1 if v >= thr else (-1 if v <= -thr else 0)


def _cycle_analyze(d):
    """单股周期传导: 周期链四层态 + 反转 + 背离 + 张力指数 + 形态分类。"""
    chain = []
    for key, sec, fld, thr in _CYCLE_LAYERS:
        v = d[sec][fld]
        chain.append({"layer": key, "name": {"day": "日·动能", "week": "周·波段",
                                             "month": "月·趋势", "year": "年·根性"}[key],
                      "pct": v, "state": _cycle_tri(v, thr)})
    day, week, month, year = (c["state"] for c in chain)
    revs = sum(1 for i in range(3) if chain[i]["state"] != 0 and chain[i + 1]["state"] != 0
               and chain[i]["state"] != chain[i + 1]["state"])
    conflict = day != year
    tension = revs * 2 + (3 if conflict else 0) + abs(year)
    if year == 1 and month == 1:
        mode, cls = "顺势·生发", "strong"
    elif year == -1 and (day == 1 or week == 1):
        mode, cls = "反弹·背离", "rebound"
    elif year == -1 and month == -1:
        mode, cls = "弱势·收敛", "weak"
    elif year == 1 and (day == -1 or week == -1):
        mode, cls = "回踩·背离", "pullback"
    else:
        mode, cls = "震荡·待变", "mixed"
    return {"chain": [{"layer": c["layer"], "name": c["name"], "pct": round(c["pct"], 2),
                       "state": c["state"], "state_name": _STATE_NAME[c["state"]]} for c in chain],
            "reversals": revs, "conflict": conflict, "tension": tension,
            "mode": mode, "mode_class": cls}


@app.get("/finance/cycle")
def finance_cycle_endpoint(code: Optional[str] = None):
    """
    周期传导认知：读 data/finance 个股 → 时间认知链(日→周→月→年) + 形态 + 背离张力指数。
    ?code=600519.SH → 单股周期传导诊断; 无code → 全部股票周期传导摘要。
    """
    data = _load_finance_data()
    if not data:
        raise HTTPException(404, "未找到金融数据 (data/finance/*.json)")
    if code:
        d = data.get(code)
        if not d:
            raise HTTPException(404, f"未找到股票 {code}")
        r = _cycle_analyze(d)
        return {"name": d["name"], "code": d["code"], "industry": d["industry"],
                "close": d["quote"]["close"], "cycle": r}
    summary = []
    for code_key, d in data.items():
        r = _cycle_analyze(d)
        summary.append({"name": d["name"], "code": d["code"], "industry": d["industry"],
                        "chain": "".join(_STATE_NAME[c["state"]] for c in r["chain"]),
                        "mode": r["mode"], "tension": r["tension"]})
    summary.sort(key=lambda x: -x["tension"])
    return {"count": len(summary), "stocks": summary}


# ============ 多层认知关注 /finance/watch ============
# 三层认知空间叠加(空间卦位+时间周期+张力信号) → 关注分级。研究辅助非荐股。
_WATCH_SPACE = {"强势·生发": 3, "稳健·蓄势": 2, "震荡·待变": 1, "弱势·收敛": 0}
_WATCH_CYC = {"顺势·生发": 3, "回踩·背离": 3, "反弹·背离": 3, "震荡·待变": 1, "弱势·收敛": 0}


def _watch_analyze(d):
    fm = _fin_mapper()
    g = fm.map_nine(d)
    r = fm.diagnose(g, d)
    cyc = _cycle_analyze(d)
    s = _WATCH_SPACE[r["mode"]] + _WATCH_CYC[cyc["mode"]]
    sigs = {x["type"] for x in r["signals"]}
    tens = {x["type"] for x in r["tensions"]}
    if {"curiosity", "novelty"} & sigs:
        s += 1.0
    if "habit_conflict" in tens:
        s += 0.5
    if "boundary" in tens:
        s -= 0.5  # 估值/风险偏高, 谨慎
    if cyc["conflict"]:
        s += 0.5  # 快慢背离=转折验证, 天然高关注
    if s >= 5.5:
        lv, lc = "高关注·转折验证", "high"
    elif s >= 3.5:
        lv, lc = "中关注·趋势跟踪", "mid"
    else:
        lv, lc = "低关注·风险规避", "low"
    return {"gua_code": r["code_str"], "gua_idx": r["gua_idx"], "space_mode": r["mode"],
            "cycle": cyc, "signals": r["signals"], "tensions": r["tensions"],
            "score": round(s, 1), "watch_level": lv, "watch_class": lc}


@app.get("/finance/watch")
def finance_watch_endpoint(code: Optional[str] = None):
    """
    多层认知关注：三层认知空间(空间卦位+时间周期+张力信号)叠加 → 关注分级。
    研究辅助非荐股。?code=600519.SH → 单股整合诊断; 无code → 全部股票关注分级。
    """
    data = _load_finance_data()
    if not data:
        raise HTTPException(404, "未找到金融数据 (data/finance/*.json)")
    if code:
        d = data.get(code)
        if not d:
            raise HTTPException(404, f"未找到股票 {code}")
        w = _watch_analyze(d)
        return {"name": d["name"], "code": d["code"], "industry": d["industry"],
                "close": d["quote"]["close"], "watch": w,
                "disclaimer": "研究辅助, 非投资建议, 不构成荐股"}
    summary = []
    for code_key, d in data.items():
        w = _watch_analyze(d)
        summary.append({"name": d["name"], "code": d["code"], "industry": d["industry"],
                        "gua": w["gua_code"], "space_mode": w["space_mode"],
                        "cycle_chain": w["cycle"]["chain"], "cycle_mode": w["cycle"]["mode"],
                        "tension": w["cycle"]["tension"], "score": w["score"],
                        "watch_level": w["watch_level"]})
    summary.sort(key=lambda x: -x["score"])
    return {"count": len(summary), "stocks": summary,
            "disclaimer": "研究辅助, 非投资建议, 不构成荐股"}
