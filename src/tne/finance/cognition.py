# -*- coding: utf-8 -*-
"""金融认知模块: 三层认知空间统一入口(空间卦位/时间周期/张力信号)。

供三类调用方复用(消除重复):
  - scripts/finance_watch.py   看板脚本
  - src/tne/api/server.py      坐标服务端点
  - scripts/finance_realtime_report.py  实时直出三层诊断

依赖 scripts/finance_mapper.py (map_nine/diagnose 九维映射)。
规则: 周期层日±1%/周月年±2%; 快慢背离=转折张力; 关注度=空间主态+周期形态+张力信号。
"""
import os
import sys

# 使 scripts/finance_mapper 可导入(项目根/scripts)
_SCRIPTS = os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import finance_mapper as _fm  # noqa: E402

S = {-1: "阴", 0: "衡", 1: "阳"}
SCODE = {-1: "-", 0: "0", 1: "+"}

# 周期传导层: (层名, 数据段, 字段, 阈值)  日层单日波动小用±1%, 周/月/年±2%
CYC_LAYERS = [("day", "quote", "chg_pct", 1.0), ("week", "trend", "two_week_pct", 2.0),
              ("month", "trend", "month_pct", 2.0), ("year", "trend", "year_pct", 2.0)]

# 空间主态 → 关注分数
WATCH_SPACE = {"强势·生发": 3, "稳健·蓄势": 2, "震荡·待变": 1, "弱势·收敛": 0}
# 周期形态 → 关注分数(背离=转折验证, 同向生发=趋势确认)
WATCH_CYC = {"顺势·生发": 3, "回踩·背离": 3, "反弹·背离": 3, "震荡·待变": 1, "弱势·收敛": 0}


def _tri(v, thr):
    return 1 if v >= thr else (-1 if v <= -thr else 0)


# ---------- 空间层: 九维卦位 ----------
def map_nine(d):
    """九维三态映射(显式阈值), 委托 finance_mapper。"""
    return _fm.map_nine(d)


def diagnose(d):
    """九维诊断: 卦位/等价类/变换/邻域/认知主态/张力信号。"""
    return _fm.diagnose(map_nine(d), d)


# ---------- 时间层: 周期传导链 ----------
def cycle_analyze(d):
    """周期传导: 日→周→月→年 链 + 形态 + 反转 + 背离 + 张力指数。"""
    chain = []
    for key, sec, fld, thr in CYC_LAYERS:
        v = d[sec][fld]
        chain.append({"layer": key, "name": {"day": "日·动能", "week": "周·波段",
                                             "month": "月·趋势", "year": "年·根性"}[key],
                      "pct": v, "state": _tri(v, thr)})
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
                       "state": c["state"], "state_name": S[c["state"]]} for c in chain],
            "chain_code": "".join(SCODE[c["state"]] for c in chain),
            "reversals": revs, "conflict": conflict, "tension": tension,
            "mode": mode, "mode_class": cls}


# ---------- 整合层: 多层认知关注 ----------
def watch_analyze(d):
    """三层认知叠加 → 关注分级。研究辅助, 非荐股。"""
    r = diagnose(d)
    cyc = cycle_analyze(d)
    s = WATCH_SPACE[r["mode"]] + WATCH_CYC[cyc["mode"]]
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
            "rec": r["rec"], "cycle": cyc, "signals": r["signals"], "tensions": r["tensions"],
            "score": round(s, 1), "watch_level": lv, "watch_class": lc}


DISCLAIMER = "研究辅助, 非投资建议, 不构成荐股"
