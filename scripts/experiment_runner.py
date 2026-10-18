#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""证伪实验 v1 · 行情维度子集

链路验证: 拉日K → 建月频截面 → 算行情四维态(g1月/g2当日/g5两周/g9年) →
未来收益标签 → 四检验(主态分层/单维RankIC/趋势分IC/随机基线)。

方法学要点:
  - 无前视: t 时点只用 ≤t 的收盘价算态; 标签取 t 之后
  - 随机基线: 打乱态重算IC, 证明不是任意三态标签都有信号
  - 结论如实: 不显著就报告"无关联"
数据: 腾讯 fqkline 前复权日K, 缓存 tmp/kline/
"""
import os, sys, json, glob, datetime
import numpy as np
import pandas as pd
import urllib.request
from scipy import stats

CACHE = os.path.join(os.path.dirname(__file__), "..", "tmp", "kline")
UA = {"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                     "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"),
      "Referer": "https://gu.qq.com/"}

# 现有14只 + 16只风格差异股, 拉开横截面差异并加大样本
POOL = ["600519", "000858", "600036", "600276", "300750", "688981", "600309",
        "601318", "600030", "000002", "601899", "000333", "601088", "603259",
        "002594", "000651", "002415", "601012", "000725", "600887",
        "601888", "002475", "600585", "000063", "002230", "300059",
        "603501", "600438", "300014", "000001"]

THR = {"g1": (2.0, -2.0), "g2": (1.0, -1.0), "g5": (2.0, -2.0), "g9": (5.0, -10.0)}
# 相对差阈值(实验版): 个股涨跌 - 截面中位数, 超过该幅度判阳/阴 (生产绝对阈值不适用于截面)
REL_THR = {"g1": (0.01, -0.01), "g2": (0.01, -0.01), "g5": (0.01, -0.01), "g9": (0.02, -0.02)}


def mkt(c):
    return "sh" if c.startswith(("60", "68")) else "sz"


def band(v, up, dn):
    return 1 if v >= up else (-1 if v <= dn else 0)


def fetch_kline(code):
    """拉日K(2021至今)并缓存。返回 DataFrame[date,close] 前复权。"""
    path = os.path.join(CACHE, f"{code}.json")
    if os.path.exists(path):
        raw = json.load(open(path, encoding="utf-8"))
        df = pd.DataFrame(raw)
        df["date"] = pd.to_datetime(df["date"])
        return df
    os.makedirs(CACHE, exist_ok=True)
    # 腾讯单次约640根(最近~2.6年), beg被忽略只回最近
    url = (f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?"
           f"param={mkt(code)}{code},day,2020-01-01,2026-12-31,640,qfq")
    req = urllib.request.Request(url, headers=UA)
    d = json.loads(urllib.request.urlopen(req, timeout=20).read().decode("utf-8"))
    node = d["data"][f"{mkt(code)}{code}"]
    kl = node.get("qfqday") or node.get("day") or []  # 科创板(688)可能用day字段
    rows = [[k[0], float(k[2])] for k in kl]
    df = pd.DataFrame(rows, columns=["date", "close"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates("date")
    json.dump({"date": df["date"].dt.strftime("%Y-%m-%d").tolist(),
               "close": df["close"].tolist()}, open(path, "w", encoding="utf-8"))
    return df


def build_panel(pool):
    """合并所有股票日K → 宽表 close_matrix。"""
    frames = {}
    for c in pool:
        df = fetch_kline(c)
        frames[c] = df.set_index("date")["close"]
    close = pd.concat(frames, axis=1).sort_index().dropna(how="all")
    close.columns = pool
    return close


def snapshot(close, pool, i):
    """截面 t=i: 先算各股 raw 涨跌, 再相对截面中位数定态(相对强弱)。
    关键: A股系统性同涨同跌, 绝对阈值在截面内无横向区分 → 必须相对化。
    返回 dict 或 None(数据不足)。"""
    raws = {}
    for c in pool:
        if i < 250 or i + 60 >= len(close):
            return None
        p = close[c].values
        p0 = p[i]
        if np.isnan(p0) or np.isnan(p[i - 250]) or np.isnan(p[i + 60]):
            continue
        raws[c] = {"g1": p0 / p[i - 20] - 1, "g2": p0 / p[i - 1] - 1,
                   "g5": p0 / p[i - 10] - 1, "g9": p0 / p[i - 250] - 1,
                   "ret20": p[i + 20] / p0 - 1, "ret60": p[i + 60] / p0 - 1}
    if len(raws) < 8:
        return None
    med = {k: float(np.median([r[k] for r in raws.values()])) for k in ("g1", "g2", "g5", "g9")}
    out = {}
    for c, v in raws.items():
        # 相对差定态(实验版): 个股涨跌相对截面中位数的强弱, 用相对阈值
        g = {k: band(v[k] - med[k], *REL_THR[k]) for k in ("g1", "g2", "g5", "g9")}
        tscore3 = g["g1"] + g["g5"] + g["g9"]
        mode = "strong" if tscore3 >= 1 else ("weak" if tscore3 <= -1 else "neutral")
        out[c] = {**g, "mode": mode, "tscore3": tscore3,
                  "ret20": v["ret20"], "ret60": v["ret60"]}
    return out


def run(pool):
    print(f"拉取日K({len(pool)}只) ...")
    close = build_panel(pool)
    print(f"日K区间: {close.index[0].date()} ~ {close.index[-1].date()}  共{len(close)}根")

    # 月频截面: 每20交易日
    rows = []
    for i in range(250, len(close), 20):
        r = snapshot(close, pool, i)
        if r:
            for c, v in r.items():
                v["date"] = close.index[i].date()
                v["code"] = c
                rows.append(v)
    df = pd.DataFrame(rows)
    print(f"有效观测: {len(df)} 行 ({df['date'].nunique()} 个截面 × {df['code'].nunique()} 只)")
    print(f"未来收益分布: ret20 mean {df['ret20'].mean()*100:+.2f}%/20d, ret60 mean {df['ret60'].mean()*100:+.2f}%/60d")

    # ---- 检验A: 主态分层单调性 ----
    print("\n[检验A] 主态分层 · 未来收益")
    grp = df.groupby("mode")[["ret20", "ret60", "tscore3"]].mean().reindex(["strong", "neutral", "weak"])
    grp["n"] = df.groupby("mode").size().reindex(["strong", "neutral", "weak"])
    for col in ("ret20", "ret60"):
        grp[col] = (grp[col] * 100).round(3)
    print(grp.round(3))
    a = df[df["mode"] == "strong"]["ret60"]
    b = df[df["mode"] == "weak"]["ret60"]
    if len(a) > 5 and len(b) > 5:
        t, p = stats.ttest_ind(a, b, equal_var=False)
        spread = (a.mean() - b.mean()) * 100
        print(f"强势-弱势 ret60 价差: {spread:+.2f}pp | Welch t={t:.2f} p={p:.4f} {'显著' if p<0.05 else '不显著'}")
    else:
        print(f"样本不足: strong n={len(a)}, weak n={len(b)}")

    # ---- 检验B: 单维 RankIC ----
    print("\n[检验B] 单维 RankIC vs ret20 (跨截面聚合, 跳常数维度)")
    ics = {}
    for gname in ("g1", "g2", "g5", "g9"):
        ic_by_cross = []
        for dt, sub in df.groupby("date"):
            if len(sub) >= 8 and sub[gname].nunique() > 1:
                ic, _ = stats.spearmanr(sub[gname], sub["ret20"])
                if not np.isnan(ic):
                    ic_by_cross.append(ic)
        arr = np.array(ic_by_cross)
        if len(arr) > 10:
            icm, ir = arr.mean(), arr.mean() / (arr.std() + 1e-9)
            ics[gname] = (icm, ir, len(arr))
            print(f"  {gname}: IC_mean={icm:+.4f} IR={ir:+.2f} ({len(arr)}截面) "
                  f"{'有效' if abs(icm)>0.03 and ir>0.3 else '—'}")
        else:
            print(f"  {gname}: 有效截面不足({len(arr)})")

    # ---- 检验C: 三趋势分 IC ----
    df["tscore3"] = df["g1"] + df["g5"] + df["g9"]
    icc = []
    for dt, sub in df.groupby("date"):
        if len(sub) >= 8 and sub["tscore3"].nunique() > 1:
            ic, _ = stats.spearmanr(sub["tscore3"], sub["ret20"])
            if not np.isnan(ic):
                icc.append(ic)
    arr = np.array(icc)
    print(f"\n[检验C] 三趋势分tscore3 vs ret20: IC_mean={arr.mean():+.4f} "
          f"IR={arr.mean()/(arr.std()+1e-9):+.2f} ({len(arr)}截面)")

    # ---- 随机基线: 打乱态重算IC ----
    rng = np.random.default_rng(42)
    rand_ics = []
    for _ in range(200):
        perm = df[["g1", "g5", "g9"]].apply(rng.permutation, axis=0)
        perm["tscore3"] = perm.sum(axis=1)
        joined = perm.join(df[["ret20", "date"]])
        icl = []
        for dt, sub in joined.groupby("date"):
            if len(sub) >= 8 and sub["tscore3"].nunique() > 1:
                ic, _ = stats.spearmanr(sub["tscore3"], sub["ret20"])
                if not np.isnan(ic):
                    icl.append(ic)
        rand_ics.append(np.mean(icl) if icl else np.nan)
    rarr = np.array([x for x in rand_ics if not np.isnan(x)])
    pct = (rarr < arr.mean()).mean() if len(arr) and len(rarr) else 0
    print(f"\n[随机基线] {len(rarr)}次打乱tscore3-IC分布: 均值{rarr.mean():+.4f} std{rarr.std():.4f} "
          f"| 真实IC位于随机之上 {pct*100:.1f}% 分位 {'有信号(>95%)' if pct>0.95 else '无信号'}")

    # 结论
    print("\n=== 结论 ===")
    verdict = "待后续完整维度后定论; 当前行情子集链路已跑通"
    print(verdict)


if __name__ == "__main__":
    pool = POOL
    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        pool = POOL + ["002594", "000651", "000333"]
    run(pool)
