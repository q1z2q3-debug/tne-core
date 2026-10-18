#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""完整九维·财务三维(g3净利率/g4ROE/g8归母净利)接入长历史检验。
周K(hfq)行情趋势维 + 东财多期财务 + 波动率对照。
前视控制: t截面用 report_date+90天<=t 的最近财报(保守披露滞后)。
检验: 双重FE回归 ret60 ~ tscore3(行情)+fscore(财务)+vol, 各自预测力。"""
import os, sys, json, numpy as np, pandas as pd, datetime
from scipy import stats
sys.path.insert(0, os.path.dirname(__file__))
from experiment_runner import panel_fe, fetch_weekly, CACHE

ROOT = os.path.join(os.path.dirname(__file__), "..")
FIN = os.path.join(ROOT, "data", "finance_history")
POOL = ["600519","000858","000568","000596","600809","603369","600036","601166","000001","002142","600919","601398","601318","601601","601336","601628","600030","601688","600999","300059","601377","300750","002594","601633","000625","600104","601012","600438","300274","300014","002466","002460","603799","600276","603259","300760","300015","600436","300122","600196","000002","600048","001979","601088","601225","600188","601899","600111","603993","600547","000333","000651","600690","600060","002415","002230","600588","600570","600893","600760","002179","600900","601985","600406","600886","601668","601390","601186","601800","600050","000063","600941","688981","603501","002371","600887","603288","000895","600031","000338","600019","000725","002475","002352"]

def band(v, up, dn):
    return 1 if v >= up else (-1 if v <= dn else 0)

def main():
    print("读周K+财务 ...")
    frames = {c: fetch_weekly(c).set_index("date")["close"] for c in POOL}
    close = pd.concat(frames, axis=1).sort_index().dropna(how="all")
    close.columns = POOL
    fin = {}
    for c in POOL:
        p = os.path.join(FIN, f"{c}.json")
        if os.path.exists(p):
            d = pd.DataFrame(json.load(open(p)))
            d["date"] = pd.to_datetime(d["date"])
            fin[c] = d
    print(f"周K {len(close)} 周; 财务缓存 {len(fin)} 只")

    rows = []
    for i in range(52, len(close), 4):
        if i + 12 >= len(close):
            break
        tdate = close.index[i]
        raws = {}
        for c in POOL:
            p = close[c].values; p0 = p[i]
            if np.isnan(p0) or np.isnan(p[i-52]) or np.isnan(p[i+12]):
                continue
            seg = p[i-52:i+1]; vret = seg[1:]/seg[:-1]-1
            fdf = fin.get(c)
            if fdf is None:
                continue
            fmask = fdf["date"] <= (tdate - pd.Timedelta(days=90))
            if not fmask.any():
                continue
            fr = fdf[fmask].iloc[-1]
            nm, roe, np_ = fr["net_margin"], fr["roe"], fr["net_parent"]
            if any(pd.isna(x) for x in (nm, roe, np_)):
                continue
            raws[c] = {"g1": p0/p[i-4]-1, "g5": p0/p[i-2]-1, "g9": p0/p[i-52]-1,
                       "vol": float(np.std(vret)) if len(vret) >= 20 else np.nan,
                       "nm": nm, "roe": roe, "np": np_,
                       "ret20": p[i+4]/p0-1, "ret60": p[i+12]/p0-1}
        if len(raws) < 10:
            continue
        mg = {k: float(np.median([r[k] for r in raws.values()])) for k in ("g1","g5","g9")}
        mf = {k: float(np.median([r[k] for r in raws.values()])) for k in ("nm","roe","np")}
        for c, v in raws.items():
            g = {k: band(v[k]-mg[k], 0.01, -0.01) for k in ("g1","g5","g9")}
            f3 = {"g3": band(v["nm"]-mf["nm"], 5.0, -5.0),
                  "g4": band(v["roe"]-mf["roe"], 3.0, -3.0),
                  "g8": band(v["np"]/mf["np"]-1, 0.2, -0.2) if mf["np"] > 0 else 0}
            rows.append({**g, **f3, "tscore3": g["g1"]+g["g5"]+g["g9"],
                         "fscore": f3["g3"]+f3["g4"]+f3["g8"], "vol": v["vol"],
                         "ret20": v["ret20"], "ret60": v["ret60"],
                         "date": tdate.date(), "code": c})
    df = pd.DataFrame(rows)
    print(f"有效观测: {len(df)} ({df['date'].nunique()}截面 × {df['code'].nunique()}只)")
    print(f"财务态分布: g3 {df['g3'].value_counts().to_dict()}, g4 {df['g4'].value_counts().to_dict()}, g8 {df['g8'].value_counts().to_dict()}")

    # 检验D: 行情+财务+vol
    print("\n[检验D] 完整九维 FE回归 ret60 ~ tscore3+fscore+vol")
    for name, b, se, t, p in panel_fe(df, "ret60", ["tscore3","fscore","vol"]):
        print(f"  {name:8s} coef={b:+.4f} SE={se:.4f} t={t:+.2f} p={p:.4f} {'显著' if p<0.05 else '不显著'}")

    # 财务单独(控制vol, 不含行情)
    print("\n[检验D2] 财务维度单独 ret60 ~ fscore+vol")
    for name, b, se, t, p in panel_fe(df, "ret60", ["fscore","vol"]):
        print(f"  {name:8s} coef={b:+.4f} SE={se:.4f} t={t:+.2f} p={p:.4f} {'显著' if p<0.05 else '不显著'}")

    # 随机基线 fscore
    real = [r[1] for r in panel_fe(df, "ret60", ["fscore","vol"]) if r[0]=="fscore"][0]
    rng = np.random.default_rng(31)
    fc = []
    for _ in range(200):
        dd = df.copy(); dd["fscore"] = rng.permutation(dd["fscore"].values)
        fc.append([r[1] for r in panel_fe(dd, "ret60", ["fscore","vol"]) if r[0]=="fscore"][0])
    fa = np.array(fc)
    pct = (fa < real).mean()
    print(f"\n[基线] 200次打乱fscore的FE系数: 均值{fa.mean():+.4f} std{fa.std():.4f} | "
          f"真实({real:+.4f})位于随机之下 {pct*100:.1f}% 分位 {'显著非偶然' if pct>0.95 or pct<0.05 else '无信号'}")

    # 检验F: 财务质量多空回测(做多高fscore/做空低fscore)
    legs = []
    for dt, sub in df.groupby("date"):
        if len(sub) < 12:
            continue
        sub = sub.sort_values("fscore")
        third = max(1, len(sub) // 3)
        legs.append({"date": dt,
                     "spread": sub.iloc[-third:]["ret60"].mean() - sub.iloc[:third]["ret60"].mean()})
    legs = pd.DataFrame(legs)
    spr = legs["spread"]
    t, p = stats.ttest_1samp(spr, 0)
    ann = (1 + spr.mean()) ** (52 / 12) - 1
    win = (spr > 0).mean()
    print(f"\n[检验F] 财务质量多空(做多高fscore/做空低fscore) · ret60")
    print(f"  截面数={len(spr)} 平均spread={spr.mean()*100:+.2f}pp/12w 年化≈{ann*100:+.1f}% "
          f"t={t:.2f} p={p:.4f} 胜率={win*100:.0f}%")
    rng5 = np.random.default_rng(37)
    fs = []
    for _ in range(200):
        dd = df.copy(); dd["fscore"] = rng5.permutation(dd["fscore"].values)
        sl = []
        for dt, sub in dd.groupby("date"):
            if len(sub) < 12:
                continue
            sub = sub.sort_values("fscore")
            third = max(1, len(sub) // 3)
            sl.append(sub.iloc[-third:]["ret60"].mean() - sub.iloc[:third]["ret60"].mean())
        fs.append(np.mean(sl) if sl else np.nan)
    ffa = np.array([x for x in fs if not np.isnan(x)])
    pct2 = (ffa < spr.mean()).mean()
    print(f"  [基线] 200次打乱fscore: 均值{ffa.mean()*100:+.2f}pp std{ffa.std()*100:.2f}pp | "
          f"真实位于随机之上 {pct2*100:.1f}% 分位 {'有信号(>95%)' if pct2>0.95 else '无信号'}")

if __name__ == "__main__":
    main()
