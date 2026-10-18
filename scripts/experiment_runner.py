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
import os, sys, json, glob, datetime, time
import numpy as np
import pandas as pd
import urllib.request
from scipy import stats

CACHE = os.path.join(os.path.dirname(__file__), "..", "tmp", "kline")
UA = {"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                     "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"),
      "Referer": "https://gu.qq.com/"}

# 80只 · 覆盖 白酒/银行/保险/券商/新能源/光伏/锂电/医药/地产/煤炭/有色/家电/计算机/军工/电力/基建/通信/半导体/食品/钢铁/机械
POOL = [
    "600519", "000858", "000568", "000596", "600809", "603369",
    "600036", "601166", "000001", "002142", "600919", "601398",
    "601318", "601601", "601336", "601628",
    "600030", "601688", "600999", "300059", "601377",
    "300750", "002594", "601633", "000625", "600104",
    "601012", "600438", "300274",
    "300014", "002466", "002460", "603799",
    "600276", "603259", "300760", "300015", "600436", "300122", "600196",
    "000002", "600048", "001979",
    "601088", "601225", "600188",
    "601899", "600111", "603993", "600547",
    "000333", "000651", "600690", "600060",
    "002415", "002230", "600588", "600570",
    "600893", "600760", "002179",
    "600900", "601985", "600406", "600886",
    "601668", "601390", "601186", "601800",
    "600050", "000063", "600941",
    "688981", "603501", "002371",
    "600887", "603288", "000895",
    "600031", "000338", "600019", "000725", "002475", "002352",
]

THR = {"g1": (2.0, -2.0), "g2": (1.0, -1.0), "g5": (2.0, -2.0), "g9": (5.0, -10.0)}
# 相对差阈值(实验版): 个股涨跌 - 截面中位数, 超过该幅度判阳/阴 (生产绝对阈值不适用于截面)
REL_THR = {"g1": (0.01, -0.01), "g2": (0.01, -0.01), "g5": (0.01, -0.01), "g9": (0.02, -0.02)}


def mkt(c):
    return "sh" if c.startswith(("60", "68")) else "sz"


def _get_json(url, retries=3):
    """带重试的GET(腾讯批量连续请求易触发501限流)。"""
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            return json.loads(urllib.request.urlopen(req, timeout=20).read().decode("utf-8"))
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(0.8 * (i + 1))
    raise last


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
    d = _get_json(url)
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


def panel_fe(df, dep, indep):
    """双重固定效应(股票+日期) OLS + 按股票聚类的稳健SE。
    返回 [(name, coef, se, t, p)]。无外部依赖(numpy实现)。"""
    d = df.dropna(subset=[dep] + indep)
    Y = d[dep].values.astype(float)
    X = np.column_stack([d[k].values.astype(float) for k in indep])
    N, K = X.shape
    # within变换: 按code去均值 + 按date去均值
    code_mean_y = d.groupby("code")[dep].transform("mean").values
    date_mean_y = d.groupby("date")[dep].transform("mean").values
    gm_y = Y.mean()
    Yt = Y - code_mean_y - date_mean_y + gm_y
    Xt = np.empty_like(X)
    for j in range(K):
        cj = d.groupby("code")[indep[j]].transform("mean").values
        dj = d.groupby("date")[indep[j]].transform("mean").values
        Xt[:, j] = X[:, j] - cj - dj + X[:, j].mean()
    Xt1 = np.column_stack([np.ones(N), Xt])          # 加常数(吸收grand mean)
    beta = np.linalg.lstsq(Xt1, Yt, rcond=None)[0]
    e = Yt - Xt1 @ beta
    # 聚类稳健SE (按code)
    XpX = Xt1.T @ Xt1
    meat = np.zeros_like(XpX)
    for key, grp in d.groupby("code"):
        mask = (d["code"] == key).to_numpy()
        Xi = Xt1[mask]
        ei = e[mask]
        s = Xi * ei[:, None]
        meat += (s.T @ s)
    cov = np.linalg.inv(XpX) @ meat @ np.linalg.inv(XpX)
    se = np.sqrt(np.diag(cov))
    dof = N - K - 1
    tstat = beta / se
    p = 2 * stats.t.sf(np.abs(tstat), dof)
    return [(f"const", beta[0], se[0], tstat[0], p[0])] + \
           [(indep[j], beta[j + 1], se[j + 1], tstat[j + 1], p[j + 1]) for j in range(K)]


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
    analyze(df, "日K", 250 / 60)


def fetch_weekly(code):
    """拉周K(2015至今, 640周≈12年)并缓存。返回 DataFrame[date,close]。
    用后复权(hfq): 价格恒正, 避免qfq前复权把历史价推成负值(污染收益)。"""
    path = os.path.join(CACHE, f"{code}_week.json")
    if os.path.exists(path):
        raw = json.load(open(path, encoding="utf-8"))
        df = pd.DataFrame(raw)
        df["date"] = pd.to_datetime(df["date"])
        return df
    os.makedirs(CACHE, exist_ok=True)
    url = (f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?"
           f"param={mkt(code)}{code},week,2015-01-01,2026-12-31,640,hfq")
    req = urllib.request.Request(url, headers=UA)
    d = _get_json(url)
    node = d["data"][f"{mkt(code)}{code}"]
    kl = node.get("hfqweek") or node.get("week") or []
    rows = [[k[0], float(k[2])] for k in kl]
    df = pd.DataFrame(rows, columns=["date", "close"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates("date")
    # 数据质量护栏: 剔除非正价格(后复权下正常全>0)
    df = df[df["close"] > 0]
    json.dump({"date": df["date"].dt.strftime("%Y-%m-%d").tolist(),
               "close": df["close"].tolist()}, open(path, "w", encoding="utf-8"))
    return df


def run_long(pool):
    """长历史模式: 周K(≈12年)重建三趋势维(g1/g5/g9), 每4周一个截面。
    周K无日内, g2 置0; ret20=未来4周, ret60=未来12周(近60交易日)。"""
    print(f"拉取周K({len(pool)}只, 长历史) ...")
    frames = {}
    for c in pool:
        frames[c] = fetch_weekly(c).set_index("date")["close"]
    close = pd.concat(frames, axis=1).sort_index().dropna(how="all")
    close.columns = pool
    print(f"周K区间: {close.index[0].date()} ~ {close.index[-1].date()}  共{len(close)}周")

    rows = []
    for i in range(52, len(close), 4):  # 每4周=月频截面
        p0row = close.iloc[i]
        if i + 12 >= len(close):
            break
        raws = {}
        for c in pool:
            p = close[c].values
            p0 = p[i]
            if np.isnan(p0) or np.isnan(p[i - 52]) or np.isnan(p[i + 12]):
                continue
            raws[c] = {"g1": p0 / p[i - 4] - 1, "g5": p0 / p[i - 2] - 1,
                       "g9": p0 / p[i - 52] - 1,
                       "ret20": p[i + 4] / p0 - 1, "ret60": p[i + 12] / p0 - 1}
        if len(raws) < 8:
            continue
        med = {k: float(np.median([r[k] for r in raws.values()])) for k in ("g1", "g5", "g9")}
        for c, v in raws.items():
            g = {k: band(v[k] - med[k], *REL_THR[k]) for k in ("g1", "g5", "g9")}
            g["g2"] = 0  # 周K无日内
            tscore3 = g["g1"] + g["g5"] + g["g9"]
            mode = "strong" if tscore3 >= 1 else ("weak" if tscore3 <= -1 else "neutral")
            rows.append({**g, "mode": mode, "tscore3": tscore3,
                         "ret20": v["ret20"], "ret60": v["ret60"],
                         "date": close.index[i].date(), "code": c})
    df = pd.DataFrame(rows)
    print(f"有效观测: {len(df)} 行 ({df['date'].nunique()} 个截面 × {df['code'].nunique()} 只)")
    print(f"未来收益(周): ret4w mean {df['ret20'].mean()*100:+.2f}%, ret12w mean {df['ret60'].mean()*100:+.2f}%")
    analyze(df, "周K长历史", 52 / 12)


def analyze(df, src, ann_period):
    print(f"\n[检验A] 主态分层 · 未来收益 ({src})")
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

    # ---- 检验D: 双重固定效应面板回归(股票+日期) + 聚类稳健SE ----
    indep = ["tscore3"] + (["g2"] if df["g2"].nunique() > 1 else [])  # g2恒定(周K)时剔除
    print(f"\n[检验D] 双重固定效应面板回归 ret60 ~ {'+'.join(indep)} ({src})")
    reg = panel_fe(df, dep="ret60", indep=indep)
    for name, b, se, t, p in reg:
        sig = "显著" if p < 0.05 else "不显著"
        print(f"  {name:10s} coef={b:+.4f}  SE={se:.4f}  t={t:+.2f}  p={p:.4f}  {sig}")

    # ---- 检验D'基线: 打乱tscore3重算FE系数 ----
    real_coef = [r[1] for r in reg if r[0] == "tscore3"][0]
    rng2 = np.random.default_rng(7)
    fe_coefs = []
    for _ in range(200):
        dd = df.copy()
        dd["tscore3"] = rng2.permutation(dd["tscore3"].values)
        regr = panel_fe(dd, dep="ret60", indep=indep)
        fe_coefs.append([r[1] for r in regr if r[0] == "tscore3"][0])
    farr = np.array(fe_coefs)
    pct2 = (farr < real_coef).mean()
    print(f"\n[检验D'基线] 200次打乱tscore3的FE系数分布: 均值{farr.mean():+.4f} std{farr.std():.4f} "
          f"| 真实系数({real_coef:+.4f})位于随机之下 {pct2*100:.1f}% 分位 "
          f"{'显著非偶然' if pct2>0.95 or pct2<0.05 else '无信号'}")

    # ---- 检验E: 反转多空组合回测(把FE负系数变成可操作信号) ----
    # 反转信号 = 个股当前tscore3 - 个股历史习惯水平(个股内相对偏强程度)
    df["rev"] = df["tscore3"] - df.groupby("code")["tscore3"].transform("mean")
    legs = []
    for dt, sub in df.groupby("date"):
        if len(sub) < 12:
            continue
        sub = sub.sort_values("rev")
        third = max(1, len(sub) // 3)
        weak = sub.iloc[:third]      # 当前最弱于自身习惯 → 反转做多
        strong = sub.iloc[-third:]   # 当前最强于自身习惯 → 反转做空
        legs.append({"date": dt,
                     "long_weak": weak["ret60"].mean(),
                     "short_strong": strong["ret60"].mean()})
    legs = pd.DataFrame(legs)
    legs["spread"] = legs["long_weak"] - legs["short_strong"]  # 做弱空强
    spr = legs["spread"]
    t, p = stats.ttest_1samp(spr, 0)
    ann = (1 + spr.mean()) ** ann_period - 1
    win = (spr > 0).mean()
    print("\n[检验E] 反转多空组合(做弱/空强) · ret60")
    print(f"  截面数={len(spr)}  平均多空spread={spr.mean()*100:+.2f}pp/60d  年化≈{ann*100:+.1f}%")
    print(f"  t={t:.2f} p={p:.4f}  胜率={win*100:.0f}%  累计={((1+spr).cumprod().iloc[-1]-1)*100:+.1f}%")
    print(f"  对照·做弱组均值={legs['long_weak'].mean()*100:+.2f}%  做空组均值={legs['short_strong'].mean()*100:+.2f}%")

    # ---- 检验E'基线: 打乱tscore3重算反转spread ----
    rng3 = np.random.default_rng(11)
    spreads = []
    for _ in range(200):
        dd = df.copy()
        dd["tscore3"] = rng3.permutation(dd["tscore3"].values)
        dd["rev"] = dd["tscore3"] - dd.groupby("code")["tscore3"].transform("mean")
        sl = []
        for dt, sub in dd.groupby("date"):
            if len(sub) < 12:
                continue
            sub = sub.sort_values("rev")
            third = max(1, len(sub) // 3)
            sl.append(sub.iloc[:third]["ret60"].mean() - sub.iloc[-third]["ret60"].mean())
        spreads.append(np.mean(sl) if sl else np.nan)
    sarr = np.array([x for x in spreads if not np.isnan(x)])
    pct3 = (sarr < spr.mean()).mean()
    print(f"\n[检验E'基线] 200次打乱tscore3的反转spread分布: 均值{sarr.mean()*100:+.2f}pp std{sarr.std()*100:.2f}pp "
          f"| 真实spread({spr.mean()*100:+.2f}pp)位于随机之上 {pct3*100:.1f}% 分位 "
          f"{'有信号(>95%)' if pct3>0.95 else '无信号'}")

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
    if len(sys.argv) > 1 and sys.argv[1] == "--long":
        run_long(pool)
    elif len(sys.argv) > 1 and sys.argv[1] == "--all":
        pool = POOL + ["002594", "000651", "000333"]
        run(pool)
    else:
        run(pool)
