#!/usr/bin/env python3
"""
TNE-Core · 九维金融 · 时序跟踪：同一股票多日快照 → 九维画像演化轨迹
输入: 某股票在多个日期(收盘/估值/趋势/财务)的快照列表
输出: 时间轴演化HTML(每点卦位+九维码+认知主态, 展示画像随行情演化)
"""
import sys, json, os
sys.path.insert(0, "/home/user/Doubao/chats/38446456964142338/tne-core/src")
sys.path.insert(0, "/home/user/Doubao/chats/38446456964142338/tne-core/scripts")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from finance_mapper import map_nine, diagnose

# 贵州茅台 · 2026年9月末→10月中 演化快照(基于同花顺日K/月K/财务, 6个交易日点)
MAOTAI = {
    "name": "贵州茅台", "code": "600519.SH", "industry": "白酒/高端消费",
    "points": [
        {"date": "2026-09-28", "close": 1243.88, "pe": 19.09, "chg": -0.55,
         "month_pct": -3.5, "two_week": -0.6, "year_pct": -7.6, "turnover": 0.32,
         "margin": 49.9, "roe": 17.72, "net_yi": 445.17},
        {"date": "2026-09-30", "close": 1258.62, "pe": 19.32, "chg": 1.20,
         "month_pct": -3.15, "two_week": -0.9, "year_pct": -7.2, "turnover": 0.34,
         "margin": 49.9, "roe": 17.72, "net_yi": 445.17},
        {"date": "2026-10-08", "close": 1255.79, "pe": 19.28, "chg": -0.22,
         "month_pct": -0.2, "two_week": 1.5, "year_pct": -6.5, "turnover": 0.30,
         "margin": 49.9, "roe": 17.72, "net_yi": 445.17},
        {"date": "2026-10-12", "close": 1288.00, "pe": 19.77, "chg": 1.96,
         "month_pct": 2.3, "two_week": 2.9, "year_pct": -5.5, "turnover": 0.35,
         "margin": 49.9, "roe": 17.72, "net_yi": 445.17},
        {"date": "2026-10-14", "close": 1286.00, "pe": 19.74, "chg": 0.87,
         "month_pct": 2.2, "two_week": 2.4, "year_pct": -5.1, "turnover": 0.32,
         "margin": 49.9, "roe": 17.72, "net_yi": 445.17},
        {"date": "2026-10-16", "close": 1329.00, "pe": 20.40, "chg": 1.22,
         "month_pct": 5.59, "two_week": 5.6, "year_pct": -3.5, "turnover": 0.31,
         "margin": 49.9, "roe": 17.72, "net_yi": 445.17},
    ],
}


def point_to_snapshot(p):
    return {"name": MAOTAI["name"], "code": MAOTAI["code"], "date": p["date"],
            "industry": MAOTAI["industry"], "sector": "白酒·龙头·板块普涨",
            "quote": {"close": p["close"], "chg_pct": p["chg"], "turnover_pct": p["turnover"],
                      "pe_ttm": p["pe"], "pb": 6.5, "mktcap_yi": round(p["close"] * 12.4)},
            "trend": {"month_pct": p["month_pct"], "two_week_pct": p["two_week"],
                      "year_pct": p["year_pct"], "year_high": 1568, "year_low": 1151},
            "finance": {"net_margin_pct": p["margin"], "roe_pct": p["roe"],
                        "net_parent_yi": p["net_yi"], "net_profit_yi": 460.33,
                        "eps": 35.57, "revenue_yi": 907.03}}


def build_html(results):
    nodes = []
    for i, r in enumerate(results):
        nodes.append(f"""<li><div class="dot"></div>
<div class="node"><div class="ndate">{r['date']}</div>
<div class="nclose">¥{r['quote']['close']:.0f} <span class="c{('up' if r['quote']['chg_pct']>0 else 'down')}">{r['quote']['chg_pct']:+.2f}%</span></div>
<div class="ncode">{r['code_str']}</div>
<div class="ngua">卦位 #{r['gua_idx']} · 类{r['class_rep']}</div>
<div class="nmode">{r['mode']}</div>
<div class="ncells">{"".join(f"<i class='{('yang' if r['vector'][k]==1 else 'yin' if r['vector'][k]==-1 else '')}'>" + ('生' if r['vector'][k]==1 else '衡' if r['vector'][k]==0 else '敛') + "</i>" for k in range(9))}</div>
<div class="nrec">{r['rec'][:18]}</div></div></li>""")
    return f"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{results[0]['name']} · 九维画像时序演化 · TNE-Core</title><style>
:root{{--ink:#d4af6e;--ink2:#e0b96a;--muted:#8a8f98;--bg:#0b0c10;--s2:#14161c;--line:#2a2d35;--yang:#e0b96a;--yin:#6b87b8;--heng:#6d717a}}
*{{margin:0;box-sizing:border-box}}body{{background:var(--bg);color:#c9ccd2;font-family:'Noto Serif SC','Songti SC',serif;padding:24px 16px;line-height:1.5}}
.wrap{{max-width:1000px;margin:0 auto}}.top{{border-bottom:1px solid var(--line);padding-bottom:14px;margin-bottom:20px}}
.brand{{font-size:11px;letter-spacing:.3em;color:var(--ink);margin-bottom:4px}}h1{{font-size:24px;color:var(--ink)}}
.sub{{color:var(--muted);font-size:13px;margin-top:4px}}
ul{{list-style:none;position:relative;padding:0;margin:0}}
ul::before{{content:"";position:absolute;left:14px;top:8px;bottom:8px;width:1px;background:var(--line)}}
li{{position:relative;padding-left:44px;padding-bottom:26px}}
.dot{{position:absolute;left:8px;top:8px;width:14px;height:14px;border-radius:50%;background:var(--ink2);border:2px solid var(--bg)}}
.node{{background:var(--s2);border:1px solid var(--line);border-radius:6px;padding:12px 16px}}
.ndate{{color:var(--ink);font-weight:600;font-size:13px;letter-spacing:.05em}}
.nclose{{font-size:12px;margin:3px 0}}.up{{color:#7fdc8a}}.down{{color:#e07f7f}}
.ncode{{font-family:monospace;font-size:14px;letter-spacing:.2em;color:var(--ink2);margin:4px 0}}
.ngua{{font-size:11px;color:var(--muted)}}.nmode{{display:inline-block;background:rgba(224,185,106,.12);border:1px solid rgba(224,185,106,.35);color:var(--ink2);padding:2px 10px;border-radius:3px;font-size:12px;margin:4px 0}}
.ncells i{{display:inline-flex;width:22px;height:22px;align-items:center;justify-content:center;border-radius:3px;font-size:12px;font-style:normal;margin:1px;background:#1c1f28}}
.ncells .yang{{color:var(--yang);border:1px solid rgba(224,185,106,.4)}}.ncells .yin{{color:var(--yin);border:1px solid rgba(107,135,184,.4)}}
.ncells i:not(.yang):not(.yin){{color:var(--heng)}}
.nrec{{font-size:11px;color:var(--muted);margin-top:5px}}
.foot{{margin-top:8px;font-size:11px;color:var(--muted);border-top:1px solid var(--line);padding-top:10px}}
.legend{{display:flex;gap:14px;font-size:11px;color:var(--muted);margin:10px 0}}i.lg{{width:10px;height:10px;border-radius:2px;display:inline-block;margin-right:4px}}.lg-yang{{background:var(--yang)}}.lg-yin{{background:var(--yin)}}
</style></head><body><div class="wrap">
<div class="top"><div class="brand">TRINARY NINE-DIMENSIONAL FINANCIAL COGNITION · TIME SERIES</div>
<h1>{results[0]['name']} · 九维画像时序演化</h1><div class="sub">同一股票 2026-09末→10月中 价格走势 → 九维画像/卦位/认知主态随行情的演化轨迹</div></div>
<div class="legend"><span><i class="lg lg-yang"></i>阳·扩张</span><span><i class="lg lg-yin"></i>阴·收敛</span><span><i class="lg"></i>衡·盘整</span></div>
<ul>{nodes}</ul>
<div class="foot">九维金融认知为对公开行情/财务数据的结构化分析，属研究辅助工具，不构成投资建议。数据来源：Wind/同花顺公开行情与财报（快照基于日K/月K构造）。</div>
</div></body></html>"""


def main():
    results = []
    for p in MAOTAI["points"]:
        snap = point_to_snapshot(p)
        g = map_nine(snap)
        r = diagnose(g, snap)
        results.append(r)
    out = "demo/贵州茅台九维画像时序演化.html"
    open(out, "w", encoding="utf-8").write(build_html(results))
    print(f"时序页已生成: {out}  ({len(results)}个时间点)")
    for r in results:
        print(f"  {r['date']}  ¥{r['quote']['close']:.0f}  {r['code_str']}  #{r['gua_idx']}  {r['mode']}")


if __name__ == "__main__":
    main()
