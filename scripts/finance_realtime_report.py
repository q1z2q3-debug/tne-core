#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实时直出三层诊断: 输入任意A股代码 → 实时拉取(腾讯行情/日K + 东财财务) →
直接跑 cognition 三层认知诊断(空间卦位/时间周期/张力信号), 无需手动入库。

用法:
  python3 scripts/finance_realtime_report.py 603259            # 直出三层诊断(终端)
  python3 scripts/finance_realtime_report.py 600519 --html     # 生成三层诊断HTML
  python3 scripts/finance_realtime_report.py 300750 --save     # 直出 + 入库 data/finance/
研究辅助, 非投资建议, 不构成荐股。
"""
import sys
import os
import json
import urllib.request
import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from tne.finance.cognition import map_nine, diagnose, cycle_analyze, watch_analyze, DISCLAIMER  # noqa: E402

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
      "Referer": "https://gu.qq.com/"}

# 常见行业映射(未收录回退"待确认")
INDUSTRY = {"600519": "白酒/高端消费", "000858": "白酒/高端消费", "600036": "银行/零售",
            "600276": "医药/创新药", "300750": "新能源/电池", "688981": "半导体/代工",
            "600309": "化工/聚氨酯", "601318": "保险/综合金融", "600030": "券商/综合",
            "000002": "地产/开发", "601899": "有色/黄金铜", "000333": "家电/白电",
            "601088": "煤炭/能源", "603259": "医药外包/CXO"}


def norm_code(code):
    code = code.strip().split(".")[0]
    if code.startswith(("60", "68")):
        return "sh", code, f"1.{code}"
    if code.startswith(("00", "30")):
        return "sz", code, f"0.{code}"
    raise ValueError(f"无法识别市场前缀: {code}")


def _get(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=15) as r:
        return r.read().decode("utf-8", "ignore")


def fetch_quote(mkt, code):
    """腾讯实时行情。返回 quote 段。"""
    raw = _get(f"https://qt.gtimg.cn/q={mkt}{code}")
    body = raw.split('="', 1)[1].rsplit('"', 1)[0]
    f = body.split("~")
    return {"name": f[1], "close": float(f[3]), "prev_close": float(f[4]), "open": float(f[5]),
            "high": float(f[33]), "low": float(f[34]), "chg_pct": float(f[32]),
            "turnover_pct": float(f[38]), "pe_ttm": float(f[39]) if f[39] else 0.0,
            "pb": float(f[46]) if f[46] else 0.0, "mktcap_yi": float(f[45]),
            "volume_wy": float(f[6]) / 100.0, "amount_yi": float(f[37]) / 10000.0,
            "date_str": f[30][:8]}


def fetch_trend(mkt, code, today):
    """腾讯日K(近2年) → 月/两周/年涨跌 + 年高低。"""
    beg = (datetime.date.today() - datetime.timedelta(days=730)).isoformat()
    end = (datetime.date.today() + datetime.timedelta(days=2)).isoformat()
    d = json.loads(_get(f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={mkt}{code},day,{beg},{end},640,qfq"))
    kl = d["data"][f"{mkt}{code}"]["qfqday"]
    latest = float(kl[-1][2])
    def c(tgt):
        for k in reversed(kl):
            if datetime.date.fromisoformat(k[0]) <= tgt:
                return float(k[2])
        return None
    m = c(today - datetime.timedelta(days=30))
    w = c(today - datetime.timedelta(days=14))
    y = c(today - datetime.timedelta(days=365))
    yr = [float(k[2]) for k in kl if datetime.date.fromisoformat(k[0]) >= (today - datetime.timedelta(days=365))]
    return {"month_pct": (latest / m - 1) * 100 if m else 0.0,
            "two_week_pct": (latest / w - 1) * 100 if w else 0.0,
            "year_pct": (latest / y - 1) * 100 if y else 0.0,
            "year_high": max(yr), "year_low": min(yr)}


def fetch_finance(secid):
    """东财 datacenter 最新一期主要财务指标。secid 形如 1.603259 / 0.000858。"""
    suffix = "SH" if secid.startswith("1.") else "SZ"
    code = secid.split(".")[1]
    url = ("https://datacenter.eastmoney.com/securities/api/data/get?"
           "type=RPT_F10_FINANCE_MAINFINADATA&sty=APP_F10_MAINFINADATA"
           f"&filter=(SECUCODE%3D%22{code}.{suffix}%22)&p=1&ps=1")
    d = json.loads(_get(url))
    rows = d["result"]["data"]
    if not rows:
        return {}
    r = rows[0]
    return {"net_margin_pct": r.get("XSMLL", 0.0), "roe_pct": r.get("ROEJQ", 0.0),
            "net_parent_yi": (r.get("PARENTNETPROFIT") or 0) / 1e8,
            "revenue_yi": (r.get("TOTALOPERATEREVE") or 0) / 1e8,
            "report": r.get("REPORT_DATE_NAME", "")}


def build(d):
    q = d["quote"]
    return {"name": d["name"], "code": d["code"], "date": d["date"],
            "industry": d["industry"], "quote": q, "trend": d["trend"],
            "finance": d["finance"], "sector": d["sector"]}


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    code = sys.argv[1]
    as_html = "--html" in sys.argv
    as_save = "--save" in sys.argv
    mkt, code6, secid = norm_code(code)
    today = datetime.date.today()
    try:
        qt = fetch_quote(mkt, code6)
        tr = fetch_trend(mkt, code6, today)
        fin = fetch_finance(secid)
        if not fin:
            print("警告: 财务数据拉取为空, 诊断将不完整")
    except Exception as e:
        print(f"数据拉取失败: {e}")
        return
    d = {"name": qt["name"], "code": f"{code6}.{'SH' if mkt=='sh' else 'SZ'}",
         "date": f"{qt['date_str'][:4]}-{qt['date_str'][4:6]}-{qt['date_str'][6:]} 收盘",
         "industry": INDUSTRY.get(code6, "待确认"),
         "quote": qt, "trend": tr, "finance": fin, "sector": "实时拉取, 行业待确认"}
    d = build(d)
    r = diagnose(d)
    cyc = cycle_analyze(d)
    w = watch_analyze(d)

    print(f"\n=== {code6} 实时三层认知诊断 ({d['date']}) ===")
    print(f"行情: 收{qt['close']} 当日{qt['chg_pct']:+.2f}% PE{qt['pe_ttm']} PB{qt['pb']} 总市值{qt['mktcap_yi']:.0f}亿")
    print(f"财务: 净利率{fin.get('net_margin_pct',0):.1f}% ROE{fin.get('roe_pct',0):.1f}% 归母净利{fin.get('net_parent_yi',0):.1f}亿")
    print(f"趋势: 月{tr['month_pct']:+.1f}% 两周{tr['two_week_pct']:+.1f}% 年{tr['year_pct']:+.1f}% 年高{tr['year_high']} 年低{tr['year_low']}")
    print(f"空间层: {r['code_str']} 卦位#{r['gua_idx']} 主态{r['mode']} | {r['rec']}")
    print(f"时间层: {cyc['chain_code']} ({cyc['mode']}) 张力{cyc['tension']} 背离{'是' if cyc['conflict'] else '否'}")
    print(f"张力层: {[s['type'] for s in r['signals']]} | 关注度 {w['score']} {w['watch_level']}")
    print(DISCLAIMER)

    if as_save:
        path = f"data/finance/{qt['name']}_{code6}.json"
        os.makedirs("data/finance", exist_ok=True)
        json.dump(d, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"\n已入库: {path}")
    if as_html:
        html = f"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{code6} 实时三层认知诊断</title><style>
:root{{--bg:#0b0c10;--s2:#11131a;--line:#2a2e3a;--gold:#d4af6e;--muted:#8a93a6;--ink:#e8e6e1}}
*{{box-sizing:border-box;margin:0}}body{{background:var(--bg);color:var(--ink);font-family:'Noto Serif SC',serif;padding:24px 16px;max-width:900px;margin:0 auto}}
h1{{color:var(--gold);font-size:22px;letter-spacing:1px;border-bottom:1px solid var(--line);padding-bottom:10px}}
.sub{{color:var(--muted);font-size:12px;margin:8px 0 16px}}
.kpi{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:14px 0}}
@media(max-width:640px){{.kpi{{grid-template-columns:repeat(2,1fr)}}}}
.k{{background:var(--s2);border:1px solid var(--line);border-radius:6px;padding:12px}}
.k b{{color:var(--gold);font-size:20px;display:block}}.k span{{color:var(--muted);font-size:11px}}
.sec{{background:var(--s2);border:1px solid var(--line);border-radius:6px;padding:14px;margin:10px 0}}
.sec b{{color:var(--gold);display:block;margin-bottom:8px}}
.chain{{display:flex;gap:8px;flex-wrap:wrap}}
.cn{{border:1px solid;border-radius:4px;padding:8px 12px;text-align:center;font-size:12px;background:#0f1120}}
.st{{font-size:18px;font-weight:700;color:var(--bg)}}
.lv{{font-size:14px;color:var(--gold);font-weight:700}}
.foot{{color:var(--muted);font-size:11px;margin-top:16px;border-top:1px solid var(--line);padding-top:8px}}
</style></head><body>
<h1>{code6} · 实时三层认知诊断</h1>
<div class="sub">{d['date']} · {d['industry']} · 实时拉取(腾讯行情/日K + 东财财务)</div>
<div class="kpi">
  <div class="k"><b>{qt['close']}</b><span>收盘价</span></div>
  <div class="k"><b>{qt['chg_pct']:+.2f}%</b><span>当日涨跌</span></div>
  <div class="k"><b>{qt['pe_ttm']}</b><span>PE</span></div>
  <div class="k"><b>{w['score']}</b><span>{w['watch_level']}</span></div>
</div>
<div class="sec"><b>空间层 · 卦位</b>九维 <span class="lv">{r['code_str']}</span> 卦位#{r['gua_idx']} · 主态 <b>{r['mode']}</b><br>
  <span style="color:var(--muted);font-size:13px">{r['rec']}</span></div>
<div class="sec"><b>时间层 · 周期传导</b>
  <div class="chain">{"".join(f"<div class='cn' style='border-color:{ {'阴':'#6b7a99','衡':'#c0a66a','阳':'#d4af6e'}[c['state_name']] }'><div class='st'>{c['state_name']}</div>{c['name']} {c['pct']:+.1f}%</div>" for c in cyc['chain'])}</div>
  <div style="margin-top:8px;color:var(--muted);font-size:13px">{cyc['mode']} · 张力{cyc['tension']} · 快慢背离{'是' if cyc['conflict'] else '否'}</div></div>
<div class="sec"><b>张力层 · 认知信号</b>
  <span style="color:var(--muted);font-size:13px">{[s['type'] for s in r['signals']]}</span>
  <span style="color:var(--gold);margin-left:10px">关注度 {w['score']} · {w['watch_level']}</span></div>
<div class="foot">研究辅助，非投资建议，不构成荐股 · 三元九维认知引擎 TNE-Core</div>
</body></html>"""
        os.makedirs("demo", exist_ok=True)
        open("demo/实时三层认知诊断.html", "w", encoding="utf-8").write(html)
        print("已生成: demo/实时三层认知诊断.html")


if __name__ == "__main__":
    main()
