# -*- coding: utf-8 -*-
"""全市场周期传导矩阵: 13只股票 × 4周期层(日→周→月→年) 周期链 + 背离张力指数。
数据复用 data/finance/*.json (真实行情2026-10-16收盘)。
日层=当日chg, 周层=两周, 月层=月趋势, 年层=年趋势。每层>=+2阳/<=-2阴/其余衡。
背离张力: 快慢(日vs年)方向不一致=转折张力; 层间反转次数累计=张力强度。
"""
import json, glob, os, math

S = {-1: "阴", 0: "衡", 1: "阳"}
# 周期层阈值不同: 日层单日波动小用±1%, 周/月/年用±2%
LAYERS = [("日·动能", "quote", "chg_pct", 1.0), ("周·波段", "trend", "two_week_pct", 2.0),
          ("月·趋势", "trend", "month_pct", 2.0), ("年·根性", "trend", "year_pct", 2.0)]

def tri(v, thr=2.0):
    return 1 if v >= thr else (-1 if v <= -thr else 0)

def load_all():
    files = sorted(glob.glob("data/finance/*.json"))
    out = []
    for f in files:
        d = json.load(open(f, encoding="utf-8"))
        chain = []
        for nm, sec, key, thr in LAYERS:
            v = d[sec][key]
            chain.append({"n": nm, "v": v, "s": tri(v, thr)})
        out.append({"d": d, "chain": chain})
    return out

def analyze(stk):
    c = stk["chain"]
    day, week, month, year = c[0]["s"], c[1]["s"], c[2]["s"], c[3]["s"]
    # 层间反转次数(方向变化)
    revs = sum(1 for i in range(3) if c[i]["s"] != 0 and c[i+1]["s"] != 0 and c[i]["s"] != c[i+1]["s"])
    # 背离张力: 快(日) vs 慢(年) 不一致
    conflict = day != year
    # 张力指数 0-10: 反转次数*2 + 快慢背离*3 + 根性绝对
    tension = revs * 2 + (3 if conflict else 0) + abs(year)
    yang = sum(1 for x in c if x["s"] == 1)
    yin = sum(1 for x in c if x["s"] == -1)
    # 形态: 年=根性定基调, 快慢背离定张力
    if year == 1 and month == 1:
        mode, cls = "顺势·生发", "strong"
    elif year == -1 and (day == 1 or week == 1):
        mode, cls = "反弹·背离", "rebound"
    elif year == -1 and month == -1:
        mode, cls = "弱势·收敛", "weak"
    elif year == 1 and (day == -1 or week == -1):
        mode, cls = "回踩·背离", "pullback"
    elif year == -1 and day == -1:
        mode, cls = "弱势·收敛", "weak"
    else:
        mode, cls = "震荡·待变", "mixed"
    return {"mode": mode, "cls": cls, "revs": revs, "conflict": conflict,
            "tension": tension, "yang": yang, "yin": yin}

def build_html(rows):
    rows = sorted(rows, key=lambda r: -r["a"]["tension"])
    trs = ""
    for r in rows:
        d, c, a = r["d"], r["chain"], r["a"]
        cells = "".join(f"<span class='s {S[x['s']]}' title='{x['n']} {x['v']:+.1f}%'>{S[x['s']]}</span>"
                        for x in c)
        conflict_tag = ("<b class='c1'>背离</b>" if a["conflict"] else "<b class='c0'>同向</b>")
        trs += f"""<tr class="{a['cls']}">
          <td><span class="nm">{d['name']}</span><span class="code">{d['code'].split('.')[0]}</span>
              <span class="ind">{d['industry']}</span></td>
          <td>{cells}</td>
          <td><span class="mode {a['cls']}">{a['mode']}</span></td>
          <td class="rev">{a['revs']}</td>
          <td>{conflict_tag}</td>
          <td class="ten">{a['tension']}</td>
        </tr>"""
    # 张力指数条
    max_t = max(r["a"]["tension"] for r in rows) or 1
    html = f"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>全市场周期传导矩阵 · 三元九维金融</title>
<style>
:root{{--bg:#0b0c10;--s2:#11131a;--line:#2a2e3a;--gold:#d4af6e;--muted:#8a93a6;--ink:#e8e6e1;--up:#d4af6e;--dn:#6b7a99}}
*{{box-sizing:border-box;margin:0}}body{{background:var(--bg);color:var(--ink);font-family:'Noto Serif SC',serif;padding:26px 18px}}
h1{{font-size:24px;letter-spacing:2px;color:var(--gold);border-bottom:1px solid var(--line);padding-bottom:12px}}
.sub{{color:var(--muted);font-size:13px;margin:10px 0 18px;line-height:1.7}}
.legend{{display:flex;gap:14px;margin:12px 0;color:var(--muted);font-size:12px;flex-wrap:wrap}}
.lg{{display:inline-flex;align-items:center;gap:5px}}.dot{{width:11px;height:11px;border-radius:2px;display:inline-block}}
table{{width:100%;border-collapse:collapse;table-layout:fixed}}
th{{color:var(--muted);font-size:12px;font-weight:500;text-align:left;padding:8px 10px;border-bottom:1px solid var(--line)}}
td{{padding:10px;border-bottom:1px solid #1d2130;font-size:13px}}
tr:hover td{{background:#151926}}
.nm{{color:var(--ink);font-weight:700;font-size:14px}}.code{{color:var(--muted);font-size:11px;margin-left:4px}}
.ind{{color:var(--muted);font-size:11px;background:#1a1e28;padding:1px 6px;border-radius:3px;margin-left:6px}}
.s{{display:inline-block;width:26px;height:26px;line-height:26px;text-align:center;border-radius:4px;margin-right:5px;font-size:13px;font-weight:700;color:#0b0c10}}
.s.阳{{background:var(--up)}}.s.衡{{background:#c0a66a}}.s.阴{{background:var(--dn)}}
.mode{{font-size:12px;padding:3px 8px;border-radius:3px}}
.mode.strong{{background:#2a2a10;color:var(--up)}}
.mode.weak{{background:#1a1e28;color:var(--dn)}}
.mode.up{{background:#1a2010;color:#c9d09a}}
.mode.down{{background:#161a26;color:#9aa7c0}}
.mode.rebound{{background:#241a10;color:var(--up)}}
.mode.pullback{{background:#101a24;color:#9ac0d0}}
.mode.mixed{{background:#1a1a1a;color:var(--muted)}}
.rev,.ten{{text-align:center}}.ten{{color:var(--gold);font-weight:700}}
.c1{{color:#d99a5e}}.c0{{color:var(--muted)}}
.tenbar{{display:inline-block;height:6px;background:var(--gold);border-radius:3px;vertical-align:middle}}
.summary{{display:flex;gap:18px;flex-wrap:wrap;margin:16px 0;font-size:13px}}
.sum{{background:var(--s2);border:1px solid var(--line);border-radius:6px;padding:10px 16px;min-width:130px}}
.sum b{{color:var(--gold);font-size:20px;display:block}}
.sum span{{color:var(--muted);font-size:11px}}
.foot{{color:var(--muted);font-size:11px;margin-top:20px;border-top:1px solid var(--line);padding-top:10px}}
</style></head><body>
<h1>全市场周期传导矩阵 · 时间认知链</h1>
<div class="sub">13 只股票 × <b style="color:var(--gold)">日(动能)→周(波段)→月(趋势)→年(根性)</b> 周期链。日层阈值 ±1%(单日波动尺度小)，周/月/年层 ±2%。快慢(日vs年)背离=转折张力；层间方向反转次数+背离=张力指数(0-10)。数据为真实行情(2026-10-16收盘)。</div>
<div class="legend">
  <span class="lg"><span class="dot" style="background:var(--up)"></span>阳·生发</span>
  <span class="lg"><span class="dot" style="background:#c0a66a"></span>衡·稳态</span>
  <span class="lg"><span class="dot" style="background:var(--dn)"></span>阴·收敛</span>
</div>
<table>
  <thead><tr><th style="width:22%">股票</th><th style="width:32%">周期链 日→周→月→年</th>
  <th style="width:14%">周期形态</th><th style="width:9%">反转</th><th style="width:9%">背离</th><th style="width:14%">张力指数</th></tr></thead>
  <tbody>{trs}</tbody>
</table>
<div class="foot">研究辅助，非投资建议 · 三元九维认知引擎 TNE-Core · 周期层数据复用 data/finance (同花顺/Wind 2026-10-16)</div>
</body></html>"""
    open("demo/全市场周期传导矩阵.html", "w", encoding="utf-8").write(html)
    print("矩阵已生成: demo/全市场周期传导矩阵.html")
    for r in rows:
        d, c, a = r["d"], r["chain"], r["a"]
        print(f"  {d['name']:<6}{''.join(S[x['s']] for x in c)}  张力{a['tension']:>2}  {a['mode']}")

if __name__ == "__main__":
    rows = []
    for item in load_all():
        a = analyze(item)
        rows.append({"d": item["d"], "chain": item["chain"], "a": a})
    build_html(rows)
