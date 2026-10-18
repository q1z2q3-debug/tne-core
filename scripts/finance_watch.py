# -*- coding: utf-8 -*-
"""多层认知关注矩阵: 把已入库股票从三层认知空间(空间卦位/时间周期/张力信号)叠加,
输出"值得继续跟踪认知"的关注分级。研究辅助, 非投资建议/非荐股。
"""
import json, glob, sys, os
sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

from tne.space.space import encode
from tne.space.equivalence import class_of
import finance_mapper as fm

S = {-1: "阴", 0: "衡", 1: "阳"}
SCODE = {-1: "-", 0: "0", 1: "+"}
MODE_SCORE = {"强势·生发": 3, "稳健·蓄势": 2, "震荡·待变": 1, "弱势·收敛": 0}
CYC_MODE_SCORE = {"顺势·生发": 3, "回踩·背离": 3, "反弹·背离": 3, "震荡·待变": 1, "弱势·收敛": 0}


def cycle_analyze(d):
    q, t = d["quote"], d["trend"]
    def tri(v, thr):
        return 1 if v >= thr else (-1 if v <= -thr else 0)
    st = [tri(q["chg_pct"], 1.0), tri(t["two_week_pct"], 2.0),
          tri(t["month_pct"], 2.0), tri(t["year_pct"], 2.0)]
    day, week, month, year = st
    revs = sum(1 for i in range(3) if st[i] != 0 and st[i + 1] != 0 and st[i] != st[i + 1])
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
    return {"chain": "".join(SCODE[x] for x in st), "mode": mode, "cls": cls,
            "tension": tension, "conflict": conflict}


def watch_score(mode, cyc_mode, tensions, signals, conflict):
    s = MODE_SCORE[mode] + CYC_MODE_SCORE[cyc_mode]
    sig = {x["type"] for x in signals}
    ten = {x["type"] for x in tensions}
    if {"curiosity", "novelty"} & sig:
        s += 1.0
    if "habit_conflict" in ten:
        s += 0.5
    if "boundary" in ten:
        s -= 0.5  # 估值/风险偏高, 谨慎
    # 快慢背离=转折验证, 天然高关注
    if conflict:
        s += 0.5
    if s >= 5.5:
        lv, lc = "高关注·转折验证", "high"
    elif s >= 3.5:
        lv, lc = "中关注·趋势跟踪", "mid"
    else:
        lv, lc = "低关注·风险规避", "low"
    return s, lv, lc


def main():
    rows = []
    for f in sorted(glob.glob("data/finance/*.json")):
        d = json.load(open(f, encoding="utf-8"))
        g = fm.map_nine(d)
        r = fm.diagnose(g, d)
        cyc = cycle_analyze(d)
        s, lv, lc = watch_score(r["mode"], cyc["mode"], r["tensions"], r["signals"], cyc["conflict"])
        rows.append({"d": d, "gua": r["code_str"], "gua_idx": r["gua_idx"],
                     "space_mode": r["mode"], "cyc": cyc, "score": round(s, 1), "lv": lv, "lc": lc,
                     "sigs": "|".join(x["type"] for x in r["signals"]), "tens": "|".join(x["type"] for x in r["tensions"])})
    rows.sort(key=lambda x: -x["score"])
    build_html(rows)
    for r in rows:
        d = r["d"]
        print(f"  {d['name']:<6}{r['gua']} 空间{r['space_mode']} 周期{r['cyc']['chain']}{r['cyc']['mode']} 张力{r['cyc']['tension']} → 关注{r['score']} {r['lv']}")


def build_html(rows):
    cards = ""
    for r in rows:
        d = r["d"]
        conflict_tag = "<b class='d'>背离</b>" if r["cyc"]["conflict"] else "<b class='s'>同向</b>"
        cards += f"""<div class="row {r['lc']}">
  <div class="nm"><span class="n">{d['name']}</span><span class="c">{d['code'].split('.')[0]}</span>
      <span class="i">{d['industry']}</span></div>
  <div class="gua">{r['gua']}<span class="gi">#{r['gua_idx']}</span></div>
  <div class="sp">{r['space_mode']}</div>
  <div class="cyc"><span class="ch">{r['cyc']['chain']}</span>{r['cyc']['mode']}{conflict_tag}</div>
  <div class="sc"><span class="num">{r['score']}</span><span class="lv">{r['lv']}</span></div>
  <div class="tg">{r['sigs'].replace('|',' ')}</div>
</div>"""
    html = f"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>多层认知关注矩阵 · 三元九维金融</title>
<style>
:root{{--bg:#0b0c10;--s2:#11131a;--line:#2a2e3a;--gold:#d4af6e;--muted:#8a93a6;--ink:#e8e6e1}}
*{{box-sizing:border-box;margin:0}}body{{background:var(--bg);color:var(--ink);font-family:'Noto Serif SC',serif;padding:26px 18px}}
h1{{font-size:23px;letter-spacing:2px;color:var(--gold);border-bottom:1px solid var(--line);padding-bottom:12px}}
.sub{{color:var(--muted);font-size:13px;margin:10px 0 16px;line-height:1.7}}
.layers{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:14px 0}}
@media(max-width:760px){{.layers{{grid-template-columns:1fr}}}}
.lay{{background:var(--s2);border:1px solid var(--line);border-radius:6px;padding:10px 12px}}
.lay b{{color:var(--gold);font-size:14px;display:block}}.lay span{{color:var(--muted);font-size:11px;line-height:1.5;display:block;margin-top:4px}}
.head,.row{{display:grid;grid-template-columns:18% 14% 14% 22% 14% 18%;gap:8px;align-items:center;padding:10px 8px}}
.head{{color:var(--muted);font-size:12px;border-bottom:1px solid var(--line);font-weight:500}}
.row{{border-bottom:1px solid #1d2130;font-size:13px}}
.row:hover{{background:#151926}}
.high{{border-left:3px solid var(--gold)}}
.mid{{border-left:3px solid #7a8a5a}}
.low{{border-left:3px solid #3a4150}}
.n{{color:var(--ink);font-weight:700;font-size:14px}}.c{{color:var(--muted);font-size:11px;margin-left:4px}}
.i{{color:var(--muted);font-size:11px;background:#1a1e28;padding:1px 6px;border-radius:3px;margin-left:6px}}
.gua{{font-family:monospace;font-size:15px;color:var(--gold)}}.gi{{color:var(--muted);font-size:10px;margin-left:4px}}
.sp,.cyc{{font-size:12px;color:var(--muted)}}.ch{{color:var(--gold);font-weight:700;margin-right:5px;letter-spacing:1px}}
.d{{color:#d99a5e;font-size:11px;margin-left:6px;padding:1px 5px;border-radius:3px;background:#241a10}}
.s{{color:#9aa7c0;font-size:11px;margin-left:6px;padding:1px 5px;border-radius:3px;background:#161a26}}
.sc{{text-align:center}}.num{{color:var(--gold);font-size:20px;font-weight:700;margin-right:8px}}
.lv{{font-size:12px}}.high .lv{{color:var(--gold)}}.mid .lv{{color:#7a8a5a}}.low .lv{{color:var(--muted)}}
.tg{{color:var(--muted);font-size:11px;word-spacing:4px}}
.foot{{color:var(--muted);font-size:11px;margin-top:18px;border-top:1px solid var(--line);padding-top:10px}}
</style></head><body>
<h1>多层认知关注矩阵</h1>
<div class="sub">三层认知空间叠加 → 关注分级。<b style="color:var(--gold)">空间层</b>(九维卦位主态) + <b style="color:var(--gold)">时间层</b>(周期传导链+张力) + <b style="color:var(--gold)">张力层</b>(认知信号)。关注≠推荐买入，是"值得继续跟踪认知的方向"。数据真实行情(2026-10-16收盘)。</div>
<div class="layers">
  <div class="lay"><b>空间层 · 卦位</b><span>九维三态画像定位到19683卦/等价类, 判当前认知主态(生发/蓄势/待变/收敛)</span></div>
  <div class="lay"><b>时间层 · 周期链</b><span>日→周→月→年传导, 快慢背离=转折张力; 顺势=趋势确认</span></div>
  <div class="lay"><b>张力层 · 信号</b><span>curiosity/novelty=探索关注, habit_conflict=分化复核, boundary=风险谨慎</span></div>
</div>
<div class="head"><div>股票</div><div>卦位</div><div>空间主态</div><div>周期传导</div><div>关注度</div><div>认知信号</div></div>
{cards}
<div class="foot">研究辅助，非投资建议，不构成荐股 · 三元九维认知引擎 TNE-Core · 数据 2026-10-16 收盘</div>
</body></html>"""
    open("demo/多层认知关注矩阵.html", "w", encoding="utf-8").write(html)
    print("看板已生成: demo/多层认知关注矩阵.html")


if __name__ == "__main__":
    main()
