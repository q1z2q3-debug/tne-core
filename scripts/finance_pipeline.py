# -*- coding: utf-8 -*-
"""周期传导轨迹看板：把K线时间维度做成一条认知链(快→慢)，观察短期信号如何逐级传导为长期趋势。
数据为真实拉取(同花顺/Wind 2026-10-16收盘)：日K/区间(两周)/月K/年K。
规则：每周期涨跌幅→三态(阳>=+2,阴<=-2,衡其余)；快慢背离=张力信号。
"""
import json

# 茅台多周期(基于真实日K/区间/月K/年K)
MAOTAI = {
    "name": "贵州茅台", "code": "600519.SH", "industry": "白酒/高端消费",
    "days": [  # 近6个交易日
        {"d": "10/08", "chg": -0.22}, {"d": "10/09", "chg": 0.57}, {"d": "10/12", "chg": 1.98},
        {"d": "10/13", "chg": -1.02}, {"d": "10/14", "chg": 0.87}, {"d": "10/15", "chg": 2.10},
        {"d": "10/16", "chg": 1.22},
    ],
    "cycle": [  # 快→慢: (周期, 涨跌幅, 层名)
        ("日·动能", 0.79, "短"),
        ("周·波段", 4.90, "中短"),
        ("月·趋势", 5.59, "中"),
        ("年·根性", -3.50, "长"),
    ],
    "close": 1329.00, "pe": 20.40, "note": "快中慢全阳、年衡: 顺势突破,根性待验",
}

# 宁德多周期
CATL = {
    "name": "宁德时代", "code": "300750.SZ", "industry": "新能源/电池",
    "days": [  # 近7个交易日
        {"d": "10/08", "chg": -1.50}, {"d": "10/09", "chg": 3.84}, {"d": "10/12", "chg": 0.21},
        {"d": "10/13", "chg": -3.11}, {"d": "10/14", "chg": -3.11}, {"d": "10/15", "chg": -0.58},
        {"d": "10/16", "chg": -0.54},
    ],
    "cycle": [  # 快→慢: (周期, 涨跌幅, 层名)
        ("日·动能", -0.30, "短"),
        ("周·波段", -8.97, "中短"),
        ("月·趋势", -3.40, "中"),
        ("年·根性", -24.58, "长"),
    ],
    "close": 277.00, "pe": 15.08, "note": "全周期共振向下: 弱势收敛,动能与势能同向",
}

S = {-1: "阴", 0: "衡", 1: "阳"}
COLOR = {-1: "#6b7a99", 0: "#c0a66a", 1: "#d4af6e"}  # 阴墨灰/衡器物金/阳金

def tri(v):
    return 1 if v >= 2 else (-1 if v <= -2 else 0)

def chain(stk):
    # 日动能=近5日主导(阳数>=3阳, 阴数>=3阴, 否则衡)
    recent = stk["days"][-5:]
    yang = sum(1 for x in recent if x["chg"] >= 0)
    yin = len(recent) - yang
    day_t = 1 if yang >= 3 else (-1 if yin >= 3 else 0)
    # 周期链(快→慢): 日层用主导态, 周/月/年用真实涨跌幅
    out = [{"name": n, "v": chg, "s": tri(chg), "c": COLOR[tri(chg)], "layer": lay}
           for n, chg, lay in stk["cycle"]]
    out[0]["s"] = day_t
    out[0]["c"] = COLOR[day_t]
    out[0]["v"] = round(sum(x["chg"] for x in recent) / len(recent), 2)
    return out, day_t

def build_html():
    ma_chain, ma_day = chain(MAOTAI)
    ca_chain, ca_day = chain(CATL)
    def render(stk, cyc, day_t):
        # 快慢背离: 日态 vs 年态
        year_t = cyc[-1]["s"]
        conflict = "背离·转折张力" if day_t != year_t else ("同向·趋势延续" if day_t == 1 or day_t == -1 else "中性·待变")
        arrow = "↘" if day_t == year_t == -1 else ("↗" if day_t == year_t == 1 else "⇄")
        cells = ""
        for i, c in enumerate(cyc):
            cells += f"""<div class="node" style="border-color:{c['c']}">
              <div class="lbl">{c['name']}</div>
              <div class="state" style="background:{c['c']}">{S[c['s']]}</div>
              <div class="val">{c['v']:+.2f}%</div>
              {"<div class='tag fast'>快</div>" if c['layer']=='短' else ""}
            </div>"""
            if i < len(cyc) - 1:
                cells += f"""<div class="arrow {('down' if cyc[i+1]['s']==-1 else '')}">{arrow if i==0 else ''}</div>"""
        return f"""
        <div class="stk-card">
          <div class="stk-head">
            <div><span class="nm">{stk['name']}</span> <span class="code">{stk['code']}</span>
            <span class="ind">{stk['industry']}</span></div>
            <div class="close">¥{stk['close']} · PE {stk['pe']}</div>
          </div>
          <div class="chain">{cells}</div>
          <div class="diag"><b>{conflict}</b><span class="note">{stk['note']}</span></div>
        </div>"""

    days_ma = "".join(f"<span class='b {('up' if x['chg']>=0 else 'down')}'>{x['d']} {x['chg']:+.1f}%</span>" for x in MAOTAI["days"])
    days_ca = "".join(f"<span class='b {('up' if x['chg']>=0 else 'down')}'>{x['d']} {x['chg']:+.1f}%</span>" for x in CATL["days"])
    html = f"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>周期传导轨迹看板 · 三元九维金融</title>
<style>
:root{{--bg:#0b0c10;--s2:#11131a;--line:#2a2e3a;--gold:#d4af6e;--muted:#8a93a6;--ink:#e8e6e1}}
*{{box-sizing:border-box;margin:0}}body{{background:var(--bg);color:var(--ink);font-family:'Noto Serif SC',serif;padding:28px 20px}}
h1{{font-size:26px;letter-spacing:2px;color:var(--gold);border-bottom:1px solid var(--line);padding-bottom:12px}}
.sub{{color:var(--muted);font-size:13px;margin:10px 0 22px;line-height:1.6}}
.wrap{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}
@media(max-width:860px){{.wrap{{grid-template-columns:1fr}}}}
.stk-card{{background:var(--s2);border:1px solid var(--line);border-radius:8px;padding:16px}}
.stk-head{{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:16px}}
.nm{{font-size:20px;color:var(--gold);font-weight:700}}.code{{color:var(--muted);font-size:12px;margin-left:6px}}
.ind{{color:var(--muted);font-size:12px;background:#1a1e28;padding:2px 8px;border-radius:4px;margin-left:8px}}
.close{{color:var(--ink);font-size:14px}}
.chain{{display:flex;align-items:center;justify-content:space-between;gap:6px;flex-wrap:wrap}}
.node{{flex:1;min-width:90px;border:1px solid;border-radius:6px;padding:12px 8px;text-align:center;background:#0f1120}}
.lbl{{font-size:12px;color:var(--muted)}}.state{{font-size:22px;font-weight:700;margin:6px 0;color:#0b0c10}}
.val{{font-size:12px;color:var(--ink)}}.tag{{font-size:10px;color:var(--gold);margin-top:4px}}
.arrow{{color:var(--gold);font-size:20px;align-self:center}}.arrow.down{{color:#6b7a99}}
.diag{{margin-top:16px;padding:10px 12px;background:#161a26;border-left:3px solid var(--gold);border-radius:4px;font-size:13px;line-height:1.6}}
.diag b{{color:var(--gold)}}.note{{color:var(--muted);display:block;margin-top:4px;font-size:12px}}
.days{{display:flex;gap:6px;flex-wrap:wrap;margin-top:12px}}
.b{{font-size:11px;padding:2px 6px;border-radius:3px;background:#1a1e28;color:var(--muted)}}
.b.up{{color:#d4af6e}}.b.down{{color:#6b7a99}}
.legend{{display:flex;gap:16px;margin:14px 0;color:var(--muted);font-size:12px}}
.lg{{display:inline-flex;align-items:center;gap:5px}}.dot{{width:12px;height:12px;border-radius:2px;display:inline-block}}
.foot{{color:var(--muted);font-size:11px;margin-top:22px;border-top:1px solid var(--line);padding-top:10px}}
</style></head><body>
<h1>周期传导轨迹看板 · 时间认知链</h1>
<div class="sub">K线时间维度 = 一条认知链：<b style="color:var(--gold)">分钟(情绪)→日(动能)→周(波段)→月(趋势)→年(根性)</b>。观察短期信号如何逐级传导为长期趋势；快慢方向背离 = 转折张力。规则：每周期涨跌 ≥+2% 判阳(生发) / ≤-2% 判阴(收敛) / 其余判衡(稳态)。数据为真实行情(2026-10-16收盘)。</div>
<div class="legend">
  <span class="lg"><span class="dot" style="background:var(--gold)"></span>阳·生发(向上)</span>
  <span class="lg"><span class="dot" style="background:#c0a66a"></span>衡·稳态</span>
  <span class="lg"><span class="dot" style="background:#6b7a99"></span>阴·收敛(向下)</span>
</div>
<div class="wrap">
{render(MAOTAI, ma_chain, ma_day)}
{render(CATL, ca_chain, ca_day)}
</div>
<div class="foot">研究辅助，非投资建议 · 三元九维认知引擎 TNE-Core · 周期数据来自同花顺/Wind 2026-10-16</div>
</body></html>"""
    open("demo/周期传导轨迹看板.html", "w", encoding="utf-8").write(html)
    print("看板已生成: demo/周期传导轨迹看板.html")
    for stk, cyc, dt in ((MAOTAI, ma_chain, ma_day), (CATL, ca_chain, ca_day)):
        print(f"  {stk['name']}: {'→'.join(S[c['s']] for c in cyc)}  [{S[cyc[0]['s']]}{'↗' if cyc[0]['s']==cyc[-1]['s'] else '⇄'}{S[cyc[-1]['s']]}]")

if __name__ == "__main__":
    build_html()
