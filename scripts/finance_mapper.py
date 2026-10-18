#!/usr/bin/env python3
"""
TNE-Core · 九维金融认知映射器
把个股真实数据(行情/趋势/财务)按显式规则映射为九维三态向量, 复用TNE引擎做认知定位。
映射规则(可复现, 非拍脑袋): 每维设定阈值 → 阳/衡/阴。
输出: 诊断JSON + 自包含HTML认知报告。
"""
import sys, json, glob, argparse, os
sys.path.insert(0, "/home/user/Doubao/chats/38446456964142338/tne-core/src")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from tne.space.space import encode
from tne.space.equivalence import class_of
from tne.space.distance import nearest_neighbors
from tne.space.operators import f_cuoo, f_zong, f_cuoo_zong, f_hu, f_bian

# ---- 九维金融语义定义 ----
# g1 愿景=行业趋势 g2 决断=资金主攻 g3 产出=盈利能力 g4 方法=经营效率 g5 主态=价格主态
# g6 协作=市场协同 g7 输入=估值水平 g8 积累=财务积累 g9 根性=资产质量/护城河
SEM = ["愿景·行业趋势", "决断·资金主攻", "产出·盈利能力", "方法·经营效率",
       "主态·价格主态", "协作·市场协同", "输入·估值水平", "积累·财务积累", "根性·护城河"]
SYM = {1: "阳·扩张", 0: "衡·盘整", -1: "阴·收敛"}
SCODE = {1: "+", 0: "0", -1: "-"}


def band(v, up, down):
    """up=阳阈值(含), down=阴阈值(含): v>=up→+1, v<=down→-1, else 0"""
    if v >= up:
        return 1
    if v <= down:
        return -1
    return 0


def map_nine(d):
    q, t, f = d["quote"], d["trend"], d["finance"]
    g = [0] * 9
    # g1 愿景=行业/月趋势
    g[0] = band(t["month_pct"], 2.0, -2.0)
    # g2 决断=资金主攻(当日涨幅+量能)
    g[1] = band(q["chg_pct"] * (1 + q["turnover_pct"]), 1.0, -1.0) if q["chg_pct"] > 0 else -1
    # g3 产出=净利率
    g[2] = band(f["net_margin_pct"], 30, 10)
    # g4 方法=ROE
    g[3] = band(f["roe_pct"], 15, 5)
    # g5 主态=两周价格主态
    g[4] = band(t["two_week_pct"], 2.0, -2.0)
    # g6 协作=市场协同(龙头领涨=阳)
    g[5] = 1 if "领涨" in d["sector"] else (0 if "板块" in d["sector"] else -1)
    # g7 输入=估值(PE相对板块: 白酒<15低=阳, >30高=阴)
    g[6] = band(30 - q["pe_ttm"], 15, -15)  # pe<=15→阳, >=45→阴... 用30-pe, 阈值15/-15
    # g8 积累=EPS/净资产质量
    g[7] = band(f["eps"], 3.0, 0.5)
    # g9 根性=护城河(龙头=阳)
    g[8] = 1 if "龙头" in d["sector"] else (0 if "板块" in d["sector"] else -1)
    return g


def diagnose(g, d):
    idx = encode(g)
    rep = class_of(idx)
    nbs = [{"i": ni, "d": round(dist, 3), "v": "".join(SCODE[x] for x in v)}
           for ni, dist, v in nearest_neighbors(g, top_n=5)]
    tr = {"错卦Fc": f_cuoo(g), "综卦Fz": f_zong(g), "错综Fcz": f_cuoo_zong(g),
          "互卦Fh": f_hu(g), "动爻Fb(主态动)": f_bian(g, {5})}
    # 认知诊断: 习惯/张力由 九维画像 + 数据健康度 判定(可复现规则)
    pos = sum(1 for x in g if x == 1)
    neg = sum(1 for x in g if x == -1)
    if pos >= 6:
        mode, rec = "强势·生发", "顺趋势, 关注高位估值与放量追高风险(增长主导)"
    elif neg >= 6:
        mode, rec = "弱势·收敛", "防御为主, 等企稳信号, 勿盲目抄底(收缩主导)"
    elif pos >= 4:
        mode, rec = "稳健·蓄势", "基本面稳健, 估值中性, 可持有观察(平衡偏强)"
    else:
        mode, rec = "震荡·待变", "多空未决, 降低仓位, 等待方向(势能平衡点)"
    tensions, signals = [], []
    if neg >= 4 or d["quote"]["pe_ttm"] > 40:
        tensions.append({"type": "boundary", "strength": 0.7, "note": "估值偏高或弱势维度多, 存在回调风险"})
        signals.append({"type": "deliberation", "strength": 0.7,
                        "text": "当前画像存在风险维度, 需审慎权衡, 勿追高, 设好止损"})
    if pos >= 2 and neg >= 2:
        tensions.append({"type": "habit_conflict", "strength": 0.5, "note": "强势与弱势维度并存, 趋势可能切换"})
        signals.append({"type": "meta_reasoning", "strength": 0.5,
                        "text": "多维信号分化, 用系统方法复核, 不凭单一指标下结论"})
    if not tensions:
        signals.append({"type": "curiosity", "strength": 0.4, "text": "画像一致, 维持当前认知框架, 持续跟踪"})
    return {
        "name": d["name"], "code": d["code"], "date": d["date"],
        "vector": g, "code_str": "".join(SCODE[x] for x in g),
        "gua_idx": idx, "class_rep": rep,
        "yao_sem": [f"{SEM[i]}={SYM[g[i]]}" for i in range(9)],
        "quote": d["quote"], "trend": d["trend"], "finance": d["finance"],
        "transform": {k: "".join(SCODE[x] for x in v) for k, v in tr.items()},
        "neighbors": nbs, "mode": mode, "rec": rec,
        "tensions": tensions, "signals": signals, "sector": d["sector"],
    }


def build_html(r):
    cells = "".join(
        f"""<div class="cell {'yang' if r['vector'][i]==1 else 'yin' if r['vector'][i]==-1 else ''}">
          <span class="yid">g{i+1}</span>
          <span class="sym">{'生' if r['vector'][i]==1 else '衡' if r['vector'][i]==0 else '敛'}</span>
          <span class="name">{SEM[i]}</span></div>""" for i in range(9))
    conf_rows = "".join(
        f"""<div class="row"><span class="clab">g{i+1} {SEM[i]}</span>
          <div class="bar"><div class="fill" style="width:{'90' if r['vector'][i]==1 else '50' if r['vector'][i]==0 else '25'}%"></div></div>
          <span class="cval">{'阳·扩张' if r['vector'][i]==1 else '衡·盘整' if r['vector'][i]==0 else '阴·收敛'}</span></div>"""
        for i in range(9))
    nbs = "".join(f"<span class='nb'>#{n['i']} <em>d={n['d']}</em> {n['v']}</span>" for n in r["neighbors"])
    tr = "".join(f"<div class='tro'><span class='tlab'>{k}</span><span class='tcode'>{v}</span></div>"
                 for k, v in r["transform"].items())
    tens = "".join(f"<span class='t'>{t['type']}·{t['strength']:.1f}</span>" for t in r["tensions"]) or "<span class='t ok'>稳定·无张力</span>"
    sigs = "".join(f"<div class='sig'><b>[{s['type']}]</b> {s['text']}</div>" for s in r["signals"])
    return f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{r['name']} · 九维金融认知报告 · TNE-Core</title>
<style>
:root{{--ink:#d4af6e;--ink2:#e0b96a;--muted:#8a8f98;--bg:#0b0c10;--s2:#14161c;--line:#2a2d35;--yang:#e0b96a;--yin:#6b87b8;--heng:#6d717a}}
*{{margin:0;box-sizing:border-box}}body{{background:var(--bg);color:#c9ccd2;font-family:'Noto Serif SC','Songti SC',serif;padding:24px 16px;line-height:1.6}}
.wrap{{max-width:1080px;margin:0 auto}}
.top{{border-bottom:1px solid var(--line);padding-bottom:14px;margin-bottom:20px}}
.brand{{font-size:11px;letter-spacing:.3em;color:var(--ink);margin-bottom:4px}}
h1{{font-size:26px;color:var(--ink);font-weight:600}}
.sub{{color:var(--muted);font-size:13px;margin-top:4px}}
.hero{{display:flex;gap:20px;flex-wrap:wrap;align-items:center;margin:18px 0;padding:16px;background:var(--s2);border:1px solid var(--line);border-radius:6px}}
.quote b{{font-size:30px;color:var(--ink)}} .quote .chg{{color:#7fdc8a}} .tag{{font-size:12px;color:var(--ink2)}}
.kpi{{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin:14px 0;flex:1;min-width:280px}}
.kpi div{{background:var(--s2);border:1px solid var(--line);padding:8px 10px;border-radius:4px;font-size:11px}}
.kpi b{{display:block;font-size:17px;color:var(--ink2);margin-top:2px}}
.grid9{{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin:14px 0}}
.cell{{aspect-ratio:1;display:flex;flex-direction:column;justify-content:center;align-items:center;background:var(--s2);border:1px solid var(--line);border-radius:5px;position:relative}}
.cell .sym{{font-size:26px}} .cell .name{{font-size:10px;color:var(--muted);margin-top:5px}} .cell .yid{{position:absolute;top:5px;left:8px;font-size:9px;color:var(--ink2)}}
.cell.yang{{border-color:rgba(224,185,106,.5)}} .cell.yang .sym{{color:var(--yang)}}
.cell.yin{{border-color:rgba(107,135,184,.5)}} .cell.yin .sym{{color:var(--yin)}}
.cell:not(.yang):not(.yin) .sym{{color:var(--heng)}}
h2{{font-size:16px;color:var(--ink);margin:22px 0 8px;letter-spacing:.1em}}
.panel{{background:var(--s2);border:1px solid var(--line);border-radius:6px;padding:16px;margin:10px 0}}
.row{{display:flex;align-items:center;gap:10px;font-size:12px;padding:3px 0}} .clab{{flex:0 0 180px;color:var(--muted)}}
.bar{{flex:1;height:8px;background:#1c1f28;border-radius:4px;overflow:hidden}} .fill{{height:100%;background:var(--ink2)}} .cval{{flex:0 0 80px;text-align:right;color:var(--ink2)}}
.nbs{{display:flex;flex-wrap:wrap;gap:8px}} .nb{{background:#1c1f28;border:1px solid var(--line);padding:6px 10px;border-radius:4px;font-size:11px;font-family:monospace}} .nb em{{color:var(--ink2);font-style:normal;margin:0 4px}}
.tro{{display:flex;justify-content:space-between;padding:4px 0;font-size:12px;border-bottom:1px solid #1c1f28}} .tlab{{color:var(--muted)}} .tcode{{font-family:monospace;letter-spacing:.15em;color:var(--ink2)}}
.t{{background:#1c1f28;border:1px solid rgba(224,185,106,.3);padding:3px 8px;border-radius:3px;font-size:11px;margin-right:6px;color:var(--ink2)}}.t.ok{{border-color:rgba(127,220,138,.3);color:#7fdc8a}}
.sig{{font-size:12px;padding:6px 0;color:#c9ccd2}} .sig b{{color:var(--ink2);margin-right:4px}}
.mode{{display:inline-block;background:rgba(224,185,106,.12);border:1px solid rgba(224,185,106,.35);color:var(--ink2);padding:4px 12px;border-radius:4px;font-size:14px;margin-top:4px}}
.note{{font-size:11px;color:var(--muted);margin-top:14px;border-top:1px solid var(--line);padding-top:10px}}
@media(max-width:760px){{.kpi{{grid-template-columns:repeat(2,1fr)}}.clab{{flex:0 0 120px}}h1{{font-size:22px}}}}
</style></head><body><div class="wrap">
<div class="top"><div class="brand">TRINARY NINE-DIMENSIONAL FINANCIAL COGNITION · TNE-CORE</div>
<h1>{r['name']} · 九维金融认知报告</h1><div class="sub">自然语言 → 九维三态金融画像 → 卦位定位 → 认知诊断 · 数据截至 {r['date']}</div></div>

<div class="hero"><div class="quote"><div class="tag">{r['code']} · {r['sector']}</div>
<b>{r['quote']['close']:.2f}</b> <span class="chg">+{r['quote']['chg_pct']}%</span></div>
<div class="kpi">
<div>市盈率TTM<b>{r['quote']['pe_ttm']:.1f}</b></div><div>市净率<b>{r['quote']['pb']:.2f}</b></div>
<div>总市值<b>{r['quote']['mktcap_yi']/10000:.2f}万亿</b></div><div>月涨跌幅<b>{r['trend']['month_pct']:.1f}%</b></div></div>
<div style="margin-left:auto;font-size:12px;color:var(--muted)">成交额 {r['quote']['amount_yi']:.0f}亿 · 换手 {r['quote']['turnover_pct']}%<br>ROE {r['finance']['roe_pct']:.1f}% · 净利率 {r['finance']['net_margin_pct']:.1f}%</div></div>

<div class="panel"><h2 style="margin-top:0">九维金融画像 · 卦位 #{r['gua_idx']} / 等价类 #{r['class_rep']}</h2>
<div class="code" style="font-family:monospace;font-size:15px;letter-spacing:.3em;color:var(--ink2)">{r['code_str']}</div>
<div class="grid9">{cells}</div>
<div class="mode">认知主态：{r['mode']}</div></div>

<div class="panel"><h2 style="margin-top:0">爻位金融语义</h2>{conf_rows}</div>

<div class="panel"><h2 style="margin-top:0">结构变换推演</h2>{tr}</div>

<div class="panel"><h2 style="margin-top:0">邻域泛化 · Top-5 邻近卦位</h2><div class="nbs">{nbs}</div></div>

<div class="panel"><h2 style="margin-top:0">认知诊断 · 习惯与张力</h2>
<div style="font-size:13px;color:var(--ink2);margin-bottom:6px">{r['rec']}</div>
<div style="margin-bottom:8px">{tens}</div>{sigs}</div>

<div class="note">本报告为三元九维认知框架对公开行情/财务数据的结构化分析, 属研究辅助工具, 不构成任何投资建议。数据来源：Wind/同花顺公开行情与财报。映射规则为显式阈值判定, 可复现。</div>
</div></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/finance/贵州茅台_600519.json")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    d = json.load(open(args.data, encoding="utf-8"))
    g = map_nine(d)
    r = diagnose(g, d)
    out = args.out or f"demo/{d['name']}九维金融认知报告.html"
    open(out, "w", encoding="utf-8").write(build_html(r))
    print(f"九维定位: {r['code_str']}  卦位#{r['gua_idx']}  等价类#{r['class_rep']}")
    print(f"认知主态: {r['mode']}  |  变换: " + "  ".join(f"{k}={v}" for k, v in r['transform'].items()))
    print(f"报告已生成: {out}")


if __name__ == "__main__":
    main()
