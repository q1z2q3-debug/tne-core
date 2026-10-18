#!/usr/bin/env python3
"""
TNE-Core · 九维金融 · 多股票横向对比页生成
4+ 只股票 → 九维画像横向对比(卦位/认知主态/逐维三态/关键指标)。
"""
import sys, json, glob, os
sys.path.insert(0, "/home/user/Doubao/chats/38446456964142338/tne-core/src")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from finance_mapper import map_nine, diagnose, SEM, SYM, SCODE

SCODE_REV = {"+": 1, "0": 0, "-": -1}


def build_compare_html(stocks, results):
    # 表头 + 每只股票行
    rows = []
    for r in results:
        cells = "".join(
            f"""<td class="{('yang' if r['vector'][i]==1 else 'yin' if r['vector'][i]==-1 else 'heng')}">{'生' if r['vector'][i]==1 else '衡' if r['vector'][i]==0 else '敛'}</td>"""
            for i in range(9))
        pos, neg = sum(1 for x in r['vector'] if x == 1), sum(1 for x in r['vector'] if x == -1)
        rows.append(f"""<tr>
<td class="sn"><div class="sn-name">{r['name']}</div><div class="sn-sub">{r['code']} · {r['industry']}</div></td>
<td>{r['code_str']}</td>
<td>#{r['gua_idx']}<div class="sub">类{r['class_rep']}</div></td>
<td class="mode">{r['mode']}</td>
<td class="pcount"><b>{pos}阳</b> / <b>{neg}阴</b></td>
{cells}</tr>""")
    # 卦位分布 + 主态
    mode_counts = {}
    for r in results:
        mode_counts[r['mode']] = mode_counts.get(r['mode'], 0) + 1
    mode_summary = " · ".join(f"{k}×{v}" for k, v in mode_counts.items())
    legend = "".join(f"<span class='lg lg-{c}'></span>{lab}"
                     for c, lab in [("yang", "阳·扩张"), ("heng", "衡·盘整"), ("yin", "阴·收敛")])
    return f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>九维金融画像 · 多股票横向对比 · TNE-Core</title><style>
:root{{--ink:#d4af6e;--ink2:#e0b96a;--muted:#8a8f98;--bg:#0b0c10;--s2:#14161c;--line:#2a2d35;--yang:#e0b96a;--yin:#6b87b8;--heng:#6d717a}}
*{{margin:0;box-sizing:border-box}}body{{background:var(--bg);color:#c9ccd2;font-family:'Noto Serif SC','Songti SC',serif;padding:24px 16px;line-height:1.5}}
.wrap{{max-width:1280px;margin:0 auto}}.top{{border-bottom:1px solid var(--line);padding-bottom:14px;margin-bottom:16px}}
.brand{{font-size:11px;letter-spacing:.3em;color:var(--ink);margin-bottom:4px}}h1{{font-size:25px;color:var(--ink);font-weight:600}}
.sub{{color:var(--muted);font-size:13px;margin-top:4px}}
.mode-sum{{font-size:13px;color:var(--ink2);margin:14px 0 8px;letter-spacing:.05em}}
table{{width:100%;border-collapse:collapse;background:var(--s2);border:1px solid var(--line);border-radius:6px;overflow:hidden;font-size:12px}}
th,td{{padding:9px 8px;border-bottom:1px solid #1c1f28;text-align:center}}
th{{color:var(--ink2);font-weight:600;letter-spacing:.1em;font-size:11px}}
td.sn{{text-align:left}} .sn-name{{color:var(--ink);font-weight:600}} .sn-sub{{color:var(--muted);font-size:10px;margin-top:2px}}
.sub{{color:var(--muted);font-size:10px}} td.mode{{color:var(--ink2);font-weight:600}}
.pcount b{{color:var(--ink2)}}.pcount b:nth-child(3){{color:var(--yin)}}
td.yang{{color:var(--yang);font-weight:700;font-size:16px}} td.yin{{color:var(--yin);font-weight:700;font-size:16px}} td.heng{{color:var(--heng);font-weight:600;font-size:16px}}
.legend{{margin-top:12px;font-size:11px;color:var(--muted);display:flex;gap:16px;align-items:center}}
.lg{{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px}}.lg-yang{{background:var(--yang)}}.lg-heng{{background:var(--heng)}}.lg-yin{{background:var(--yin)}}
.note{{font-size:11px;color:var(--muted);margin-top:16px;border-top:1px solid var(--line);padding-top:10px}}
@media(max-width:760px){{.wrap{{overflow-x:auto}}table{{min-width:860px}}}}
</style></head><body><div class="wrap">
<div class="top"><div class="brand">TRINARY NINE-DIMENSIONAL FINANCIAL COGNITION · MULTI-STOCK</div>
<h1>九维金融画像 · 多股票横向对比</h1><div class="sub">不同行业龙头 · 九维三态画像 + 卦位定位 + 认知主态 · 数据截至 2026-10-16 收盘（Wind/同花顺）</div></div>
<div class="mode-sum">认知主态分布：{mode_summary}</div>
<table><tr><th>股票</th><th>九维码</th><th>卦位</th><th>认知主态</th><th>阳/阴</th>
<th>愿景</th><th>决断</th><th>产出</th><th>方法</th><th>主态</th><th>协作</th><th>输入</th><th>积累</th><th>根性</th></tr>
{rows}</table>
<div class="legend">{legend}<span style="margin-left:auto">逐维阈值：g1月趋势 g2资金 g3净利率 g4ROE g5两周 g6换手 g7PE g8净利规模 g9年趋势</span></div>
<div class="note">九维金融认知为对公开行情/财务数据的结构化分析，属研究辅助工具，不构成投资建议。映射规则为显式阈值判定，可复现。数据来源：Wind/同花顺公开行情与财报。</div>
</div></body></html>"""


def main():
    files = sorted(glob.glob("data/finance/*.json"))
    results = []
    for f in files:
        d = json.load(open(f, encoding="utf-8"))
        g = map_nine(d)
        r = diagnose(g, d)
        r["industry"] = d["industry"]
        results.append(r)
    html = build_compare_html(files, results)
    out = "demo/九维金融多股票对比.html"
    open(out, "w", encoding="utf-8").write(html)
    print(f"对比页已生成: {out}  ({len(results)}只股票)")
    for r in results:
        print(f"  {r['name']:<6} {r['code_str']}  卦位#{r['gua_idx']}  {r['mode']}")


if __name__ == "__main__":
    main()
