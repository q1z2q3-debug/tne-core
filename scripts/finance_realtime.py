#!/usr/bin/env python3
"""
TNE-Core · 九维金融 · 实时接入工具
输入任意A股代码 → 输出"数据采集清单"(需拉取的字段) + 校验现有数据json完整性。
流程: 1)本工具生成模板/清单 → 2)采集实时数据(行情/趋势/财务) → 3)填入data/finance/<名>_<代码>.json → 4)finance_mapper.py生成报告
支持: --new <代码> 生成空白数据模板; --check <代码> 校验已有json是否可出报告
"""
import sys, json, glob, os, re

REQUIRED = {
    "quote": ["close", "chg_pct", "turnover_pct", "pe_ttm", "pb", "mktcap_yi"],
    "trend": ["month_pct", "two_week_pct", "year_pct", "year_high", "year_low"],
    "finance": ["net_margin_pct", "roe_pct", "net_parent_yi", "revenue_yi"],
}
SAMPLE = {
    "name": "示例", "code": "600519.SH", "date": "2026-10-16 收盘", "industry": "行业",
    "quote": {"close": 1329.0, "chg_pct": 1.22, "turnover_pct": 0.31, "pe_ttm": 20.4,
              "pb": 6.61, "mktcap_yi": 16613},
    "trend": {"month_pct": 5.59, "two_week_pct": 5.8, "year_pct": -3.5, "year_high": 1568, "year_low": 1151},
    "finance": {"net_margin_pct": 49.9, "roe_pct": 17.72, "net_parent_yi": 445.2, "revenue_yi": 907.0},
    "sector": "板块定位描述",
}


def check(path):
    d = json.load(open(path, encoding="utf-8"))
    missing = []
    for section, fields in REQUIRED.items():
        for f in fields:
            if f not in d.get(section, {}):
                missing.append(f"{section}.{f}")
    name = d.get("name", "?")
    code = d.get("code", "?")
    if missing:
        print(f"[{name} {code}] ✗ 缺字段, 无法出报告: {', '.join(missing)}")
        print("  采集清单:")
        for f in missing:
            print(f"    - {f}")
        return False
    print(f"[{name} {code}] ✓ 字段完整, 可立即生成报告")
    return True


def new_template(code):
    path = f"data/finance/模板_{code}.json"
    tpl = dict(SAMPLE, code=code, name=f"待填-{code}")
    json.dump(tpl, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"空白模板已生成: {path}")
    print("需采集字段清单(实时数据来源):")
    for section, fields in REQUIRED.items():
        print(f"  {section}:")
        for f in fields:
            print(f"    - {f}")
    print("  industry: 行业 | sector: 板块定位/龙头描述")


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return
    cmd, code = args[0], args[1] if len(args) > 1 else None
    if cmd == "--new" and code:
        new_template(code)
    elif cmd == "--check" and code:
        cands = glob.glob(f"data/finance/*{code}*.json") or glob.glob("data/finance/*.json")
        hit = None
        for c in cands:
            if code in c or (hit is None and c.endswith(".json")):
                hit = c
        check(hit) if hit else print("未找到对应数据文件")
    else:
        # 校验全部
        print("=== 九维金融 · 数据完整性校验(全部) ===")
        for f in sorted(glob.glob("data/finance/*.json")):
            check(f)


if __name__ == "__main__":
    main()
