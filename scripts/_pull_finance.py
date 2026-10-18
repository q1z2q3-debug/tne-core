#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""一次性: 拉80只历史财务(多期季报) 缓存 data/finance_history/。分批+重试。"""
import os, sys, json, time, urllib.request
ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "data", "finance_history")
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0 Safari/537.36",
      "Referer": "https://emweb.securities.eastmoney.com/"}
POOL80 = ["600519","000858","000568","000596","600809","603369","600036","601166","000001","002142","600919","601398","601318","601601","601336","601628","600030","601688","600999","300059","601377","300750","002594","601633","000625","600104","601012","600438","300274","300014","002466","002460","603799","600276","603259","300760","300015","600436","300122","600196","000002","600048","001979","601088","601225","600188","601899","600111","603993","600547","000333","000651","600690","600060","002415","002230","600588","600570","600893","600760","002179","600900","601985","600406","600886","601668","601390","601186","601800","600050","000063","600941","688981","603501","002371","600887","603288","000895","600031","000338","600019","000725","002475","002352"]

def fetch_finance(code):
    path = os.path.join(OUT, f"{code}.json")
    if os.path.exists(path):
        return True
    os.makedirs(OUT, exist_ok=True)
    secu = code + (".SH" if code.startswith(("60","68")) else ".SZ")
    rows, page = [], 1
    for _ in range(4):
        url = ("https://datacenter.eastmoney.com/securities/api/data/get?"
               f"type=RPT_F10_FINANCE_MAINFINADATA&sty=APP_F10_MAINFINADATA"
               f"&filter=(SECUCODE%3D%22{secu}%22)&p={page}&ps=50")
        try:
            d = json.loads(urllib.request.urlopen(
                urllib.request.Request(url, headers=UA), timeout=20).read().decode("utf-8"))
            data = (d.get("result") or {}).get("data") or []
            rows.extend(data)
            if page >= ((d.get("result") or {}).get("pages") or 1) or not data:
                break
            page += 1
        except Exception as e:
            time.sleep(1.0); continue
    if len(rows) < 12:
        print(f"  {code}: 期数不足({len(rows)}) 跳过"); return False
    out = {}
    for r in rows:
        rd = (r.get("REPORT_DATE") or "")[:10]
        if rd:
            out[rd] = {"date": rd, "net_margin": r.get("XSMLL"), "roe": r.get("ROEJQ"),
                       "net_parent": r.get("PARENTNETPROFIT"), "revenue": r.get("TOTALOPERATEREVE")}
    json.dump(sorted(out.values(), key=lambda x: x["date"]),
              open(path, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"  {code}: ok {len(out)} 期 {min(out)}~{max(out)}"); return True

if __name__ == "__main__":
    batch = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    codes = POOL80
    if batch > 0:
        n = 27; codes = POOL80[(batch-1)*n:(batch-1)*n+n]
    ok = fail = 0
    for c in codes:
        if fetch_finance(c): ok += 1
        else: fail += 1
        time.sleep(0.3)
    print(f"batch{batch}: ok={ok} fail={fail}")
