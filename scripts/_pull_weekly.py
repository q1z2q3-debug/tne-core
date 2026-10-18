#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""一次性: 扩池到~80只, 分批拉周K hfq 缓存(腾讯限流→分批+重试)。"""
import os, sys, json, time, urllib.request
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

CACHE = os.path.join(os.path.dirname(__file__), "..", "tmp", "kline")
UA = {"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                     "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"),
      "Referer": "https://gu.qq.com/"}

# 80只 · 覆盖 白酒/银行/保险/券商/新能源/光伏/锂电/医药/地产/煤炭/有色/家电/计算机/军工/电力/基建/通信/半导体/食品/钢铁/机械
POOL80 = [
    # 白酒
    "600519", "000858", "000568", "000596", "600809", "603369",
    # 银行
    "600036", "601166", "000001", "002142", "600919", "601398",
    # 保险
    "601318", "601601", "601336", "601628",
    # 券商
    "600030", "601688", "600999", "300059", "601377",
    # 新能源车
    "300750", "002594", "601633", "000625", "600104",
    # 光伏
    "601012", "600438", "300274",
    # 锂电材料
    "300014", "002466", "002460", "603799",
    # 医药
    "600276", "603259", "300760", "300015", "600436", "300122", "600196",
    # 地产
    "000002", "600048", "001979",
    # 煤炭
    "601088", "601225", "600188",
    # 有色
    "601899", "600111", "603993", "600547",
    # 家电
    "000333", "000651", "600690", "600060",
    # 计算机/软件
    "002415", "002230", "600588", "600570",
    # 军工
    "600893", "600760", "002179",
    # 电力
    "600900", "601985", "600406", "600886",
    # 基建
    "601668", "601390", "601186", "601800",
    # 通信
    "600050", "000063", "600941",
    # 半导体
    "688981", "603501", "002371",
    # 食品
    "600887", "603288", "000895",
    # 机械/钢铁/其他
    "600031", "000338", "600019", "000725", "002475", "002352",
]

def mkt(c):
    return "sh" if c.startswith(("60", "68")) else "sz"


def fetch_weekly(code):
    path = os.path.join(CACHE, f"{code}_week.json")
    if os.path.exists(path):
        return True
    os.makedirs(CACHE, exist_ok=True)
    url = (f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?"
           f"param={mkt(code)}{code},week,2015-01-01,2026-12-31,640,hfq")
    last = None
    for i in range(4):
        try:
            d = json.loads(urllib.request.urlopen(
                urllib.request.Request(url, headers=UA), timeout=20).read().decode("utf-8"))
            node = d["data"][f"{mkt(code)}{code}"]
            kl = node.get("hfqweek") or node.get("week") or []
            rows = [[k[0], float(k[2])] for k in kl]
            rows = [r for r in rows if r[1] > 0]
            if len(rows) < 100:
                print(f"  {code}: 数据过少({len(rows)}) 跳过")
                return False
            df = {"date": [r[0] for r in rows], "close": [r[1] for r in rows]}
            json.dump(df, open(path, "w", encoding="utf-8"))
            print(f"  {code}: ok {len(rows)} 周 {rows[0][0]}~{rows[-1][0]}")
            return True
        except Exception as e:
            last = e
            time.sleep(1.0 * (i + 1))
    print(f"  {code}: FAIL {type(last).__name__} {last}")
    return False


if __name__ == "__main__":
    batch = int(sys.argv[1]) if len(sys.argv) > 1 else 0  # 0=全部
    codes = POOL80
    if batch > 0:
        n = 26
        codes = POOL80[(batch - 1) * n:(batch - 1) * n + n]
    ok = fail = 0
    for c in codes:
        if fetch_weekly(c):
            ok += 1
        else:
            fail += 1
        time.sleep(0.3)
    print(f"batch{batch}: ok={ok} fail={fail}")
