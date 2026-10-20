#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""变算子（Bian Operators）单元测试。逐条验证变之七律的形式化。"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from tne.space.space import gua_str
from tne.space.bian import (exhaustion, is_exhausted, exhaustion_positions,
                            tension, timing_weights, sensation, benefit,
                            contingent, f_v, EX_POS)

PASS = 0


def check(name, got, expect, tol=1e-6):
    global PASS
    ok = abs(got - expect) < tol if isinstance(got, (int, float)) else got == expect
    print(f"  {'✓' if ok else '✗'} {name}: 得 {got} | 期 {expect}")
    PASS += ok


print("[律3] 穷度检测器 Ex（穷则变）")
check("纯阳卦穷度", exhaustion([1,1,1,1,1,1,1,1,1])["total"], 1.0)
check("纯阴卦穷度", exhaustion([-1,-1,-1,-1,-1,-1,-1,-1,-1])["total"], 1.0)
check("全衡卦穷度", exhaustion([0,0,0,0,0,0,0,0,0])["total"], 0.0)
check("半阴阳穷度", exhaustion([1,1,1,-1,-1,-1,0,0,0])["total"], round(0.4*0+0.4*(6/9)+0.2*(3/9),4))
check("纯阳触发", is_exhausted([1,1,1,1,1,1,1,1,1]), True)
check("全衡不触发", is_exhausted([0,0,0,0,0,0,0,0,0]), False)
check("穷点(纯阳两端)", exhaustion_positions([1,1,1,1,1,1,1,1,1]), [1,9])
check("穷点(中段连排)", exhaustion_positions([1,0,1,1,1,0,1,0,1]), [3,5])

print("[律1] 推荡（刚柔相推）")
t = tension([1,1,-1,-1,1,0,-1,1,-1])
check("阳数", t["yang"], 4)
check("阴数", t["yin"], 4)
check("对张", t["conflict"], 4)
check("推荡边界", t["boundaries"], [2,4,7,8])
check("偏张(全阳)", tension([1]*9)["divergence"], 1.0)

print("[律4] 时中（趣时）")
w = timing_weights()
check("中位权重最低", w[5], 1.0)
check("上爻权重最高", w[1], 2.0)
check("下爻权重最高", w[9], 2.0)

print("[律5] 感（屈信相感）")
check("同卦距离0", sensation([1,0,-1]*3, [1,0,-1]*3), 0.0)
check("反卦距离", sensation([1]*9, [-1]*9), 6.0)

print("[律6] 利评估（变动以利言）")
G = [1,1,1,1,1,1,1,1,1]
b1 = benefit(G, [1,1,1,1,0,1,1,1,1])  # 动中爻(5) → 趋衡
check("变则通(中爻动→平衡增益>0)", b1["balance_gain"] > 0, True)
b2 = benefit(G, [1,1,1,1,1,1,1,1,1], [1,0,-1]*3)  # 不变
check("不变距离0", b2["target_gain"], 0.0)

print("[律7] 权变（唯变所适）")
cands = contingent(G, [5, 1])  # 动中爻(稳) / 上爻(激)
check("候选非空且通", len(cands) >= 1, True)
check("只留通/利", all(c[2]["benefit"] >= 0 for c in cands), True)

print("[整合] 变易流程 F_V（穷→变→通→久）")
r = f_v(G, target=[1,0,-1]*3)
check("穷则触发", r["triggered"], True)
check("穷度满", r["exhaustion"]["total"], 1.0)
check("候选集(权变)", len(r["candidates"]) > 0, True)
r2 = f_v([0,0,0,0,0,0,0,0,0])
check("未穷不触发(通则久)", r2["triggered"], False)

print(f"\n== 通过 {PASS}/22 ==")
sys.exit(0 if PASS == 22 else 1)
