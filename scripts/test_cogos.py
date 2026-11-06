"""
CogOS 单元测试：内核公理 + 映射确定性 + 空间先于内容 + 闭环写回 + 输出脱结构。

运行：PYTHONPATH=src python3 scripts/test_cogos.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tne.space.space import TOTAL_STATES, encode, decode, gua_str
from tne.space.operators import f_cuoo, f_zong, f_cuoo_zong, f_hu, f_bian
from tne.space.equivalence import count_equivalence_classes, class_of
from tne.cogos import CogOS, map_text, MemorySpace

FAILS = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILS.append(name)


print("== 1. 相空间公理 ==")
check("全集 3^9 = 19683", TOTAL_STATES == 19683, TOTAL_STATES)
check("等价类 = 5002", count_equivalence_classes() == 5002, count_equivalence_classes())
check("编码往返一致", all(decode(encode(g)) == g for g in [[1, 0, -1] * 3, [0] * 9, [1] * 9]))

print("== 2. 五算子（单元测试样例）==")
G = [1, 1, 0, -1, 1, 0, -1, -1, 0]
check("F_C(G)", f_cuoo(G) == [-1, -1, 0, 1, -1, 0, 1, 1, 0], f_cuoo(G))
check("F_Z(G)", f_zong(G) == [0, -1, -1, 0, 1, -1, 0, 1, 1], f_zong(G))
check("F_CZ(G)", f_cuoo_zong(G) == [0, 1, 1, 0, -1, 1, 0, -1, -1], f_cuoo_zong(G))
check("F_H(G) 5维", f_hu(G) == [0, -1, 1, 0, -1], f_hu(G))
check("第5爻动", f_bian(G, {5}) == [1, 1, 0, -1, 0, 0, -1, -1, 0], f_bian(G, {5}))
check("错卦自反", f_cuoo(f_cuoo(G)) == G)
check("综卦自反", f_zong(f_zong(G)) == G)
check("互卦不补齐9维", len(f_hu(G)) == 5)

print("== 3. 映射确定性 ==")
r1 = map_text("先定目标，方法拆成三步，资料补齐", domain="technical")
r2 = map_text("先定目标，方法拆成三步，资料补齐", domain="technical")
check("同输入同卦位", r1.gua == r2.gua, (r1.gua, r2.gua))
check("置信度在 [0,1]", 0 <= r1.confidence <= 1, r1.confidence)

print("== 4. 空间先于内容 ==")
empty = MemorySpace()
check("空库必 miss", empty.recall("任意内容", [1, 0, 0, 0, 0, 0, 0, 0, 0]).status == "miss")
check("空库空间候选为 0", empty.spatial_candidates([1, 0, 0, 0, 0, 0, 0, 0, 0]) == [])

os_ = CogOS()
t1 = os_.process("先定目标，方法拆成三步，资料补齐", domain="technical")
check("首轮为 miss", t1.recall.status == "miss", t1.recall.status)
check("方向成立则写入", len(os_.memory) == 1, len(os_.memory))

t2 = os_.process("数据中台按三步走，资料已补齐，目标也定了", domain="technical")
check("相近内容命中", t2.recall.status == "hit", t2.recall.status)
check("命中不新增记忆", len(os_.memory) == 1, len(os_.memory))

t3 = os_.process("团队沟通很累，成员之间对立，我很犹豫", domain="emotional")
check("结构近但内容远 → 联想", t3.recall.status == "association", t3.recall.status)
check("联想不新增记忆", len(os_.memory) == 1, len(os_.memory))

print("== 5. 方向未立不记忆 ==")
os2 = CogOS()
t4 = os2.process("嗯，随便吧", domain="general")
check("低置信不记忆", len(os2.memory) == 0 and not t4.remembered, (t4.mapping.confidence, len(os2.memory)))

print("== 6. 共振反馈只由强结构触发 ==")
os3 = CogOS()
os3.memory.add([1, 1, 1, 1, 0, 0, 1, 0, 0], ["目标"], "目标明确方法清楚", os3._frame_for("technical"))
prov = os3.memory.frames_near([0, -1, 0, 0, 0, -1, 0, 0, 0])
check("邻近关系不提供调轴", prov == [], len(prov))
prov2 = os3.memory.frames_near(f_cuoo([1, 1, 1, 1, 0, 0, 1, 0, 0]))
check("对偶关系提供调轴", len(prov2) == 1, len(prov2))

print("== 7. 输出层不泄露结构 ==")
BANNED = ["卦位", "九维", "爻", "g1", "g2", "算子", "19683", "欧氏", "等价类", "错卦", "综卦", "互卦"]
for tag, t in (("轮1", t1), ("轮2", t2), ("轮3", t3)):
    leaked = [w for w in BANNED if w in t.reply]
    check(f"{tag} 回复无结构术语", not leaked, leaked)

print("== 8. 等价类索引一致 ==")
check("同卦同类", class_of(encode(G)) == class_of(encode(f_cuoo(G))))
check("类代表可查", len(os_.memory.spatial_candidates(t2.mapping.gua)) >= 1)

print()
if FAILS:
    print(f"FAILED {len(FAILS)}: {FAILS}")
    sys.exit(1)
print("ALL TESTS PASSED")
