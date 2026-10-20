"""
TNE-Core · 变算子（Bian Operators）——第 6 类算子

从古人智慧（《周易》系辞上传/下传、《易纬·乾凿度》郑玄注）逆向提炼"变"的
七条规律，形式化为可执行的算子。公理层保持不可修改；本模块是在既有五算子
（F_C 错 / F_Z 综 / F_CZ 错综 / F_H 互 / F_B 动爻）之上新增的"变机"。

变之七律 → 算子映射：
  律1 刚柔相推而生变化        → tension()  推荡度（阴阳对张 + 推荡点）
  律2 一阖一辟谓之变，往来不穷谓之通 → F_C/F_Z 对合性（已证），本模块引用
  律3 穷则变，变则通，通则久  → exhaustion() / is_exhausted()  穷点检测器
  律4 变通者，趣时者也        → timing_weights()  时中（变位权重）
  律5 屈信相感而利生焉        → sensation()  感（检索触发）
  律6 变动以利言，变而通之以尽利 → benefit()  利评估（通/利）
  律7 唯变所适，不可为典要    → contingent()  权变（候选集，不唯一）

整合入口：f_v()  变易流程（穷→变→通→久 → 输出候选与通久标记）。
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from .space import DIM, validate
from .operators import f_bian


# ---------------------------------------------------------------------------
# 律3 · 穷点检测器（Exhaustion）
# ---------------------------------------------------------------------------
def exhaustion(gua: List[int]) -> Dict[str, float]:
    """穷度 Ex(G) ∈ [0,1]，卦的饱和/壅塞/极 程度。

    三分量：
      polarity   净极性张度 = |Σg|/9      全同号=1（阳极/阴极穷）
      saturation 非零爻比例 = 非零爻数/9   全实=1（饱和穷）
      congestion 最长连续同号段/9          连排=1（壅塞穷）
    total = 0.4*polarity + 0.4*saturation + 0.2*congestion
    """
    validate(gua)
    s = sum(gua)
    polarity = abs(s) / DIM
    saturation = sum(1 for g in gua if g != 0) / DIM
    # 最长连续同号（非零）段；中和(0)不构成壅塞，遇0重置
    run = best = 0
    prev = None
    for g in gua:
        if g == 0:
            run, prev = 0, None
            continue
        if g != prev:
            run, prev = 1, g
        else:
            run += 1
        best = max(best, run)
    congestion = best / DIM
    total = 0.4 * polarity + 0.4 * saturation + 0.2 * congestion
    return {"polarity": round(polarity, 4), "saturation": round(saturation, 4),
            "congestion": round(congestion, 4), "total": round(total, 4)}


def is_exhausted(gua: List[int], threshold: float = 0.6) -> bool:
    """穷则变的判定：穷度超过阈值即触发变。"""
    return exhaustion(gua)["total"] >= threshold


def exhaustion_positions(gua: List[int]) -> List[int]:
    """穷点：最长连续同号段两端的爻位（1-based），穷则于此发动。"""
    validate(gua)
    best_run, best_start, run, start = 0, -1, 0, 0
    prev = 0
    for i, g in enumerate(gua):
        if g != 0 and g == prev:
            run += 1
        else:
            run, start, prev = 1, i, g
        if run > best_run:
            best_run, best_start = run, start
    if best_start < 0:
        return []
    end = best_start + best_run - 1
    pos = {best_start + 1, end + 1}  # 两端（1-based）
    return sorted(pos)


# ---------------------------------------------------------------------------
# 律1 · 推荡（刚柔相推）
# ---------------------------------------------------------------------------
def tension(gua: List[int]) -> Dict:
    """刚柔推荡度：阴阳对张 + 推荡点（阴阳交接位，最可能变处）。"""
    validate(gua)
    yang = sum(1 for g in gua if g == 1)
    yin = sum(1 for g in gua if g == -1)
    boundaries = [i + 1 for i in range(DIM - 1)
                  if gua[i] != 0 and gua[i + 1] != 0 and (gua[i] > 0) != (gua[i + 1] > 0)]
    return {"yang": yang, "yin": yin, "conflict": min(yang, yin),
            "boundaries": boundaries, "divergence": round(abs(yang - yin) / DIM, 4)}


# ---------------------------------------------------------------------------
# 律4 · 时中（变通者，趣时者也）
# ---------------------------------------------------------------------------
def timing_weights(gua: Optional[List[int]] = None) -> Dict[int, float]:
    """变位时中权重：中位(g5)稳、边缘(g1/g9)激。权 = 1 + |爻位-中位|/4 ∈ [1,2]。"""
    mid = 5
    return {i + 1: round(1 + abs(i + 1 - mid) / 4, 3) for i in range(DIM)}


def chosen_position(gua: List[int]) -> int:
    """择时定变位：优先穷点端，次推荡边界，最后取时中权重最高位。"""
    pos = exhaustion_positions(gua)
    if not pos:
        t = tension(gua)
        pos = t["boundaries"]
    if not pos:
        w = timing_weights()
        pos = [max(w, key=w.get)]
    # 穷点/边界多重时取时中权重最"激"者（边缘优先，穷则变重机变）
    w = timing_weights()
    return max(pos, key=lambda p: (exhaustion_positions(gua) and p in exhaustion_positions(gua), w[p]))


# ---------------------------------------------------------------------------
# 律5 · 感（屈信相感而利生焉）
# ---------------------------------------------------------------------------
def euclid(a: List[int], b: List[int]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


def sensation(gua: List[int], target: List[int]) -> float:
    """感：卦与目标态的欧氏距离（越小=感越强/越被触发）。"""
    validate(gua)
    return euclid(gua, target)


# ---------------------------------------------------------------------------
# 律6 · 利评估（变动以利言，变而通之以尽利）
# ---------------------------------------------------------------------------
def benefit(gua: List[int], new_gua: List[int],
            target: Optional[List[int]] = None) -> Dict[str, float]:
    """变后是否通/利：
      balance_gain = 穷度极性下降（变后更趋衡=通）
      target_gain  = 距目标更近（感之增益，若给目标）
      benefit      = balance_gain + target_gain
    benefit > 0 为"通而利"。
    """
    validate(gua); validate(new_gua)
    bal = exhaustion(gua)["polarity"] - exhaustion(new_gua)["polarity"]
    tg = 0.0
    if target is not None:
        tg = euclid(gua, target) - euclid(new_gua, target)
    return {"balance_gain": round(bal, 4), "target_gain": round(tg, 4),
            "benefit": round(bal + tg, 4)}


# ---------------------------------------------------------------------------
# 律7 · 权变（唯变所适，不可为典要）
# ---------------------------------------------------------------------------
def contingent(gua: List[int], moving_set: List[int],
               target: Optional[List[int]] = None) -> List[Tuple[int, List[int], Dict]]:
    """对候选动爻逐一变卦，保留通/利者，返回候选集（不唯一，交主体裁量）。
    每项: (发动爻位, 变后卦, 利评估)。"""
    out = []
    for p in moving_set:
        ng = f_bian(gua, {p})
        b = benefit(gua, ng, target)
        if b["benefit"] >= 0:  # 唯变所适：只留通/利
            out.append((p, ng, b))
    return out


# ---------------------------------------------------------------------------
# 整合 · 变易流程 F_V（穷 → 变 → 通 → 久）
# ---------------------------------------------------------------------------
def f_v(gua: List[int], target: Optional[List[int]] = None,
        ex_threshold: float = 0.6, k: int = 3) -> Dict:
    """变易流程（第6类算子综合）：
      1 测穷   exhaustion(gua)   ——穷则变
      2 择时   chosen_position() ——趣时
      3 推荡   tension() 提供推荡边界
      4 发动   f_bian() 变卦
      5 通久   benefit() 评估，只留通/利
      6 权变   contingent() 输出候选集（唯变所适）
    若未穷则不触发（通则久，保持稳定）。
    """
    validate(gua)
    rep = exhaustion(gua)
    if not is_exhausted(gua, ex_threshold):
        # 未穷 → 通则久（保持稳定，不触发）
        return {"triggered": False, "exhaustion": rep, "chosen": None,
                "candidates": [(None, gua, None)], "通": True, "久": True}
    # 穷点端 + 推荡边界 并集作为候选发动位
    moving_set = list(set(exhaustion_positions(gua)) | set(tension(gua)["boundaries"]))
    if not moving_set:
        moving_set = [chosen_position(gua)]
    cands = contingent(gua, moving_set, target)
    cands.sort(key=lambda x: -x[2]["benefit"])
    cands = cands[:k]
    return {"triggered": True, "exhaustion": rep,
            "chosen": chosen_position(gua), "candidates": cands,
            "通": bool(cands), "久": False if cands else True}


# 别名（第6类算子）
EX = exhaustion
EX_POS = exhaustion_positions
T = tension
TM = timing_weights
S = sensation
L = benefit
Q = contingent
F_V = f_v
