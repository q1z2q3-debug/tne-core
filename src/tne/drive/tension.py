"""
TNE-Core · 认知动力引擎（Drive）

把"任务→定位→习惯"的被动认知循环，升级为"主动驱动"：

  任务定位 → 张力检测 → 动力信号 → 注入下一轮推理

三类张力（健康的认知状态，不是错误）：
  - novelty        新奇：等价类经验不足 / 邻域稀疏 → 好奇心（主动探索）
  - boundary       边界：判断在多卦位间摇摆 / 置信度低 → 深思（审慎权衡）
  - habit_conflict 习惯冲突：习惯建议 vs 当前语义相悖 → 元认知（质疑旧行为）

三类动力信号：
  - curiosity       探索：不套用旧经验，主动尝试新行为并记录
  - deliberation    深思：在多个候选间更审慎地权衡
  - meta_reasoning  元认知：质疑"习惯在这里是否适用"

设计：
  TensionDetector 检测当前态的三类张力
  SignalGenerator 张力 → 动力信号（注入文本 + 结构化信号）
"""
from __future__ import annotations
from typing import List, Dict, Any, Optional


class TensionDetector:
    """
    张力检测。输入：习惯查询结果 + 九维置信度 + 邻域信息。
    """

    def __init__(self, boundary_conf_threshold: float = 0.60,
                 novelty_freq_threshold: int = 3,
                 boundary_neighbor_gap: float = 0.15):
        self.boundary_conf = boundary_conf_threshold
        self.novelty_freq = novelty_freq_threshold
        self.neighbor_gap = boundary_neighbor_gap

    def detect(self, habit: Dict[str, Any],
               confidence: Optional[List[float]] = None,
               neighbors: Optional[List[Dict]] = None) -> List[Dict[str, Any]]:
        """
        返回检测到的张力列表：[{type, strength, note}]
        """
        tensions = []
        freq = habit.get("freq", 0)
        mode = habit.get("mode", "explore")

        # 1. novelty：等价类经验不足（explore 或命中 < 阈值）
        if mode == "explore" or freq < self.novelty_freq:
            tensions.append({
                "type": "novelty",
                "strength": 0.9 if freq == 0 else 0.6,
                "note": f"等价类经验不足（命中 {freq} 次），这是新领域",
            })

        # 2. boundary：某维置信度低，或邻域前两卦距离接近（判断摇摆）
        if confidence:
            low = [i for i, c in enumerate(confidence) if c < self.boundary_conf]
            if low:
                tensions.append({
                    "type": "boundary",
                    "strength": 0.5 + 0.1 * len(low),
                    "note": f"第 {[i+1 for i in low]} 维判断置信度偏低，边界模糊",
                })
        if neighbors and len(neighbors) >= 2:
            d0, d1 = neighbors[0].get("distance", 0), neighbors[1].get("distance", 1)
            if abs(d0 - d1) < self.neighbor_gap and d1 > 0:
                tensions.append({
                    "type": "boundary",
                    "strength": 0.7,
                    "note": f"两个邻近卦位距离接近（{d0} vs {d1}），判断在边缘摇摆",
                })

        # 3. habit_conflict：模式是 explore 但有明确习惯建议 / 或 habit 但语义冲突
        #    简单判定：explore 模式却给出明确方向建议 → 潜在冲突（元认知）
        if mode == "habit_weak":
            tensions.append({
                "type": "habit_conflict",
                "strength": 0.8,
                "note": "历史行为成功率不足，习惯可能不适用，需质疑旧行为",
            })

        # 去重（同 type 保留最强）
        seen = {}
        for t in tensions:
            if t["type"] not in seen or t["strength"] > seen[t["type"]]["strength"]:
                seen[t["type"]] = t
        return list(seen.values())


class SignalGenerator:
    """
    张力 → 动力信号。
    """

    # 张力类型 → (动力类型, 注入文本)
    SIGNAL_MAP = {
        "novelty": ("curiosity",
            "当前认知状态处于新领域（等价类经验不足）。注入「探索动力」：不要套用旧经验，主动尝试多种新行为并记录结果，未知本身就是机会。"),
        "boundary": ("deliberation",
            "当前判断在多个邻近卦位/维度间摇摆（置信度不足或边界接近）。注入「深思动力」：不要急于定论，先对每个候选做更审慎的权衡，量化不确定性后再决策。"),
        "habit_conflict": ("meta_reasoning",
            "检测到习惯可能不适用（历史行为成功率不足/与当前语义冲突）。注入「元认知动力」：质疑'我的习惯在这里是否适用'，主动重新评估旧行为，必要时打破惯例。"),
    }

    def generate(self, tensions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        张力 → 动力信号。返回 {signals:[{type,text}], prompt, strength}
        """
        signals = []
        max_strength = 0.0
        for t in tensions:
            stype, text = self.SIGNAL_MAP.get(t["type"], ("reflective", "审慎处理当前状态。"))
            strength = t.get("strength", 0.5)
            max_strength = max(max_strength, strength)
            signals.append({"type": stype, "strength": round(strength, 2), "text": text})

        # 组合注入文本（多信号拼接）
        if signals:
            kinds = "、".join(s["type"] for s in signals)
            prompt = (
                f"【认知动力注入 · {kinds}】\n"
                + "\n".join(f"- {s['text']}" for s in signals)
            )
        else:
            prompt = "当前认知状态稳定，习惯可复用，无需额外动力注入。"

        return {
            "signals": signals,
            "prompt": prompt,
            "max_strength": round(max_strength, 2),
        }


def detect_and_generate(habit: Dict[str, Any],
                        confidence: Optional[List[float]] = None,
                        neighbors: Optional[List[Dict]] = None) -> Dict[str, Any]:
    """一步完成：张力检测 + 动力信号生成。"""
    det = TensionDetector()
    gen = SignalGenerator()
    tensions = det.detect(habit, confidence, neighbors)
    return {"tensions": tensions, **gen.generate(tensions)}
