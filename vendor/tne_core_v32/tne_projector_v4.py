"""TNE-Core v3.0 - Projector V4"""

import json
from dataclasses import dataclass
from typing import List


@dataclass
class EncodeResult:
    vector: List[int]
    confidence: float
    reasoning: str
    ambiguity_flags: List[str]


class YuanbaoProjectorV4:
    DIMENSION_ANCHORS = {
        0: {"+1": "创造、生成、从无到有", "0": "维持现状", "-1": "破坏、损耗"},
        1: {"+1": "习惯延续、自动化", "0": "首次尝试", "-1": "习惯中断"},
        2: {"+1": "外部驱动、紧迫", "0": "平稳", "-1": "外部阻力"},
        3: {"+1": "环境切换、新场景", "0": "环境不变", "-1": "环境退化"},
        4: {"+1": "工具/方法运用", "0": "无工具", "-1": "工具缺失"},
        5: {"+1": "状态转变、升级", "0": "状态稳定", "-1": "状态降级"},
        6: {"+1": "关系、协作", "0": "独立", "-1": "孤立、冲突"},
        7: {"+1": "自律、内省", "0": "无自律", "-1": "放纵、失控"},
        8: {"+1": "内在驱动、热情", "0": "无驱动", "-1": "内在耗竭"},
    }

    def encode(self, user_input: str) -> EncodeResult:
        return self._mock_encode(user_input)

    def _mock_encode(self, user_input: str) -> EncodeResult:
        vector = [0] * 9
        ambiguity = []

        if any(k in user_input for k in ["写", "创作", "设计", "实现", "构建", "做"]):
            vector[0] = 1
        elif any(k in user_input for k in ["删", "破坏", "终止"]):
            vector[0] = -1

        if any(k in user_input for k in ["继续", "再次", "重复", "延续"]):
            vector[1] = 1

        if any(k in user_input for k in ["紧急", "马上", "立刻", "催"]):
            vector[2] = 1

        if any(k in user_input for k in ["切换", "转到", "回到", "跨"]):
            vector[3] = 1

        if any(k in user_input for k in ["代码", "测试", "优化", "函数", "类", "模块"]):
            vector[4] = 1

        if any(k in user_input for k in ["优化", "升级", "改进", "突破", "重构"]):
            vector[5] = 1

        if any(k in user_input for k in ["你", "我", "我们", "协作", "信任", "团队"]):
            vector[6] = 1

        if any(k in user_input for k in ["自主", "自律", "推进", "坚持"]):
            vector[7] = 1

        if any(k in user_input for k in ["推", "驱动", "热情", "使命"]):
            vector[8] = 1

        determined = sum(1 for v in vector if v != 0)
        confidence = min(0.95, 0.6 + determined * 0.05)

        if determined < 4:
            ambiguity.append("信息不足，部分维度默认0")

        return EncodeResult(
            vector=vector,
            confidence=confidence,
            reasoning=f"关键词推断，{determined}/9维确定",
            ambiguity_flags=ambiguity
        )

    def batch_encode(self, inputs: List[str]) -> List[EncodeResult]:
        return [self.encode(inp) for inp in inputs]


projector_v4 = YuanbaoProjectorV4()
