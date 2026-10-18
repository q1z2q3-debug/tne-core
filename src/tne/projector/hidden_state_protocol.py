"""
TNE-Core 阶段二 · hidden state 抽取协议

定义从推理框架（vLLM/TGI）抽取每轮 hidden states 的标准接口。
该协议是"运行时投影"的入口：LLM 内部状态 → 九维三态。

当前环境无 GPU，此协议是给 GPU 环境（阶段二真实运行）的契约。
所有实现类都必须实现 `extract_hidden_states`。
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any


@dataclass
class HiddenStateSample:
    """一轮 LLM 推理的 hidden state 样本。"""
    # 原始 hidden state 向量（dim=4096 或对应模型维度）
    hidden_states: List[float]
    # 该轮的 tool 调用轨迹（可选，作为辅助信号）
    tool_trace: List[Dict[str, Any]] = field(default_factory=list)
    # 该轮的 reasoning 文本（可选）
    reasoning: str = ""
    # 模型名称
    model_name: str = ""
    # 时间戳
    timestamp: float = 0.0
    # 该轮对应的九维标签（人工标注，训练用；预测时为空）
    label: Optional[List[int]] = None

    def dim(self) -> int:
        return len(self.hidden_states)

    def validate(self) -> List[str]:
        errs = []
        if not self.hidden_states:
            errs.append("hidden_states 为空")
        if self.label and len(self.label) != 9:
            errs.append(f"label 长度应为 9，实际 {len(self.label)}")
        if self.label and any(x not in (-1, 0, 1) for x in self.label):
            errs.append(f"label 含非 -1/0/1 值: {self.label}")
        return errs


class HiddenStateExtractor:
    """
    hidden state 抽取器抽象基类。
    子类实现：从 vLLM / TGI / 本地模型 抽取每轮 hidden states。
    """

    def extract_hidden_states(self, request_id: str,
                              model_input: str) -> HiddenStateSample:
        """根据一次推理请求，返回该轮的 hidden state 样本。"""
        raise NotImplementedError("子类必须实现 extract_hidden_states")


class SyntheticExtractor(HiddenStateExtractor):
    """
    合成抽取器：为架构验证生成伪 hidden state。
    用"已知九维标签 → 构造带区分度的 hidden state"，
    验证 Projector 能从 hidden state 中恢复九维标签。
    """

    # 每个维度对应的 hidden state 基向量方向（正交）
    # 真实场景里 hidden state 的各维度是纠缠的，这里用正交简化验证
    DIM_PROJECTIONS = [
        [0.9, 0.1, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1, 0.0],   # d0
        [0.0, 0.9, 0.1, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1],   # d1
        [0.1, 0.0, 0.9, 0.1, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0],   # d2
        [0.0, 0.1, 0.0, 0.9, 0.1, 0.0, 0.0, 0.1, 0.0, 0.0],   # d3
        [0.0, 0.0, 0.1, 0.0, 0.9, 0.1, 0.0, 0.0, 0.1, 0.0],   # d4
        [0.1, 0.0, 0.0, 0.1, 0.0, 0.9, 0.1, 0.0, 0.0, 0.1],   # d5
        [0.0, 0.1, 0.0, 0.0, 0.1, 0.0, 0.9, 0.1, 0.0, 0.0],   # d6
        [0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.0, 0.9, 0.1, 0.0],   # d7
        [0.1, 0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.0, 0.9, 0.1],   # d8
    ]

    def __init__(self, hidden_dim: int = 10, noise: float = 0.05):
        self.hidden_dim = hidden_dim
        self.noise = noise

    def extract_hidden_states(self, request_id: str,
                              model_input: str) -> HiddenStateSample:
        raise NotImplementedError("SyntheticExtractor 用于批量生成，请用 synthesize()")

    def synthesize(self, vector: List[int]) -> HiddenStateSample:
        """从九维标签合成 hidden state。每个维度按其三态值投影到隐向量。"""
        import random
        h = [0.0] * self.hidden_dim
        for d in range(9):
            val = vector[d]  # -1/0/+1
            for k, proj in enumerate(self.DIM_PROJECTIONS[d]):
                h[k] += val * proj
        # 加噪声
        h = [x + random.uniform(-self.noise, self.noise) for x in h]
        return HiddenStateSample(
            hidden_states=h,
            model_name="synthetic",
            label=list(vector),
        )
