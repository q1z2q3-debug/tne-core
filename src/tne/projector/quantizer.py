"""
TNE-Core 阶段二 · 量化器

把 Projector 输出的 9×3 logits 转换为硬三态 (-1/0/+1) + 置信度。

设计决策（来自终极方案）：
  - 输出硬三态，不做连续回归：三态本身就是认知判断（收敛/驻留/生发）
  - 硬分类迫使模型做出判断，避免连续值模糊语义
"""
from __future__ import annotations
from typing import List, Dict, Any


# logits 三态索引
IDX_TO_STATE = {0: -1, 1: 0, 2: 1}   # 类别索引 → 三态值


def quantize(logits_3x3: List[List[float]]) -> Dict[str, Any]:
    """
    把 9×3 logits 矩阵量化为九维三态向量 + 置信度。

    Args:
        logits_3x3: 9 行 × 3 列的 logits 矩阵（每行对应一个维度）
    Returns:
        {vector, confidence, per_dim_conf, argmax_probs}
    """
    import math

    vector = []
    per_dim_conf = []
    argmax_probs = []

    for row in logits_3x3:
        # softmax 求每类概率
        exps = [math.exp(v) for v in row]
        total = sum(exps)
        probs = [e / total for e in exps]
        # 取 argmax
        idx = max(range(3), key=lambda i: probs[i])
        vector.append(IDX_TO_STATE[idx])
        argmax_probs.append(probs[idx])
        per_dim_conf.append(probs[idx])

    # 整体置信度 = 各维 argmax 概率的均值
    confidence = sum(per_dim_conf) / len(per_dim_conf) if per_dim_conf else 0.0

    return {
        "vector": vector,
        "confidence": round(confidence, 3),
        "per_dim_conf": [round(p, 3) for p in per_dim_conf],
        "argmax_probs": [round(p, 3) for p in argmax_probs],
    }
