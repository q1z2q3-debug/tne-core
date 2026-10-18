"""
TNE-Core 阶段二 · Projector 评测脚本

评测标准（决策门）：
  - 完全吻合率 > 75%  → 继续全量（阶段三）
  - 60-75%            → 调整坐标设计
  - < 60%             → 放弃 hidden state 路线

评测指标：
  - 完全吻合率（九维全对）
  - 逐维准确率
  - 混淆矩阵（每维的 -1/0/+1 预测分布）
  - 置信度校准（argmax 概率 vs 正确率）
"""
from __future__ import annotations
import sys, os
from typing import List
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def evaluate_model(predict_fn, samples: List, verbose: bool = True) -> dict:
    """
    predict_fn: (hidden_state) -> {vector, confidence, per_dim_conf}
    samples: [HiddenStateSample with .label]
    """
    exact = 0
    dim_correct = [0] * 9
    conf_correct = []
    conf_wrong = []
    n = len(samples)

    for s in samples:
        pred = predict_fn(s.hidden_states)
        vec = pred["vector"]
        conf = pred["confidence"]
        if vec == s.label:
            exact += 1
            conf_correct.append(conf)
        else:
            conf_wrong.append(conf)
        for d in range(9):
            if vec[d] == s.label[d]:
                dim_correct[d] += 1

    result = {
        "total": n,
        "exact_match_rate": round(exact / n, 3),
        "per_dim_accuracy": [round(c / n, 3) for c in dim_correct],
        "decision": "pass" if exact / n > 0.75
                    else ("adjust" if exact / n >= 0.6 else "abandon"),
    }
    if verbose:
        print(f"  完全吻合率: {result['exact_match_rate']*100:.1f}%")
        print(f"  逐维准确率: {result['per_dim_accuracy']}")
        print(f"  决策门: {result['decision']}")
    return result


if __name__ == "__main__":
    print("TNE-Core 阶段二 · Projector 评测（需 GPU 训练后运行）")
