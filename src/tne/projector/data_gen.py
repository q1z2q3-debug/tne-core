"""
TNE-Core 阶段二 · 合成训练数据生成器

生成 (hidden_state, 九维标签) 对，用于：
  1. 验证 Projector 架构端到端逻辑正确（不依赖 GPU）
  2. 真实 GPU 环境的 warm start / sanity check

注意（v3.2 教训）：
  合成数据训练 + 真实数据测试 = 分布不匹配，必然失败。
  这里的合成数据只用于【架构验证】，绝不用于宣称真实泛化能力。
"""
from __future__ import annotations
import random
from typing import List, Tuple
from .hidden_state_protocol import SyntheticExtractor, HiddenStateSample


def random_vector() -> List[int]:
    """随机生成一个九维三态向量（均匀分布）。"""
    return [random.choice([-1, 0, 1]) for _ in range(9)]


def synthesize_dataset(n: int, hidden_dim: int = 10,
                       noise: float = 0.05,
                       seed: int = 42) -> List[HiddenStateSample]:
    """合成 n 条 (hidden_state, label) 样本。"""
    random.seed(seed)
    extractor = SyntheticExtractor(hidden_dim=hidden_dim, noise=noise)
    samples = []
    for _ in range(n):
        vec = random_vector()
        samples.append(extractor.synthesize(vec))
    return samples


def split(samples: List[HiddenStateSample],
          train_ratio: float = 0.8,
          seed: int = 42) -> Tuple[List, List]:
    """按固定 seed 划分训练/验证集（划分在任何预测之前完成）。"""
    random.seed(seed)
    idx = list(range(len(samples)))
    random.shuffle(idx)
    n_train = int(len(samples) * train_ratio)
    train = [samples[i] for i in idx[:n_train]]
    val = [samples[i] for i in idx[n_train:]]
    return train, val


if __name__ == "__main__":
    data = synthesize_dataset(1000)
    train, val = split(data)
    print(f"合成数据集: {len(data)} 条")
    print(f"训练集: {len(train)} | 验证集: {len(val)}")
    print(f"样例向量: {data[0].label}, hidden_dim={data[0].dim()}")
