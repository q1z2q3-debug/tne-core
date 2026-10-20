"""
TNE-Core · 集中参数配置（Config）

把所有预设数字集中管理 + 基于全集/数据经验校准 + 支持自定义覆盖。

覆盖优先级（低 → 高）：
  1. DEFAULTS（经验校准默认值）
  2. 环境变量  TNE_<SECTION>_<KEY>   （如 TNE_BIAN_EXHAUSTION_THRESHOLD=0.42）
  3. Config(overrides={...}) 代码覆盖
  4. 调用处函数实参（最优先，单次调用自定义）

经验校准记录（勿覆盖本源，仅记依据）：
  - exhaustion_weights=(0.5,0.1,0.4)  全集19683卦对比：原(0.4,0.4,0.2)让saturation主导
    致穷度均值0.405、交替平衡卦错判0.47穷；改后交替0.20(不穷)/壅塞0.53(穷)/纯阳1.0/全衡0
  - exhaustion_threshold=0.42          全集穷度 P90（约10%卦触发"穷"，穷则变不过于频繁）
"""
from __future__ import annotations
import os, copy
from typing import Any, Dict, Optional

DEFAULTS: Dict[str, Dict[str, Any]] = {
    "bian": {
        "exhaustion_weights": (0.5, 0.1, 0.4),   # polarity/saturation/congestion
        "exhaustion_threshold": 0.42,            # 全集穷度 P90
        "timing_divisor": 4.0,                   # 时中权重 1+|爻位-中位|/divisor
        "f_v_k": 3,                              # 变易候选数(权变)
    },
    "tension": {
        "boundary_conf": 0.60,      # 维度置信度阈值
        "novelty_freq": 3,          # 等价类命中频次阈值
        "neighbor_gap": 0.15,       # 邻域前两卦距离接近阈值
        "exhaustion_threshold": 0.42,  # 与 bian 一致
        "strength": {               # 各张力强度
            "novelty_zero": 0.9, "novelty_low": 0.6,
            "boundary_base": 0.5, "boundary_per_dim": 0.1,
            "boundary_near": 0.7, "habit_conflict": 0.8,
            "default": 0.5,
        },
    },
}


def _parse_env(raw: str) -> Any:
    """环境变量字符串 → 数值/元组/字符串。"""
    raw = raw.strip()
    if "," in raw:
        try:
            return tuple(float(x) for x in raw.split(","))
        except ValueError:
            return tuple(x.strip() for x in raw.split(","))
    try:
        return float(raw)
    except ValueError:
        return raw


class Config:
    """集中配置：默认 + 环境变量 + 代码覆盖，支持运行时 set。"""

    def __init__(self, overrides: Optional[Dict[str, Dict[str, Any]]] = None):
        self.d = copy.deepcopy(DEFAULTS)
        # 环境变量覆盖
        for sec, keys in DEFAULTS.items():
            for k in keys:
                if isinstance(keys[k], dict):
                    continue  # 嵌套(如 strength)走整体 env 或 overrides
                env = os.environ.get(f"TNE_{sec.upper()}_{k.upper()}")
                if env is not None:
                    self.d[sec][k] = _parse_env(env)
        # 代码覆盖
        if overrides:
            for sec, kv in overrides.items():
                self.d.setdefault(sec, {}).update(kv)

    def get(self, section: str, key: str, default: Any = None) -> Any:
        return self.d.get(section, {}).get(key, default)

    def set(self, section: str, key: str, value: Any) -> None:
        self.d.setdefault(section, {})[key] = value

    def section(self, section: str) -> Dict[str, Any]:
        return self.d.get(section, {})

    def dump(self) -> Dict[str, Any]:
        return copy.deepcopy(self.d)


# 全局默认配置实例
default_config = Config()
