# TNE-Core · 三元九维认知引擎

> **Trinary Nine-Dimensional Cognitive Engine**
> 让大模型在推理的同时，自然地在九维三态相空间中留下轨迹；轨迹积累成习惯；习惯产生张力；张力驱动认知——一个自我运转的认知闭环。

## 一句话定位

在大模型之上加一层**可定位、可泛化、可审计**的符号认知坐标。不是外挂查询工具，而是运行时内化的认知层。

## 公理底座（锁定，不可修改）

| 项 | 定义 |
|---|---|
| 单爻三态 | 阳(+1) / 中和(0) / 阴(-1) |
| 卦体 | 9 维有序向量 G=[g₁…g₉]，g₁=上爻，g₉=下爻 |
| 全集 | 3⁹ = **19683** 卦 |
| 等价类 | {G, F_C(G), F_Z(G), F_CZ(G)}，合并后 **5002** 组 |
| 距离 | 欧氏距离，Top-N 邻域检索 |

五类算子：错卦 F_C / 综卦 F_Z / 错综 F_CZ / 互卦 F_H（5维内核）/ 动爻变卦 F_B。

学术论文：[Zenodo DOI 10.5281/zenodo.23437930](https://zenodo.org/records/23437930)
官方主页：https://q1z2q3-debug.github.io/sanjiu-19683/

## 当前实现（阶段一：外挂认知层产品化）

```
tne-core/
├── src/tne/
│   ├── space/           # 相空间内核
│   │   ├── space.py         # 19683态枚举、三态编码/解码
│   │   ├── operators.py     # 错/综/错综/互/变算子（论文锁定）
│   │   ├── equivalence.py   # 5002等价类计算与缓存
│   │   └── distance.py      # 欧氏距离 + Top-N邻域检索
│   ├── trajectory/      # 认知轨迹库
│   │   ├── schema.sql       # 表结构
│   │   ├── store.py         # SQLite 读写，每轮自动落盘
│   │   └── habit.py         # 习惯偏置查询（命中→习惯/未命中→探索）
│   ├── api/             # 坐标服务
│   │   └── server.py        # FastAPI：encode/search/transform/habit/trajectory/stats
│   └── projector/       # 阶段二预留（运行时投影器）
└── tests/
    └── test_core.py         # 7项单元测试，全部通过
```

## 快速开始

```bash
# 安装依赖
pip install fastapi uvicorn

# 运行坐标服务
cd tne-core
PYTHONPATH=src uvicorn tne.api.server:app --host 0.0.0.0 --port 8000

# 跑测试
python3 tests/test_core.py
```

## API 端点

| 端点 | 方法 | 功能 |
|---|---|---|
| `/health` | GET | 健康检查 + 全集/等价类统计 |
| `/encode` | POST | 九维向量 → 卦位索引 + 等价类 |
| `/search` | POST | 目标向量 → Top-N 最邻近卦 |
| `/transform` | POST | 卦位 → 错/综/错综/互/变全套推演 |
| `/habit` | POST | 当前卦位 → 习惯偏置查询 |
| `/trajectory` | POST | 记录一轮 (态,行为,结果) 轨迹 |
| `/stats` | GET | 轨迹库等价类覆盖率 |

## 测试结果

```
✓ space encode/decode OK
✓ operators paper example ALL PASS
✓ operator involutions OK
✓ equivalence classes = 5002 OK
✓ equivalence closure OK (100 samples)
✓ distance & nearest_neighbors OK
✓ trajectory store & habit query OK
🎉 ALL TESTS PASSED
```

## 升级路线

| 阶段 | 内容 | 周期 | 资源 |
|---|---|---|---|
| **阶段一**（当前） | 外挂认知层产品化：API + 轨迹库 + 习惯偏置 + 接入真实 Agent | 2-4 周 | 无 GPU |
| **阶段二** | 运行时投影实验：hidden state → 九维三态，验证根本假设 | 2-4 周 | 1× GPU |
| **阶段三** | TNE-Core 全量：张力检测 + 动态等价 + 内化蒸馏 | 3-6 月 | 4× A100 |

详见 [docs/PLAN.md](docs/PLAN.md)。

## 开源协议

MIT License · 论文 CC-BY-4.0
