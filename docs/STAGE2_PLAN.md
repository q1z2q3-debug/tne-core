# TNE-Core 阶段二 · 运行时投影实验方案

> 核心问题：**九维三态是否是 LLM hidden state 的自然坐标？**
> 决策门：Projector 准确率 >75% 继续全量；60-75% 调整坐标设计；<60% 放弃 hidden state 路线。

## 一、前置信与教训（来自 tne_core_v32）

整合的 v3.2 版本已证明两条死路：

1. **关键词法（吃文本）到天花板**：val 泛化 25%，FALSE（多义词）无解
   - 「重复」可以是形容词或习惯延续；「临时」方向取决于施动者
2. **离线小模型走不通**：预训练权重 403 取不到、torch 装不上、从零训练曲线下降（4.3%）
   - 0.858⁹≈25% 吻合独立概率乘积 → 没学到维度间关联

**一条活路**：在线 LLM 通道（OpenAI 兼容）已接好但缺 API Key 未实测。

## 二、阶段二实验设计（吃 hidden state，不吃文本）

### 为什么必须吃 hidden state
吃文本 = 外挂语义查询（v3.2 已证明天花板 25%）。
吃 hidden state = 内化映射（模型内部状态携带任务态势）。

### 数据流
```
LLM 推理（vLLM/TGI）
  → 每轮 hidden_states (dim=4096)  +  tool_trace  +  reasoning
  → Projector (MLP: 4096 → 9×3 logits)
  → quantizer → 硬三态 (-1/0/+1) + 置信度
  → encode 定位 → 等价类 → 轨迹落盘
```

### 三阶段训练管线
- **阶段A · Projector 冷启动**：人工标注 (hidden, 9d) 对 5000-10000 条，训练 MLP
- **阶段B · 轨迹积累**：跑 5000-10000 轮真实 Agent 任务，覆盖等价类
- **阶段C · 内化蒸馏**：SFT + RL，让模型自带九维感

## 三、本目录结构

```
src/tne/projector/
├── hidden_state_protocol.py   从推理框架抽取 hidden state 的接口协议
├── model.py                   轻量 MLP (4096 → 9×3 logits)，torch 实现
├── quantizer.py               logits → 硬三态 (-1/0/+1) + 置信度
├── data_gen.py                合成 hidden state + 九维标签（架构验证用）
├── train.py                   训练脚本（需 GPU）
├── eval.py                    评测脚本（准确率/置信度/混淆矩阵）
└── verify_arch.py             不依赖 torch 的最小架构验证（numpy 模拟 MLP）
```

## 四、当前环境约束（来自 v3.2）

| 约束 | 影响 |
|---|---|
| 无 GPU | 真实投影训练跑不了 |
| 装不上 torch（File too large） | torch 实现无法本地运行 |
| 拿不到预训练权重（403） | 无法加载基座模型抽 hidden state |

**对策**：
- 用 `verify_arch.py`（numpy）验证 Projector 架构端到端逻辑正确
- torch 实现保留为"待 GPU 环境运行"
- hidden_state_protocol 定义标准接口，等 GPU 环境接入 vLLM

## 五、决策记录

| 决策 | 理由 |
|---|---|
| Projector 输出硬三态不做连续回归 | 三态是认知判断（收敛/驻留/生发），连续值模糊语义 |
| 先验证架构后投入训练 | 环境约束下先证明逻辑正确 |
| 保留 v3.2 作前置参考 | 避免重复踩关键词法/离线小模型的坑 |
