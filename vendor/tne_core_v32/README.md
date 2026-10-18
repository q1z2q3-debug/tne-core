# TNE-Core v3.2

三元九维认知框架。从对话轨迹提取可复用模式，泛化到新任务。

## 运行

```bash
cd tne_core
python main.py                  # 校准 + 批量入库 + 推荐演示
python calibrate.py             # 单独跑泛化测量（含失败归因）
python -m pytest tests/ -q      # 45 项测试
```

## 泛化能力：四个集，身份必须分清

| 集 | 条数 | 吻合率 | 身份 |
|---|---|---|---|
| `golden_set.json` | 20 | 100% | 拟合集，回归保护 |
| `train_set.json` | 60 | **93.3%** | 训练集，可调 |
| `val_set.json` | 40 | **25.0%** | 验证集，只测 |
| `holdout_set.json` | 30 | 36.7% | 独立集，只测 |

**真实泛化水平 = 25%**（两个只测集取较低者）。过拟合缺口 68.3 个百分点。

train/val 由 `split_dataset.py` 用固定 seed 从 `expanded_set.json`（100 条）划分，
**划分在看到任何预测结果之前完成**。100 条标注一次性写完，标注期间未跑任何预测。

### 补词实验的真实收益（可复现）

| 阶段 | train | val | holdout |
|---|---|---|---|
| 基线（调词前） | 10.0% | 7.5% | 16.7% |
| 第 1 轮：补通用词汇 | 70.0% | 22.5% | 30.0% |
| 第 2 轮：修 FALSE 个案 | 93.3% | 25.0% | 36.7% |

第 1 轮三个集同步大涨（val +15.0），是真实收益。
第 2 轮 train +23.3 而 val 仅 +2.5 —— **收益急剧递减，这就是过拟合的样子**。
继续在 train 上打补丁只会把缺口拉得更大，因此在第 2 轮后收手。

### 剩余错误的性质

- **MISS 为主**：词表覆盖不足，补词仍有效，但需要**新样本**才不算自证
- **FALSE 是关键词法的天花板**：多义词靠词表无解
  ——「把**重复**的判断逻辑抽成公共函数」里的「重复」是形容词不是习惯延续
- **方向依赖施动者的词无法表达**：「客户**临时**改验收标准」是阻力（-1），
  而「**临时**加了个会」是驱动（+1）。同一个词，主语决定方向。
  holdout 上现存 1 例，train 上为 0 例 —— 不通过针对 holdout 改词来消除

### 离线小模型：试过，走不通

完整记录见 `data/model_experiment.md`，`python tne_lean_model.py` 可复现。

**预训练小模型装不上**：HuggingFace / hf-mirror / ModelScope 全部 403，
模型权重取不到；`pip install torch` 报 `File too large`（沙箱 5G 磁盘 2 CPU）。

**从零训练也没用**：LLM 标注 235 条语料 → char n-gram + LinearSVC

| 语料量 | 60 | 120 | 180 | 235 |
|---|---|---|---|---|
| 留出集吻合率 | 41.7% | 33.3% | 5.6% | **4.3%** |

曲线**下降** —— 加数据反而更差。val 上小模型 15.0% < 词表 25.0%。

原因：语料按语义族构造，同族句子字面相近，模型学到字面不是语义。
逐维 85.8% 但九维全对 4.3% ≈ 0.858⁹ = 24%，与独立概率乘积吻合，
说明**没学到任何维度间关联**，只是九个独立任务各猜一次相乘。

合成语料训练 + 真实数据测试，分布不匹配是必然的。
要再试需要：真实分布语料数千条 / 可用的预训练权重 / 建模维度间关联。

### 几类典型的误触发与处置

| 现象 | 例子 | 处置 |
|---|---|---|
| 话题词 ≠ 态势词 | 「接口**三个月没动过**」 | IDLE_CONTEXT 抑制 d4 |
| 事故描述 ≠ 运用工具 | 「服务**崩了**」 | FAULT_CONTEXT 抑制 d0/d4 |
| 组织架构 ≠ 技术模块 | 「**模块**负责人换成小李」 | ORG_CONTEXT 仅抑制歧义词 |
| 行政审批 ≠ 工程方法 | 「**审批流程**太长」 | ADMIN_CONTEXT 抑制 d4 |
| 词素 ≠ 否定副词 | 「状态**不错**，效率很高」 | NOT_NEGATION 白名单 |
| 请求 ≠ 协作 | 「帮我看看**为什么**慢」 | QUESTION_CONTEXT 抑制 d6 |

## 批量入库

v3.1 每跑一次 `run_pipeline` 就 new 一个 Distiller 再 load/export 一次，
模式状态全靠文件往返累积。v3.2 用 `TNESession` 把生命周期拉长到整批：

```python
from tne_ingest import TNESession, build_batch_from_texts

with TNESession(fresh=True) as s:              # fresh=True 从零累积
    rep = s.ingest_file("data/batch_demo.jsonl")
    # → {'input':25,'ingested':21,'dupe_skipped':1,'invalid':3,'new_patterns':[...]}

    recs = s.recommend(task_type="创作叙事", text="再写一段父女在车站的重逢")
    # → [{'pattern_id':..., 'compat':1.0, 'success_rate':0.92, 'usage_count':7}]
# 退出 with 时落盘一次
```

`ingest_batch` 的 item 字段：

| 字段 | 必填 | 说明 |
|---|---|---|
| `text` | ✓ | 原始输入文本，自动 encode |
| `task_type` | ✓ | 任务类型，模式按此分桶 |
| `efficacy` | ✓ | 0~1 真实成效。**不可用 confidence 顶替** |
| `task_id` | | 不填自动生成；重复会被跳过 |
| `habits` | | 至少 2 项，否则永远蒸馏不出模式 |

内置校验会拦下脏数据并计数，不中断整批：
`efficacy` 越界、`habits` 少于 2 项、空文本、向量非法值。

## 目录

```
tne_core/
├── tne_schema.py              统一数据契约（TrajectoryRecord / 九维签名）
├── tne_projector_v5.py        双通道 encode（本地语义 + 在线 LLM）
├── tne_distiller.py           蒸馏与推荐
├── tne_trajectory_collector.py 轨迹落盘（批量 / 幂等）
├── tne_ingest.py              TNESession 批量入库入口
├── tne_lean_model.py          离线小模型实验（默认不启用，结论见下）
├── tne_annotator.py           人在回路标注缓存（命中即返回 LLM 判定值）
├── main.py                    pipeline 演示
├── calibrate.py               泛化测量与失败归因
├── tne_projector_v4.py        legacy，仅作对照保留
└── data/
    ├── golden_set.json        拟合集 20 条
    ├── holdout_set.json       独立集 30 条
    ├── batch_demo.jsonl       批量入库演示数据（含故意混入的脏数据）
    ├── trajectories.jsonl     轨迹库
    └── patterns.json          模式库
```

## encode 双通道

`YuanbaoProjectorV4.encode(text)` 接口不变，内部按配置自动选路：

| 模式 | 触发条件 | 说明 |
|---|---|---|
| `local`（默认） | 无 API Key | 由 LLM 标注语义固化而来的九维判定器，零依赖零延迟 |
| `llm` | 设了 `TNE_LLM_API_KEY` | OpenAI 兼容协议在线 encode |
| `llm_fallback` | 在线调用失败 | 自动回落本地，结果里带失败原因 |

```bash
export TNE_LLM_API_KEY=sk-xxx
export TNE_LLM_BASE_URL=https://api.deepseek.com/v1   # 可选，默认 OpenAI
export TNE_LLM_MODEL=deepseek-chat                    # 可选
```

```python
proj = YuanbaoProjectorV4(use_llm="auto")   # auto | local | llm
proj.encode("优化数据库查询性能")
# → vector=[0,0,0,0,1,1,0,0,0] conf=0.60 backend='local'
#   reasoning='d4地·习(+):+数据库|+查询；d5地·运(+):+优化|+性能'
```

## 九维定义

| # | 维度 | +1 | 0 | -1 |
|---|---|---|---|---|
| 0 | 天·性 | 创造生成 | 维持 | 破坏损耗 |
| 1 | 天·习 | 习惯延续 | 首次 | 习惯中断 |
| 2 | 天·运 | 外部驱动 | 平稳 | 外部阻力 |
| 3 | 地·性 | 环境切换 | 不变 | 环境退化 |
| 4 | 地·习 | 工具方法 | 无 | 缺失 |
| 5 | 地·运 | 状态升级 | 稳定 | 降级 |
| 6 | 人·性 | 关系协作 | 独立 | 孤立冲突 |
| 7 | 人·习 | 自律内省 | 无 | 放纵失控 |
| 8 | 人·运 | 内在驱动 | 无 | 耗竭 |

## V5 相对 V4 修掉的硬伤

1. **-1 不可达**：V4 有 8 个维度只能输出 +1/0，3^9 空间实际只剩 2^9
2. **d6 恒为 1**：用「你/我/我们」判定协作，几乎任何中文句都命中，零区分度
3. **d4/d5 语义耦合**：「优化」同时点亮工具维和状态维

新增的语义处理：
- **最长覆盖优先**：「没工具」(否定短语) 不再被词根「工具」抵消
- **否定即归零**：「别删」不等于「创造」，回落到 0 而非反向
- **词素保护**：「特**别**来劲」的「别」不是否定副词
- **上下文抑制**：「接口全挂」是事故不是工具运用；「团队又闹僵」冲突压过协作
- **0 锚点**：d1 的「首次」是框架中位，优先级高于任何正负信号

## 校准

`data/golden_set.json` 是 LLM 逐条语义标注的 20 条基准集。

```python
proj.evaluate_against(golden)   # → exact_match_rate / per_dim_accuracy / weakest_dims
```

当前本地判定器对该集 **100% 吻合**。扩充词表后跑一次即可发现回归与盲区。

## v3.2 修掉的 bug

1. **`typical_state_signature` 恒为默认值** —— v3.1 的 `_compat()` 读这个键，但
   它从来没被写过，所有模式共享同一个假签名，推荐排序等于没用状态信息
2. **signature 累积漂移** —— 增量平均时把均值存回再除 `cnt`，均值逐次衰减
   （0.5 → 0.17 → …），第三次就失真成 `p0z1n8`。改为存累积和
3. **signature 出现负分量** —— `9 - round(p) - round(z)` 在舍入后和超过 9 时
   算出 `p2z8n-1`。改用最大余数法，保证三分量非负且和为 9
4. **批内去重失效** —— `_seen_ids` 只登记落盘时读到的 id，同一批里的重复
   task_id 会全部写入，导致 usage_count 虚高
5. **`TrajectoryRecord` 双份定义** —— 统一到 `tne_schema.py`

## 已知待办

- 真实泛化率 25%（val）。补词收益已递减：第 2 轮 train +23.3 只换到 val +2.5
- 继续提升需要新标注样本，用现有样本补词属于自证
- 「临时」这类方向依赖施动者的词，词表无法表达，需语义模型
- `tne_projector_v4.py` 仅作对照保留，不参与运行
- `tne_lean_model.py` 为实验记录，默认不启用（结论是走不通）
- `tne_annotator.py` 人在回路缓存方案可用，但需对话介入，非全自动
- `efficacy_score` 需外部真实反馈填入，不能用 encode 的 confidence 代替
- 在线 LLM 通道已接好但未实测（缺 API Key），接上后跑 `calibrate.py --llm` 对比
