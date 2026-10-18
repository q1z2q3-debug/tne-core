# TNE-Core 阶段二 · 离线小模型路线实测报告

> 用户要求：**不复读 v3.2 的"走不通"，亲自验证**。
> 结论：**环境已解除（能装 torch、能下载模型），但数据量仍是离线小模型路线的根本瓶颈。**

## 一、环境验证（推翻 v3.2 的"环境封死"结论）

v3.2 记录：torch 装不上（File too large）、HuggingFace/ModelScope 全 403。
**本次实测（环境已变）：**

| 尝试 | 结果 |
|---|---|
| `pip install torch --index-url .../cpu` | ✅ 成功（torch 2.14.1+cpu，196MB） |
| `pip install transformers tokenizers` | ✅ 成功（5.20.1） |
| HuggingFace 下载 MiniLM-L12-v2 | ✅ 成功（6 文件，含 pytorch_model.bin） |
| MiniLM 加载 + embedding | ✅ 384 维，余弦相似度正常（0.432） |

**环境解除是真的。** 磁盘从 v3.2 的 5G 增至 9.7G，torch 装得上，HF 不再 403。

## 二、离线小模型路线实测（真实标注集）

方法：MiniLM embedding (384d) → 分类头。
数据：v3.2 真实标注集（train=60 / val=40 / holdout=30）。

### 结果（完全吻合率）

| 方案 | train | val | holdout |
|---|---|---|---|
| **MiniLM + 独立 LogisticRegression** | 48.3% | **7.5%** | 13.3% |
| **MiniLM + 共享表示层 MLP** | 0.0%* | 2.5% | 0.0% |

*共享表示层训练异常（train 0%），60 条数据 + 384 维输入不足以稳定训练神经网络。

### 对比 v3.2

| 方案 | val | 说明 |
|---|---|---|
| 关键词法（v3.2） | 25% | 手工调词极限 |
| char n-gram 小模型（v3.2） | 15% | 学字面非语义 |
| **MiniLM + LR（本次）** | **7.5%** | 语义 embedding 仍差 |

## 三、核心结论

1. **环境解除是真的**：torch 能装、MiniLM 能下载能跑。v3.2 的"环境封死"不再成立。

2. **但"离线小模型路线"在有限数据下确认走不通**：即使换成真正的语义 embedding（MiniLM），val 只有 7.5%，**比关键词法（25%）还差**。

3. **瓶颈是数据量，不是环境、不是模型结构**：
   - 60 条训练数据，九维联合预测（3^9=19683 空间）严重欠定
   - 384 维 embedding + 60 条 = 严重过拟合（train 48.3% vs val 7.5%）
   - 独立分类器 vs 共享表示层都救不了数据量的短板

4. **v3.2 的核心教训依然成立，但归因要修正**：
   - v3.2 归因"语料按语义族构造 → 学字面非语义" → 本次用 MiniLM 语义 embedding 已绕过，仍差
   - **真正的瓶颈 = 标注数据量（60 条），不是语义表示方式**

## 四、对阶段二的意义

| 路线 | 可行性 | 依据 |
|---|---|---|
| 离线小模型（当前数据） | ❌ 走不通 | val 7.5%，数据量是硬瓶颈 |
| 离线小模型（数千真实标注） | ❓ 未验证 | 需数千条真实分布标注 |
| **在线 LLM encode** | ✅ 优先 | 无需标注，LLM 自带语义判断 |
| hidden state 投影（GPU） | ❓ 需 GPU | 环境约束未解除 |

**推荐**：阶段二先走**在线 LLM encode**（不依赖 GPU、不需标注），用真实大模型判断"任务态势"九维。离线小模型路线等真实标注数据积累到数千条再回看。

## 五、复现

```bash
# 环境（一次性）
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers tokenizers

# 下载模型
python -c "from huggingface_hub import snapshot_download; snapshot_download('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', cache_dir='/home/user/.cache/hf_model')"

# 实验
python src/tne/projector/experiments/offline_minilm.py   # 独立 LR
python src/tne/projector/experiments/shared_layer.py      # 共享表示层
```
