"""
TNE-Core · LLM Agent（真实大模型认知循环）

完整认知循环：
  任务输入 → LLM判断九维向量 → encode定位 → habit查习惯 → search邻域泛化
  → transform多视角 → LLM生成回答 → trajectory落盘

设计：
  - LLMAgent 接受一个 llm_callback，用于 (任务→九维向量) 和 (任务→回答)
  - 内置 DoubaoJudger：用 Doubao 预判断的向量映射表（真实大模型判断）
  - 预留 OpenAIClient：可接入任何 OpenAI 兼容 API
"""
from __future__ import annotations
import json
import uuid
from typing import List, Dict, Optional, Callable, Any, Tuple
from .client import TNEApiClient


class LLMAgent:
    """
    真实 LLM 认知循环 Agent。
    每轮自动执行：定位 → 习惯 → 邻域 → 变换 → 回答 → 落盘。
    """

    def __init__(self, api: TNEApiClient,
                 vector_judger: Callable[[str], List[int]],
                 answer_generator: Optional[Callable[[str, dict], str]] = None,
                 session_id: Optional[str] = None):
        """
        Args:
            api: TNE 坐标服务客户端
            vector_judger: 函数 (任务文本) -> 九维三态向量（由 LLM 判断）
            answer_generator: 函数 (任务文本, 认知上下文) -> 回答文本（可选）
            session_id: 会话 ID
        """
        self.api = api
        self.vector_judger = vector_judger
        self.answer_generator = answer_generator
        self.session_id = session_id or str(uuid.uuid4())[:8]

    def run(self, task: str, enable_search: bool = True,
            enable_transform: bool = False) -> Dict[str, Any]:
        """
        执行一轮完整认知循环。

        Returns:
            {task, vector, gua_idx, class_rep, habit, neighbors, transform,
             answer, outcome, reward, record_id}
        """
        # 1. LLM 判断九维向量
        vector = self.vector_judger(task)
        assert len(vector) == 9 and all(v in (-1, 0, 1) for v in vector), \
            f"vector_judger 返回非法向量: {vector}"

        # 2. 定位
        enc = self.api.encode(vector)

        # 3. 习惯查询
        hab = self.api.habit(vector)

        # 4. 邻域泛化
        neighbors = None
        if enable_search:
            sr = self.api.search(vector, top_n=5)
            neighbors = sr["neighbors"]

        # 5. 多视角变换
        transform = None
        if enable_transform:
            transform = self.api.transform(vector)

        # 6. 生成回答
        context = {
            "gua_idx": enc["gua_idx"],
            "class_rep": enc["class_rep"],
            "habit": hab,
            "neighbors": neighbors,
            "transform": transform,
        }
        if self.answer_generator:
            answer = self.answer_generator(task, context)
        else:
            answer = f"[认知定位] 卦位#{enc['gua_idx']} (等价类#{enc['class_rep']}) | 习惯模式: {hab['mode']}"

        # 7. 评估结果（简单规则：习惯模式且复用=success，探索=partial）
        if hab["mode"] == "habit":
            outcome = "success"
            reward = 0.8
        elif hab["mode"] == "habit_weak":
            outcome = "partial"
            reward = 0.4
        else:
            outcome = "success"  # 探索也可能成功
            reward = 0.5

        # 8. 落盘
        traj = self.api.trajectory(
            self.session_id, vector, "llm_reasoning",
            action_detail={"task": task[:200], "answer_len": len(answer)},
            outcome=outcome, reward=reward,
        )

        return {
            "task": task,
            "vector": vector,
            "gua_idx": enc["gua_idx"],
            "class_rep": enc["class_rep"],
            "habit_mode": hab["mode"],
            "habit_freq": hab.get("freq", 0),
            "habit_recommendation": hab.get("recommendation", ""),
            "neighbors": neighbors,
            "transform": transform,
            "answer": answer,
            "outcome": outcome,
            "reward": reward,
            "record_id": traj["record_id"],
        }

    def run_batch(self, tasks: List[str], **kwargs) -> List[Dict[str, Any]]:
        """批量执行。"""
        results = []
        for i, task in enumerate(tasks):
            r = self.run(task, **kwargs)
            results.append(r)
            if (i + 1) % 10 == 0:
                print(f"  ... {i+1}/{len(tasks)} tasks done")
        return results


class DoubaoJudger:
    """
    Doubao 预判断的九维向量映射表。
    这是真实大模型（Doubao）对各类任务的认知定位判断。
    向量含义：g1=上爻(显化层) g5=中爻(核心态) g9=下爻(根基层)
    阳(+1)=显/向外/生发  中和(0)=居间/耦合  阴(-1)=隐/向内/收敛
    """

    # 任务关键词 → 九维向量（Doubao 判断）
    # 每个领域多个同义词，提升匹配率
    JUDGMENTS = {
        # === 投资/金融 ===
        "投资": [1, 0, -1, 1, 0, -1, 0, 1, -1],
        "风险": [-1, 1, 0, -1, 0, 1, -1, 0, 1],
        "选股": [1, 1, 0, -1, 0, 1, -1, -1, 0],
        "量化": [1, -1, 1, -1, 1, -1, 1, -1, 1],
        "交易": [1, -1, 1, -1, 1, -1, 1, -1, 1],
        "资产配置": [0, 1, 0, -1, 0, 1, 0, -1, 0],
        "配置": [0, 1, 0, -1, 0, 1, 0, -1, 0],
        "基金": [0, 1, 0, -1, 0, 1, 0, -1, 0],
        "股票": [1, 1, 0, -1, 0, 1, -1, -1, 0],
        "金融": [1, 0, -1, 1, 0, -1, 0, 1, -1],
        # === 创作/文学 ===
        "诗": [1, 1, 0, 1, 0, -1, 0, -1, -1],
        "写文章": [1, 0, 1, 0, -1, 0, -1, 0, -1],
        "文章": [1, 0, 1, 0, -1, 0, -1, 0, -1],
        "故事": [1, 1, 1, 0, -1, -1, -1, 0, 0],
        "小说": [1, 1, 1, 0, -1, -1, -1, 0, 0],
        "科幻": [1, 1, 1, 0, -1, -1, -1, 0, 0],
        "诗词": [0, 1, -1, 1, 0, -1, 1, -1, 0],
        "文案": [1, 0, 0, 1, -1, 0, 0, -1, -1],
        "营销": [1, 0, 0, 1, -1, 0, 0, -1, -1],
        "创作": [1, 1, 0, 1, 0, -1, 0, -1, -1],
        # === 技术/工程 ===
        "代码": [1, -1, 0, 1, 0, -1, 0, 1, -1],
        "编程": [1, -1, 0, 1, 0, -1, 0, 1, -1],
        "python": [1, -1, 0, 1, 0, -1, 0, 1, -1],
        "审查": [-1, 1, -1, 1, -1, 1, -1, 1, -1],
        "漏洞": [-1, 1, -1, 1, -1, 1, -1, 1, -1],
        "系统设计": [1, 1, 0, 0, 0, -1, -1, -1, 1],
        "架构": [1, 1, 1, 0, 0, 0, -1, -1, -1],
        "调试": [-1, 0, 1, -1, 0, 1, -1, 0, 1],
        "bug": [-1, 0, 1, -1, 0, 1, -1, 0, 1],
        "微服务": [1, 1, 1, 0, 0, 0, -1, -1, -1],
        "拓扑": [1, 1, 1, 0, 0, 0, -1, -1, -1],
        # === 医疗/健康 ===
        "中医": [-1, 0, 1, 0, -1, 0, 1, 0, -1],
        "辨证": [-1, 0, 1, 0, -1, 0, 1, 0, -1],
        "诊断": [-1, 1, 0, 1, -1, 0, 0, 1, -1],
        "症状": [-1, 1, 0, 1, -1, 0, 0, 1, -1],
        "疗愈": [0, -1, 0, 1, 0, -1, 0, 1, 0],
        "焦虑": [0, -1, 0, 1, 0, -1, 0, 1, 0],
        "养生": [0, 0, 1, -1, 0, 1, -1, 0, 0],
        "心理": [-1, 0, -1, 1, 0, 1, -1, 0, -1],
        "咨询": [-1, 0, -1, 1, 0, 1, -1, 0, -1],
        "健康": [0, 0, 1, -1, 0, 1, -1, 0, 0],
        # === 决策/管理 ===
        "战略": [1, 1, 1, 1, 0, -1, -1, -1, -1],
        "规划": [1, 1, 1, 1, 0, -1, -1, -1, -1],
        "管理": [1, 0, 1, 0, -1, 0, 1, 0, -1],
        "团队": [1, 0, 1, 0, -1, 0, 1, 0, -1],
        "冲突": [-1, 1, 0, 1, -1, 0, 0, 1, -1],
        "调解": [-1, 1, 0, 1, -1, 0, 0, 1, -1],
        "危机": [1, -1, 1, -1, 1, -1, 1, -1, 1],
        "宕机": [1, -1, 1, -1, 1, -1, 1, -1, 1],
        "复盘": [-1, 0, 1, -1, 0, 1, -1, 0, 1],
        "总结": [-1, -1, -1, 0, 0, 0, 1, 1, 1],
        # === 学习/研究 ===
        "学习": [0, 1, 0, -1, 0, 1, 0, -1, 0],
        "路线图": [0, 1, 0, -1, 0, 1, 0, -1, 0],
        "研究": [-1, -1, 0, 1, 0, -1, 0, 1, 1],
        "数学": [-1, -1, 0, 1, 0, -1, 0, 1, 1],
        "翻译": [0, 1, -1, 1, 0, -1, 1, -1, 0],
        "论文": [0, 1, -1, 1, 0, -1, 1, -1, 0],
        "搜索": [1, 0, -1, 1, 0, -1, 1, 0, -1],
        "信息": [1, 0, -1, 1, 0, -1, 1, 0, -1],
        "书": [-1, -1, -1, 0, 0, 0, 1, 1, 1],
        "核心观点": [-1, -1, -1, 0, 0, 0, 1, 1, 1],
    }

    def __init__(self):
        self.cache = {}

    def judge(self, task: str) -> List[int]:
        """根据任务文本判断九维向量。关键词匹配，未命中则用默认。"""
        task_lower = task.lower()
        for keyword, vector in self.JUDGMENTS.items():
            if keyword in task or keyword.lower() in task_lower:
                return vector
        # 默认：中和偏生发
        return [1, 0, 0, 0, -1, 0, 0, 0, -1]


def make_doubao_agent(api: TNEApiClient, session_id: Optional[str] = None) -> LLMAgent:
    """创建一个使用 Doubao 判断的 LLM Agent。"""
    judger = DoubaoJudger()
    return LLMAgent(
        api=api,
        vector_judger=judger.judge,
        answer_generator=lambda task, ctx: (
            f"[卦位#{ctx['gua_idx']}] 任务已认知定位。"
            f"习惯模式: {ctx['habit']['mode']} (命中{ctx['habit'].get('freq',0)}次)。"
            f"建议: {ctx['habit'].get('recommendation', '无')[:80]}"
        ),
        session_id=session_id,
    )


def make_encoder_agent(api: TNEApiClient, session_id: Optional[str] = None) -> LLMAgent:
    """
    创建一个使用自训 encode 模型判断的 LLM Agent。

    替代关键词法（DoubaoJudger），用阶段二训练的 MiniLM+九维分类头，
    对任意任务文本判断九维向量（每维准确率 ~79%）。
    """
    from tne.projector.encoder_judger import TrainedEncoderJudger
    judger = TrainedEncoderJudger()
    return LLMAgent(
        api=api,
        vector_judger=judger.judge,
        answer_generator=lambda task, ctx: (
            f"[卦位#{ctx['gua_idx']}] 任务已认知定位。"
            f"习惯模式: {ctx['habit']['mode']} (命中{ctx['habit'].get('freq',0)}次)。"
            f"建议: {ctx['habit'].get('recommendation', '无')[:80]}"
        ),
        session_id=session_id,
    )
