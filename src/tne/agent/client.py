"""
TNE-Core · Agent 客户端
封装坐标服务 API，提供"定位→习惯→决策→落盘"的认知循环。

支持两种模式：
  - SimulatedAgent：基于规则的模拟 Agent（无 LLM，用于批量积累轨迹）
  - LLMAgent：预留真实 LLM 接口（需配置 API key）

每轮自动执行：
  1. encode 定位（将处境映射为九维向量）
  2. habit 查询习惯偏置
  3. 决策（模拟/LLM）
  4. trajectory 落盘
"""
from __future__ import annotations
import json
import random
import uuid
from typing import List, Dict, Optional, Any
import urllib.request


class TNEApiClient:
    """TNE-Core 坐标服务 HTTP 客户端。"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")

    def _post(self, path: str, data: dict) -> dict:
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(data).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())

    def encode(self, vector: List[int]) -> dict:
        return self._post("/encode", {"vector": vector})

    def search(self, vector: List[int], top_n: int = 10) -> dict:
        return self._post("/search", {"vector": vector, "top_n": top_n})

    def transform(self, vector: List[int], moving_yaos: Optional[List[int]] = None) -> dict:
        return self._post("/transform", {"vector": vector, "moving_yaos": moving_yaos})

    def habit(self, vector: List[int]) -> dict:
        return self._post("/habit", {"vector": vector})

    def trajectory(self, session_id: str, vector: List[int], action: str,
                   action_detail: Optional[dict] = None,
                   outcome: Optional[str] = None, reward: float = 0.0) -> dict:
        return self._post("/trajectory", {
            "session_id": session_id, "vector": vector, "action": action,
            "action_detail": action_detail, "outcome": outcome, "reward": reward,
        })


class SimulatedAgent:
    """
    基于规则的模拟 Agent。
    将预设情境映射为九维向量，根据习惯偏置做决策，记录轨迹。
    用于批量积累轨迹库，验证等价类覆盖率与行为收敛性。
    """

    # 预设情境模板：(情境名, 九维向量, 可选行为列表)
    SCENARIOS = [
        ("新领域探索", [1, 0, -1, 1, 0, -1, 1, 0, -1], ["search_web", "ask_clarify", "try_tool", "defer"]),
        ("稳定执行", [1, 1, 1, 0, 0, 0, -1, -1, -1], ["execute_plan", "reuse_skill", "monitor", "finalize"]),
        ("冲突调解", [-1, 1, 0, 1, -1, 0, 0, 1, -1], ["mediate", "gather_info", "propose_compromise", "escalate"]),
        ("危机响应", [1, -1, 1, -1, 1, -1, 1, -1, 1], ["immediate_action", "assess_damage", "notify", "contingency"]),
        ("学习积累", [0, 1, 0, -1, 0, 1, 0, -1, 0], ["read_doc", "take_notes", "practice", "summarize"]),
        ("决策犹豫", [0, 0, 0, 1, -1, 0, -1, 1, 0], ["list_options", "weigh_pros", "consult", "defer_decision"]),
        ("创意发散", [1, 1, 0, 1, 0, -1, 0, -1, -1], ["brainstorm", "analogy", "sketch", "combine"]),
        ("深度聚焦", [-1, -1, -1, 0, 0, 0, 1, 1, 1], ["deep_work", "eliminate_noise", "iterate", "review"]),
        ("社交协作", [1, 0, 1, 0, -1, 0, 1, 0, -1], ["sync", "delegate", "feedback", "celebrate"]),
        ("复盘反思", [-1, 0, 1, -1, 0, 1, -1, 0, 1], ["log", "analyze", "extract_lesson", "plan_next"]),
        ("资源紧张", [-1, 1, -1, 1, -1, 1, -1, 1, -1], ["prioritize", "cut_scope", "negotiate", "improvise"]),
        ("机遇窗口", [1, 1, 1, 1, 0, -1, -1, -1, -1], ["seize", "scale", "pitch", "lock_in"]),
    ]

    def __init__(self, api: TNEApiClient, seed: int = 42):
        self.api = api
        self.rng = random.Random(seed)
        self.session_id = str(uuid.uuid4())[:8]

    def _pick_scenario(self) -> tuple:
        return self.rng.choice(self.SCENARIOS)

    def _decide(self, vector: List[int], habit_info: dict, actions: List[str]) -> tuple:
        """根据习惯偏置决策。返回 (action, outcome, reward)。"""
        mode = habit_info.get("mode", "explore")
        top = habit_info.get("top_actions", [])

        if mode == "habit" and top:
            # 习惯模式：80% 概率复用历史最优行为，20% 探索
            if self.rng.random() < 0.8:
                action = top[0]["action"]
                # 习惯行为成功率较高
                outcome = "success" if self.rng.random() < 0.75 else "partial"
                reward = 0.8 if outcome == "success" else 0.3
                return action, outcome, reward

        # 探索模式：随机选行为
        action = self.rng.choice(actions)
        # 探索行为成功率较低
        r = self.rng.random()
        if r < 0.5:
            outcome, reward = "success", 0.6
        elif r < 0.8:
            outcome, reward = "partial", 0.2
        else:
            outcome, reward = "failure", -0.3
        return action, outcome, reward

    def run_round(self) -> Dict[str, Any]:
        """执行一轮认知循环。返回本轮记录。"""
        scenario_name, vector, actions = self._pick_scenario()

        # 1. 定位
        enc = self.api.encode(vector)
        # 2. 习惯查询
        hab = self.api.habit(vector)
        # 3. 决策
        action, outcome, reward = self._decide(vector, hab, actions)
        # 4. 落盘
        traj = self.api.trajectory(
            self.session_id, vector, action,
            action_detail={"scenario": scenario_name},
            outcome=outcome, reward=reward,
        )
        return {
            "scenario": scenario_name,
            "gua_idx": enc["gua_idx"],
            "class_rep": enc["class_rep"],
            "habit_mode": hab["mode"],
            "action": action,
            "outcome": outcome,
            "reward": reward,
            "record_id": traj["record_id"],
        }

    def run_batch(self, n_rounds: int = 1000) -> List[Dict[str, Any]]:
        """批量执行 n 轮。"""
        results = []
        for i in range(n_rounds):
            r = self.run_round()
            results.append(r)
            if (i + 1) % 200 == 0:
                print(f"  ... {i+1}/{n_rounds} rounds done")
        return results
