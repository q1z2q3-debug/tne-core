"""
TNE-Core · TrainedEncoderJudger

用阶段二自训的 encode 模型（MiniLM embedding + 九维分类头）替代关键词法，
对任意任务文本判断九维三态向量。

- 不再依赖关键词表（v3.2 关键词法 val 25% 天花板）
- 对未见过的新任务也能判断（每维准确率 ~79%，2.4 倍随机基线）
- 本地运行，不依赖外部 API

用法：
    judger = TrainedEncoderJudger()
    vector = judger.judge("用户投诉产品有严重问题")
"""
from __future__ import annotations
import os
import sys
from typing import List

# 确保 src 在路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
from transformers import AutoTokenizer, AutoModel

from tne.projector.experiments.train_teacher_model import Encoder, STATE

MODEL_PATH = "/home/user/.cache/minilm_local"
HEAD_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data",
                         "trained_encoder.pt")


class TrainedEncoderJudger:
    """
    基于自训 encode 模型的任务 → 九维向量判断器。
    """

    def __init__(self, model_path: str = MODEL_PATH,
                 head_path: str = HEAD_PATH, max_len: int = 64):
        self.max_len = max_len
        # 加载 MiniLM + 分类头
        self.tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.base = AutoModel.from_pretrained(model_path, local_files_only=True)
        self.model = Encoder(self.base)
        head = torch.load(head_path, map_location="cpu")["head"]
        self.model.load_state_dict(head, strict=False)
        self.model.eval()
        # 记录用的向量缓存（同任务避免重复前向）
        self.cache: dict = {}

    def predict(self, texts: List[str]) -> List[List[int]]:
        """批量：文本 → 九维向量。"""
        enc = self.tok(texts, padding=True, truncation=True,
                       max_length=self.max_len, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(enc.input_ids, enc.attention_mask)
            pred = logits.argmax(-1).numpy()
        return [[STATE[int(v)] for v in row] for row in pred]

    def judge(self, task: str) -> List[int]:
        """单条：任务文本 → 九维三态向量。"""
        if task in self.cache:
            return self.cache[task]
        vec = self.predict([task])[0]
        self.cache[task] = vec
        return vec

    def __call__(self, task: str) -> List[int]:
        """使实例可调用（作为 vector_judger 传入 agent，保留 judge_with_confidence）。"""
        return self.judge(task)

    def judge_with_confidence(self, task: str) -> dict:
        """返回向量 + 每维置信度（用于可审计）。"""
        enc = self.tok([task], padding=True, truncation=True,
                       max_length=self.max_len, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(enc.input_ids, enc.attention_mask)
            probs = torch.softmax(logits, dim=-1)[0].numpy()
            pred = logits.argmax(-1)[0].numpy()
        vec = [STATE[int(v)] for v in pred]
        conf = [round(float(probs[d][int(pred[d])]), 3) for d in range(9)]
        return {"vector": vec, "confidence": conf}


def vec_to_symbols(vec: List[int]) -> str:
    """向量 → 符号串（+ / 0 / -）。"""
    return "".join("+" if v == 1 else "0" if v == 0 else "-" for v in vec)


if __name__ == "__main__":
    judger = TrainedEncoderJudger()
    tests = [
        "用户投诉产品有严重问题",
        "从零开发一款独立游戏",
        "写一首关于春天的词",
        "学习如何做数据分析",
        "解决两个部门之间的长期矛盾",
        "团队内部互相推诿，没人愿意干活",
        "公司连续亏损，不知道该不该继续",
        "按时上线了重构后的核心模块",
        "业务进入爆发式增长期",
        "静心打坐，安顿身心",
    ]
    print("=== 自训 encode 模型判断 ===")
    for t in tests:
        vec = judger.judge(t)
        print(f"  「{t[:22]}」→ {vec_to_symbols(vec)}  {vec}")
