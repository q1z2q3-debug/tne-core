"""
TNE-Core · 内化蒸馏解码器 Judger
封装 distill_zh_best.pt（中文 GPT2，并行输出九维态），
接口与 TrainedEncoderJudger 完全一致：
  - judge(task) -> List[int]             九维三态向量
  - judge_with_confidence(task) -> dict  向量 + 置信度
  - __call__(task)                       使实例可调用（作为 vector_judger）
这是"内化模式"：模型推理时自带九维输出，不再依赖外挂 encode 模型。
"""
from __future__ import annotations
import os, torch
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from transformers import GPT2LMHeadModel, AutoTokenizer

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
MODEL_NAME = "uer/gpt2-chinese-cluecorpussmall"
CKPT = os.path.join(BASE, "tmp", "distill_zh_best.pt")
INSTR = "判断以下情境在九维三态下的认知坐标\n任务："
OUT = "九维向量："

# 爻位语义（g1..g9）
SEM = ["愿景", "决断", "产出", "方法", "主态", "协作", "输入", "积累", "根性"]
STATE = {-1: 0, 0: 1, 1: 2}


class DistillDecoderJudger:
    """内化蒸馏解码器：文本 → 九维三态向量（模型并行输出）。"""

    def __init__(self, ckpt: str = CKPT, confidence: float = 0.75):
        self.conf_fallback = confidence
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
        self.model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        self.model.eval()
        self.cache = {}

    def _decode(self, task: str) -> str:
        if task in self.cache:
            return self.cache[task]
        src = INSTR + task + "\n" + OUT
        with torch.no_grad():
            inp = self.tokenizer.encode(src, return_tensors="pt")
            out = self.model.generate(
                input_ids=inp, max_new_tokens=16, do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id)
            gen = self.tokenizer.decode(out[0][inp.shape[1]:], skip_special_tokens=True)
        code = "".join(c for c in gen if c in "+0-")[:9]
        # 非法则回退全 0（势能平衡），标注置信度低
        self.cache[task] = code if len(code) == 9 else "000000000"
        return self.cache[task]

    def judge(self, task: str) -> list:
        code = self._decode(task)
        return [1 if c == "+" else -1 if c == "-" else 0 for c in code]

    def judge_with_confidence(self, task: str) -> dict:
        code = self._decode(task)
        vec = [1 if c == "+" else -1 if c == "-" else 0 for c in code]
        valid = len(code) == 9 and all(c in "+0-" for c in code)
        conf = self.conf_fallback if valid else 0.33
        return {"vector": vec, "confidence": [conf] * 9,
                "decode": code, "valid": valid}

    def __call__(self, task: str) -> list:
        return self.judge(task)

    def __repr__(self):
        return "<DistillDecoderJudger 内化解码器>"
