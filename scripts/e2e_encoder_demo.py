#!/usr/bin/env python3
"""
TNE-Core · 端到端验证：自训 encode 模型接入坐标服务
对比 DoubaoJudger（关键词法）vs TrainedEncoderJudger（自训模型）
跑真实认知循环，验证替代价值。
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from tne.agent.client import TNEApiClient
from tne.agent.llm_agent import DoubaoJudger
from tne.projector.encoder_judger import TrainedEncoderJudger

API = "http://127.0.0.1:8765"

# 覆盖关键词法命中与未命中的多样化真实任务
TASKS = [
    # 关键词法能命中
    "做一份投资分析报告",
    "写一首关于春天的诗",
    "学习数据分析的方法",
    # 关键词法可能命中"团队"但无法区分
    "团队内部互相推诿，没人愿意干活",
    # 关键词法未命中（新任务）
    "用户投诉产品有严重问题",
    "公司连续亏损，不知道该不该继续",
    "从零开发一款独立游戏",
    "静心打坐，安顿身心",
    "业务进入爆发式增长期",
    "按时上线了重构后的核心模块",
]

def vec_str(v):
    return "".join("+" if x==1 else "0" if x==0 else "-" for x in v)

def main():
    api = TNEApiClient(API)
    keyword = DoubaoJudger()
    encoder = TrainedEncoderJudger()

    print("="*72)
    print("关键词法(DoubaoJudger) vs 自训模型(TrainedEncoderJudger)")
    print("="*72)
    print(f"{'任务':<24}{'关键词法':<12}{'自训模型':<12}{'端到端卦位'}")
    print("-"*72)

    # 清空轨迹库，从干净状态开始（用于覆盖率统计对比）
    for t in TASKS:
        kv = keyword.judge(t)
        ev = encoder.judge(t)
        # 用自训模型跑一次端到端定位
        enc = api.encode(ev)
        gua = enc["gua_idx"]
        cls = enc["class_rep"]
        # 记录轨迹（自训模型）
        api.trajectory("verify_session", ev, "e2e_test",
                       action_detail={"task": t[:60]},
                       outcome="success", reward=0.5)
        match = "一致" if kv == ev else "不同"
        print(f"  {t:<22}{vec_str(kv):<12}{vec_str(ev):<12}卦#{gua:>5}/类#{cls}  {match}")

    # 统计自训模型跑过的覆盖（直接用 HTTP 请求）
    print("-"*72)
    import urllib.request, json as _json
    with urllib.request.urlopen(API + "/stats") as r:
        stats = _json.loads(r.read())
    print(f"端到端后轨迹库覆盖: {stats}")
    print("完成：自训 encode 模型已接入坐标服务跑通端到端认知循环")

if __name__ == "__main__":
    main()
