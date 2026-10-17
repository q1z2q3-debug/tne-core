#!/usr/bin/env python3
"""
TNE-Core 真实 LLM 认知循环演示
使用 DoubaoJudger（真实大模型判断的九维向量）跑 30 个多样化任务。
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from tne.agent.client import TNEApiClient
from tne.agent.llm_agent import make_doubao_agent

TASKS = [
    # 投资金融
    "帮我做一份新能源行业的投资分析报告",
    "评估这个创业项目的风险等级",
    "从科技股中选出3只最有潜力的选股",
    "设计一个量化交易策略的回测框架",
    "给我做一份50万资金的资产配置方案",
    # 创作文学
    "写一首关于秋天的七言绝句",
    "写一篇关于人工智能未来的深度文章",
    "创作一个短篇科幻故事，主题是时间旅行",
    "把这些诗词按年龄段分类整理",
    "为新产品写一句吸引人的营销文案",
    # 技术工程
    "用Python写一个快速排序的实现",
    "审查这段代码有没有安全漏洞",
    "设计一个高并发消息队列的系统架构",
    "帮我调试这个内存泄漏的问题",
    "画出微服务架构的整体拓扑图",
    # 医疗健康
    "根据症状描述做中医辨证分析",
    "患者持续低烧一周，可能的诊断方向",
    "设计一个焦虑情绪的自我疗愈方案",
    "制定一份适合中年人的养生计划",
    "分析这个心理咨询案例的干预策略",
    # 决策管理
    "为公司制定未来三年的战略规划",
    "如何管理一个跨部门的远程团队",
    "调解两个团队之间的资源冲突",
    "服务器宕机了，写一份危机处理预案",
    "对本季度的工作做一次全面复盘",
    # 学习研究
    "帮我制定一个学习机器学习的路线图",
    "研究一下三元九维符号体系的数学基础",
    "把这篇英文论文翻译成中文",
    "总结一下这本书的核心观点",
    "搜索最新的大模型认知架构相关论文",
]

def main():
    api = TNEApiClient("http://127.0.0.1:8765")
    agent = make_doubao_agent(api, session_id="doubao-real-01")
    print(f"Session: {agent.session_id}")
    print(f"Running {len(TASKS)} real LLM cognition tasks...\n")

    results = agent.run_batch(TASKS, enable_search=True, enable_transform=False)

    # 统计
    from collections import Counter
    classes = set(r["class_rep"] for r in results)
    guas = set(r["gua_idx"] for r in results)
    modes = Counter(r["habit_mode"] for r in results)
    outcomes = Counter(r["outcome"] for r in results)

    print(f"\n{'='*60}")
    print(f"真实 LLM 认知循环结果（{len(results)} 轮）")
    print(f"{'='*60}")
    print(f"覆盖卦位: {len(guas)} / 19683")
    print(f"覆盖等价类: {len(classes)} / 5002")
    print(f"习惯模式分布: {dict(modes)}")
    print(f"结果分布: {dict(outcomes)}")

    # 打印每轮详情
    print(f"\n{'='*60}")
    print("每轮详情")
    print(f"{'='*60}")
    for i, r in enumerate(results, 1):
        vec = "".join(["+" if v==1 else "0" if v==0 else "-" for v in r["vector"]])
        print(f"{i:2d}. [{vec}] 卦#{r['gua_idx']:5d} 类#{r['class_rep']:5d} "
              f"| {r['habit_mode']:10s} (freq={r['habit_freq']}) "
              f"| {r['task'][:30]}")

    # 保存
    with open("/tmp/tne_real_results.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: /tmp/tne_real_results.json")

if __name__ == "__main__":
    main()
