#!/usr/bin/env python3
"""
TNE-Core · A2 实验：轨迹积累 → 习惯偏置涌现

用自训 encode 模型跑真实任务，积累轨迹库，验证"习惯从轨迹中涌现"。

设计：
- 8 个认知场景，每场景 10 条同义不同表述的真实任务
- 每场景绑定一个行为动作（认知策略）
- 同一场景的任务映射到相近/相同等价类，积累 ≥3 次 → 习惯偏置涌现

验证指标：
- 等价类覆盖
- 每场景等价类命中频次
- 涌现的 habit 模式数量 / 习惯复用率
"""
import os, sys, urllib.request, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from tne.agent.client import TNEApiClient
from tne.projector.encoder_judger import TrainedEncoderJudger
from tne.space.space import encode as _enc
from tne.space.equivalence import class_of

API = "http://127.0.0.1:8765"

# 认知场景 → (行为动作, 同义任务列表)
SCENARIOS = {
    "危机投诉": ("crisis_response", [
        "用户投诉产品有严重问题",
        "客户集体反馈质量不过关",
        "产品被曝出安全故障，用户不满",
        "用户大量投诉体验糟糕",
        "消费者对产品质量强烈不满",
        "客户投诉我们服务出了大问题",
        "用户反馈产品存在严重缺陷",
        "产品质量遭到大量用户投诉",
        "客户服务出现问题，引发投诉",
        "产品出状况了，用户很不满",
    ]),
    "从零创造": ("creative_build", [
        "从零开始做一个全新产品",
        "白手起家创立一家新公司",
        "从无到有研发一款新应用",
        "独立启动一个全新的创意项目",
        "从零打造一个原创品牌",
        "全新出发，做一个没人做过的东西",
        "从零搭建一套全新的业务体系",
        "无中生有地创造一个新品类",
        "从零起步孵化一个创新项目",
        "全新的起点，开创一份新事业",
    ]),
    "团队协作": ("team_repair", [
        "解决两个部门之间的长期矛盾",
        "团队内部互相推诿，没人愿意干活",
        "跨部门协作彻底停摆",
        "团队成员内耗严重，需要调和",
        "两个小组因为抢资源闹翻了",
        "团队士气低落，配合出了问题",
        "部门之间互相扯皮，效率低下",
        "协调团队内部的分歧和冲突",
        "把内耗的团队重新拧成一股绳",
        "化解团队之间的积怨和隔阂",
    ]),
    "方向迷茫": ("strategic_rethink", [
        "公司连续亏损，不知道该不该继续",
        "做了三年没起色，怀疑这条路对不对",
        "行业要变天了，我们该往哪走",
        "越来越不知道项目意义在哪",
        "两个方向都有道理，但看不到未来",
        "团队士气涣散，大家看不到盼头",
        "没有清晰的战略，四处乱撞",
        "公司经营困难，方向越来越模糊",
        "项目做久了，价值感在流失",
        "前路迷茫，不知道该怎么转型",
    ]),
    "学习研究": ("focused_study", [
        "学习如何做数据分析",
        "系统研究一个新领域",
        "深入钻研某个专业课题",
        "读书学习，提升自己",
        "研究市场趋势，形成判断",
        "系统性学习一套新技能",
        "深入研究一个行业问题",
        "学习数据分析的方法",
        "钻研新技术，武装自己",
        "研究竞品的成功规律",
    ]),
    "上升增长": ("scale_up", [
        "业务进入爆发式增长期",
        "公司正处上升通道，越做越顺",
        "抓住窗口期把市场份额做大",
        "人气正旺，加快上新节奏",
        "势头好，趁热打铁推新品",
        "增长快，赶紧补产能和团队",
        "业务曲线持续上扬",
        "发展驶入快车道",
        "正处于发展最快的时期",
        "乘势扩大规模，加速扩张",
    ]),
    "日常维持": ("steady_ops", [
        "按时上线了重构后的核心模块",
        "按流程走完这个季度的例行审计",
        "把现有客户关系维护好",
        "保障系统按既定节奏正常交付",
        "做好每天的运营值守",
        "推进团队例会的正常运转",
        "把现有的流程优化到更顺手",
        "保持产品当前的水准不滑坡",
        "稳步执行已定好的年度计划",
        "按部就班推进常规工作",
    ]),
    "静心收敛": ("withdraw_recharge", [
        "静心打坐，安顿身心",
        "喝茶放空，回归平静",
        "规律作息，养精蓄锐",
        "给身心做个断舍离",
        "慢下来，重新感受生活",
        "闭关休整，调整状态",
        "放慢节奏，好好休息",
        "独处冥想，恢复能量",
        "停下来，给自己充电",
        "隐居静养，蓄势待发",
    ]),
}

def vec_str(v):
    return "".join("+" if x==1 else "0" if x==0 else "-" for x in v)

def main():
    api = TNEApiClient(API)
    judger = TrainedEncoderJudger()
    session = "a2_habit_exp"

    # 清空本次实验会话的历史（保留已有）
    results = {}   # scenario -> {class_rep, gua, freq}
    all_tasks = []

    print("="*74)
    print("A2 实验：8 场景 × 10 同义任务 → 轨迹积累 → 习惯涌现")
    print("="*74)
    for scen, (action, tasks) in SCENARIOS.items():
        classes = {}
        for task in tasks:
            vec = judger.judge(task)
            idx = _enc(vec)
            rep = class_of(idx)
            classes[rep] = classes.get(rep, 0) + 1
            api.trajectory(session, vec, action,
                           action_detail={"scenario": scen, "task": task[:40]},
                           outcome="success", reward=0.8)
        # 该场景的主等价类
        top_rep, top_freq = max(classes.items(), key=lambda x: x[1])
        results[scen] = {"top_rep": top_rep, "top_freq": top_freq,
                         "n_classes": len(classes), "action": action}
        print(f"\n  【{scen}】动作={action}")
        print(f"    映射到 {len(classes)} 个等价类, 主类#{top_rep} 命中{top_freq}次")

    # 习惯偏置涌现验证
    print("\n" + "="*74)
    print("习惯偏置涌现验证（min_freq=3, 成功率≥0.6）")
    print("="*74)
    habit_count = 0
    for scen, r in results.items():
        vec = judger.judge(SCENARIOS[scen][1][0])  # 用场景代表任务
        hab = api.habit(vec)
        mode = hab["mode"]
        freq = hab["freq"]
        if mode in ("habit", "habit_weak"):
            habit_count += 1
            mark = "✅习惯" if mode == "habit" else "⚠️弱习惯"
        else:
            mark = "○探索"
        print(f"  {scen:<6} 模式={mode:<12} 命中{freq:>3}次 {mark}")
        if mode == "habit":
            print(f"         ↳ {hab['recommendation'][:70]}")

    # 汇总
    total = len(SCENARIOS)
    print("\n" + "="*74)
    print(f"场景总数 {total} | 涌现习惯偏置 {habit_count} 个（含弱习惯）")
    print(f"习惯涌现率: {habit_count}/{total}")

    # 轨迹库最终覆盖
    with urllib.request.urlopen(API + "/stats") as resp:
        stats = json.loads(resp.read())
    print(f"轨迹库: {stats['total_records']} 条记录 / 覆盖 {stats['covered_classes']} 等价类")
    print("完成：验证习惯偏置是否从轨迹中涌现")

if __name__ == "__main__":
    main()
