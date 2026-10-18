"""TNE-Core v3.1 - Projector V5

双通道 encode：
  1) LocalSemanticBackend —— 由 LLM 离线标注固化而来的九维语义判定器（默认，零依赖、零延迟）
  2) LLMBackend          —— OpenAI 兼容协议的在线 encode（配了 TNE_LLM_* 环境变量自动启用，失败回落本地）

相对 V4 修掉的三个硬伤：
  - V4 的 9 个维度里有 8 个只能输出 +1/0，-1 永远不可达 → 3^9 空间实际只剩 2^9
  - dim6 用「你/我/我们」判定，几乎任何中文句都命中 → 该维恒为 1，零区分度
  - dim4 与 dim5 共用「优化」，语义耦合
"""

import json
import os
import re
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple


# ---------------------------------------------------------------------------
# 九维语义词表
#
# 这份词表不是拍脑袋的关键词堆砌：它是以 DIMENSION_ANCHORS 为判定基准，对
# 每一维的 +1 / 0 / -1 三种锚点做语义展开得到的加权信号集。
#   weight 1.0 = 强信号（出现即基本可定维）
#   weight 0.5 = 弱信号（需要与其他信号叠加）
# 同一组内的词只计一次，避免「优化 / 优化一下」重复累加。
# ---------------------------------------------------------------------------

DIMENSION_ANCHORS = {
    0: {"name": "天·性", "+1": "创造、生成、从无到有", "0": "维持现状", "-1": "破坏、损耗"},
    1: {"name": "天·习", "+1": "习惯延续、自动化", "0": "首次尝试", "-1": "习惯中断"},
    2: {"name": "天·运", "+1": "外部驱动、紧迫", "0": "平稳", "-1": "外部阻力"},
    3: {"name": "地·性", "+1": "环境切换、新场景", "0": "环境不变", "-1": "环境退化"},
    4: {"name": "地·习", "+1": "工具/方法运用", "0": "无工具", "-1": "工具缺失"},
    5: {"name": "地·运", "+1": "状态转变、升级", "0": "状态稳定", "-1": "状态降级"},
    6: {"name": "人·性", "+1": "关系、协作", "0": "独立", "-1": "孤立、冲突"},
    7: {"name": "人·习", "+1": "自律、内省", "0": "无自律", "-1": "放纵、失控"},
    8: {"name": "人·运", "+1": "内在驱动、热情", "0": "无驱动", "-1": "内在耗竭"},
}

# 每维: {"pos": [(词表, 权重), ...], "neg": [(词表, 权重), ...]}
NEGATORS_STRICT = ["不", "没", "无", "未", "非", "并非", "谈不上", "停止", "放弃", "拒绝", "不再"]
NEGATORS_ADJACENT = ["别", "甭"]
# 「别」作词素时的前接字：特别 / 区别 / 级别 / 类别 ... 这些不是否定
BIE_PREFIX = set("特区级类差个分离告性职识派种")

# 「不」作词素时的后接字：不错 / 不仅 / 不妨 / 不至于 ...
# 「状态不错，效率很高」里的「不」曾被当成否定副词，把后面的「效率」误判为 ~（否定）
NOT_NEGATION = ["不错", "不赖", "不仅", "不但", "不妨", "不免", "不至于",
                "不愧", "不失", "不亚于", "不外乎", "不外", "不啻", "不单",
                "不只", "不光", "不已", "不亦", "不无", "未尝", "未必"]

SATURATION = 1.5  # 信号强度饱和阈值
WEAK_THRESHOLD = 0.34  # 局部置信低于此值记为模糊维

# ---------------------------------------------------------------------------
# 上下文抑制表
#
# 这些表解决的是「话题词 vs 态势词」的问题：提到「接口」「流程」「设计」只说明
# 在谈论这个东西，不等于正在运用它 / 正在创造它。v3.2 把这类词一律按正向信号
# 处理，是 FALSE（误触发）的主要来源。
# ---------------------------------------------------------------------------

# 事故描述：说「接口全挂」是在报案，不是在运用工具，也不是在创造
FAULT_CONTEXT = ["崩了", "挂了", "全挂", "宕机", "故障", "跑不起来",
                 "起不来", "不可用", "坏了", "崩", "有问题", "有缺陷",
                 "缺陷", "坏味道", "烂", "报错", "错误", "泄漏", "重启",
                 "频繁重启", "丢失", "丢了", "没了", "损坏"]

# 停滞描述：久未触碰的东西不是当前在用的工具
IDLE_CONTEXT = ["没动过", "久未", "很久没", "好久没", "三个月没", "几年没",
                "没人维护", "停止维护", "已废弃", "年久"]

# 疑问句式：「帮我看看为什么慢」是求助，不是协作
QUESTION_CONTEXT = ["为什么", "怎么", "如何", "是不是", "能不能", "为何",
                    "哪里出", "什么情况"]

# 行政审批语境：这里的「流程」是组织程序，不是工具方法
ADMIN_CONTEXT = ["审批", "审核", "签字", "报销", "走流程", "流程太长"]

# 强冲突：一出现，「团队/一起」等协作词降级为场景名词
CONFLICT_CONTEXT = ["闹僵", "闹翻", "吵架", "冲突", "撕", "甩锅", "冷战",
                    "对立", "扯皮", "推诿", "被孤立", "没人配合",
                    "没人愿意", "没人愿意接", "没人接", "不愿接", "不愿意",
                    "推卸", "没人可问", "互相推", "一个人扛"]

# 组织架构语境：这里的「模块/负责人」指人和组织，不是技术工具
ORG_CONTEXT = ["负责人", "同事", "小组", "部门", "离职", "入职", "新人",
               "带新人", "换人", "交接"]
# 仅在组织架构语境下被抑制的歧义词。README / 文档 / 代码 等技术名词不受影响
ORG_AMBIGUOUS = ["模块", "负责人", "组件", "接口", "岗位"]

# 显式 0 锚点：命中即把该维钉在中位，优先于任何正负信号
NEUTRAL_ANCHORS: Dict[int, List[List[str]]] = {
    1: [["第一次", "首次", "新手", "初学", "没做过", "头一回", "刚接触", "入门", "初次"]],
}

LEXICON: Dict[int, Dict[str, List[Tuple[List[str], float]]]] = {
    0: {  # 天·性：创造 / 维持 / 破坏损耗
        "pos": [
            (["创造", "创作", "写作", "写", "设计", "实现", "生成",
              "从零", "从无到有", "搭建", "起草", "构思", "新建", "开发",
              "产出", "编写", "画", "谱", "编曲", "孵化", "立项", "开创",
              "新增", "加上", "起名", "起个", "造一个", "做出来", "落地"], 1.0),
            (["做一个", "搞一个", "弄个", "想做个", "整一个", "出个",
              "补一下", "补齐", "补个", "做个", "写成", "做成", "沉淀成",
              "做成产品", "自己的项目", "分享"], 0.5),
            (["故事", "章节", "小说", "剧本", "文案", "稿子", "文章", "诗",
              "歌词", "播客", "短视频", "插画", "漫画", "README",
              "分享"], 0.5),
        ],
        "neg": [
            # 成果作废是最强的损耗信号，权重压过「写」这类创造动词
            (["搞错", "方向错", "白做", "白干", "作废", "废弃掉", "全没了",
              "废掉", "砍掉", "砍了", "砍一半", "被砍", "不做", "不做了",
              "推倒", "推翻重做", "重做了三遍"], 1.5),
            (["删除", "删掉", "删", "破坏", "摧毁", "终止", "废弃", "拆掉",
              "拆除", "移除", "清空", "抹掉", "报废", "关停", "撤销",
              "下线", "回滚掉", "没了", "丢失"], 1.0),
            (["损耗", "消耗", "磨损", "折损", "亏空", "缩水"], 0.5),
        ],
    },
    1: {  # 天·习：习惯延续 / 首次 / 习惯中断
        "pos": [
            (["继续", "再次", "重复", "延续", "照例", "老规矩", "惯常", "例行",
              "定期", "每天", "每周", "常态化", "又来", "还是按", "照旧做",
              "接着做", "接着", "一如既往", "日常", "固定流程", "坚持",
              "留一小时", "固定时间"], 1.0),
            (["很熟", "我熟", "熟门熟路", "轻车熟路", "手到擒来", "闭着眼",
              "三十天", "又", "再来", "下一轮", "第二轮"], 0.5),
        ],
        "neg": [
            (["中断", "断掉", "搁置", "荒废", "半途而废", "停更", "断了",
              "放弃", "破例", "不再", "好久没", "生疏", "撂下", "停摆",
              "断更", "弃坑", "没写", "没碰", "没开始", "全乱了", "半夜"], 1.0),
            (["拖延", "摸鱼", "停滞", "一直没动", "拖了很久", "临时"], 0.5),
        ],
        "neutral": [
            ["第一次", "首次", "新手", "初学", "没做过", "头一回", "刚接触", "入门", "初次"],
        ],
    },
    2: {  # 天·运：外部驱动紧迫 / 平稳 / 外部阻力
        "pos": [
            (["紧急", "马上", "立刻", "催", "截止", "deadline", "明天要交",
              "老板要", "客户催", "加急", "限时", "今晚就要", "火烧眉毛",
              "赶在", "来不及", "优先处理", "插队", "急", "必须", "上线"], 1.0),
            (["尽快", "越快越好", "这周内", "今天内", "临时", "加班",
              "半夜", "明天开始", "这个月必须"], 0.5),
        ],
        "neg": [
            (["受阻", "卡住", "卡着", "被拒", "不通", "阻塞", "等审批", "拖着",
              "阻碍", "资源不够", "被挡", "延期", "迟迟", "批不下来", "被卡",
              "依赖别人", "等别人", "卡在", "没批", "没批下来", "砍",
              "被砍", "对不上", "变卦", "反复变"], 1.0),
            (["不顺", "碰壁", "推不动", "流程慢", "流程太长", "太长",
              "审批", "走流程"], 0.5),
        ],
    },
    3: {  # 地·性：环境切换 / 不变 / 环境退化
        "pos": [
            (["切换", "转到", "回到", "跨", "换个", "迁移", "换环境", "新场景",
              "换个地方", "转岗", "换项目", "换个方向", "搬家", "换个城市",
              "切到", "转入", "转移到", "换到", "改用", "换成", "换人",
              "搬过来", "新框架", "新工具", "新语言", "新项目", "新环境",
              "熟悉环境", "别的仓库", "另一个"], 1.0),
            (["不同", "新地方", "播客", "短视频", "新账号", "新渠道",
              "新领域", "新平台", "新同事", "入职"], 0.5),
        ],
        "neg": [
            (["崩了", "挂了", "宕机", "不可用", "崩坏", "损坏", "失修",
              "环境坏了", "跑不起来", "起不来", "中毒", "被黑", "数据丢了",
              "坏了", "泄漏", "内存泄漏", "磁盘满", "带宽打满", "打满",
              "配置低", "配置太低", "没人维护", "停止维护", "没动过",
              "久未维护", "耦合太重", "耦合重", "设计有问题", "有问题"], 1.0),
            (["退化", "老化", "破败", "混乱不堪", "重启"], 0.5),
        ],
    },
    4: {  # 地·习：工具方法运用 / 无 / 工具缺失
        "pos": [
            (["代码", "测试", "函数", "类", "模块", "脚本", "工具", "框架",
              "git", "python", "命令", "库", "插件", "模板", "方法论",
              "单测", "单元测试", "数据库", "查询", "接口", "api", "脚本化",
              "自动化", "用脚本", "写脚本", "调库", "开个分支", "云环境",
              "服务器", "部署", "docker", "k8s", "SQL", "sql", "bug", "BUG",
              "README", "日志", "缓存", "索引", "错误码", "配置文件",
              "组件", "Makefile", "micrometer", "熔断", "监控", "均衡",
              "响应式", "布局", "首页", "数据", "硬盘", "机器", "缓存策略",
              "跑通"], 1.0),
            (["用法", "手段", "套路", "技巧", "流程", "沉淀", "归档"], 0.5),
        ],
        "neg": [
            (["没工具", "工具缺失", "缺少", "找不到", "没法", "无从下手",
              "裸", "没环境", "没有库", "缺库", "装不上", "没权限",
              "工具不够", "缺少工具", "不知道从哪入手", "不知从哪入手"], 1.0),
            (["手工", "纯手", "手动", "手动改"], 0.5),
        ],
    },
    5: {  # 地·运：状态升级 / 稳定 / 状态降级
        "pos": [
            (["优化", "升级", "改进", "突破", "重构", "提升", "加速", "增强",
              "提高", "翻倍", "提速", "完善", "迭代", "进化", "性能", "效率",
              "更快", "更好", "上一个台阶", "解决", "跑通", "自动化"], 1.0),
            (["统一", "合并", "抽取", "抽成", "收敛", "拆解", "拆成", "细化",
              "填坑", "填平", "清洗", "重建", "固化", "调细", "改成", "提到",
              "覆盖率", "均衡", "熔断", "上线", "修复",
              "打磨", "改善", "调一下"], 0.5),
        ],
        "neg": [
            (["降级", "变慢", "退步", "恶化", "下滑", "性能下降", "变卡",
              "拖慢", "回退", "劣化", "越来越差", "变糟", "退化了",
              "崩了", "挂了", "全挂", "崩坏", "跑不起来", "慢", "很慢",
              "启动慢", "启动很慢", "泄漏", "重启", "频繁重启", "配置低",
              "配置太低", "扛不动", "有问题", "耦合太重", "故障",
              "耦合重", "拖垮", "没动过", "停止维护", "没人维护",
              "被砍", "砍一半"], 1.0),
            (["不稳", "抖动", "时好时坏"], 0.5),
        ],
    },
    6: {  # 人·性：关系协作 / 独立 / 孤立冲突
        "pos": [
            (["协作", "一起", "团队", "合作", "配合", "对接", "商量", "请教",
              "评审", "pair", "共同", "和你", "帮我", "陪我", "拉通", "对齐",
              "同步给", "拉个会", "约个会", "请教一下", "一起搞", "一起做",
              "带带我", "指教", "搭把手", "review", "Review", "确认",
              "对口径", "对一下", "同步", "组决定", "带新人", "新人",
              "负责人", "熟悉环境", "分享"], 1.0),
            (["沟通", "开会", "约了"], 0.5),
        ],
        "neg": [
            (["孤立", "冲突", "吵架", "不合", "甩锅", "对立", "冷战", "争执",
              "没人配合", "被孤立", "分歧", "闹僵", "撕", "扯皮", "推诿",
              "互不理", "闹翻", "没人愿意", "没人接", "推卸责任", "推卸",
              "没人可问", "一个人扛", "扛不动", "单打独斗", "没人管",
              "孤军"], 1.0),
        ],
    },
    7: {  # 人·习：自律内省 / 无 / 放纵失控
        "pos": [
            (["自律", "自主", "推进", "坚持", "复盘", "反思", "内省", "计划",
              "按时", "打卡", "约束", "克制", "总结", "回顾", "自检",
              "按计划", "有节奏", "稳住节奏", "想清楚", "定了",
              "留一小时", "归档", "列清单", "清单", "拆成", "拆解"], 1.0),
            (["整理", "梳理", "沉淀", "一个人做", "效率", "规划", "安排"], 0.5),
        ],
        "neg": [
            (["放纵", "失控", "摆烂", "拖延", "摸鱼", "熬夜", "上瘾", "沉迷",
              "拖延症", "走神", "分心", "颓", "躺平", "刷到停不下来",
              "管不住", "乱了阵脚", "通宵", "浮躁", "静不下心", "没开始"], 1.0),
            (["没规划", "随缘", "想到哪"], 0.5),
        ],
    },
    8: {  # 人·运：内在驱动 / 无 / 内在耗竭
        "pos": [
            (["热爱", "热情", "使命", "意义", "渴望", "动力", "主动", "我愿意",
              "发自", "初心", "愿景", "兴奋", "期待", "很想", "特别想", "来劲",
              "有劲儿", "打心底", "想做成", "笃定", "想做", "想做好", "想把",
              "做成产品", "自己的项目", "有感情"], 1.0),
            (["喜欢", "感兴趣", "好奇", "很熟", "我熟", "状态不错", "效率高",
              "值得", "有成就感"], 0.5),
        ],
        "neg": [
            (["耗竭", "耗尽", "疲惫", "倦怠", "没劲", "心累", "burnout", "枯竭",
              "麻木", "没动力", "提不起", "空虚", "撑不住", "累", "乏力",
              "不想动", "燃尽", "没意思", "扛不住", "扛不动", "半夜"], 1.0),
            (["应付", "凑合", "走过场"], 0.5),
        ],
    },
}

# 否定词分两级，避免「特别来劲」的「别」被当成否定副词的误伤
#   STRICT   —— 允许 3 字窗口内出现即反转
#   ADJACENT —— 必须紧贴关键词前一字（"别""甭" 常作为词素出现在 特别/区别/级别 里）
@dataclass
class EncodeResult:
    vector: List[int]
    confidence: float
    reasoning: str
    ambiguity_flags: List[str] = field(default_factory=list)
    backend: str = "local"          # local | llm | llm_fallback
    dim_scores: List[float] = field(default_factory=list)  # 每维净信号强度，调试用


@dataclass
class DimEvidence:
    dim: int
    pos: float
    neg: float
    hits: List[str]


class LocalSemanticBackend:
    """由 LLM 标注语义固化而来的本地九维判定器。

    算法：
      1. 逐维扫描 pos / neg 词表，同组词只计一次，累加权重
      2. 命中词前 3 字内出现否定词 → 该词信号反转
      3. net = pos - neg；|net| 达阈值则定维，否则归 0
      4. 局部置信 = min(1, |net| / SATURATION)，整体 confidence 由各维局部置信聚合
    """

    def encode(self, user_input: str) -> EncodeResult:
        text = user_input or ""
        vector: List[int] = []
        scores: List[float] = []
        certainties: List[float] = []
        weak_dims: List[str] = []
        evidence: List[DimEvidence] = []

        for dim in range(9):
            ev = self._scan_dim(text, dim)
            evidence.append(ev)
            if self._neutral_hit(text, dim):
                # 框架的 0 锚点（如 d1「首次」）优先级高于任何信号
                vector.append(0)
                certainties.append(0.6)
                scores.append(0.0)
                continue
            net = ev.pos - ev.neg
            if net >= 0.5:
                vector.append(1)
            elif net <= -0.5:
                vector.append(-1)
            else:
                vector.append(0)

            strength = max(ev.pos, ev.neg)
            cert = min(1.0, strength / SATURATION) if strength > 0 else 0.0
            certainties.append(cert)
            scores.append(round(net, 3))
            if cert < WEAK_THRESHOLD:
                weak_dims.append(f"d{dim}{DIMENSION_ANCHORS[dim]['name']}")

        determined = sum(1 for v in vector if v != 0)
        # 全维无信号时 confidence 落到 0.5（完全不确定），信号充分时逼近 0.95
        confidence = 0.5 + 0.45 * (sum(certainties) / 9.0)

        ambiguity = []
        if determined == 0:
            ambiguity.append("全维无有效信号，向量不可信")
        elif weak_dims:
            ambiguity.append(f"弱信号维度: {', '.join(weak_dims)}")

        reasoning = self._explain(vector, evidence)
        return EncodeResult(
            vector=vector,
            confidence=round(confidence, 3),
            reasoning=reasoning,
            ambiguity_flags=ambiguity,
            backend="local",
            dim_scores=scores,
        )

    def _scan_dim(self, text: str, dim: int) -> DimEvidence:
        lex = LEXICON.get(dim, {})

        # 1) 收集全部命中：(start, end, word, weight, polarity)
        raw: List[Tuple[int, int, str, float, int]] = []
        for pol, sign in (("pos", 1), ("neg", -1)):
            for words, weight in lex.get(pol, []):
                for w in words:
                    start = 0
                    while True:
                        i = text.find(w, start)
                        if i < 0:
                            break
                        raw.append((i, i + len(w), w, weight, sign))
                        start = i + 1

        # 2) 最长覆盖优先：短命中被更长的命中完全包住就丢弃。
        #    解决「没工具」(neg短语) 与「工具」(pos词根) 互相抵消的经典冲突
        kept = []
        for m in raw:
            covered = any(
                o is not m and o[0] <= m[0] and m[1] <= o[1] and (o[1] - o[0]) > (m[1] - m[0])
                for o in raw
            )
            if not covered:
                kept.append(m)

        pos = 0.0
        neg = 0.0
        hits: List[str] = []
        for start, _end, w, weight, sign in kept:
            if self._is_negated_at(text, start):
                # 被否定的信号不计分 —— 「别删」不等于「创造」，它只是回到中位 0
                hits.append(f"~{w}")
                continue
            if sign > 0:
                pos += weight
                hits.append(f"+{w}")
            else:
                neg += weight
                hits.append(f"-{w}")

        # 3) 上下文抑制：某些信号一出现，同维的反向信号就降级为背景描述。
        #    这是修复 FALSE（误触发）的核心机制 —— 「接口/流程/模块/设计」
        #    这类话题词只说明在谈论什么，不代表正在运用什么。
        if dim in (0, 4) and any(f in text for f in FAULT_CONTEXT):
            # 「接口全挂」「设计有问题」是在报案，不是在用工具、也不是在创造
            pos = 0.0
            hits = [h for h in hits if h.startswith("-")]
        if dim == 4 and any(f in text for f in IDLE_CONTEXT):
            # 「三个月没动过」的东西不是当前在用的工具
            pos = 0.0
            hits = [h for h in hits if h.startswith("-")]
        if dim == 4 and any(o in text for o in ORG_CONTEXT):
            # 「模块负责人」里的模块指组织单元，不是技术工具；
            # 但「补 README」里的技术名词不能被一并清掉
            drop = {h[1:] for h in hits if h.startswith("+")} & set(ORG_AMBIGUOUS)
            if drop:
                pos -= sum(wt for _s, _e, wd, wt, sg in kept
                           if sg > 0 and wd in drop)
                hits = [h for h in hits if not (h.startswith("+") and h[1:] in drop)]
        if dim == 4 and any(a in text for a in ADMIN_CONTEXT):
            # 「审批流程太长」里的流程是组织程序，不是工程方法
            pos = 0.0
            hits = [h for h in hits if h.startswith("-")]
        if dim == 6 and any(c in text for c in CONFLICT_CONTEXT):
            # 「团队又闹僵」时，「团队」只是场景名词，冲突才是态势
            pos = 0.0
            hits = [h for h in hits if h.startswith("-")]
        if dim == 6 and any(q in text for q in QUESTION_CONTEXT):
            # 「帮我看看为什么慢」是求助，不是协作关系
            pos = 0.0
            hits = [h for h in hits if h.startswith("-")]

        return DimEvidence(dim=dim, pos=pos, neg=neg, hits=hits)

    @staticmethod
    def _is_negated_at(text: str, idx: int) -> bool:
        if idx <= 0:
            return False
        window = text[max(0, idx - 3):idx]
        for n in NEGATORS_STRICT:
            j = window.rfind(n)
            if j >= 0:
                # 「不错」「不仅」里的「不」是词素，不是否定副词
                at = max(0, idx - 3) + j
                if any((text[at:at + len(n) + 1]).startswith(nn) for nn in NOT_NEGATION):
                    continue
                return True
        prev = text[idx - 1]
        if prev in NEGATORS_ADJACENT:
            # 「特别来劲」的「别」是词素，不是否定副词
            if prev == "别" and idx >= 2 and text[idx - 2] in BIE_PREFIX:
                return False
            return True
        return False

    @staticmethod
    def _neutral_hit(text: str, dim: int) -> bool:
        for words in NEUTRAL_ANCHORS.get(dim, []):
            if any(w in text for w in words):
                return True
        return False

    @staticmethod
    def _explain(vector: List[int], evidence: List[DimEvidence]) -> str:
        parts = []
        for dim in range(9):
            ev = evidence[dim]
            if not ev.hits:
                continue
            sign = "+" if vector[dim] > 0 else ("-" if vector[dim] < 0 else "0")
            parts.append(f"d{dim}{DIMENSION_ANCHORS[dim]['name']}({sign}):{'|'.join(ev.hits)}")
        return "；".join(parts) if parts else "无命中信号"


# ---------------------------------------------------------------------------
# 在线 LLM 通道（OpenAI 兼容协议）
# ---------------------------------------------------------------------------

ENCODE_PROMPT = """你是三元九维认知框架的编码器。把用户输入映射为 9 维向量，每维取 +1 / 0 / -1。

维度定义：
0 天·性  +1创造生成 / 0维持 / -1破坏损耗
1 天·习  +1习惯延续 / 0首次 / -1习惯中断
2 天·运  +1外部驱动紧迫 / 0平稳 / -1外部阻力
3 地·性  +1环境切换 / 0不变 / -1环境退化
4 地·习  +1工具方法运用 / 0无工具 / -1工具缺失
5 地·运  +1状态升级 / 0稳定 / -1状态降级
6 人·性  +1关系协作 / 0独立 / -1孤立冲突
7 人·习  +1自律内省 / 0无自律 / -1放纵失控
8 人·运  +1内在驱动 / 0无驱动 / -1内在耗竭

注意：
- 判断的是「任务本身的态势」，不是文本里提到的题材。写一段关于父女对话的小说，不等于存在协作关系。
- 只有明确信号才给 ±1，判断不了就给 0。

只输出 JSON，不要其他文字：
{{"vector": [9个整数], "confidence": 0到1的小数, "reasoning": "一句话说明"}}

用户输入：
{text}"""


class LLMBackend:
    """OpenAI 兼容协议的在线 encode。

    配置（任一缺失则该通道不可用，自动回落本地）：
      TNE_LLM_API_KEY   必填
      TNE_LLM_BASE_URL 默认 https://api.openai.com/v1
      TNE_LLM_MODEL    默认 gpt-4o-mini
    """

    def __init__(self, api_key: str = None, base_url: str = None,
                 model: str = None, timeout: int = 25):
        self.api_key = api_key or os.getenv("TNE_LLM_API_KEY")
        self.base_url = (base_url or os.getenv("TNE_LLM_BASE_URL")
                         or "https://api.openai.com/v1").rstrip("/")
        self.model = model or os.getenv("TNE_LLM_MODEL") or "gpt-4o-mini"
        self.timeout = timeout

    def available(self) -> bool:
        return bool(self.api_key)

    def encode(self, user_input: str) -> EncodeResult:
        if not self.available():
            raise RuntimeError("LLM backend 未配置 TNE_LLM_API_KEY")

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "你是精确的九维状态编码器，只输出 JSON。"},
                {"role": "user", "content": ENCODE_PROMPT.format(text=user_input)},
            ],
            "temperature": 0.0,
            "max_tokens": 300,
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        content = body["choices"][0]["message"]["content"]
        return self._parse(content)

    @staticmethod
    def _parse(content: str) -> EncodeResult:
        # 容忍 markdown 代码围栏
        cleaned = re.sub(r"^```(?:json)?|```$", "", content.strip(),
                         flags=re.MULTILINE).strip()
        data = json.loads(cleaned)
        vec = [int(x) for x in data["vector"]]
        if len(vec) != 9:
            raise ValueError(f"向量长度应为 9，实际 {len(vec)}")
        return EncodeResult(
            vector=[max(-1, min(1, v)) for v in vec],
            confidence=float(data.get("confidence", 0.8)),
            reasoning=data.get("reasoning", ""),
            ambiguity_flags=[],
            backend="llm",
        )


# ---------------------------------------------------------------------------
# 对外主类：V4 同名同签名，可直接替换
# ---------------------------------------------------------------------------

class YuanbaoProjectorV4:
    """保持与 V4 完全一致的调用接口：.encode(str) / .batch_encode(List[str])。

    use_llm:
      "auto"  —— 配了 key 走在线，失败回落本地（默认）
      "local" —— 强制本地
      "llm"   —— 强制在线，失败直接抛错
    """

    DIMENSION_ANCHORS = DIMENSION_ANCHORS

    def __init__(self, use_llm: str = "auto", llm_backend: LLMBackend = None):
        self.use_llm = use_llm
        self.local = LocalSemanticBackend()
        self.llm = llm_backend or LLMBackend()
        self.stats = {"llm_ok": 0, "llm_fallback": 0, "local": 0}

    def encode(self, user_input: str) -> EncodeResult:
        if self.use_llm in ("auto", "llm") and self.llm.available():
            try:
                r = self.llm.encode(user_input)
                self.stats["llm_ok"] += 1
                return r
            except Exception as e:  # noqa: BLE001 - 任何异常都回落，保证 pipeline 不中断
                if self.use_llm == "llm":
                    raise
                self.stats["llm_fallback"] += 1
                r = self.local.encode(user_input)
                r.backend = "llm_fallback"
                r.ambiguity_flags.append(f"LLM 调用失败已回落: {type(e).__name__}")
                return r

        r = self.local.encode(user_input)
        self.stats["local"] += 1
        return r

    def batch_encode(self, inputs: List[str]) -> List[EncodeResult]:
        return [self.encode(i) for i in inputs]

    # --- 校准支持：用 LLM 标注的 golden set 反查本地判定器偏差 ---

    @staticmethod
    def diagnose(golden: List[Dict[str, Any]], predict_fn=None) -> Dict[str, Any]:
        """失败归因。把每个错误的维度分成三类，指明该往哪个方向改：

          MISS      漏召回 —— 该有信号却判成 0，词表缺词，应当补词
          FALSE     误触发 —— 不该有信号却判出 ±1，多义词或上下文缺失
          POLARITY  极性错 —— 方向判反，正/负词表串了

        MISS 占主导说明词表覆盖率不够；FALSE 占主导说明关键词法已到天花板，
        补词只会越补越糟，该换语义匹配方案。
        """
        import collections
        counts = collections.Counter()
        dim_miss = collections.Counter()
        dim_false = collections.Counter()
        samples = {"MISS": [], "FALSE": [], "POLARITY": []}

        for g in golden:
            pred = (predict_fn(g["text"]) if predict_fn
                    else YuanbaoProjectorV4(use_llm="local").local.encode(g["text"])).vector
            exp = g["vector"]
            for d in range(9):
                if pred[d] == exp[d]:
                    continue
                if exp[d] != 0 and pred[d] == 0:
                    kind, dim_miss[d] = "MISS", dim_miss[d] + 1
                elif exp[d] == 0 and pred[d] != 0:
                    kind, dim_false[d] = "FALSE", dim_false[d] + 1
                else:
                    kind = "POLARITY"
                counts[kind] += 1
                if len(samples[kind]) < 6:
                    samples[kind].append(f"{g['text'][:20]} d{d}:{pred[d]}≠{exp[d]}")

        total_err = sum(counts.values())
        return {
            "total_errors": total_err,
            "MISS": counts["MISS"],
            "FALSE": counts["FALSE"],
            "POLARITY": counts["POLARITY"],
            "miss_rate": round(counts["MISS"] / total_err, 3) if total_err else 0.0,
            "dominant": max(counts, key=counts.get) if total_err else "none",
            "miss_by_dim": dict(sorted(dim_miss.items(), key=lambda x: -x[1])),
            "false_by_dim": dict(sorted(dim_false.items(), key=lambda x: -x[1])),
            "samples": samples,
        }

    def evaluate_against(self, golden: List[Dict[str, Any]]) -> Dict[str, Any]:
        """golden: [{"text": "...", "vector": [...]}, ...]
        本地判定器对标注集的吻合率，用于发现词表盲区。"""
        if not golden:
            return {"total": 0}
        dim_correct = [0] * 9
        exact = 0
        for g in golden:
            pred = self.local.encode(g["text"]).vector
            expect = g["vector"]
            ok = True
            for d in range(9):
                if pred[d] == expect[d]:
                    dim_correct[d] += 1
                else:
                    ok = False
            exact += 1 if ok else 0
        n = len(golden)
        return {
            "total": n,
            "exact_match_rate": round(exact / n, 3),
            "per_dim_accuracy": [round(c / n, 3) for c in dim_correct],
            "weakest_dims": sorted(range(9), key=lambda d: dim_correct[d])[:3],
        }


projector_v4 = YuanbaoProjectorV4()
