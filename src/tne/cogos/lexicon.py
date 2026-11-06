"""
CogOS · 轻量语义词典（映射阶段的文本特征抽取）

不依赖大模型即可跑通链路：用维度关键词打分 + 强度词放大，
把一段自然语言压成 9 维原始分数，交给 mapping 做符号化。

消歧策略（针对中文口语的三条硬规则）：
  1) 长词优先：先匹配整段负向短语（如「方向没想明白」），命中即挖掉，避免被短词重复计分；
  2) 先负后正：负向短语先扫，扫过的片段不再进入正向匹配；
  3) 否定窗口：正向词前 3 字内出现否定词，极性翻转为负；负向词同理翻正。

对外接口 raw_scores(text) -> (9维分, 命中解释) 保持不变，后续可整体替换为模型投影。
"""

import re
from typing import Dict, List, Tuple

DIM = 9

LEXICON: Dict[int, Dict[str, List[str]]] = {
    0: {  # g1 愿景定向
        "pos": ["先定目标", "明确方向", "愿景", "目标", "未来", "长期", "意义", "追求", "使命", "规划", "方向"],
        "neg": ["方向没想明白", "没想明白", "想不清楚", "迷茫", "没方向", "随便", "无所谓", "走一步看一步", "空转"],
    },
    1: {  # g2 决断定夺
        "pos": ["决定", "选择", "确定", "拍板", "定下", "立场", "取舍", "下决心", "结论", "先定"],
        "neg": ["犹豫", "纠结", "两难", "拿不准", "不确定", "反复", "拖着", "拖延", "回避"],
    },
    2: {  # g3 产出输出
        "pos": ["产出", "交付", "完成", "发布", "写出来", "做出来", "成果", "落地", "上线", "补齐"],
        "neg": ["没产出", "憋着", "空想", "半途", "卡住", "没做"],
    },
    3: {  # g4 方法路径
        "pos": ["方法", "流程", "步骤", "怎么", "如何", "工具", "路径", "框架", "拆解", "策略", "三步"],
        "neg": ["没想清楚方法", "没方法", "乱的", "瞎试", "无序", "硬扛"],
    },
    4: {  # g5 核心主态
        "pos": ["核心问题", "本质", "关键", "重心", "主线", "根本问题"],
        "neg": ["边缘", "次要", "枝节", "跑题", "表面"],
    },
    5: {  # g6 协作关联
        "pos": ["合作", "协作", "一起", "对齐", "伙伴", "配合", "支持"],
        "neg": ["沟通很累", "各干各的", "孤立", "冲突", "吵架", "割裂", "对立", "内耗", "很累", "疲惫"],
    },
    6: {  # g7 输入素材
        "pos": ["资料", "信息", "数据", "素材", "输入", "调研", "证据", "参考"],
        "neg": ["资料也缺", "资料不足", "缺资料", "资料缺", "没信息", "信息不足", "不知道", "盲区"],
    },
    7: {  # g8 积累沉淀
        "pos": ["经验", "积累", "复盘", "沉淀", "历史", "迭代", "熟练", "做过"],
        "neg": ["经验也不够", "经验不够", "经验不足", "没经验", "也不够", "新手", "第一次", "从零", "空白"],
    },
    8: {  # g9 根性立身
        "pos": ["根基", "基础", "底层", "底子", "身体", "本源", "扎实"],
        "neg": ["撑不住", "动摇", "薄弱", "不牢", "透支", "虚"],
    },
}

INTENSIFIERS = ["非常", "特别", "极其", "完全", "一直", "深度", "彻底", "严重", "越来越"]
NEGATIONS = ["不", "没", "无", "别", "非", "未"]
_MASK = "□"


def _scan(text: str, words: List[str]) -> Tuple[int, List[str]]:
    """长词优先扫描；命中后挖掉；前置否定翻转极性。返回 (得分, 命中词)。"""
    score = 0
    hits: List[str] = []
    masked = text
    for w in sorted(words, key=len, reverse=True):
        idx = masked.find(w)
        while idx != -1:
            prefix = masked[max(0, idx - 3): idx]
            negated = any(n in prefix for n in NEGATIONS)
            score += -1 if negated else 1
            hits.append(("¬" if negated else "") + w)
            masked = masked[:idx] + _MASK * len(w) + masked[idx + len(w):]
            idx = masked.find(w)
    return score, hits


def raw_scores(text: str) -> Tuple[List[float], List[str]]:
    """返回 (9维原始分, 命中解释词列表)。"""
    t = text or ""
    boost = 1.6 if any(w in t for w in INTENSIFIERS) else 1.0
    scores: List[float] = []
    all_hits: List[str] = []

    for dim in range(DIM):
        neg_words = LEXICON[dim]["neg"]
        pos_words = LEXICON[dim]["pos"]
        # 先负后正：负向短语优先占位，避免「方向没想明白」再被「方向」计正分
        neg_score, neg_hits = _scan(t, neg_words)
        pos_score, pos_hits = _scan(_mask_out(t, neg_words), pos_words)

        s = (pos_score - neg_score) * boost
        scores.append(round(s, 3))
        if pos_score:
            all_hits.append(f"g{dim+1}+{pos_hits[0]}")
        if neg_score:
            all_hits.append(f"g{dim+1}-{neg_hits[0]}")
    return scores, all_hits


def _mask_out(text: str, words: List[str]) -> str:
    """把已判定过的词段挖掉，防止同一片段被正负两侧重复计分。"""
    masked = text
    for w in sorted(words, key=len, reverse=True):
        masked = re.sub(re.escape(w), _MASK * len(w), masked)
    return masked
