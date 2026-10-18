"""TNE-Core - 校准与泛化测量

用法：
    python calibrate.py             # 四个集全跑 + 失败归因
    python calibrate.py --llm       # 额外用在线 LLM 通道对比（需 TNE_LLM_API_KEY）
    python calibrate.py --detail    # 打印每条不一致样本

四个集的身份必须分清，否则数字没有意义：

    golden_set   拟合集  词表是照着它调出来的，高分只能证明「记住了」
    train_set    训练集  可以拿它调词、看归因、逐条分析
    val_set      验证集  只测量。用它调词就等于把它变成第二个拟合集
    holdout_set  独立集  只测量。全程未参与任何调词，最终仲裁

train/val 由 split_dataset.py 用固定 seed 从 expanded_set（100 条）划分，
划分在看到任何预测结果之前完成。
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tne_projector_v5 import YuanbaoProjectorV4

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DIMS = ["天·性", "天·习", "天·运", "地·性", "地·习", "地·运", "人·性", "人·习", "人·运"]

SETS = [
    ("golden_set.json", "拟合集", "可调·回归保护"),
    ("train_set.json", "训练集", "可调"),
    ("val_set.json", "验证集", "只测"),
    ("holdout_set.json", "独立集", "只测"),
]


def load(name):
    with open(os.path.join(DATA_DIR, name), 'r', encoding='utf-8') as f:
        return json.load(f)["samples"]


def measure(samples, projector, title, role, diagnose=True, detail=False):
    r = projector.evaluate_against(samples)
    d = YuanbaoProjectorV4.diagnose(samples)
    print(f"\n  {title:<10}{len(samples):>3} 条   吻合率 {r['exact_match_rate']*100:>5.1f}%"
          f"   [{role}]")
    print(f"  {'':<10}MISS={d['MISS']:<4} FALSE={d['FALSE']:<4} POLARITY={d['POLARITY']}")
    if diagnose:
        if d["miss_by_dim"]:
            top = list(d["miss_by_dim"].items())[:3]
            print(f"  {'':<10}盲区最大 " +
                  "  ".join(f"d{i}{DIMS[i]}({c})" for i, c in top))
        if d["false_by_dim"]:
            top = list(d["false_by_dim"].items())[:3]
            print(f"  {'':<10}误触发最多 " +
                  "  ".join(f"d{i}{DIMS[i]}({c})" for i, c in top))
    if detail:
        for g in samples:
            pred = projector.local.encode(g["text"]).vector if hasattr(projector, "local") else None
            if pred is None:
                continue
            diff = [f"d{i}:{pred[i]}≠{g['vector'][i]}"
                    for i in range(9) if pred[i] != g["vector"][i]]
            if diff:
                print(f"      !! {g['text'][:30]:<32} {', '.join(diff)}")
    return r


def main():
    detail = "--detail" in sys.argv
    use_llm = "--llm" in sys.argv
    local = YuanbaoProjectorV4(use_llm="local")

    print("TNE-Core 校准与泛化测量")
    print("=" * 66)

    results = {}
    for name, title, role in SETS:
        samples = load(name)
        # 只测集不给逐条归因，避免被拿去调词
        show_diag = role.startswith("可调")
        results[name] = measure(samples, local, title, role,
                                diagnose=show_diag, detail=detail and show_diag)

    print(f"\n{'='*66}")
    print("  泛化读数")
    print(f"{'='*66}")
    g = results["golden_set.json"]["exact_match_rate"]
    t = results["train_set.json"]["exact_match_rate"]
    v = results["val_set.json"]["exact_match_rate"]
    h = results["holdout_set.json"]["exact_match_rate"]
    print(f"  拟合集 {g*100:>5.1f}%   训练集 {t*100:>5.1f}%")
    print(f"  验证集 {v*100:>5.1f}%   独立集 {h*100:>5.1f}%")
    print(f"\n  过拟合缺口（训练集 − 验证集）: {(t-v)*100:>5.1f} 个百分点")
    print(f"  真实泛化水平取两个只测集的较低者: {min(v,h)*100:>5.1f}%")

    if use_llm:
        online = YuanbaoProjectorV4(use_llm="llm")
        if not online.llm.available():
            print("\n  [跳过] 未设置 TNE_LLM_API_KEY，在线通道不可用")
        else:
            print(f"\n  在线模型: {online.llm.model}")
            for name in ("val_set.json", "holdout_set.json"):
                measure(load(name), online, name.replace(".json", ""), "在线LLM")

    print(f"\n{'='*66}")
    print("  读法")
    print("    · golden/train 高、val 低 → 过拟合，词表在背答案")
    print("    · MISS 为主 → 词表覆盖不足，补词有效")
    print("    · FALSE 为主 → 关键词法到顶，补词只会越补越糟")
    print("    · 方向依赖施动者的词（如「临时」）词表无法表达，属固有局限")
    print(f"{'='*66}")


if __name__ == "__main__":
    main()
