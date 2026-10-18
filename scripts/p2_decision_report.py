#!/usr/bin/env python3
"""
TNE-Core · 阶段四 P2 · 应用试点：认知诊断报告
输入一段任务/决策文本 → 输出完整认知诊断：
九维定位 → 卦位 → 等价类 → 结构变换 → 邻域 → 习惯偏置 → 认知张力 → 动力信号。
可作投资/管理/内容/生活多场景的"认知体检"。
"""
import sys, json, argparse
sys.path.insert(0, "/home/user/Doubao/chats/38446456964142338/tne-core/src")
os_env = __import__("os")
os_env.environ["TOKENIZERS_PARALLELISM"] = "false"

from tne.space.space import encode, gua_yao_str, TOTAL_STATES
from tne.space.equivalence import class_of
from tne.space.distance import nearest_neighbors
from tne.space.operators import f_cuoo, f_zong, f_cuoo_zong, f_hu, f_bian
from tne.trajectory.store import TrajectoryStore
from tne.trajectory.habit import HabitQuery
from tne.drive.tension import detect_and_generate
from tne.projector.encoder_judger import TrainedEncoderJudger

SYM = {1: "阳·生发", 0: "衡·居间", -1: "阴·收敛"}
SEM = ["愿景", "决断", "产出", "方法", "主态", "协作", "输入", "积累", "根性"]


def diagnose(text, judger, store, habit, mode="encode"):
    vec = judger.judge(text)
    idx = encode(vec)
    rep = class_of(idx)
    hb = habit.query(vec)
    nbs = [{"卦位": ni, "距离": round(d, 3), "向量": ng}
           for ni, d, ng in nearest_neighbors(vec, top_n=5)]
    drive = detect_and_generate(hb, judger.judge_with_confidence(text)["confidence"],
                                [{"gua_idx": ni, "distance": round(d, 3), "gua_vector": ng,
                                  "class_rep": class_of(ni)}
                                 for ni, d, ng in nearest_neighbors(vec, top_n=3)])
    return {
        "text": text,
        "mode": mode,
        "vector": vec,
        "gua_idx": idx,
        "class_rep": rep,
        "yao_sem": [f"{SEM[i]}={SYM[v]}" for i, v in enumerate(vec)],
        "transform": {
            "错卦Fc": f_cuoo(vec), "综卦Fz": f_zong(vec),
            "错综Fcz": f_cuoo_zong(vec), "互卦Fh(5维)": f_hu(vec),
            "动爻Fb(主态动)": f_bian(vec, {5}),
        },
        "neighbors": nbs,
        "habit": hb,
        "drive": drive,
    }


def report(d):
    lines = []
    lines.append(f"┌─ 认知诊断 · 卦位 #{d['gua_idx']} / 等价类 #{d['class_rep']}")
    lines.append(f"│ 任务：{d['text']}")
    v = "".join("+" if x == 1 else "0" if x == 0 else "-" for x in d["vector"])
    lines.append(f"│ 九维定位：{v}")
    lines.append(f"│ 爻位语义：{'  '.join(d['yao_sem'])}")
    lines.append(f"├─ 结构变换推演")
    for k, t in d["transform"].items():
        lines.append(f"│   {k}：{''.join('+' if x==1 else '0' if x==0 else '-' for x in t)}")
    lines.append(f"├─ 邻域泛化 Top-5")
    for n in d["neighbors"]:
        lines.append(f"│   #{n['卦位']} d={n['距离']}  {n['向量']}")
    hb = d["habit"]
    lines.append(f"├─ 习惯偏置：{hb.get('mode','')}")
    lines.append(f"│   推荐：{hb.get('recommendation','')}")
    if d["drive"] and d["drive"].get("tensions"):
        tensions_str = " ".join(f"{t['type']}({t['strength']:.1f})" for t in d["drive"]["tensions"])
        lines.append(f"├─ 认知张力：{tensions_str}")
        for s in d["drive"]["signals"]:
            lines.append(f"│   动力信号[{s['type']}]：{s.get('text','')[:40]}...")
    else:
        lines.append(f"├─ 认知张力：稳定习惯，无需额外动力注入")
    lines.append(f"└─")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default=None)
    ap.add_argument("--demo", action="store_true", help="跑内置演示场景")
    ap.add_argument("--json", action="store_true", help="输出JSON")
    args = ap.parse_args()

    store = TrajectoryStore("tne_trajectories.db")
    habit = HabitQuery(store)
    judger = TrainedEncoderJudger()

    demos = [
        "是继续追加投资还是及时止损",
        "新招的技术负责人迟迟不出方案",
        "要不要趁着行情好扩张线下门店",
        "核心员工突然提出离职",
        "尝试做个人知识库副业",
    ]
    texts = [args.text] if args.text else (demos if args.demo else demos)

    out = [diagnose(t, judger, store, habit) for t in texts]
    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        for d in out:
            print(report(d))
            print()
    store.close()


if __name__ == "__main__":
    main()
