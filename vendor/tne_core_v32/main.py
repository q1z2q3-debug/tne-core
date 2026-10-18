"""TNE-Core v3.2 - Main Pipeline"""

import time
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tne_projector_v5 import YuanbaoProjectorV4
from tne_ingest import TNESession, PATTERNS_PATH, build_batch_from_texts
from tne_distiller import TNE_Distiller, TrajectoryRecord
from tne_trajectory_collector import TrajectoryCollector

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
GOLDEN_PATH = os.path.join(DATA_DIR, "golden_set.json")
BATCH_PATH = os.path.join(DATA_DIR, "batch_demo.jsonl")

DEFAULT_HABITS = ["以终为始", "统合综效", "不断更新"]


def run_pipeline(user_input: str, task_type: str, task_id: str,
                 description: str = "", efficacy: float = None,
                 habits: list = None, session: TNESession = None) -> dict:
    """单条 pipeline。传 session 则复用其状态（推荐），否则自建临时 session。

    efficacy 必须显式传入 —— 它是「这次任务做得好不好」，与 encode 的
    confidence（「九维判断有多确定」）是两个独立概念，混用会让模式库永远空转。
    """
    own = session is None
    if own:
        session = TNESession()

    proj = session.projector
    result = proj.encode(user_input)

    print(f"\n  {task_id}  「{user_input[:34]}」")
    print(f"    [1] Encode → {result.vector} (conf={result.confidence:.2f}, {result.backend})")
    if efficacy is None:
        efficacy = result.confidence
        print(f"    [!] efficacy 未传入，回落为 confidence={efficacy:.2f}（准入门槛 0.85）")

    habits = habits or DEFAULT_HABITS
    rec = TrajectoryRecord.create(
        task_id=task_id, task_type=task_type,
        state_before=result.vector, habits_used=habits,
        efficacy_score=efficacy, description=description or user_input[:60],
    )
    errs = rec.validate()
    if errs:
        print(f"    [x] 校验失败，拒绝入库: {errs}")
        return {"ok": False, "errors": errs}

    written = session.collector.record(rec)
    if not written:
        print(f"    [x] task_id 重复，跳过")
        return {"ok": False, "errors": ["duplicate"]}

    new = session.distiller.add_trajectory(rec)
    print(f"    [2] Distill → {len(new)} 新模式 (efficacy={efficacy:.2f})")

    recs = session.distiller.recommend_with_scores(task_type, result.vector, top_k=1)
    if recs:
        r = recs[0]
        print(f"    [3] Recommend → {r['pattern_id'][:40]} "
              f"(success={r['success_rate']:.2f}, 状态匹配={r['compat']:.2f})")
    else:
        print(f"    [3] Recommend → 暂无匹配")

    print(f"    [4] Collect → 已入库")

    if own:
        session.close()
    return {"vector": result.vector, "confidence": result.confidence, "efficacy": efficacy}


def run_batch(batch_path: str = BATCH_PATH, fresh: bool = True,
              show_progress: bool = False) -> dict:
    """批量入库：单次加载模式库 → 整批 encode/校验/蒸馏 → 单次落盘。"""
    print(f"\n{'='*60}")
    print(f"  批量入库 · {os.path.basename(batch_path)}")
    print(f"{'='*60}")

    with TNESession(fresh=fresh) as s:
        t0 = time.time()
        report = s.ingest_file(batch_path, show_progress=show_progress)
        print(f"  输入 {report['input']} 条 | 入库 {report['ingested']} 条 | "
              f"跳过重复 {report['dupe_skipped']} | 校验失败 {report['invalid']}")
        print(f"  耗时 {report['elapsed_sec']}s | 新增模式 {len(report['new_patterns'])} 个")
        if report["invalid_detail"]:
            print(f"  脏数据样本:")
            for d in report["invalid_detail"]:
                print(f"    - {d['task_id']}: {d['errors'][0]}")

        print(f"\n  --- 模式库 ---")
        st = s.distiller.get_stats()
        print(f"  模式总数 {st.get('total', 0)} | 可推荐 {st.get('ready_to_recommend', 0)} | "
              f"累积调用 {st.get('total_usage', 0)}")
        for tt, cnt in st.get("by_type", {}).items():
            print(f"    {tt}: {cnt}")

        print(f"\n  --- 各类型 Top 模式 ---")
        for tt in st.get("by_type", {}):
            print(f"  [{tt}]")
            for p in s.distiller.patterns.values():
                if p.task_type != tt:
                    continue
                mark = "✓可推荐" if (p.usage_count >= 2 and p.success_rate >= 0.85) else "·样本不足"
                print(f"    {mark} usage={p.usage_count} success={p.success_rate:.2f} "
                      f"sig={p.state_signature()} chain={'>'.join(p.execution_chain)}")

        print(f"\n  --- 轨迹库 ---")
        print(f"  {s.collector.stats()}")

    return report


def run_recommend_demo():
    """新任务进来时的推荐演示：直接给文本，自动 encode 后匹配。"""
    print(f"\n{'='*60}")
    print(f"  新任务推荐（复用已累积的模式库）")
    print(f"{'='*60}")

    with TNESession(fresh=False) as s:
        cases = [
            ("再写一段父女在车站的重逢", "创作叙事"),
            ("优化首页首屏渲染速度", "核心模块优化"),
            ("服务又崩了，赶紧看日志", "故障处置"),
        ]
        for text, tt in cases:
            recs = s.recommend(task_type=tt, text=text, top_k=2)
            print(f"\n  「{text}」→ {tt}")
            if recs:
                for r in recs:
                    print(f"    ✓ {r['pattern_id'][:44]}")
                    print(f"      匹配度={r['compat']:.2f} 成功率={r['success_rate']:.2f} "
                          f"调用次数={r['usage_count']} 特征={r['signature']}")
                    print(f"      执行链: {' > '.join(r['execution_chain'])}")
            else:
                print(f"    ✗ 无匹配（该类型模式尚不足 2 次调用）")


def run_calibration() -> dict:
    """用 golden_set 校准本地判定器。"""
    import json
    with open(GOLDEN_PATH, 'r', encoding='utf-8') as f:
        golden = json.load(f)["samples"]
    p = YuanbaoProjectorV4(use_llm="local")
    report = p.evaluate_against(golden)
    print(f"\n{'='*60}")
    print(f"  本地判定器 vs LLM 标注集 ({report['total']} 条)")
    print(f"{'='*60}")
    print(f"  完全吻合率: {report['exact_match_rate']*100:.0f}%")
    print(f"  逐维准确率: {report['per_dim_accuracy']}")
    return report


if __name__ == "__main__":
    print("TNE-Core v3.2 Pipeline")
    print("=" * 60)

    run_calibration()
    run_batch()
    run_recommend_demo()

    print(f"\n{'='*60}")
    print("  Pipeline Complete!")
    print(f"{'='*60}")
