#!/usr/bin/env python3
"""
TNE-Core 批量轨迹积累脚本
用法：
  python3 scripts/run_batch.py --rounds 1000 --seed 42 --api http://localhost:8000
"""
import argparse
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from tne.agent.client import TNEApiClient, SimulatedAgent


def main():
    parser = argparse.ArgumentParser(description="TNE-Core 批量轨迹积累")
    parser.add_argument("--rounds", type=int, default=1000, help="运行轮数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--api", type=str, default="http://localhost:8000", help="坐标服务地址")
    parser.add_argument("--output", type=str, default=None, help="结果输出 JSON 路径")
    args = parser.parse_args()

    api = TNEApiClient(args.api)
    agent = SimulatedAgent(api, seed=args.seed)
    print(f"Session: {agent.session_id}")
    print(f"Running {args.rounds} rounds against {args.api}...")

    results = agent.run_batch(args.rounds)

    # 统计
    from collections import Counter
    outcomes = Counter(r["outcome"] for r in results)
    modes = Counter(r["habit_mode"] for r in results)
    classes = set(r["class_rep"] for r in results)

    print(f"\n=== {args.rounds} 轮完成 ===")
    print(f"覆盖等价类: {len(classes)} / 5002")
    print(f"结果分布: {dict(outcomes)}")
    print(f"习惯模式: {dict(modes)}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"结果已保存: {args.output}")


if __name__ == "__main__":
    main()
