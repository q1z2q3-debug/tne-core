"""TNE-Core - 数据集划分

划分必须在看到预测结果之前完成，否则「挑对自己有利的分法」本身就是作弊。
这里用固定 seed + 分层抽样：按向量的非零维数分层，保证 train/val 的
稀疏分布一致，避免某一侧全是简单样本。
"""

import json
import os
import random
import hashlib
from collections import defaultdict

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SEED = 20261018
TRAIN_RATIO = 0.6


def split(samples, seed=SEED, ratio=TRAIN_RATIO):
    """按非零维数分层，再在层内按 seed 洗牌切分。"""
    strata = defaultdict(list)
    for s in samples:
        key = sum(1 for v in s["vector"] if v != 0)
        strata[key].append(s)

    train, val = [], []
    for key in sorted(strata):
        items = strata[key][:]
        rnd = random.Random(seed + key)  # 每层独立但确定的随机序列
        rnd.shuffle(items)
        n_train = max(1, round(len(items) * ratio)) if len(items) > 1 else 1
        train.extend(items[:n_train])
        val.extend(items[n_train:])

    # 两侧都按原 id 排序，便于人工核对
    train.sort(key=lambda s: s.get("id", 0))
    val.sort(key=lambda s: s.get("id", 0))
    return train, val


def write(name, samples, src_hash):
    payload = {
        "_meta": {
            "split": name,
            "count": len(samples),
            "seed": SEED,
            "source_sha256": src_hash,
            "rule": "按非零维数分层 + 固定 seed 层内洗牌，train:val = 60:40",
            "warning": "此集可用于调词；val 集调完后仍需用 holdout_set.json 做终检",
        },
        "samples": samples,
    }
    path = os.path.join(DATA_DIR, f"{name}.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def coverage(samples):
    """每维 +1/0/-1 的样本数，用于检查两侧覆盖是否失衡。"""
    cov = {d: {"+1": 0, "0": 0, "-1": 0} for d in range(9)}
    for s in samples:
        for d in range(9):
            v = s["vector"][d]
            cov[d]["+1" if v > 0 else ("-1" if v < 0 else "0")] += 1
    return cov


def main():
    src = os.path.join(DATA_DIR, "expanded_set.json")
    raw = open(src, 'rb').read()
    src_hash = hashlib.sha256(raw).hexdigest()[:16]
    samples = json.loads(raw.decode('utf-8'))["samples"]

    train, val = split(samples)
    print(f"源集 {len(samples)} 条  sha256={src_hash}")
    print(f"划分 seed={SEED}  rule=非零维数分层 + 层内洗牌")
    print(f"  train: {len(train)} 条")
    print(f"  val  : {len(val)} 条")

    ct, cv = coverage(train), coverage(val)
    print(f"\n{'维':<8}{'train(+/0/-)':<18}{'val(+/0/-)':<18}")
    NAMES = ["天·性", "天·习", "天·运", "地·性", "地·习", "地·运", "人·性", "人·习", "人·运"]
    for d in range(9):
        t = f"{ct[d]['+1']}/{ct[d]['0']}/{ct[d]['-1']}"
        v = f"{cv[d]['+1']}/{cv[d]['0']}/{cv[d]['-1']}"
        print(f"d{d}{NAMES[d]:<6}{t:<18}{v:<18}")

    # 覆盖告警：某侧某极性样本为 0 时，该极性在那侧无法被评估
    warn = []
    for d in range(9):
        for p in ("+1", "0", "-1"):
            if cv[d][p] == 0:
                warn.append(f"d{d}{NAMES[d]} 的 {p} 在 val 中无样本")
    if warn:
        print("\n覆盖告警:")
        for w in warn:
            print(f"  ! {w}")

    write("train_set", train, src_hash)
    write("val_set", val, src_hash)
    print(f"\n已写出 train_set.json / val_set.json")


if __name__ == "__main__":
    main()
