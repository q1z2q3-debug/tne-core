"""TNE-Core - 离线小模型 encode（实验性）

结论先行：**这条路在当前沙箱里走不通，本模块保留为可复现的实验记录，
默认不启用。** 所有数字见 data/model_experiment.md。

试过什么：
  1. 装预训练小模型（MiniLM / 中文 sentence-bert）
     → HuggingFace / hf-mirror / ModelScope 全部 403，模型权重下载不到
     → pip 装 torch 报 "File too large"（沙箱磁盘 5G，2 CPU）
  2. 用 LLM 标注语料从零训练统计模型（本模块做的事）
     → char n-gram TF-IDF + LinearSVC，语料 235 条
     → val 15.0%，低于词表基线 25.0%
     → 学习曲线是**下降**的（60 条 41.7% → 235 条 4.3%），增加数据反而更差

为什么更差：
  语料按「语义族」构造，同一族句子字面高度相近。模型学到的是族内字面
  特征而非语义，遇到真实分布里说法不同的句子就失效。逐维准确率有 85%，
  但九维全对 ≈ 0.85^9 ≈ 23%，就是独立概率的乘积 —— 说明模型没学到维度
  间的关联，只是在九个独立的字符匹配任务上各猜一次。

什么情况下可以再试：
  - 拿到真实分布的语料（不是我构造的），至少数千条
  - 或有可用的预训练中文小模型权重
"""

import os
import sys
import json
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CORPUS = "llm_train_corpus"
MODEL_PATH = os.path.join(DATA_DIR, "lean_model.json")


def _load_corpus():
    sys.path.insert(0, DATA_DIR)
    mod = __import__(CORPUS)
    return [{"text": t, "vector": v} for t, v in mod.SAMPLES]


def _load_json(name):
    if not name.endswith(".json"):
        name += ".json"
    with open(os.path.join(DATA_DIR, name), 'r', encoding='utf-8') as f:
        return json.load(f)["samples"]


class LeanModelEncoder:
    """char n-gram + 逐维线性分类器。

    之所以逐维单独训练而不 MultiOutputClassifier：某些维度在子集上只有
    单一类别，MultiOutputClassifier 会直接抛错，这里退化成常量预测。
    """

    def __init__(self, extra_sets=("golden_set.json", "train_set.json")):
        self.samples = _load_corpus()
        for name in extra_sets or ():
            self.samples += _load_json(name)
        self._fitted = False

    def _fit(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.svm import LinearSVC
        from sklearn.dummy import DummyClassifier

        X = [s["text"] for s in self.samples]
        Y = np.array([s["vector"] for s in self.samples])
        self.vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 5),
                                   min_df=1, sublinear_tf=True)
        A = self.vec.fit_transform(X)
        self.ests = []
        for d in range(9):
            if len(set(Y[:, d])) < 2:
                self.ests.append(DummyClassifier(strategy="constant",
                                                 constant=int(Y[:, d][0])))
            else:
                self.ests.append(LinearSVC(C=4.0, max_iter=5000))
            self.ests[-1].fit(A, Y[:, d])
        self._fitted = True

    def encode(self, texts):
        if not self._fitted:
            self._fit()
        A = self.vec.transform(texts)
        return np.column_stack([e.predict(A) for e in self.ests]).astype(int)

    def encode_one(self, text):
        return list(self.encode([text])[0])


class HybridEncoder:
    """词表优先融合：词表给得出非零信号就信词表，否则用模型补。

    holdout 上 36.7% → 46.7%，但 val 上 25.0% → 25.0% 无变化。
    两个只测集取较低者仍是 25%，所以**不构成泛化提升**，只是可选的兜底策略。
    """

    def __init__(self, local_projector=None, lean: LeanModelEncoder = None):
        from tne_projector_v5 import YuanbaoProjectorV4
        self.lex = local_projector or YuanbaoProjectorV4(use_llm="local").local
        self.lean = lean or LeanModelEncoder()

    def encode(self, text):
        lv = self.lex.encode(text)
        if all(v == 0 for v in lv.vector):
            mv = self.lean.encode_one(text)
            return {"vector": mv, "source": "lean", "confidence": lv.confidence}
        return {"vector": lv.vector, "source": "lexicon", "confidence": lv.confidence}


def run_experiment(verbose=True):
    """完整实验：学习曲线 + 三方案对比。复现 model_experiment.md 里的数字。"""
    from sklearn.model_selection import train_test_split
    from tne_projector_v5 import YuanbaoProjectorV4

    p = YuanbaoProjectorV4(use_llm="local")
    val = _load_json("val_set.json")
    hold = _load_json("holdout_set")
    corpus = _load_corpus()

    out = {"learning_curve": [], "compare": {}}

    X = [s["text"] for s in corpus]
    Y = np.array([s["vector"] for s in corpus])
    for n in (60, 120, 180, len(corpus)):
        Xs, Xt, Ys, Yt = train_test_split(X[:n], Y[:n], test_size=0.2, random_state=42)
        m = LeanModelEncoder(extra_sets=())
        m.samples = [{"text": t, "vector": v} for t, v in zip(Xs, Ys)]
        m._fit()
        P = m.encode(Xt)
        r = {"n": int(n), "exact": float((P == Yt).all(1).mean()),
             "per_dim": float((P == Yt).mean())}
        out["learning_curve"].append(r)
        if verbose:
            print(f"  语料 {n:>4} 条 → 留出集 {r['exact']*100:>5.1f}%  "
                  f"逐维 {r['per_dim']*100:>5.1f}%")

    lean = LeanModelEncoder()
    hy = HybridEncoder(lean=lean)
    for name, ds in (("val", val), ("holdout", hold)):
        Yg = np.array([g["vector"] for g in ds])
        texts = [g["text"] for g in ds]
        lex = sum(1 for g in ds if p.local.encode(g["text"]).vector == g["vector"]) / len(ds)
        mdl = float((lean.encode(texts) == Yg).all(1).mean())
        hyb = sum(1 for g in ds if hy.encode(g["text"])["vector"] == g["vector"]) / len(ds)
        out["compare"][name] = {"lexicon": lex, "lean": mdl, "hybrid": hyb}
        if verbose:
            print(f"  {name:<9} 词表 {lex*100:>5.1f}%   小模型 {mdl*100:>5.1f}%   "
                  f"融合 {hyb*100:>5.1f}%")

    return out


if __name__ == "__main__":
    print("离线小模型实验（可复现）")
    print("=" * 60)
    print("\n[1] 学习曲线 —— 语料越多越好吗？")
    run_experiment()
    print("\n[2] 读法：曲线下降 = 语料分布与真实分布不一致，加数据无用")
