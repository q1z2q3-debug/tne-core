"""TNE-Core 集成测试"""

import unittest
import sys
import os
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tne_projector_v5 import YuanbaoProjectorV4, LocalSemanticBackend
from tne_distiller import TNE_Distiller, TrajectoryRecord
from tne_trajectory_collector import TrajectoryCollector
from tne_schema import TrajectoryRecord as SchemaRecord, state_signature, signature_distance
from tne_ingest import TNESession, build_batch_from_texts

import json as _json
import os as _os
import sys as _sys
import numpy as np


def _load_set(filename):
    path = _os.path.join(_os.path.dirname(__file__), '..', 'data', filename)
    with open(path, 'r', encoding='utf-8') as f:
        return _json.load(f)["samples"]


class TestProjector(unittest.TestCase):
    def setUp(self):
        self.p = YuanbaoProjectorV4()

    def test_encode_9d(self):
        r = self.p.encode("写代码")
        self.assertEqual(len(r.vector), 9)

    def test_encode_creative(self):
        r = self.p.encode("创作一首诗")
        self.assertEqual(r.vector[0], 1)

    def test_encode_continuation(self):
        r = self.p.encode("继续写文档")
        self.assertEqual(r.vector[1], 1)

    def test_batch(self):
        rs = self.p.batch_encode(["写代码", "测试"])
        self.assertEqual(len(rs), 2)


class TestProjectorV5(unittest.TestCase):
    """V5 相对 V4 新增能力的回归测试"""

    def setUp(self):
        self.p = YuanbaoProjectorV4(use_llm="local")

    def test_negative_dim_reachable(self):
        """V4 的硬伤：-1 几乎不可达。V5 必须能打出负维"""
        r = self.p.encode("把旧代码删掉")
        self.assertEqual(r.vector[0], -1)

    def test_multi_negative(self):
        r = self.p.encode("服务崩了，环境跑不起来")
        self.assertEqual(r.vector[3], -1)
        self.assertEqual(r.vector[5], -1)

    def test_dim6_not_triggered_by_pronoun(self):
        """V4 的硬伤：裸「我/你」就点亮协作维。V5 必须免疫"""
        r = self.p.encode("我打算换到另一个项目组")
        self.assertEqual(r.vector[6], 0)  # 换组是环境切换，不是协作
        self.assertEqual(r.vector[3], 1)

    def test_dim6_real_collaboration(self):
        r = self.p.encode("和团队一起拉通对齐")
        self.assertEqual(r.vector[6], 1)

    def test_conflict_suppresses_collaboration(self):
        r = self.p.encode("团队又闹僵了")
        self.assertEqual(r.vector[6], -1)

    def test_fault_suppresses_tool(self):
        r = self.p.encode("接口全挂了")
        self.assertEqual(r.vector[4], 0)

    def test_negation_not_polarity_flip(self):
        """「别删」不等于「创造」，应回到 0"""
        r = self.p.encode("别删了我的文件")
        self.assertEqual(r.vector[0], 0)

    def test_bie_morpheme_not_negation(self):
        """「特别来劲」的「别」是词素，不能当否定词"""
        r = self.p.encode("特别来劲想做")
        self.assertEqual(r.vector[8], 1)

    def test_phrase_beats_word_root(self):
        """「没工具」不能被词根「工具」抵消"""
        r = self.p.encode("没工具，缺库，装不上")
        self.assertEqual(r.vector[4], -1)

    def test_neutral_anchor(self):
        """框架里 d1=0 的锚点是「首次」，不是负向"""
        r = self.p.encode("第一次尝试做播客")
        self.assertEqual(r.vector[1], 0)

    def test_golden_set_match(self):
        """golden_set 是拟合集（边调词表边跑），只作回归保护，不代表泛化能力"""
        golden = _load_set("golden_set.json")
        report = self.p.evaluate_against(golden)
        self.assertGreaterEqual(report["exact_match_rate"], 0.90)

    def test_holdout_baseline(self):
        """holdout 是未参与调词的独立集，测量真实泛化率。

        这里只做下限保护，不设高阈值 —— 基线约 17%，把它断言成 90%
        等于自欺欺人。作用是在词表被改动后捕捉崩溃式退化。
        """
        holdout = _load_set("holdout_set.json")
        report = self.p.evaluate_against(holdout)
        self.assertGreaterEqual(report["exact_match_rate"], 0.10)

    def test_holdout_stays_honest(self):
        """防作弊：holdout 不能被改造成第二个拟合集。

        如果哪天有人对着 holdout 调词表到 100%，说明这个集已经失效。
        """
        holdout = _load_set("holdout_set.json")
        report = self.p.evaluate_against(holdout)
        self.assertLess(report["exact_match_rate"], 0.99,
                        "holdout 吻合率接近 100%，说明该集已被污染，需重建")

    def test_no_polarity_errors(self):
        """方向判断不能串。只检查可调集（golden/train）——
        holdout 上现存 1 例「客户临时改验收标准」（期望 d2=-1，预测 +1）：
        「临时」的方向取决于施动者（内部安排=驱动，客户变更=阻力），
        这是关键词法的固有局限，不通过针对 holdout 改词来消除。"""
        for name in ("golden_set.json", "train_set.json"):
            d = YuanbaoProjectorV4.diagnose(_load_set(name))
            self.assertEqual(d["POLARITY"], 0, f"{name} 出现极性错判")

    def test_bucuo_not_negation(self):
        """「不错」里的「不」是词素，不能把它后面的词判为否定"""
        r = self.p.encode("这周状态不错，效率很高")
        self.assertNotEqual(r.vector[5], 0, "「效率」被「不错」的「不」误判为否定")
        self.assertNotEqual(r.vector[7], 0)

    def test_val_baseline(self):
        """val 是只测集，此处只设下限防崩溃，不代表目标值"""
        report = self.p.evaluate_against(_load_set("val_set.json"))
        self.assertGreaterEqual(report["exact_match_rate"], 0.20)

    def test_split_is_deterministic(self):
        """划分必须可复现：同一 seed 必须得到同一结果"""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from split_dataset import split
        s = _load_set("expanded_set.json")
        a1, b1 = split(s)
        a2, b2 = split(s)
        self.assertEqual([x["id"] for x in a1], [x["id"] for x in a2])
        self.assertEqual([x["id"] for x in b1], [x["id"] for x in b2])
        self.assertEqual(len(a1) + len(b1), len(s))
        self.assertEqual(set(x["id"] for x in a1) & set(x["id"] for x in b1), set())

    def test_train_val_disjoint(self):
        tr = {x["id"] for x in _load_set("train_set.json")}
        va = {x["id"] for x in _load_set("val_set.json")}
        self.assertEqual(tr & va, set(), "train/val 有重叠，测量会失真")

    def test_diagnose_reports_miss_dominant(self):
        """归因应指出主矛盾是漏召回（词表盲区），而非方向错误"""
        d = YuanbaoProjectorV4.diagnose(_load_set("holdout_set.json"))
        self.assertEqual(d["dominant"], "MISS")
        self.assertGreater(d["miss_rate"], 0.5)

    def test_result_has_backend_field(self):
        r = self.p.encode("写代码")
        self.assertEqual(r.backend, "local")
        self.assertEqual(len(r.dim_scores), 9)


class TestDistiller(unittest.TestCase):
    def setUp(self):
        self.d = TNE_Distiller(min_samples=2, min_success_rate=0.85)

    def test_create_pattern(self):
        rec = TrajectoryRecord(
            task_id="t1", timestamp=1.0,
            task_type="创作", state_before=[1]*9, state_after=[1]*9,
            habits_used=["以终为始", "统合综效"],
            efficacy_score=0.95, patterns_extracted=[]
        )
        ext = self.d.add_trajectory(rec)
        self.assertEqual(len(ext), 1)

    def test_low_efficacy(self):
        rec = TrajectoryRecord(
            task_id="t2", timestamp=2.0,
            task_type="测试", state_before=[0]*9, state_after=[0]*9,
            habits_used=["以终为始"],
            efficacy_score=0.5, patterns_extracted=[]
        )
        ext = self.d.add_trajectory(rec)
        self.assertEqual(len(ext), 0)

    def test_accumulate(self):
        for i in range(3):
            rec = TrajectoryRecord(
                task_id=f"a{i}", timestamp=float(i),
                task_type="创作", state_before=[1,1,0,1,1,0,1,1,1],
                state_after=[1,1,0,1,1,0,1,1,1],
                habits_used=["以终为始", "统合综效"],
                efficacy_score=0.9, patterns_extracted=[]
            )
            self.d.add_trajectory(rec)
        self.assertEqual(self.d.get_stats()["total"], 1)

    def test_recommend(self):
        for i in range(3):
            rec = TrajectoryRecord(
                task_id=f"r{i}", timestamp=float(i),
                task_type="创作", state_before=[1,1,0,1,1,0,1,1,1],
                state_after=[1,1,0,1,1,0,1,1,1],
                habits_used=["以终为始", "统合综效"],
                efficacy_score=0.95, patterns_extracted=[]
            )
            self.d.add_trajectory(rec)
        rec = self.d.recommend_pattern("创作", [1,1,0,1,1,0,1,1,1])
        self.assertIsNotNone(rec)


class TestSchema(unittest.TestCase):
    """v3.2：TrajectoryRecord 必须只有一份定义"""

    def test_single_definition(self):
        import tne_distiller as d
        import tne_trajectory_collector as c
        self.assertIs(d.TrajectoryRecord, c.TrajectoryRecord)
        self.assertIs(d.TrajectoryRecord, SchemaRecord)

    def test_validate_rejects_bad_efficacy(self):
        r = SchemaRecord.create("t", "创作", [1, 0, 0, 0, 0, 0, 0, 0, 0], ["a", "b"], 1.8)
        self.assertTrue(any("efficacy_score" in e for e in r.validate()))

    def test_validate_rejects_short_habits(self):
        r = SchemaRecord.create("t", "创作", [1, 0, 0, 0, 0, 0, 0, 0, 0], ["a"], 0.9)
        self.assertTrue(any("habits_used" in e for e in r.validate()))

    def test_validate_rejects_bad_vector(self):
        r = SchemaRecord.create("t", "创作", [2] * 9, ["a", "b"], 0.9)
        self.assertTrue(r.validate())

    def test_roundtrip(self):
        r = SchemaRecord.create("t", "创作", [1, 0, 0, 0, 0, 0, 0, 0, 0], ["a", "b"], 0.9)
        self.assertEqual(SchemaRecord.from_dict(r.to_dict()).task_id, "t")

    def test_unknown_field_not_silently_dropped(self):
        d = SchemaRecord.create("t", "创作", [0] * 9, ["a", "b"], 0.9).to_dict()
        d["surprise"] = 1
        with self.assertRaises(ValueError):
            SchemaRecord.from_dict(d)

    def test_state_signature_sums_to_nine(self):
        for v in ([1] * 9, [-1] * 9, [0] * 9, [1, 0, -1, 0, 0, 0, 0, 0, 0]):
            s = state_signature(v)
            self.assertEqual(s["p"] + s["z"] + s["n"], 9)

    def test_signature_distance_bounds(self):
        self.assertEqual(signature_distance([1] * 9, "p9z0n0"), 0.0)
        self.assertGreater(signature_distance([1] * 9, "p0z0n9"), 0.9)

    def test_round_to_nine(self):
        from tne_distiller import _round_to_nine
        for case in [(2.0, 7.5, -0.5), (4.5, 4.5, 0.0), (1.33, 7.33, 0.34), (0.0, 9.0, 0.0)]:
            r = _round_to_nine(*case)
            self.assertEqual(sum(r), 9)
            self.assertTrue(all(x >= 0 for x in r), f"{case} 出现负分量 {r}")


class TestSignatureAccumulation(unittest.TestCase):
    """v3.2 修掉的真 bug：均值被当成累积和存回，signature 三次后失真成 p0z1n8"""

    def test_signature_stays_stable_over_many_updates(self):
        d = TNE_Distiller()
        vec = [1, 0, 0, 0, 0, 0, 0, 0, 0]  # 恒定 p1z8n0
        for i in range(10):
            d.add_trajectory(TrajectoryRecord.create(
                task_id=f"s{i}", task_type="创作", state_before=vec,
                habits_used=["以终为始", "统合综效"], efficacy_score=0.95))
        p = list(d.patterns.values())[0]
        self.assertEqual(p.state_signature(), "p1z8n0",
                         "signature 随样本数漂移，说明又存成了均值")

    def test_signature_of_two_positives(self):
        d = TNE_Distiller()
        for i in range(5):
            d.add_trajectory(TrajectoryRecord.create(
                task_id=f"t{i}", task_type="开发",
                state_before=[1, 0, 0, 0, 1, 0, 0, 0, 0],
                habits_used=["以终为始", "统合综效"], efficacy_score=0.95))
        self.assertEqual(list(d.patterns.values())[0].state_signature(), "p2z7n0")

    def test_recommend_uses_state(self):
        """负向维不再被忽略：全负状态不该匹配到全正模式"""
        d = TNE_Distiller(min_samples=1)
        for i in range(3):
            d.add_trajectory(TrajectoryRecord.create(
                task_id=f"p{i}", task_type="创作", state_before=[1] * 9,
                habits_used=["以终为始", "统合综效"], efficacy_score=0.95))
        hi = d.recommend_with_scores("创作", [1] * 9)[0]["compat"]
        lo = d.recommend_with_scores("创作", [-1] * 9)[0]["compat"]
        self.assertGreater(hi, lo)
        self.assertAlmostEqual(hi, 1.0, places=2)


class TestBatchIngest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_batch_ingest(self):
        with TNESession(data_dir=self.tmpdir, fresh=True) as s:
            items = build_batch_from_texts(
                ["写一段对话", "继续写下一章", "再写一段"],
                "创作叙事", [0.92, 0.94, 0.93])
            rep = s.ingest_batch(items)
            self.assertEqual(rep["ingested"], 3)
            self.assertEqual(len(rep["new_patterns"]), 1)  # 同 habits 链 → 一个模式

    def test_dedupe_across_batches(self):
        """同一 session 内重复 task_id 只入库一次"""
        with TNESession(data_dir=self.tmpdir, fresh=True) as s:
            items = build_batch_from_texts(["写一段对话"], "创作叙事", [0.92])
            items[0]["task_id"] = "fixed_id"
            s.ingest_batch(items)
            s.ingest_batch(items)  # 完全相同的一批
            self.assertEqual(s.stats["ingested"], 1)
            self.assertEqual(s.stats["dupe_skipped"], 1)

    def test_invalid_not_ingested(self):
        with TNESession(data_dir=self.tmpdir, fresh=True) as s:
            rep = s.ingest_batch([
                {"text": "写一段对话", "task_type": "创作", "efficacy": 1.8,
                 "task_id": "bad", "habits": ["a", "b"]},
                {"text": "写一段对话", "task_type": "创作", "efficacy": 0.9,
                 "task_id": "ok", "habits": ["a", "b"]},
            ])
            self.assertEqual(rep["ingested"], 1)
            self.assertEqual(rep["invalid"], 1)

    def test_session_persists_patterns(self):
        """跨 session 累积：第二次入库应看到第一次的 usage_count"""
        with TNESession(data_dir=self.tmpdir, fresh=True) as s1:
            s1.ingest_batch(build_batch_from_texts(
                ["写一段对话", "继续写"], "创作叙事", [0.92, 0.93]))
        with TNESession(data_dir=self.tmpdir, fresh=False) as s2:
            self.assertEqual(s2.distiller.get_stats()["total"], 1)
            s2.ingest_batch(build_batch_from_texts(["再写一段"], "创作叙事", [0.94]))
            p = list(s2.distiller.patterns.values())[0]
            self.assertEqual(p.usage_count, 3)
        with TNESession(data_dir=self.tmpdir, fresh=False) as s3:
            self.assertEqual(list(s3.distiller.patterns.values())[0].usage_count, 3)

    def test_recommend_from_text(self):
        with TNESession(data_dir=self.tmpdir, fresh=True) as s:
            s.ingest_batch(build_batch_from_texts(
                ["写一段对话", "继续写下一章"], "创作叙事", [0.92, 0.94]))
            recs = s.recommend(task_type="创作叙事", text="再写一段父女对白")
            self.assertEqual(len(recs), 1)
            self.assertGreater(recs[0]["compat"], 0.8)

    def test_low_efficacy_never_enters_patterns(self):
        with TNESession(data_dir=self.tmpdir, fresh=True) as s:
            s.ingest_batch(build_batch_from_texts(
                ["服务崩了", "线上挂了"], "故障处置", [0.3, 0.4]))
            self.assertEqual(s.distiller.get_stats()["total"], 0)
            self.assertEqual(s.collector.stats()["total"], 2)  # 轨迹仍入库


class TestCollectorBatch(unittest.TestCase):
    def test_record_many_writes_all(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            tmp = f.name
        try:
            c = TrajectoryCollector(storage_path=tmp)
            recs = [TrajectoryRecord.create(f"m{i}", "测试", [0] * 9, ["a", "b"], 0.9)
                    for i in range(5)]
            r = c.record_many(recs)
            self.assertEqual(r["written"], 5)
            self.assertEqual(len(c.load_all()), 5)
        finally:
            os.unlink(tmp)

    def test_batch_dedupe(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            tmp = f.name
        try:
            c = TrajectoryCollector(storage_path=tmp)
            recs = [TrajectoryRecord.create("same", "测试", [0] * 9, ["a", "b"], 0.9)
                    for _ in range(3)]
            r = c.record_many(recs)
            self.assertEqual(r["written"], 1)
            self.assertEqual(r["skipped_dupe"], 2)
        finally:
            os.unlink(tmp)

    def test_stats_range(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            tmp = f.name
        try:
            c = TrajectoryCollector(storage_path=tmp)
            c.record_many([TrajectoryRecord.create(f"s{i}", "测试", [0] * 9, ["a", "b"], 0.1 * i)
                           for i in range(1, 10)])
            st = c.stats()
            self.assertAlmostEqual(st["min_efficacy"], 0.1, places=2)
            self.assertAlmostEqual(st["max_efficacy"], 0.9, places=2)
        finally:
            os.unlink(tmp)


class TestLeanModelExperiment(unittest.TestCase):
    """小模型路线的实验结论固化。

    这些不是「功能正确」的测试，是把「这条路走不通」这件事写进代码，
    免得以后有人（包括我）又重跑一遍得出同样结论。
    """

    def test_lean_model_exists_and_runs(self):
        from tne_lean_model import LeanModelEncoder
        m = LeanModelEncoder()
        v = m.encode_one("优化数据库查询性能")
        self.assertEqual(len(v), 9)
        self.assertTrue(all(x in (-1, 0, 1) for x in v))

    def test_lean_model_no_better_than_lexicon_on_val(self):
        """核心负结论：小模型在只测集上不能超过词表。

        注意断言的是 val（只测集）。holdout 上融合确实更高（40% > 36.7%），
        但按「两个只测集取较低者」的口径，真实水平仍是 val 的 25%。
        """
        from tne_lean_model import LeanModelEncoder
        m = LeanModelEncoder()
        val = _load_set("val_set.json")
        Yg = np.array([g["vector"] for g in val])
        acc = float((m.encode([g["text"] for g in val]) == Yg).all(1).mean())
        self.assertLess(acc, 0.25,
                        "小模型超过词表基线 25%，说明学习曲线结论已失效，需重做实验")

    def test_learning_curve_is_not_monotonic_up(self):
        """学习曲线不能是单调上升的 —— 它实际上在下降。

        若哪天变成上升，说明语料分布问题已解决，本实验结论作废。
        """
        curve = [0.417, 0.333, 0.056, 0.043]  # 60 / 120 / 180 / 235
        self.assertGreater(curve[0], curve[-1],
                           "学习曲线变上升，语料分布问题可能已解决")

    def test_corpus_leakage_check(self):
        """语料不能与只测集重叠，否则实验无效"""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data'))
        from llm_train_corpus import SAMPLES
        texts = {t for t, _ in SAMPLES}
        for name in ("val_set.json", "holdout_set.json"):
            overlap = texts & {s["text"] for s in _load_set(name)}
            self.assertEqual(overlap, set(), f"语料与 {name} 重叠: {overlap}")


if __name__ == "__main__":
    unittest.main()
