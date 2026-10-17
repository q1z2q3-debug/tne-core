"""
TNE-Core · 单元测试
验证公理算子、等价类、距离、轨迹库。
论文锁定的单元测试样例必须全部通过。
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tne.space.space import encode, decode, TOTAL_STATES, DIM, gua_str
from tne.space.operators import f_cuoo, f_zong, f_cuoo_zong, f_hu, f_bian, equivalence_set
from tne.space.equivalence import count_equivalence_classes, class_of, get_equivalence
from tne.space.distance import euclidean, nearest_neighbors, squared_distance


def test_space_encode_decode():
    """编码/解码可逆，全集 19683。"""
    assert TOTAL_STATES == 19683
    assert DIM == 9
    # 边界
    assert decode(0) == [-1, -1, -1, -1, -1, -1, -1, -1, -1]
    assert decode(19682) == [1, 1, 1, 1, 1, 1, 1, 1, 1]
    # 可逆
    for idx in [0, 1, 100, 9999, 19682]:
        assert encode(decode(idx)) == idx
    print("✓ space encode/decode OK")


def test_operators_paper_example():
    """论文锁定的单元测试样例，必须精确匹配。"""
    G = [1, 1, 0, -1, 1, 0, -1, -1, 0]
    # 1. F_C
    assert f_cuoo(G) == [-1, -1, 0, 1, -1, 0, 1, 1, 0], f"F_C fail: {f_cuoo(G)}"
    # 2. F_Z
    assert f_zong(G) == [0, -1, -1, 0, 1, -1, 0, 1, 1], f"F_Z fail: {f_zong(G)}"
    # 3. F_CZ
    assert f_cuoo_zong(G) == [0, 1, 1, 0, -1, 1, 0, -1, -1], f"F_CZ fail: {f_cuoo_zong(G)}"
    # 4. F_H (5维内核)
    assert f_hu(G) == [0, -1, 1, 0, -1], f"F_H fail: {f_hu(G)}"
    # 5. 第5爻动
    assert f_bian(G, {5}) == [1, 1, 0, -1, 0, 0, -1, -1, 0], f"F_B fail: {f_bian(G, {5})}"
    print("✓ operators paper example ALL PASS")


def test_operator_involutions():
    """算子对合性质：F_C(F_C(G))=G, F_Z(F_Z(G))=G。"""
    G = [1, 0, -1, 1, 0, -1, 1, 0, -1]
    assert f_cuoo(f_cuoo(G)) == G
    assert f_zong(f_zong(G)) == G
    # F_CZ 也是对合
    assert f_cuoo_zong(f_cuoo_zong(G)) == G
    print("✓ operator involutions OK")


def test_equivalence_count():
    """等价类总数应为 5002。"""
    n = count_equivalence_classes()
    assert n == 5002, f"等价类总数应为 5002，实际 {n}"
    print(f"✓ equivalence classes = {n} OK")


def test_equivalence_closure():
    """等价类内任意卦的等价集合都指向同一组。"""
    class_of_map, groups = get_equivalence()
    # 抽查 100 个卦
    import random
    random.seed(42)
    for idx in random.sample(range(TOTAL_STATES), 100):
        gua = decode(idx)
        members = equivalence_set(gua)
        reps = set(class_of_map[encode(m)] for m in members)
        assert len(reps) == 1, f"卦 {idx} 等价类不闭合: {reps}"
    print("✓ equivalence closure OK (100 samples)")


def test_distance():
    """距离函数正确性 + 最近邻检索。"""
    a = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    b = [-1, -1, -1, -1, -1, -1, -1, -1, -1]
    assert squared_distance(a, b) == 9 * 4  # 每维差2，平方4，9维=36
    assert abs(euclidean(a, b) - 6.0) < 1e-9  # sqrt(36)=6
    # 最近邻：自己到自己距离为0
    results = nearest_neighbors(a, top_n=3)
    assert results[0][1] == 0.0
    assert results[0][2] == a
    print("✓ distance & nearest_neighbors OK")


def test_trajectory_store():
    """轨迹库写入与查询。"""
    import tempfile
    from tne.trajectory.store import TrajectoryStore
    from tne.trajectory.habit import HabitQuery
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db = f.name
    store = TrajectoryStore(db)
    G = [1, 1, 0, -1, 1, 0, -1, -1, 0]
    # 写入 5 条同类成功轨迹
    for i in range(5):
        store.record("test_session", G, "tool_call", {"tool": f"t{i}"}, "success", reward=0.8)
    # 写入 1 条失败
    store.record("test_session", G, "final_answer", None, "failure", reward=-0.2)
    cov = store.class_coverage()
    assert cov["total_records"] == 6
    assert cov["covered_classes"] >= 1
    # 习惯查询
    hq = HabitQuery(store, min_freq=3)
    result = hq.query(G)
    assert result["mode"] in ("habit", "habit_weak")
    assert result["freq"] == 6
    assert len(result["top_actions"]) >= 1
    store.close()
    os.unlink(db)
    print("✓ trajectory store & habit query OK")


if __name__ == "__main__":
    test_space_encode_decode()
    test_operators_paper_example()
    test_operator_involutions()
    test_equivalence_count()
    test_equivalence_closure()
    test_distance()
    test_trajectory_store()
    print("\n🎉 ALL TESTS PASSED")
