import numpy as np
import pytest

from secom_spc import spc


@pytest.mark.parametrize("n,expected", [(2, 0.7979), (5, 0.9400), (10, 0.9727), (25, 0.9896)])
def test_c4_matches_published_table(n, expected):
    assert spc.c4(n) == pytest.approx(expected, abs=1e-4)


def test_xbar_s_constants_n5():
    # Montgomery, Appendix VI: A3 = 1.427, B3 = 0, B4 = 2.089 for n = 5
    assert spc.a3(5) == pytest.approx(1.427, abs=1e-3)
    b3, b4 = spc.b3_b4(5)
    assert b3 == 0.0
    assert b4 == pytest.approx(2.089, abs=1e-3)


def test_mr_chart_constants():
    d3, d4 = spc.d3_d4(2)
    assert d3 == 0.0
    assert d4 == pytest.approx(3.267, abs=2e-3)


def test_imr_sigma_recovers_true_sigma():
    rng = np.random.default_rng(1)
    x = rng.normal(10, 2, 5000)
    li, _ = spc.imr_limits(x, np.ones_like(x, dtype=bool), max_iter=0)
    assert li.center == pytest.approx(10, abs=0.1)
    assert li.sigma == pytest.approx(2, rel=0.05)


def test_imr_false_alarm_rate_close_to_nominal():
    rng = np.random.default_rng(2)
    x = rng.normal(0, 1, 20000)
    base = np.zeros(x.size, dtype=bool)
    base[:5000] = True
    ch = spc.imr_chart(x, base, max_iter=0)
    # P(|Z| > 3) = 0.27 %
    assert ch.rule1[~base].mean() == pytest.approx(0.0027, abs=0.0015)


def test_imr_detects_shift():
    rng = np.random.default_rng(3)
    x = np.r_[rng.normal(0, 1, 300), rng.normal(4, 1, 50)]
    base = np.arange(x.size) < 300
    ch = spc.imr_chart(x, base)
    assert ch.rule1[~base].mean() > 0.4


def test_phase1_cleanup_removes_outliers():
    rng = np.random.default_rng(4)
    x = rng.normal(0, 1, 500)
    x[::50] = 25.0
    li, _ = spc.imr_limits(x, np.ones(500, dtype=bool))
    assert li.n_removed >= 10
    assert abs(li.center) < 0.2


def test_xbar_s_chart_shapes_and_phase():
    rng = np.random.default_rng(5)
    x = rng.normal(0, 1, 503)
    base = np.arange(503) < 250
    st = spc.xbar_s_chart(x, base, size=5)
    assert len(st) == 100  # ragged tail of 3 dropped
    assert st.phase_I.sum() == 50
    assert (st.ucl > st.cl).all()


def test_p_chart_limits():
    fails = np.array([5, 4, 6, 5, 30])
    totals = np.array([100, 100, 100, 100, 100])
    base = np.array([True, True, True, True, False])
    pc = spc.p_chart(fails, totals, base)
    assert pc.pbar.iloc[0] == pytest.approx(0.05)
    assert pc.ooc_high.iloc[-1]
    assert not pc.ooc_high.iloc[:4].any()


def test_ewma_and_cusum_detect_small_shift_faster_than_shewhart():
    rng = np.random.default_rng(6)
    x = np.r_[rng.normal(0, 1, 200), rng.normal(1.0, 1, 200)]
    ew = spc.ewma_chart(x, 0.0, 1.0)
    cu = spc.cusum_chart(x, 0.0, 1.0)
    first_ewma = np.argmax(ew.ooc.values[200:])
    first_cusum = np.argmax(cu.ooc.values[200:])
    first_shew = np.argmax(np.abs(x[200:]) > 3)
    assert ew.ooc.values[200:].any() and cu.ooc.values[200:].any()
    assert min(first_ewma, first_cusum) <= first_shew


def test_phase_label_follows_time_mask_not_reference_mask():
    # A failed run inside Phase I is excluded from the limit baseline but
    # must still be labelled Phase I.
    x = np.random.default_rng(7).normal(size=100)
    phase1 = np.arange(100) < 50
    ref = phase1.copy()
    ref[10] = False
    ch = spc.imr_chart(x, ref, phase1=phase1)
    assert ch.phase.iloc[10] == "I"
    assert (ch.phase == "I").sum() == 50
    st = spc.xbar_s_chart(x, ref, size=5, phase1=phase1)
    assert st.phase_I.sum() == 10
    assert st.baseline.sum() == 9
