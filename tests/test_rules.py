import numpy as np

from secom_spc import rules as R


def test_rule1():
    z = np.array([0, 3.1, -3.5, 2.9])
    assert R.rule1(z).tolist() == [False, True, True, False]


def test_rule2_nine_same_side():
    z = np.r_[np.full(8, 0.5), -0.1, np.full(9, 0.2)]
    f = R.rule2(z)
    assert not f[:9].any()
    assert f[-1] and f.sum() == 1


def test_rule3_six_trending():
    z = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.4])
    f = R.rule3(z)
    assert f[5] and f.sum() == 1


def test_rule4_fourteen_alternating():
    z = np.array([(-1) ** i * 0.5 for i in range(14)])
    f = R.rule4(z)
    assert f[13] and f.sum() == 1
    assert not R.rule4(z[:13]).any()


def test_rule5_two_of_three():
    z = np.array([0, 2.5, 0.0, 2.2, 0])
    f = R.rule5(z)
    assert f[3]


def test_rule6_four_of_five():
    z = np.array([1.2, 1.5, 0.2, 1.1, 1.3])
    assert R.rule6(z)[4]


def test_rule7_stratification():
    z = np.full(15, 0.3)
    assert R.rule7(z)[14] and R.rule7(z).sum() == 1


def test_rule8_mixture():
    z = np.array([1.5, -1.5] * 4)
    assert R.rule8(z)[7]


def test_nan_breaks_patterns():
    z = np.r_[np.full(5, 0.5), np.nan, np.full(5, 0.5)]
    res = R.evaluate(z)
    assert not res[2].any()
    assert not np.isnan(z[res[1]]).any()


def test_in_control_noise_rarely_trips_rule1():
    z = np.random.default_rng(0).normal(size=10000)
    assert R.rule1(z).mean() < 0.005
