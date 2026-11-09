import numpy as np
import pandas as pd
import pytest

from secom_spc import drift as DR
from secom_spc import features as FE
from secom_spc.modeling import TopKByAUC
from secom_spc.preprocess import SensorScreen


def test_screen_drops_missing_constant_duplicate():
    rng = np.random.default_rng(0)
    a = rng.normal(size=200)
    X = pd.DataFrame(
        {
            "good": a,
            "dup": a * 2 + 1,
            "const": np.ones(200),
            "mostly_nan": np.where(np.arange(200) < 150, np.nan, 1.0),
            "other": rng.normal(size=200),
        }
    )
    s = SensorScreen().fit(X)
    assert s.report_.kept == ["good", "other"]
    assert s.report_.dropped_duplicate == {"dup": "good"}
    assert "const" in s.report_.dropped_constant
    assert "mostly_nan" in s.report_.dropped_missing


def test_imputation_uses_fit_medians_only():
    X1 = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
    X2 = pd.DataFrame({"a": [np.nan, 100.0, 200.0]})
    s = SensorScreen(corr_dedup_threshold=None).fit(X1)
    assert s.transform(X2).iloc[0, 0] == 3.0


def test_bh_fdr_matches_reference():
    p = np.array([0.01, 0.04, 0.03, 0.005])
    # sorted: .005 .01 .03 .04 -> *4/rank: .02 .02 .04 .04
    assert FE.bh_fdr(p) == pytest.approx([0.02, 0.04, 0.04, 0.02])


def test_univariate_tests_find_planted_signal():
    rng = np.random.default_rng(1)
    y = (rng.random(600) < 0.1).astype(int)
    X = pd.DataFrame({"signal": rng.normal(size=600) + 1.5 * y, "noise": rng.normal(size=600)})
    u = FE.univariate_tests(X, y)
    assert u.loc["signal", "mw_q"] < 1e-6
    assert u.loc["noise", "mw_q"] > 0.01
    assert u.loc["signal", "direction"] == "higher in fails"


def test_within_week_auc_exposes_time_confound():
    rng = np.random.default_rng(2)
    weeks = np.repeat(np.arange(10), 60)
    fail_rate = np.linspace(0.25, 0.02, 10)[weeks]
    y = (rng.random(600) < fail_rate).astype(int)
    trend = -weeks + rng.normal(scale=0.5, size=600)  # tracks time, not failure
    real = rng.normal(size=600) + 1.2 * y
    X = pd.DataFrame({"trend": trend, "real": real})
    pooled = FE.univariate_tests(X, y).auc
    within = DR.within_period_auc(X, y, weeks)
    assert pooled["trend"] > 0.65
    assert abs(within["trend"] - 0.5) < 0.1
    assert within["real"] > 0.75


def test_topk_by_auc_selects_signal():
    rng = np.random.default_rng(3)
    y = (rng.random(500) < 0.2).astype(int)
    X = np.c_[rng.normal(size=(500, 5)), rng.normal(size=500) + 2 * y]
    t = TopKByAUC(k=1).fit(X, y)
    assert t.support_.tolist() == [5]
    assert t.transform(X).shape == (500, 1)
