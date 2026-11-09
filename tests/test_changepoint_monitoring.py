import numpy as np
import pandas as pd

from secom_spc import changepoint, drift, monitoring
from secom_spc.multivariate import fit_mspc


def _frame(n=600, p=20, shift_at=None, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_t(df=4, size=(n, p))
    if shift_at is not None:
        X[shift_at:, :5] += 1.5
    dates = pd.Series(pd.date_range("2008-07-01", periods=n // 20, freq="D").repeat(20))
    ts = dates + pd.to_timedelta(np.tile(np.arange(20), n // 20), unit="h")
    return pd.DataFrame(X, columns=[f"S{i}" for i in range(p)]), dates, ts


def test_changepoint_recovers_planted_shift_at_day_boundary():
    X, dates, ts = _frame(shift_at=380)
    cp = changepoint.detect(X, dates, ts)
    assert cp.index == 380
    assert cp.timestamp == ts.iloc[380]


def test_changepoint_no_shift_gives_weak_profile():
    X, dates, ts = _frame(shift_at=None)
    Xs, _, _ = _frame(shift_at=300)
    flat = changepoint.detect(X, dates, ts).statistic
    shifted = changepoint.detect(Xs, dates, ts).statistic
    assert shifted > 5 * flat


def test_shift_spread_measures_shift_and_spread():
    rng = np.random.default_rng(1)
    early = np.r_[np.ones(500, bool), np.zeros(500, bool)]
    X = pd.DataFrame({
        "moved": np.r_[rng.normal(0, 1, 500), rng.normal(3, 1, 500)],
        "wider": np.r_[rng.normal(0, 1, 500), rng.normal(0, 4, 500)],
        "same": rng.normal(0, 1, 1000),
    })
    t = drift.shift_spread(X, early, ~early)
    assert abs(t.loc["moved", "shift_robust_sd"] - 3) < 0.4
    assert abs(t.loc["wider", "spread_ratio"] - 4) < 0.6
    assert abs(t.loc["same", "shift_robust_sd"]) < 0.3
    assert t.index[-1] == "same"


def test_out_of_fold_scoring_removes_in_sample_optimism():
    rng = np.random.default_rng(2)
    X = pd.DataFrame(rng.normal(size=(400, 60)))
    ref = np.arange(400) < 200
    in_sample = fit_mspc(X, ref, cv_folds=None)
    oof = fit_mspc(X, ref, cv_folds=5)
    # in-sample SPE of reference runs is optimistic relative to new runs; out-of-fold is not
    assert in_sample.spe[ref].mean() < 0.8 * in_sample.spe[~ref].mean()
    assert 0.8 < oof.spe[ref].mean() / oof.spe[~ref].mean() < 1.25


def test_burden_counts():
    alarm = np.array([True, False, True, False, False, True])
    y = np.array([1, 0, 0, 0, 1, 0])
    dates = pd.Series(pd.to_datetime(["2008-01-01"] * 3 + ["2008-01-02"] * 3))
    b = monitoring.burden(alarm, y, dates, np.ones(6, bool))
    assert b["alarms_per_day"] == 1.5
    assert b["false_alarms_per_day"] == 1.0
    assert b["fails_flagged_pct"] == 50.0
