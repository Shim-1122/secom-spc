"""Process drift diagnostics.

Fab data is not stationary: tools get maintained, recipes are tuned,
sensors are recalibrated. These checks quantify how much the process
moved between the baseline and the monitoring period, and whether a
sensor's link to failure is real or just a shared time trend.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline


def ks_drift(X: pd.DataFrame, early: np.ndarray, late: np.ndarray) -> pd.DataFrame:
    """Two-sample Kolmogorov-Smirnov statistic per sensor (early vs late)."""
    rows = []
    for c in X.columns:
        a = X[c].values[early]
        b = X[c].values[late]
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if a.size < 10 or b.size < 10:
            continue
        ks = stats.ks_2samp(a, b)
        sd = np.std(a, ddof=1)
        rows.append(
            {
                "sensor": c,
                "ks_stat": ks.statistic,
                "ks_p": ks.pvalue,
                "median_shift_in_early_sd": (np.median(b) - np.median(a)) / sd if sd > 0 else np.inf,
                "sd_ratio_late_early": np.std(b, ddof=1) / sd if sd > 0 else np.inf,
            }
        )
    return pd.DataFrame(rows).set_index("sensor").sort_values("ks_stat", ascending=False)


def adversarial_validation(X: pd.DataFrame, period_label: np.ndarray, seed: int = 0) -> tuple[float, pd.Series]:
    """How well can a model tell *which period* a run came from?

    AUC ~ 0.5 means the two periods look alike; AUC near 1.0 means the
    sensor distributions shifted enough that time is trivially
    recoverable, and a failure model trained on one period should not be
    expected to transfer to the other without retraining.
    """
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=seed),
    )
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    p = cross_val_predict(model, X.values, period_label, cv=cv, method="predict_proba")[:, 1]
    auc = roc_auc_score(period_label, p)
    model.fit(X.values, period_label)
    imp = pd.Series(model[-1].feature_importances_, index=X.columns).sort_values(ascending=False)
    return float(auc), imp


def frozen_sensors(X: pd.DataFrame, period: pd.Series, min_runs: int = 20) -> pd.DataFrame:
    """Sensors that report one repeated value for whole weeks, then move.

    A long stretch of identical readings usually means a held / stale
    value (sensor offline, PLC tag not updating) rather than a perfectly
    stable process, and it silently breaks control limits fit on it.
    """
    rows = []
    for c in X.columns:
        v = X[c]
        weekly = v.groupby(period.values).agg(lambda s: s.dropna().nunique() if s.notna().sum() >= min_runs else np.nan)
        frozen = (weekly == 1).sum()
        live = (weekly > 3).sum()
        if frozen >= 2 and live >= 2:
            first_live = weekly.index[weekly > 3][0]
            rows.append({"sensor": c, "frozen_weeks": int(frozen), "live_weeks": int(live), "first_live_week": str(first_live)})
    return pd.DataFrame(rows, columns=["sensor", "frozen_weeks", "live_weeks", "first_live_week"]).set_index("sensor")


def within_period_auc(X: pd.DataFrame, y: np.ndarray, strata: np.ndarray, min_fail: int = 2, min_pass: int = 5) -> pd.Series:
    """Fail-weighted average of per-stratum AUCs (e.g. per week).

    Compares fails only with passes from the *same* week, so a sensor
    that merely trends with time (while the fail rate also trends) gets
    no credit. A sensor whose pooled AUC is high but within-week AUC is
    ~0.5 is a time confound, not a failure driver.
    """
    y = np.asarray(y)
    strata = np.asarray(strata)
    out = {}
    groups = [np.where(strata == s)[0] for s in np.unique(strata)]
    groups = [g for g in groups if y[g].sum() >= min_fail and (y[g] == 0).sum() >= min_pass]
    for c in X.columns:
        v = X[c].values
        num = den = 0.0
        for g in groups:
            ok = g[np.isfinite(v[g])]
            yy = y[ok]
            if yy.sum() < min_fail or (yy == 0).sum() < min_pass or np.ptp(v[ok]) == 0:
                continue
            w = yy.sum()
            num += w * roc_auc_score(yy, v[ok])
            den += w
        out[c] = num / den if den else np.nan
    return pd.Series(out, name="within_week_auc")
