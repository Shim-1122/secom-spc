"""Ranking which sensors predict wafer failure.

Five complementary views are combined into a consensus rank:

1. **Mann-Whitney U** two-sided test (pass vs fail distributions), with
   Benjamini-Hochberg FDR control. Non-parametric because most SECOM
   sensors are skewed or heavy tailed.
2. **Univariate AUC** effect size, reported as |AUC - 0.5| so direction
   does not matter.
3. **Mutual information**, which also catches non-monotonic links.
4. **L1 logistic stability selection** (Meinshausen & Buehlmann, 2010):
   how often a sensor survives a sparse fit on random half-samples.
5. **Extra-Trees importance** with balanced class weights, averaged
   over seeds (multivariate, interaction-aware).

An SPC-specific view is added on top: **alarm lift**, the fail rate
among runs where the sensor's I-chart signalled divided by the overall
fail rate.

All rankings are fit on the training period only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from ._compat import l1_logistic


def bh_fdr(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (q-values)."""
    p = np.asarray(p, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


def univariate_tests(X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    rows = []
    y = np.asarray(y)
    for c in X.columns:
        x = X[c].values
        ok = np.isfinite(x)
        a, b = x[ok & (y == 1)], x[ok & (y == 0)]
        if a.size < 3 or b.size < 3 or np.ptp(x[ok]) == 0:
            rows.append((c, 1.0, 0.5, np.nan))
            continue
        _, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        auc = roc_auc_score(y[ok], x[ok])
        d = (np.median(a) - np.median(b)) / (stats.iqr(x[ok]) / 1.349 or 1.0)
        rows.append((c, p, auc, d))
    df = pd.DataFrame(rows, columns=["sensor", "mw_p", "auc", "median_shift_robust_sd"]).set_index("sensor")
    df["mw_q"] = bh_fdr(df.mw_p.values)
    df["auc_effect"] = (df.auc - 0.5).abs()
    df["direction"] = np.where(df.auc >= 0.5, "higher in fails", "lower in fails")
    return df


def mutual_info(X: pd.DataFrame, y: np.ndarray, seeds=(0, 1, 2, 3, 4)) -> pd.Series:
    mis = [mutual_info_classif(X.values, y, random_state=s, n_neighbors=5) for s in seeds]
    return pd.Series(np.mean(mis, axis=0), index=X.columns, name="mutual_info")


def stability_selection(
    X: pd.DataFrame, y: np.ndarray, n_boot: int = 200, C: float = 0.05, seed: int = 0
) -> pd.Series:
    rng = np.random.default_rng(seed)
    Z = StandardScaler().fit_transform(X.values)
    Z = np.clip(Z, -10, 10)
    y = np.asarray(y)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    counts = np.zeros(X.shape[1])
    for _ in range(n_boot):
        idx = np.r_[
            rng.choice(pos, pos.size // 2, replace=False),
            rng.choice(neg, neg.size // 2, replace=False),
        ]
        m = l1_logistic(C, class_weight="balanced", max_iter=2000).fit(Z[idx], y[idx])
        counts += (np.abs(m.coef_[0]) > 1e-8)
    return pd.Series(counts / n_boot, index=X.columns, name="stability_freq")


def tree_importance(X: pd.DataFrame, y: np.ndarray, seeds=(0, 1, 2, 3, 4)) -> pd.Series:
    imps = []
    for s in seeds:
        et = ExtraTreesClassifier(
            n_estimators=400,
            min_samples_leaf=3,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=s,
            n_jobs=-1,
        ).fit(X.values, y)
        imps.append(et.feature_importances_)
    return pd.Series(np.mean(imps, axis=0), index=X.columns, name="tree_importance")


def alarm_lift(alarms: pd.DataFrame, y: np.ndarray, min_alarms: int = 5) -> pd.DataFrame:
    """Fail rate given an I-chart alarm on this sensor, relative to base."""
    y = np.asarray(y)
    base = y.mean()
    n_al = alarms.sum(axis=0)
    fails_al = (alarms.values & (y[:, None] == 1)).sum(axis=0)
    rate = np.where(n_al > 0, fails_al / np.maximum(n_al, 1), np.nan)
    df = pd.DataFrame(
        {"n_alarms": n_al.values, "fails_in_alarms": fails_al, "fail_rate_given_alarm": rate},
        index=alarms.columns,
    )
    df["alarm_lift"] = df.fail_rate_given_alarm / base
    df.loc[df.n_alarms < min_alarms, "alarm_lift"] = np.nan
    # one-sided Fisher exact test: are fails over-represented among alarms?
    pvals = []
    n_fail = int(y.sum())
    n = y.size
    for c in alarms.columns:
        a = int(df.at[c, "fails_in_alarms"])
        b = int(df.at[c, "n_alarms"]) - a
        cc = n_fail - a
        d = (n - n_fail) - b
        pvals.append(stats.fisher_exact([[a, b], [cc, d]], alternative="greater")[1] if df.at[c, "n_alarms"] else 1.0)
    df["fisher_p"] = pvals
    return df


def consensus_rank(X_raw: pd.DataFrame, X_imp: pd.DataFrame, y: np.ndarray, n_boot: int = 200, C: float = 0.05, seed: int = 0) -> pd.DataFrame:
    """Combine the views; lower ``consensus_rank`` = more predictive.

    X_raw keeps NaNs (used for rank tests); X_imp is median-imputed
    (used by the model-based methods).
    """
    uni = univariate_tests(X_raw, y)
    mi = mutual_info(X_imp, y)
    st = stability_selection(X_imp, y, n_boot=n_boot, C=C, seed=seed)
    tr = tree_importance(X_imp, y)
    df = uni.join([mi, st, tr])
    ranks = pd.DataFrame(
        {
            "r_mw": df.mw_p.rank(method="average"),
            "r_auc": (-df.auc_effect).rank(method="average"),
            "r_mi": (-df.mutual_info).rank(method="average"),
            "r_stab": (-df.stability_freq).rank(method="average"),
            "r_tree": (-df.tree_importance).rank(method="average"),
        }
    )
    df = df.join(ranks)
    df["mean_rank"] = ranks.mean(axis=1)
    df["consensus_rank"] = df.mean_rank.rank(method="first").astype(int)
    df["n_methods_top20"] = (ranks <= 20).sum(axis=1)
    return df.sort_values("consensus_rank")


def method_agreement(rank_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["r_mw", "r_auc", "r_mi", "r_stab", "r_tree"]
    return rank_df[cols].corr(method="spearman")
