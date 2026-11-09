"""Failure prediction models with leakage-safe, time-ordered evaluation.

Design choices
--------------
* **Temporal hold-out.** The final ``test_frac`` of runs (by timestamp)
  is never touched until the end. Random shuffling would let the model
  learn from the future, which overstates real-world performance on
  drifting fab data.
* **Everything inside the pipeline.** Imputation, scaling and the
  top-k sensor filter are refit in every CV fold.
* **Imbalance-aware metrics.** With ~6.6 % fails, accuracy is
  meaningless (always predicting "pass" scores 93 %). We report ROC AUC,
  PR AUC (vs. the base rate), balanced error rate (BER, the metric in
  the original UCI baseline), recall and specificity.
* **Threshold chosen on training data only** (out-of-fold predictions,
  minimise BER), then frozen for the test period.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from ._compat import l1_logistic


class TopKByAUC(BaseEstimator, TransformerMixin):
    """Keep the k columns with the largest |AUC - 0.5| on the fit data."""

    def __init__(self, k: int = 20):
        self.k = k

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        eff = []
        for j in range(X.shape[1]):
            col = X[:, j]
            if np.ptp(col) == 0:
                eff.append(0.0)
                continue
            eff.append(abs(roc_auc_score(y, col) - 0.5))
        self.scores_ = np.asarray(eff)
        self.support_ = np.argsort(-self.scores_)[: self.k]
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)[:, self.support_]


def _clip(z):
    return np.clip(z, -10, 10)


def model_zoo(cfg) -> dict[str, Pipeline]:
    rs = cfg.random_state
    return {
        "L1 logistic (all sensors)": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("clip", FunctionTransformer(_clip)),
                ("clf", l1_logistic(0.05, class_weight="balanced", max_iter=5000)),
            ]
        ),
        f"Logistic (top {cfg.top_k_sensors} by AUC, in-fold)": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("select", TopKByAUC(k=cfg.top_k_sensors)),
                ("scale", StandardScaler()),
                ("clip", FunctionTransformer(_clip)),
                ("clf", LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000)),
            ]
        ),
        "Random forest (balanced)": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=500, min_samples_leaf=5, max_features="sqrt",
                        class_weight="balanced_subsample", random_state=rs, n_jobs=-1,
                    ),
                ),
            ]
        ),
        "Gradient boosting (hist, NaN-native)": Pipeline(
            [
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        learning_rate=0.05, max_iter=300, max_leaf_nodes=15, min_samples_leaf=20,
                        l2_regularization=1.0, class_weight="balanced", random_state=rs,
                        early_stopping=False,
                    ),
                )
            ]
        ),
    }


def ber_threshold(y, p) -> float:
    """Threshold that minimises balanced error rate."""
    fpr, tpr, thr = roc_curve(y, p)
    ber = 0.5 * (fpr + (1 - tpr))
    i = int(np.argmin(ber))
    return float(thr[i]) if np.isfinite(thr[i]) else 0.5


def metrics_at(y, p, thr: float) -> dict:
    y = np.asarray(y)
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn) if (tp + fn) else np.nan
    tnr = tn / (tn + fp) if (tn + fp) else np.nan
    return {
        "roc_auc": roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan,
        "pr_auc": average_precision_score(y, p) if len(np.unique(y)) > 1 else np.nan,
        "base_rate": float(np.mean(y)),
        "ber": 0.5 * ((1 - tpr) + (1 - tnr)),
        "recall_fail": tpr,
        "specificity": tnr,
        "precision_fail": tp / (tp + fp) if (tp + fp) else np.nan,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "threshold": thr,
    }


def bootstrap_auc_ci(y, p, n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, y.size, y.size)
        if y[i].min() == y[i].max():
            continue
        vals.append(roc_auc_score(y[i], p[i]))
    return tuple(np.percentile(vals, [2.5, 97.5]))


@dataclass
class EvalResult:
    table: pd.DataFrame
    test_scores: dict[str, np.ndarray]
    oof_scores: dict[str, np.ndarray]
    fitted: dict[str, Pipeline]


def evaluate_models(X: pd.DataFrame, y: np.ndarray, train: np.ndarray, cfg, extra_scores: dict | None = None) -> EvalResult:
    """Fit every model on the training period and score the test period.

    ``extra_scores`` maps a name to a per-run score array (e.g. the count
    of SPC alarms or Hotelling T^2) that is evaluated as a model-free
    baseline with the same threshold procedure.
    """
    y = np.asarray(y)
    test = ~train
    Xtr, Xte, ytr, yte = X.values[train], X.values[test], y[train], y[test]
    cv = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.random_state)
    rows, test_scores, oof_scores, fitted = [], {}, {}, {}

    for name, pipe in model_zoo(cfg).items():
        oof = cross_val_predict(clone(pipe), Xtr, ytr, cv=cv, method="predict_proba")[:, 1]
        thr = ber_threshold(ytr, oof)
        m = clone(pipe).fit(Xtr, ytr)
        p = m.predict_proba(Xte)[:, 1]
        cvm = metrics_at(ytr, oof, thr)
        tm = metrics_at(yte, p, thr)
        lo, hi = bootstrap_auc_ci(yte, p, seed=cfg.random_state)
        rows.append(_row(name, cvm, tm, lo, hi))
        test_scores[name], oof_scores[name], fitted[name] = p, oof, m

    for name, s in (extra_scores or {}).items():
        s = np.asarray(s, dtype=float)
        thr = ber_threshold(ytr, s[train])
        cvm = metrics_at(ytr, s[train], thr)
        tm = metrics_at(yte, s[test], thr)
        lo, hi = bootstrap_auc_ci(yte, s[test], seed=cfg.random_state)
        rows.append(_row(name, cvm, tm, lo, hi))
        test_scores[name], oof_scores[name] = s[test], s[train]

    return EvalResult(pd.DataFrame(rows).set_index("model"), test_scores, oof_scores, fitted)


def _row(name, cvm, tm, lo, hi) -> dict:
    return {
        "model": name,
        "train_cv_roc_auc": cvm["roc_auc"],
        "train_cv_ber": cvm["ber"],
        "test_roc_auc": tm["roc_auc"],
        "test_roc_auc_ci95": f"[{lo:.2f}, {hi:.2f}]",
        "test_pr_auc": tm["pr_auc"],
        "test_base_rate": tm["base_rate"],
        "test_ber": tm["ber"],
        "test_recall_fail": tm["recall_fail"],
        "test_specificity": tm["specificity"],
        "test_precision_fail": tm["precision_fail"],
        "test_tp": tm["tp"], "test_fp": tm["fp"], "test_fn": tm["fn"], "test_tn": tm["tn"],
    }


def random_split_cv(X: pd.DataFrame, y: np.ndarray, cfg, n_splits: int = 10) -> pd.DataFrame:
    """Shuffled stratified CV on all data, for comparison with the
    UCI baseline (which used 10-fold CV). Expect this to look better
    than the temporal hold-out."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg.random_state)
    rows = []
    for name, pipe in model_zoo(cfg).items():
        oof = cross_val_predict(clone(pipe), X.values, y, cv=cv, method="predict_proba")[:, 1]
        thr = ber_threshold(y, oof)
        m = metrics_at(y, oof, thr)
        rows.append({"model": name, "cv10_roc_auc": m["roc_auc"], "cv10_pr_auc": m["pr_auc"], "cv10_ber_optimistic": m["ber"]})
    return pd.DataFrame(rows).set_index("model")


def walk_forward(
    X: pd.DataFrame, y: np.ndarray, cfg, start_frac: float = 0.5, block: int = 100, window: int | None = None,
    models: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Rolling-origin evaluation with periodic retraining.

    The first ``start_frac`` of runs seed the model. Each following block
    of ``block`` runs is scored by a model trained on everything before it
    (``window=None``, expanding) or on only the most recent ``window`` runs
    (sliding). A "static" variant trained once on the seed period is scored
    on the same blocks for comparison. This is how a fab would actually
    deploy a yield model: retrain on a schedule, never on the future.
    """
    y = np.asarray(y)
    zoo = model_zoo(cfg)
    names = models or list(zoo)
    n = len(y)
    start = int(round(n * start_frac))
    edges = list(range(start, n, block)) + [n]
    preds = {f"{nm} | {mode}": np.full(n, np.nan) for nm in names for mode in ("static", "retrained")}
    rows = []
    for nm in names:
        static = clone(zoo[nm]).fit(X.values[:start], y[:start])
        for a, b in zip(edges[:-1], edges[1:]):
            lo = 0 if window is None else max(0, a - window)
            m = clone(zoo[nm]).fit(X.values[lo:a], y[lo:a])
            preds[f"{nm} | retrained"][a:b] = m.predict_proba(X.values[a:b])[:, 1]
            preds[f"{nm} | static"][a:b] = static.predict_proba(X.values[a:b])[:, 1]
    mask = np.arange(n) >= start
    for k, p in preds.items():
        nm, mode = k.split(" | ")
        block_aucs = []
        for a, b in zip(edges[:-1], edges[1:]):
            yy = y[a:b]
            if 0 < yy.sum() < len(yy):
                block_aucs.append(roc_auc_score(yy, p[a:b]))
        lo_ci, hi_ci = bootstrap_auc_ci(y[mask], p[mask], seed=cfg.random_state)
        rows.append(
            {
                "model": nm,
                "mode": mode,
                "pooled_roc_auc": roc_auc_score(y[mask], p[mask]),
                "pooled_auc_ci95": f"[{lo_ci:.2f}, {hi_ci:.2f}]",
                "pooled_pr_auc": average_precision_score(y[mask], p[mask]),
                "median_block_auc": float(np.median(block_aucs)) if block_aucs else np.nan,
                "n_blocks": len(edges) - 1,
                "eval_runs": int(mask.sum()),
                "eval_fails": int(y[mask].sum()),
            }
        )
    return pd.DataFrame(rows).set_index(["model", "mode"]), preds
