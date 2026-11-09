"""Small shims so the code runs on scikit-learn 1.3 through 1.9+."""

from __future__ import annotations

import sklearn
from sklearn.linear_model import LogisticRegression

_MAJOR, _MINOR = (int(p) for p in sklearn.__version__.split(".")[:2])
_NEW_API = (_MAJOR, _MINOR) >= (1, 8)


def l1_logistic(C: float, **kw) -> LogisticRegression:
    """Lasso-penalised logistic regression (liblinear)."""
    if _NEW_API:
        return LogisticRegression(l1_ratio=1.0, C=C, solver="liblinear", **kw)
    return LogisticRegression(penalty="l1", C=C, solver="liblinear", **kw)
