import numpy as np
import pandas as pd

from secom_spc import multivariate as MV


def _latent(n, p, k, rng):
    L = rng.normal(size=(k, p))
    return rng.normal(size=(n, k)) @ L + 0.3 * rng.normal(size=(n, p))


def test_false_alarm_rate_near_alpha_on_gaussian_data():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(_latent(4000, 30, 4, rng))
    ref = np.arange(4000) < 2000
    r = MV.fit_mspc(X, ref, var_explained=0.8, alpha=0.01, clip_z=None)
    assert 0.003 < (r.t2[~ref] > r.t2_ucl).mean() < 0.025
    assert 0.003 < (r.spe[~ref] > r.spe_ucl).mean() < 0.03


def test_spe_catches_broken_correlation_and_contrib_points_to_sensor():
    rng = np.random.default_rng(1)
    X = pd.DataFrame(_latent(1200, 20, 3, rng), columns=[f"S{i}" for i in range(20)])
    ref = np.arange(1200) < 800
    X.loc[~ref, "S7"] += 4.0  # one sensor drifts on its own
    r = MV.fit_mspc(X, ref)
    assert (r.spe[~ref] > r.spe_ucl).mean() > 0.8
    top = MV.top_contributors(r.spe_contrib, X.index[-1], k=1)[0][0]
    assert top == "S7"
