import numpy as np
import pandas as pd
import pytest

from secom_spc import capability as C


def test_cpk_centered_normal():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 20000)
    c = C.capability_indices(x, -4, 4)
    assert c["Pp"] == pytest.approx(4 / 3, rel=0.03)
    assert c["Ppk"] == pytest.approx(4 / 3, rel=0.03)
    assert c["Cpk"] == pytest.approx(4 / 3, rel=0.05)
    assert c["Cpk_pct"] == pytest.approx(4 / 3, rel=0.1)


def test_cpk_off_center_penalised():
    rng = np.random.default_rng(1)
    x = rng.normal(1, 1, 20000)
    c = C.capability_indices(x, -4, 4)
    assert c["Ppk"] == pytest.approx(1.0, rel=0.03)
    assert c["Pp"] > c["Ppk"]


def test_capability_table_flags_drift():
    rng = np.random.default_rng(2)
    stable = rng.normal(0, 1, 1000)
    drift = np.r_[rng.normal(0, 1, 500), rng.normal(3, 1, 500)]
    X = pd.DataFrame({"stable": stable, "drift": drift})
    p1 = np.arange(1000) < 500
    t = C.capability_table(X, p1, spec_k=4)
    assert t.loc["drift", "Ppk_II"] < 0.5
    assert t.loc["stable", "Ppk_II"] > 1.1
    assert t.index[0] == "drift"
