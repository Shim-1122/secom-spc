"""Static report figures (matplotlib).

Visual conventions used everywhere:

* data series: blue; second series (fails / Phase II): orange
* out-of-control points: red *and* a distinct marker (x), so the alarm
  never relies on colour alone
* control limits: dashed mid-gray; centre line: solid dark gray
* Phase I / Phase II boundary: dotted vertical line with a text label
* hairline grid, no top/right spines, single y axis per panel
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
CRIT = "#d03b3b"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK2,
            "axes.titlecolor": INK,
            "axes.titleweight": "bold",
            "axes.titlesize": 11,
            "axes.titlelocation": "left",
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "font.family": "DejaVu Sans",
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "lines.linewidth": 1.2,
        }
    )


def _phase_line(ax, x_boundary, label_y=None):
    ax.axvline(x_boundary, color=INK2, ls=":", lw=1)
    ymax = ax.get_ylim()[1] if label_y is None else label_y
    ax.text(x_boundary, ymax, "  Phase II  →", color=INK2, fontsize=8, va="top", ha="left")
    ax.text(x_boundary, ymax, "←  Phase I  ", color=INK2, fontsize=8, va="top", ha="right")


def _limits(ax, x, cl, lcl, ucl):
    ax.plot(x, np.broadcast_to(cl, len(x)), color=INK2, lw=1)
    ax.plot(x, np.broadcast_to(ucl, len(x)), color=MUTED, lw=1, ls="--")
    ax.plot(x, np.broadcast_to(lcl, len(x)), color=MUTED, lw=1, ls="--")


def save(fig, path: Path) -> None:
    fig.savefig(path)
    plt.close(fig)


# ----------------------------------------------------------------------
def imr_figure(ch: pd.DataFrame, sensor: str, boundary: int, path: Path, fails: np.ndarray | None = None) -> None:
    x = np.arange(1, len(ch) + 1)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 5.2), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    a1.plot(x, ch.value, color=BLUE, lw=0.8, marker="o", ms=2.2)
    _limits(a1, x, ch.cl.values, ch.lcl.values, ch.ucl.values)
    other = ch.any_rule & ~ch.rule1
    a1.scatter(x[other], ch.value[other], s=22, facecolors="none", edgecolors=ORANGE, lw=1, zorder=3, label="Nelson rules 2 to 8")
    a1.scatter(x[ch.rule1], ch.value[ch.rule1], s=30, color=CRIT, marker="x", lw=1.4, zorder=4, label="Rule 1: beyond 3σ")
    if fails is not None:
        fy = a1.get_ylim()[0]
        a1.scatter(x[fails == 1], np.full((fails == 1).sum(), fy), marker="|", color=INK, s=40, lw=1, label="failed wafer run", clip_on=False)
    a1.set_ylabel(sensor)
    li = ch.attrs["limits_i"]
    a1.set_title(f"Individuals chart: {sensor}   (CL {li['center']:.4g}, σ within {li['sigma']:.3g}, limits from Phase I passing runs)")
    a1.legend(loc="upper left", ncol=3, bbox_to_anchor=(0, -0.02))
    _phase_line(a1, boundary + 0.5)
    a2.plot(x, ch.mr, color=BLUE, lw=0.7)
    _limits(a2, x, ch.mr_cl.values, np.zeros(len(x)), ch.mr_ucl.values)
    a2.scatter(x[ch.mr_ooc], ch.mr[ch.mr_ooc], s=26, color=CRIT, marker="x", lw=1.3, zorder=3)
    a2.set_ylabel("moving range")
    a2.set_xlabel("run (time order)")
    fig.tight_layout()
    save(fig, path)


def xbar_s_figure(st: pd.DataFrame, sensor: str, size: int, path: Path) -> None:
    g = st.g.values + 1
    b = int(st.phase_I.sum())
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
    a1.plot(g, st.xbar, color=BLUE, marker="o", ms=2.5, lw=0.8)
    _limits(a1, g, st.cl.values, st.lcl.values, st.ucl.values)
    a1.scatter(g[st.rule1], st.xbar[st.rule1], color=CRIT, marker="x", s=30, lw=1.4, zorder=3, label="beyond 3σ")
    o = st.any_rule & ~st.rule1
    a1.scatter(g[o], st.xbar[o], s=22, facecolors="none", edgecolors=ORANGE, lw=1, zorder=3, label="other Nelson rule")
    a1.set_title(f"X̄ chart: {sensor}  (subgroups of {size} consecutive runs)")
    a1.set_ylabel("subgroup mean")
    a1.legend(loc="upper left")
    _phase_line(a1, b + 0.5)
    a2.plot(g, st.s, color=BLUE, marker="o", ms=2.5, lw=0.8)
    _limits(a2, g, st.s_cl.values, st.s_lcl.values, st.s_ucl.values)
    a2.scatter(g[st.s_ooc], st.s[st.s_ooc], color=CRIT, marker="x", s=30, lw=1.4, zorder=3)
    a2.set_title("S chart")
    a2.set_ylabel("subgroup std dev")
    a2.set_xlabel("subgroup")
    fig.tight_layout()
    save(fig, path)


def yield_figure(daily: pd.DataFrame, pch: pd.DataFrame, boundary_date, path: Path) -> None:
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    d = daily.date
    a1.plot(d, daily.yield_pct, color=BLUE, lw=0, marker="o", ms=3, label="daily yield")
    a1.plot(d, daily.yield_7d, color=BLUE, lw=1.8, label="7-day rolling yield (run-weighted)")
    a1.axhline(100 * (1 - pch.pbar.iloc[0]), color=INK2, lw=1, label="Phase I baseline yield")
    a1.set_ylabel("yield, % of runs passing")
    a1.set_title("Line yield KPI")
    a1.legend(loc="lower left", ncol=3)
    a1.axvline(boundary_date, color=INK2, ls=":", lw=1)
    a2.step(d, 100 * pch.ucl, where="mid", color=MUTED, ls="--", lw=1, label="UCL (varies with runs/day)")
    a2.axhline(100 * pch.pbar.iloc[0], color=INK2, lw=1, label="p̄ (Phase I)")
    a2.plot(d, 100 * pch.p, color=ORANGE, marker="o", ms=3, lw=0.8, label="daily fail rate")
    hi = pch.ooc_high.values
    a2.scatter(d[hi], 100 * pch.p[hi], color=CRIT, marker="x", s=40, lw=1.5, zorder=3, label="out of control")
    a2.set_ylabel("fail rate, %")
    a2.set_title("p chart of daily fail fraction (variable limits)")
    a2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4)
    a2.axvline(boundary_date, color=INK2, ls=":", lw=1)
    a2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.tight_layout()
    save(fig, path)


def mspc_figure(ms, boundary: int, fails: np.ndarray, path: Path) -> None:
    """T^2 and SPE charts (log scale) with parametric and empirical limits."""
    x = np.arange(1, len(ms.t2) + 1)
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)
    panels = (
        (axes[0], ms.t2, ms.t2_ucl, ms.t2_ucl_emp, "Hotelling T\u00b2"),
        (axes[1], ms.spe, ms.spe_ucl, ms.spe_ucl_emp, "SPE (Q)"),
    )
    for ax, v, u, ue, name in panels:
        ok = v <= ue
        ax.scatter(x[ok & (fails == 0)], v[ok & (fails == 0)], s=6, color=BLUE, label="pass run")
        ax.scatter(x[ok & (fails == 1)], v[ok & (fails == 1)], s=14, color=ORANGE, marker="D", label="fail run")
        ax.scatter(x[~ok & (fails == 0)], v[~ok & (fails == 0)], s=18, color=CRIT, marker="x", lw=1.1, label="alarm, pass run")
        ax.scatter(x[~ok & (fails == 1)], v[~ok & (fails == 1)], s=36, color=CRIT, marker="D", label="alarm, fail run")
        ax.axhline(u, color=MUTED, ls="--", lw=1)
        ax.axhline(ue, color=INK2, ls="-.", lw=1)
        ax.set_yscale("log")
        ax.set_ylabel(f"{name} (log)")
        ax.set_title(f"{name}   (dashed: parametric 99% limit {u:.0f}; dash-dot: empirical 99th percentile of baseline {ue:.0f})")
        ax.axvline(boundary + 0.5, color=INK2, ls=":", lw=1)
    axes[0].legend(loc="upper left", ncol=4)
    axes[1].set_xlabel("run (time order)")
    fig.tight_layout()
    save(fig, path)


def ranking_figure(rank: pd.DataFrame, k: int, path: Path) -> None:
    top = rank.head(k).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.32 * k + 1.2))
    y = np.arange(len(top))
    ax.barh(y, top.auc_effect + 0.5, left=0, color=BLUE, height=0.6)
    for i, (_, r) in enumerate(top.iterrows()):
        ax.text(r.auc_effect + 0.5 + 0.004, i, f"{r.direction}, q={r.mw_q:.2g}", va="center", fontsize=7, color=INK2)
    ax.set_yticks(y, top.index)
    ax.set_xlim(0.5, max(0.8, top.auc_effect.max() + 0.62))
    ax.axvline(0.5, color=AXIS, lw=1)
    ax.set_xlabel("univariate separating power, max(AUC, 1 - AUC)")
    ax.set_title(f"Top {k} failure-predictive sensors (consensus of 5 methods, training period only)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, path)


def rank_heatmap(rank: pd.DataFrame, k: int, path: Path) -> None:
    cols = ["r_mw", "r_auc", "r_mi", "r_stab", "r_tree"]
    labels = ["Mann-Whitney", "AUC", "Mutual info", "L1 stability", "Extra-Trees"]
    top = rank.head(k)
    M = top[cols].values
    from matplotlib.colors import LinearSegmentedColormap, LogNorm

    cmap = LinearSegmentedColormap.from_list("seq", SEQ[::-1])
    fig, ax = plt.subplots(figsize=(6.5, 0.3 * k + 1.4))
    im = ax.imshow(M, cmap=cmap, norm=LogNorm(1, max(M.max(), 2)), aspect="auto")
    ax.set_xticks(range(len(labels)), labels, rotation=20, ha="right")
    ax.set_yticks(range(k), top.index)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6.5, color="#ffffff" if v < 30 else INK)
    ax.grid(False)
    ax.set_title("Rank of each top sensor under each method (1 = best)")
    cb = fig.colorbar(im, ax=ax, shrink=0.6)
    cb.set_label("rank (log scale)")
    fig.tight_layout()
    save(fig, path)


def distribution_figure(X: pd.DataFrame, y: np.ndarray, sensors: list[str], path: Path) -> None:
    n = len(sensors)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(11, 2.6 * rows))
    for ax, s in zip(np.ravel(axes), sensors):
        v = X[s].values
        ok = np.isfinite(v)
        lo, hi = np.nanpercentile(v, [0.5, 99.5])
        bins = np.linspace(lo, hi, 40)
        ax.hist(v[ok & (y == 0)], bins=bins, density=True, color=BLUE, alpha=0.55, label="pass")
        ax.hist(v[ok & (y == 1)], bins=bins, density=True, color=ORANGE, alpha=0.75, histtype="stepfilled", label="fail")
        ax.set_title(s, fontsize=9)
        ax.set_yticks([])
    for ax in np.ravel(axes)[n:]:
        ax.axis("off")
    np.ravel(axes)[0].legend(loc="upper right")
    fig.suptitle("Pass vs fail distributions of the top-ranked sensors (density, 0.5 to 99.5 pct range)", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    save(fig, path)


def capability_figure(cap: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6))
    c = cap.dropna(subset=["Ppk_I", "Ppk_II"])
    lim = (0, max(2.0, np.nanpercentile(c[["Ppk_I", "Ppk_II"]].values, 99) * 1.05))
    ax.scatter(c.Ppk_I.clip(*lim), c.Ppk_II.clip(*lim), s=10, color=BLUE, alpha=0.7)
    ax.plot(lim, lim, color=AXIS, lw=1)
    ax.axhline(1.33, color=MUTED, ls="--", lw=1)
    ax.axhline(1.0, color=CRIT, ls="--", lw=1)
    ax.text(lim[1], 1.33, "1.33 capable ", ha="right", va="bottom", fontsize=8, color=INK2)
    ax.text(lim[1], 1.0, "1.00 marginal ", ha="right", va="bottom", fontsize=8, color=CRIT)
    worst = c.sort_values("Ppk_II").head(6)
    for s, r in worst.iterrows():
        ax.annotate(s, (min(r.Ppk_I, lim[1]), max(r.Ppk_II, lim[0])), fontsize=7, color=INK2, xytext=(4, 2), textcoords="offset points")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Ppk, Phase I (baseline)")
    ax.set_ylabel("Ppk, Phase II (monitoring)")
    ax.set_title("Capability drift vs proxy spec limits (points below the diagonal degraded)")
    fig.tight_layout()
    save(fig, path)


def roc_pr_figure(y_test, scores: dict, path: Path) -> None:
    from sklearn.metrics import precision_recall_curve, roc_curve

    colors = [BLUE, ORANGE, AQUA, "#eda100", "#e87ba4", "#008300", "#6250d6"]
    styles = ["-", "-", "-", "-", "--", "--", "--"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))
    for (name, s), c, ls in zip(scores.items(), colors, styles):
        fpr, tpr, _ = roc_curve(y_test, s)
        a1.plot(fpr, tpr, color=c, ls=ls, lw=1.6, label=name)
        pr, rc, _ = precision_recall_curve(y_test, s)
        a2.plot(rc, pr, color=c, ls=ls, lw=1.6, label=name)
    a1.plot([0, 1], [0, 1], color=AXIS, lw=1)
    a1.set_xlabel("false positive rate")
    a1.set_ylabel("true positive rate (fails caught)")
    a1.set_title("ROC, temporal hold-out (last 30% of runs)")
    a2.axhline(np.mean(y_test), color=AXIS, lw=1)
    a2.text(1, np.mean(y_test), "base fail rate ", ha="right", va="bottom", fontsize=8, color=INK2)
    a2.set_xlabel("recall (fails caught)")
    a2.set_ylabel("precision")
    a2.set_title("Precision-recall, temporal hold-out")
    a1.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    save(fig, path)


def missing_figure(miss: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.4))
    v = miss.missing_frac.values * 100
    ax.hist(v[v > 0], bins=50, color=BLUE)
    ax.axvline(50, color=CRIT, ls="--", lw=1)
    ax.text(50, ax.get_ylim()[1], " drop threshold (50%)", color=CRIT, fontsize=8, va="top")
    ax.set_xlabel("% of runs missing")
    ax.set_ylabel("number of sensors")
    ax.set_title(f"Missingness per sensor ({(v == 0).sum()} sensors fully observed, not shown)")
    fig.tight_layout()
    save(fig, path)


def alarms_over_time(meta: pd.DataFrame, n_alarms: np.ndarray, boundary: int, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 3.4))
    x = np.arange(1, len(meta) + 1)
    roll = pd.Series(n_alarms).rolling(25, min_periods=5).mean()
    ax.plot(x, n_alarms, color=BLUE, lw=0.5, alpha=0.5, label="sensors beyond 3σ in this run")
    ax.plot(x, roll, color=BLUE, lw=1.8, label="25-run rolling mean")
    f = meta.fail.values == 1
    ax.scatter(x[f], n_alarms[f], color=ORANGE, marker="D", s=14, zorder=3, label="failed run")
    ax.axvline(boundary + 0.5, color=INK2, ls=":", lw=1)
    ax.set_xlabel("run (time order)")
    ax.set_ylabel("count of sensors in alarm")
    ax.set_title("Univariate alarm load per run (why per-sensor charts alone do not scale)")
    ax.legend(loc="upper left", ncol=3)
    fig.tight_layout()
    save(fig, path)


def contribution_figure(contrib: list[tuple[str, float]], run_id: int, stat: str, path: Path) -> None:
    names = [c[0] for c in contrib][::-1]
    vals = [100 * c[1] for c in contrib][::-1]
    fig, ax = plt.subplots(figsize=(6, 0.35 * len(names) + 1.2))
    ax.barh(names, vals, color=BLUE, height=0.6)
    for i, v in enumerate(vals):
        ax.text(v + 0.5, i, f"{v:.0f}%", va="center", fontsize=8, color=INK2)
    ax.set_xlabel(f"share of {stat}")
    ax.set_title(f"Top {stat} contributors, run {run_id}")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, path)


def drift_figure(X: pd.DataFrame, meta: pd.DataFrame, sensors: list[str], boundary_ts, path: Path) -> None:
    """Small multiples: weekly median with 10-90 pct band per drifting sensor."""
    n = len(sensors)
    cols = 2
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(11, 2.4 * rows), sharex=True)
    wk = meta.timestamp.dt.to_period("W").dt.start_time
    for ax, s in zip(np.ravel(axes), sensors):
        g = X[s].groupby(wk.values)
        med, lo, hi = g.median(), g.quantile(0.1), g.quantile(0.9)
        ax.fill_between(med.index, lo, hi, color=SEQ[1], lw=0, label="10th to 90th pct")
        ax.plot(med.index, med, color=BLUE, lw=1.8, marker="o", ms=3, label="weekly median")
        ax.axvline(boundary_ts, color=INK2, ls=":", lw=1)
        ax.set_title(s, fontsize=9)
    for ax in np.ravel(axes)[n:]:
        ax.axis("off")
    np.ravel(axes)[0].legend(loc="best")
    for ax in np.ravel(axes):
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.suptitle("Sensors that moved most between Phase I and Phase II (dotted line = end of Phase I)", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    save(fig, path)


def confound_figure(rank: pd.DataFrame, path: Path, k: int = 40) -> None:
    """Pooled train AUC vs within-week AUC for the top sensors."""
    r = rank.head(k).dropna(subset=["within_week_auc"])
    fig, ax = plt.subplots(figsize=(6.8, 6))
    ax.axhline(0.5, color=AXIS, lw=1)
    ax.axvline(0.5, color=AXIS, lw=1)
    ax.plot([0.25, 0.85], [0.25, 0.85], color=AXIS, lw=1, ls="--")
    ax.scatter(r.auc, r.within_week_auc, s=18, color=BLUE)
    for s, row in r.head(12).iterrows():
        ax.annotate(s, (row.auc, row.within_week_auc), fontsize=7, color=INK2, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("pooled AUC, training period")
    ax.set_ylabel("within-week AUC (fails vs passes from the same week)")
    ax.set_title(f"Is the signal real or a time trend? (top {k} sensors)")
    ax.text(0.27, 0.83, "on dashed line: signal holds within a week\nnear 0.5 horizontal: time confound", fontsize=7.5, color=INK2, va="top")
    fig.tight_layout()
    save(fig, path)


def walk_forward_figure(wf: pd.DataFrame, path: Path) -> None:
    d = wf.reset_index()
    models = list(dict.fromkeys(d.model))
    fig, ax = plt.subplots(figsize=(8.5, 0.6 * len(models) + 1.6))
    for i, m in enumerate(models):
        s = d[(d.model == m) & (d["mode"] == "static")].pooled_roc_auc.iloc[0]
        r = d[(d.model == m) & (d["mode"] == "retrained")].pooled_roc_auc.iloc[0]
        ax.plot([s, r], [i, i], color=AXIS, lw=2, zorder=1)
        ax.scatter([s], [i], color=ORANGE, s=60, marker="s", zorder=2, label="trained once" if i == 0 else None)
        ax.scatter([r], [i], color=BLUE, s=60, marker="o", zorder=2, label="retrained every block" if i == 0 else None)
    ax.axvline(0.5, color=CRIT, ls="--", lw=1)
    ax.text(0.5, -0.45, " chance", color=CRIT, fontsize=8, va="center")
    ax.set_yticks(range(len(models)), models)
    ax.set_xlabel("pooled ROC AUC on Phase II runs")
    ax.set_title("Walk-forward evaluation: does periodic retraining help?")
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, path)
