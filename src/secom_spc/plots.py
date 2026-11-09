"""Report figures (matplotlib).

Shared conventions
------------------
* Bold title with a smaller gray subtitle for detail, both left-aligned.
* Phase I is a light gray background band; Phase II is unshaded.
* Data are blue; failing runs or a second series are orange.
* Alarms are red *and* drawn with a distinct marker shape.
* Control limits are dashed gray; centre lines are solid dark gray.
* Control-chart y-axes are clipped to a robust range so that a handful of
  extreme readings does not flatten the rest; clipped points are drawn as
  triangles on the axis edge.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter, PercentFormatter  # noqa: E402

BLUE = "#2a78d6"
BLUE_LIGHT = "#9ec5f4"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
PURPLE = "#6250d6"
CRIT = "#d03b3b"
INK = "#1a1a1a"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#ebeae4"
AXIS = "#c3c2b7"
PHASE_BG = "#f1f0ea"
SURFACE = "#ffffff"
SEQ = ["#e6f0fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

WIDE = 11.0  # inches; every full-width figure uses this width


def style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.labelcolor": INK2,
            "axes.labelsize": 9,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlecolor": INK,
            "axes.titlelocation": "left",
            "axes.titlepad": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "legend.handlelength": 1.6,
            "font.family": "DejaVu Sans",
            "figure.dpi": 110,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.15,
            "lines.linewidth": 1.2,
            "lines.solid_capstyle": "round",
        }
    )


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path)
    plt.close(fig)


def _title(ax: plt.Axes, title: str, subtitle: str | None = None) -> None:
    """Bold title with an optional gray subtitle underneath."""
    ax.set_title(title, pad=22 if subtitle else 8)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=8.5, color=MUTED, va="bottom", ha="left")


def _fig_title(fig: plt.Figure, title: str, subtitle: str | None = None) -> None:
    fig.text(0.01, 1.0, title, fontsize=11, fontweight="bold", color=INK, va="bottom", ha="left")
    if subtitle:
        fig.text(0.01, 0.985, subtitle, fontsize=8.5, color=MUTED, va="top", ha="left")


def _phase_band(ax: plt.Axes, x_end: float, x_start: float = 0.0, label: bool = False) -> None:
    """Shade Phase I; optionally label the boundary above the axes."""
    ax.axvspan(x_start, x_end, color=PHASE_BG, lw=0, zorder=0)
    if label:
        trans = ax.get_xaxis_transform()
        ax.text(x_end, 1.0, "Phase I \u2190 ", transform=trans, ha="right", va="bottom", fontsize=7.5, color=MUTED)
        ax.text(x_end, 1.0, " \u2192 Phase II", transform=trans, ha="left", va="bottom", fontsize=7.5, color=MUTED)


def _limit_lines(ax: plt.Axes, cl: float, lcl: float, ucl: float) -> None:
    ax.axhline(cl, color=INK2, lw=1, zorder=1)
    for v in (lcl, ucl):
        ax.axhline(v, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)


def _robust_ylim(values: np.ndarray, lcl: float, ucl: float, pad: float = 0.08) -> tuple[float, float]:
    """Y range covering the limits and the 0.5 to 99.5 percentile of the data."""
    v = values[np.isfinite(values)]
    lo, hi = np.percentile(v, [0.5, 99.5]) if v.size else (lcl, ucl)
    lo, hi = min(lo, lcl), max(hi, ucl)
    span = hi - lo or 1.0
    return lo - pad * span, hi + pad * span


def _plot_clipped(ax: plt.Axes, x: np.ndarray, v: np.ndarray, ylim: tuple[float, float], color: str = BLUE) -> None:
    """Mark values outside ``ylim`` with triangles on the axis edge."""
    lo, hi = ylim
    above, below = v > hi, v < lo
    ax.scatter(x[above], np.full(above.sum(), hi), marker="^", s=22, color=color, clip_on=False, zorder=4)
    ax.scatter(x[below], np.full(below.sum(), lo), marker="v", s=22, color=color, clip_on=False, zorder=4)


def _log_plain(ax: plt.Axes) -> None:
    """Log-scale y axis with plain numeric tick labels (40, 100, 200 ...)."""
    from matplotlib.ticker import LogLocator, NullFormatter

    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())


# ----------------------------------------------------------------------
# Control charts
# ----------------------------------------------------------------------
def imr_figure(ch: pd.DataFrame, sensor: str, boundary: int, path: Path, fails: np.ndarray | None = None) -> None:
    x = np.arange(1, len(ch) + 1)
    v = ch.value.to_numpy()
    li = ch.attrs["limits_i"]
    fig, (a1, a2, a3) = plt.subplots(
        3, 1, figsize=(WIDE, 5.6), sharex=True, gridspec_kw={"height_ratios": [3, 0.25, 1.2], "hspace": 0.12}
    )

    ylim = _robust_ylim(v, li["lcl"], li["ucl"])
    _phase_band(a1, boundary + 0.5)
    _limit_lines(a1, li["center"], li["lcl"], li["ucl"])
    a1.plot(x, np.clip(v, *ylim), color=BLUE, lw=0.6, alpha=0.55, zorder=2)
    a1.scatter(x, np.clip(v, *ylim), s=3, color=BLUE, zorder=3)
    other = (ch.any_rule & ~ch.rule1).to_numpy()
    r1 = ch.rule1.to_numpy()
    a1.scatter(x[other], np.clip(v[other], *ylim), s=14, facecolors="none", edgecolors=ORANGE, lw=0.8, zorder=4)
    a1.scatter(x[r1], np.clip(v[r1], *ylim), s=16, color=CRIT, marker="x", lw=1.0, zorder=5)
    _plot_clipped(a1, x, v, ylim, CRIT)
    a1.set_ylim(ylim)
    a1.set_ylabel(sensor)
    _title(
        a1,
        f"Individuals chart, {sensor}",
        f"CL {li['center']:.4g}, σ (MR-bar/d2) {li['sigma']:.3g}; limits from Phase I passing runs; shaded = Phase I",
    )
    handles = [
        Line2D([], [], color=MUTED, ls=(0, (4, 3)), label="3σ limits"),
        Line2D([], [], marker="x", color=CRIT, ls="", label="beyond 3σ (rule 1)"),
        Line2D([], [], marker="o", mfc="none", mec=ORANGE, ls="", label="Nelson rules 2 to 8"),
        Line2D([], [], marker="^", color=CRIT, ls="", label="off scale"),
    ]
    fig.legend(handles=handles + [Line2D([], [], color=ORANGE, lw=1, label="failed run")],
               loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.03))

    # failed-run rug
    a2.set_yticks([])
    a2.grid(False)
    a2.spines["left"].set_visible(False)
    _phase_band(a2, boundary + 0.5, label=False)
    if fails is not None:
        a2.vlines(x[fails == 1], 0, 1, color=ORANGE, lw=1)
    a2.set_ylim(0, 1)
    a2.set_ylabel("fails", rotation=0, ha="right", va="center")

    mr = ch.mr.to_numpy()
    mr_ylim = (0, _robust_ylim(mr, 0, float(ch.mr_ucl.iloc[0]))[1])
    _phase_band(a3, boundary + 0.5, label=False)
    a3.axhline(ch.mr_cl.iloc[0], color=INK2, lw=1)
    a3.axhline(ch.mr_ucl.iloc[0], color=MUTED, lw=1, ls=(0, (4, 3)))
    a3.plot(x, np.clip(mr, *mr_ylim), color=BLUE, lw=0.6)
    ooc = ch.mr_ooc.to_numpy()
    a3.scatter(x[ooc], np.clip(mr[ooc], *mr_ylim), s=12, color=CRIT, marker="x", lw=0.9, zorder=3)
    _plot_clipped(a3, x, mr, mr_ylim, CRIT)
    a3.set_ylim(mr_ylim)
    a3.set_ylabel("moving range")
    a3.set_xlabel("run (time order)")
    a3.set_xlim(0, len(x) + 1)
    save(fig, path)


def xbar_s_figure(st: pd.DataFrame, sensor: str, size: int, path: Path) -> None:
    g = st.g.to_numpy() + 1
    boundary = int(st.phase_I.sum()) + 0.5
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(WIDE, 4.8), sharex=True, gridspec_kw={"hspace": 0.3})

    xb = st.xbar.to_numpy()
    ylim = _robust_ylim(xb, st.lcl.iloc[0], st.ucl.iloc[0])
    _phase_band(a1, boundary)
    _limit_lines(a1, st.cl.iloc[0], st.lcl.iloc[0], st.ucl.iloc[0])
    a1.plot(g, np.clip(xb, *ylim), color=BLUE, lw=0.8, marker="o", ms=2)
    other = (st.any_rule & ~st.rule1).to_numpy()
    r1 = st.rule1.to_numpy()
    a1.scatter(g[other], np.clip(xb[other], *ylim), s=16, facecolors="none", edgecolors=ORANGE, lw=0.8, zorder=3)
    a1.scatter(g[r1], np.clip(xb[r1], *ylim), s=20, color=CRIT, marker="x", lw=1.1, zorder=4)
    _plot_clipped(a1, g, xb, ylim, CRIT)
    a1.set_ylim(ylim)
    a1.set_ylabel("subgroup mean")
    _title(a1, f"X̄ and S charts, {sensor}", f"subgroups of {size} consecutive runs; shaded = Phase I")
    handles = [
        Line2D([], [], marker="x", color=CRIT, ls="", label="beyond 3σ"),
        Line2D([], [], marker="o", mfc="none", mec=ORANGE, ls="", label="other Nelson rule"),
    ]
    a1.legend(handles=handles, loc="upper right", ncol=2, bbox_to_anchor=(1, 1.17))

    s = st.s.to_numpy()
    s_ylim = (0, _robust_ylim(s, 0, st.s_ucl.iloc[0])[1])
    _phase_band(a2, boundary, label=False)
    _limit_lines(a2, st.s_cl.iloc[0], st.s_lcl.iloc[0], st.s_ucl.iloc[0])
    a2.plot(g, np.clip(s, *s_ylim), color=BLUE, lw=0.8, marker="o", ms=2)
    so = st.s_ooc.to_numpy()
    a2.scatter(g[so], np.clip(s[so], *s_ylim), color=CRIT, marker="x", s=20, lw=1.1, zorder=3)
    _plot_clipped(a2, g, s, s_ylim, CRIT)
    a2.set_ylim(s_ylim)
    a2.set_ylabel("subgroup std dev")
    a2.set_xlabel("subgroup")
    a2.set_xlim(0, g.max() + 1)
    save(fig, path)


def pass_rate_figure(daily: pd.DataFrame, pch: pd.DataFrame, boundary_date, path: Path) -> None:
    d = daily.date
    start = d.min() - pd.Timedelta(days=1)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(WIDE, 6), sharex=True, gridspec_kw={"hspace": 0.38})

    for ax in (a1, a2):
        ax.axvspan(start, boundary_date, color=PHASE_BG, lw=0, zorder=0)
    a1.axhline(100 * (1 - pch.pbar.iloc[0]), color=INK2, lw=1)
    a1.scatter(d, daily.pass_rate_pct, s=10, color=BLUE_LIGHT, zorder=2)
    a1.plot(d, daily.pass_rate_7d, color=BLUE, lw=2, zorder=3)
    a1.set_ylim(-3, 103)
    a1.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    _title(a1, "Line-test pass rate", "share of runs passing the downstream line test; shaded = Phase I (before the detected change)")
    a1.legend(
        handles=[
            Line2D([], [], marker="o", color=BLUE_LIGHT, ls="", label="daily"),
            Line2D([], [], color=BLUE, lw=2, label="7-day rolling (run-weighted)"),
            Line2D([], [], color=INK2, lw=1, label="Phase I mean"),
        ],
        loc="upper right",
        ncol=3,
        bbox_to_anchor=(1, 1.2),
    )

    cap = 60.0
    ucl = (100 * pch.ucl).clip(upper=cap)
    a2.step(d, ucl, where="mid", color=MUTED, lw=1, ls=(0, (4, 3)))
    a2.axhline(100 * pch.pbar.iloc[0], color=INK2, lw=1)
    p = 100 * pch.p
    a2.plot(d, p.clip(upper=cap), color=ORANGE, lw=0.8, marker="o", ms=3)
    hi = pch.ooc_high.to_numpy()
    a2.scatter(d[hi], p[hi].clip(upper=cap), color=CRIT, marker="x", s=46, lw=1.6, zorder=4)
    over = (p > cap).to_numpy()
    a2.scatter(d[over], np.full(over.sum(), cap), marker="^", s=30, color=CRIT, clip_on=False, zorder=5)
    a2.set_ylim(0, cap)
    a2.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    _title(a2, "p chart of the daily fail fraction", f"limits vary with daily volume; values and limits above {cap:.0f}% drawn at the top edge")
    a2.legend(
        handles=[
            Line2D([], [], color=ORANGE, marker="o", ms=3, lw=0.8, label="daily fail rate"),
            Line2D([], [], color=INK2, lw=1, label="p̄ (Phase I)"),
            Line2D([], [], color=MUTED, ls=(0, (4, 3)), label="UCL"),
            Line2D([], [], marker="x", color=CRIT, ls="", label="out of control"),
        ],
        loc="upper right",
        ncol=4,
        bbox_to_anchor=(1, 1.2),
    )
    a2.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
    a2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    a2.set_xlim(start, d.max() + pd.Timedelta(days=1))
    save(fig, path)


def alarms_over_time(meta: pd.DataFrame, n_alarms: np.ndarray, boundary: int, path: Path) -> None:
    x = np.arange(1, len(meta) + 1)
    s = pd.Series(n_alarms, dtype=float)
    roll = s.rolling(50, center=True, min_periods=10)
    fig, ax = plt.subplots(figsize=(WIDE, 3.6))
    _phase_band(ax, boundary + 0.5)
    ax.fill_between(x, roll.quantile(0.1), roll.quantile(0.9), color=BLUE_LIGHT, alpha=0.6, lw=0)
    ax.plot(x, roll.median(), color=BLUE, lw=2)
    f = meta.fail.to_numpy() == 1
    ax.scatter(x[f], n_alarms[f], color=ORANGE, marker="D", s=12, zorder=3, edgecolors="white", linewidths=0.4)
    ax.set_ylim(0, None)
    ax.set_xlim(0, len(x) + 1)
    ax.set_xlabel("run (time order)")
    ax.set_ylabel("sensors beyond 3σ")
    _title(ax, "Sensors in alarm per run", "count of sensors beyond their I-chart 3\u03c3 limits; shaded = Phase I")
    ax.legend(
        handles=[
            Line2D([], [], color=BLUE, lw=2, label="50-run rolling median"),
            Patch(color=BLUE_LIGHT, alpha=0.6, label="10th to 90th percentile"),
            Line2D([], [], marker="D", color=ORANGE, ls="", ms=4, label="failed run"),
        ],
        loc="upper right",
        ncol=3,
        bbox_to_anchor=(1, 1.16),
    )
    save(fig, path)


# ----------------------------------------------------------------------
# Multivariate SPC
# ----------------------------------------------------------------------
def mspc_figure(ms, boundary: int, fails: np.ndarray, path: Path) -> None:
    """T^2 and SPE charts (log scale) with parametric and empirical limits."""
    x = np.arange(1, len(ms.t2) + 1)
    fig, axes = plt.subplots(2, 1, figsize=(WIDE, 6), sharex=True, gridspec_kw={"hspace": 0.32})
    panels = (
        (axes[0], ms.t2, ms.t2_ucl, ms.t2_ucl_emp, "Hotelling T²", "variation within the PCA model"),
        (axes[1], ms.spe, ms.spe_ucl, ms.spe_ucl_emp, "SPE (Q)", "variation outside the PCA model"),
    )
    for ax, v, u, ue, name, sub in panels:
        _phase_band(ax, boundary + 0.5)
        alarm = v > ue
        ax.scatter(x[~alarm & (fails == 0)], v[~alarm & (fails == 0)], s=5, color=BLUE_LIGHT, lw=0, zorder=2)
        ax.scatter(x[alarm & (fails == 0)], v[alarm & (fails == 0)], s=9, color=BLUE, lw=0, zorder=2)
        ax.scatter(x[fails == 1], v[fails == 1], s=18, color=ORANGE, marker="D", edgecolors="white", linewidths=0.4, zorder=3)
        ax.axhline(u, color=MUTED, ls=(0, (4, 3)), lw=1)
        ax.axhline(ue, color=CRIT, lw=1.1)
        ax.text(len(x) + 8, ue, f"empirical {ue:.0f}", color=CRIT, fontsize=7.5, va="center", clip_on=False)
        ax.text(len(x) + 8, u, f"parametric {u:.0f}", color=MUTED, fontsize=7.5, va="center", clip_on=False)
        ax.set_yscale("log")
        _log_plain(ax)
        ax.set_ylabel(name)
        _title(ax, name, sub + "; 99% limits from Phase I passing runs; shaded = Phase I")
        ax.set_xlim(0, len(x) + 1)
    axes[0].legend(
        handles=[
            Line2D([], [], marker="o", color=BLUE_LIGHT, ls="", ms=4, label="pass, below limit"),
            Line2D([], [], marker="o", color=BLUE, ls="", ms=4, label="pass, above empirical limit"),
            Line2D([], [], marker="D", color=ORANGE, ls="", ms=4, label="failed run"),
        ],
        loc="upper right",
        ncol=3,
        bbox_to_anchor=(1, 1.22),
    )
    axes[1].set_xlabel("run (time order)")
    save(fig, path)


def contribution_figure(contrib: list[tuple[str, float]], run_id: int, stat: str, path: Path) -> None:
    names = [c[0] for c in contrib][::-1]
    vals = [100 * c[1] for c in contrib][::-1]
    fig, ax = plt.subplots(figsize=(6.2, 0.36 * len(names) + 1.2))
    ax.barh(names, vals, color=[BLUE] * (len(vals) - 1) + [ORANGE], height=0.62)
    for i, v in enumerate(vals):
        ax.text(v + max(vals) * 0.015, i, f"{v:.1f}%", va="center", fontsize=8, color=INK2)
    ax.set_xlim(0, max(vals) * 1.15)
    ax.xaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelcolor=INK2, labelsize=8.5)
    _title(ax, f"{stat} contributions, run {run_id}", f"share of the run's {stat} from each sensor (top {len(names)})")
    save(fig, path)


# ----------------------------------------------------------------------
# Capability and drift
# ----------------------------------------------------------------------
def drift_figure(X: pd.DataFrame, meta: pd.DataFrame, sensors: list[str], boundary_ts, path: Path) -> None:
    """Small multiples: weekly median with 10 to 90 percentile band."""
    n = len(sensors)
    cols = 2
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(WIDE, 2.3 * rows + 0.6), sharex=True, squeeze=False)
    wk = meta.timestamp.dt.to_period("W").dt.start_time
    start = wk.min() - pd.Timedelta(days=3)
    for ax, s in zip(axes.ravel(), sensors):
        g = X[s].groupby(wk.to_numpy())
        med, lo, hi = g.median(), g.quantile(0.1), g.quantile(0.9)
        ax.axvspan(start, boundary_ts, color=PHASE_BG, lw=0, zorder=0)
        ax.fill_between(med.index, lo, hi, color=BLUE_LIGHT, alpha=0.7, lw=0)
        ax.plot(med.index, med, color=BLUE, lw=1.8, marker="o", ms=2.5)
        ax.set_title(s, fontsize=9.5, pad=4)
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    for ax in axes.ravel():
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _fig_title(
        fig,
        "Sensors with the largest change between phases",
        "weekly median (line) and 10th to 90th percentile (band); shaded = Phase I",
    )
    save(fig, path)


def missing_figure(miss: pd.DataFrame, path: Path) -> None:
    v = np.sort(miss.missing_frac.to_numpy() * 100)[::-1]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.fill_between(np.arange(1, v.size + 1), v, color=BLUE_LIGHT, lw=0, step="mid")
    ax.step(np.arange(1, v.size + 1), v, color=BLUE, where="mid", lw=1.2)
    ax.axhline(50, color=CRIT, ls=(0, (4, 3)), lw=1)
    ax.text(v.size, 51, "screening threshold (applied to Phase I runs) ", color=CRIT, fontsize=8, ha="right", va="bottom")
    ax.set_xlim(0, v.size)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.set_xlabel("sensors, sorted by missingness")
    ax.set_ylabel("runs missing")
    _title(ax, "Missing values per sensor", f"all runs; {(v == 0).sum()} of {v.size} sensors are fully observed")
    save(fig, path)


# ----------------------------------------------------------------------
# Sensor ranking
# ----------------------------------------------------------------------
def ranking_figure(rank: pd.DataFrame, k: int, path: Path) -> None:
    top = rank.head(k).iloc[::-1]
    eff = top.auc_effect + 0.5
    higher = (top.direction == "higher in fails").to_numpy()
    fig, ax = plt.subplots(figsize=(7.5, 0.3 * k + 1.4))
    y = np.arange(len(top))
    ax.hlines(y, 0.5, eff, color=AXIS, lw=1.2)
    ax.scatter(eff[higher], y[higher], color=ORANGE, s=34, zorder=3, label="higher in failing runs")
    ax.scatter(eff[~higher], y[~higher], color=BLUE, s=34, zorder=3, label="lower in failing runs")
    for i, q in enumerate(top.mw_q):
        ax.text(eff.iloc[i] + 0.006, i, f"q = {q:.2g}", va="center", fontsize=7, color=MUTED)
    ax.set_yticks(y, top.index)
    ax.tick_params(axis="y", labelcolor=INK2)
    ax.set_xlim(0.5, max(0.8, eff.max() + 0.06))
    ax.set_xlabel("max(AUC, 1 − AUC), training partition")
    ax.grid(axis="y", visible=False)
    _title(ax, f"Top {k} sensors by consensus rank", "ordered by consensus of five criteria; q = Benjamini-Hochberg adjusted p")
    ax.legend(loc="lower right")
    save(fig, path)


def rank_heatmap(rank: pd.DataFrame, k: int, path: Path) -> None:
    cols = ["r_mw", "r_auc", "r_mi", "r_stab", "r_tree"]
    labels = ["Mann-\nWhitney", "AUC", "Mutual\ninfo", "L1\nstability", "Extra-\nTrees"]
    top = rank.head(k)
    M = top[cols].to_numpy()
    cmap = LinearSegmentedColormap.from_list("seq", SEQ[::-1])
    fig, ax = plt.subplots(figsize=(5.6, 0.28 * k + 1.5))
    im = ax.imshow(M, cmap=cmap, norm=LogNorm(1, max(M.max(), 2)), aspect="auto")
    ax.set_xticks(range(len(labels)), labels, fontsize=8)
    ax.xaxis.tick_top()
    ax.set_yticks(range(k), top.index)
    ax.tick_params(axis="y", labelcolor=INK2)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7, color="white" if v < 20 else INK)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Rank under each criterion (1 = strongest)", pad=34)
    cb = fig.colorbar(im, ax=ax, shrink=0.5, pad=0.03)
    cb.outline.set_visible(False)
    cb.set_ticks([1, 10, 100])
    cb.set_ticklabels(["1", "10", "100"])
    cb.set_label("rank", color=INK2)
    save(fig, path)


def confound_figure(rank: pd.DataFrame, path: Path, k: int = 40) -> None:
    """Pooled training AUC against within-week AUC for the top sensors."""
    r = rank.head(k).dropna(subset=["within_week_auc"])
    fig, ax = plt.subplots(figsize=(6.2, 6))
    lim = (0.3, 0.8)
    ax.axhline(0.5, color=AXIS, lw=1)
    ax.axvline(0.5, color=AXIS, lw=1)
    ax.plot(lim, lim, color=AXIS, lw=1, ls=(0, (4, 3)))
    holds = (np.sign(r.auc - 0.5) == np.sign(r.within_week_auc - 0.5)) & ((r.within_week_auc - 0.5).abs() > 0.05)
    ax.scatter(r.auc[~holds], r.within_week_auc[~holds], s=22, color=BLUE_LIGHT, label="within-week AUC within 0.05 of 0.5")
    ax.scatter(r.auc[holds], r.within_week_auc[holds], s=26, color=BLUE, label="within-week AUC beyond 0.05 of 0.5")
    for s in ["S059", "S021", "S247", "S033", "S103", "S129"]:
        if s in r.index:
            ax.annotate(s, (r.at[s, "auc"], r.at[s, "within_week_auc"]), fontsize=7.5, color=INK2,
                        xytext=(5, -3), textcoords="offset points")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.set_xlabel("pooled AUC, training partition")
    ax.set_ylabel("within-week AUC")
    _title(ax, "Signal or time trend?", f"top {k} sensors; on the dashed line the association holds within a week")
    ax.legend(loc="upper left")
    save(fig, path)


def distribution_figure(X: pd.DataFrame, y: np.ndarray, sensors: list[str], path: Path) -> None:
    """Pass vs fail distributions: filled for passes, outline for fails."""
    n = len(sensors)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(WIDE, 2.2 * rows + 0.7), squeeze=False)
    for ax, s in zip(axes.ravel(), sensors):
        v = X[s].to_numpy()
        ok = np.isfinite(v)
        lo, hi = np.nanpercentile(v, [1, 99])
        bins = np.linspace(lo, hi, 36)
        ax.hist(np.clip(v[ok & (y == 0)], lo, hi), bins=bins, density=True, color=BLUE_LIGHT, lw=0)
        ax.hist(np.clip(v[ok & (y == 1)], lo, hi), bins=bins, density=True, histtype="step", color=ORANGE, lw=1.6)
        ax.set_title(s, fontsize=9.5, pad=4)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.grid(False)
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=1.2)
    fig.legend(
        handles=[Patch(color=BLUE_LIGHT, label="passing runs"), Line2D([], [], color=ORANGE, lw=1.6, label="failing runs")],
        loc="lower right",
        ncol=2,
        bbox_to_anchor=(0.99, 0.985),
    )
    _fig_title(fig, "Distributions of the top-ranked sensors", "training partition, density scale, 1st to 99th percentile")
    save(fig, path)


# ----------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------
def roc_pr_figure(y_test: np.ndarray, scores: dict, path: Path) -> None:
    """ROC and precision-recall curves; classifiers solid, baselines dashed."""
    from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve

    colors = [BLUE, ORANGE, AQUA, YELLOW, PURPLE, MUTED, INK2]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(WIDE, 4.9), gridspec_kw={"wspace": 0.22})
    handles = []
    for (name, s), c in zip(scores.items(), colors):
        baseline = "baseline" in name.lower()
        ls = (0, (4, 2)) if baseline else "-"
        lw = 1.1 if baseline else 1.4
        fpr, tpr, _ = roc_curve(y_test, s)
        a1.plot(fpr, tpr, color=c, ls=ls, lw=lw, drawstyle="steps-post")
        prec, rec, _ = precision_recall_curve(y_test, s)
        a2.plot(rec[:-1], prec[:-1], color=c, ls=ls, lw=lw, drawstyle="steps-post")
        label = name.split("baseline: ")[-1]
        handles.append(Line2D([], [], color=c, ls=ls, lw=lw, label=f"{label} ({roc_auc_score(y_test, s):.2f})"))
    a1.plot([0, 1], [0, 1], color=AXIS, lw=1)
    a1.set_xlim(0, 1)
    a1.set_ylim(0, 1.01)
    a1.set_aspect("equal")
    a1.set_xlabel("false positive rate")
    a1.set_ylabel("true positive rate (fails caught)")
    _title(a1, "ROC", "temporal hold-out: last 30% of runs")
    base = float(np.mean(y_test))
    a2.axhline(base, color=INK2, lw=1, ls=(0, (1, 2)))
    a2.set_xlim(0, 1)
    a2.set_ylim(0, 0.4)
    a2.set_xlabel("recall (fails caught)")
    a2.set_ylabel("precision")
    _title(a2, "Precision-recall", f"dotted line: base fail rate {100 * base:.1f}%; y-axis truncated at 0.4")
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.1), title="solid: classifiers; dashed: model-free baselines (test AUC)",
               title_fontsize=8)
    save(fig, path)


def walk_forward_figure(wf: pd.DataFrame, path: Path) -> None:
    d = wf.reset_index()
    models = list(dict.fromkeys(d.model))
    fig, ax = plt.subplots(figsize=(8, 0.62 * len(models) + 1.5))
    for i, m in enumerate(models):
        s = d[(d.model == m) & (d["mode"] == "static")].pooled_roc_auc.iloc[0]
        r = d[(d.model == m) & (d["mode"] == "retrained")].pooled_roc_auc.iloc[0]
        ax.annotate("", xy=(r, i), xytext=(s, i), arrowprops={"arrowstyle": "->", "color": AXIS, "lw": 1.5, "shrinkA": 5, "shrinkB": 5})
        ax.scatter([s], [i], color=ORANGE, s=55, marker="s", zorder=3)
        ax.scatter([r], [i], color=BLUE, s=55, marker="o", zorder=3)
        for v in (s, r):
            ax.text(v, i + 0.26, f"{v:.2f}", ha="center", fontsize=7.5, color=INK2,
                    bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.0}, zorder=4)
    ax.axvline(0.5, color=CRIT, ls=(0, (4, 3)), lw=1)
    ax.text(0.5, -0.6, " chance", color=CRIT, fontsize=7.5, va="center")
    ax.set_yticks(range(len(models)), models)
    ax.tick_params(axis="y", labelcolor=INK2)
    ax.set_ylim(-0.8, len(models) - 0.4)
    ax.set_xlim(0.4, 0.7)
    ax.set_xlabel("pooled ROC AUC over Phase II runs")
    ax.grid(axis="y", visible=False)
    _title(ax, "Walk-forward evaluation", "each block of 100 runs scored by a model trained on all earlier runs")
    ax.legend(
        handles=[
            Line2D([], [], marker="s", color=ORANGE, ls="", label="trained once on Phase I"),
            Line2D([], [], marker="o", color=BLUE, ls="", label="retrained before each block"),
        ],
        loc="upper right",
        ncol=2,
        bbox_to_anchor=(1, 1.2),
    )
    save(fig, path)


# ----------------------------------------------------------------------
# Change point, shift and monitoring burden
# ----------------------------------------------------------------------
def changepoint_figure(profile: pd.Series, timestamps: pd.Series, cp_index: int, secondary: list[int], path: Path) -> None:
    """Change-point statistic against the candidate boundary date."""
    t = timestamps.iloc[profile.index]
    fig, ax = plt.subplots(figsize=(WIDE, 3.2))
    ax.plot(t, profile.to_numpy() / profile.max(), color=BLUE, lw=1.6, marker="o", ms=2.5)
    ax.axvline(timestamps.iloc[cp_index], color=CRIT, lw=1.2)
    ax.text(timestamps.iloc[cp_index], 0.04, f" {timestamps.iloc[cp_index]:%d %b}", color=CRIT, fontsize=8.5, va="bottom",
            transform=ax.get_xaxis_transform())
    for k in secondary:
        ax.axvline(timestamps.iloc[k], color=MUTED, lw=1, ls=(0, (4, 3)))
        ax.text(timestamps.iloc[k], 0.04, f" {timestamps.iloc[k]:%d %b}", color=MUTED, fontsize=8.5, va="bottom",
                transform=ax.get_xaxis_transform())
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("statistic (scaled to max)")
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    _title(ax, "Where did the process change?",
           "mean-shift statistic for a boundary at the start of each day (all sensors, pass/fail labels not used); "
           "red = estimate, dashed = secondary")
    save(fig, path)


def shift_spread_figure(ss: pd.DataFrame, n_label: int, path: Path) -> None:
    """Median shift against spread ratio for every sensor (no spec limits)."""
    from matplotlib.patches import Rectangle

    d = ss.dropna(subset=["shift_robust_sd", "spread_ratio"])
    xlim, ylim = (-12, 12), (1 / 12, 12)
    x = d.shift_robust_sd.clip(*xlim)
    y = d.spread_ratio.clip(*ylim)
    off = ((x != d.shift_robust_sd) | (y != d.spread_ratio)).to_numpy()
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    ax.add_patch(Rectangle((-1, 1 / 1.5), 2, 1.5 - 1 / 1.5, color=PHASE_BG, lw=0, zorder=0))
    ax.axvline(0, color=AXIS, lw=1)
    ax.axhline(1, color=AXIS, lw=1)
    big = (d.ks_stat >= 0.3).to_numpy()
    ax.scatter(x[~big & ~off], y[~big & ~off], s=12, color=BLUE_LIGHT, lw=0, label="KS < 0.3")
    ax.scatter(x[big & ~off], y[big & ~off], s=16, color=BLUE, lw=0, label="KS \u2265 0.3")
    ax.scatter(x[off], y[off], s=20, color=ORANGE, marker="s", lw=0, clip_on=False, zorder=4, label="off scale (drawn at edge)")
    # label the largest movers; off-scale ones are listed in a corner note
    top = d.sort_values("ks_stat", ascending=False).head(n_label)
    off_names = [s_ for s_ in top.index if off[d.index.get_loc(s_)]]
    for s_ in top.index.difference(off_names):
        right = x[s_] > 0
        ax.annotate(s_, (x[s_], y[s_]), xytext=(-6 if right else 6, -10), textcoords="offset points",
                    fontsize=7.5, color=INK2, ha="right" if right else "left")
    if off_names:
        ax.text(0.98, 0.97, "off scale: " + ", ".join(sorted(off_names)), transform=ax.transAxes,
                ha="right", va="top", fontsize=7.5, color=ORANGE)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_yscale("log")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("median shift, in Phase I robust SDs (symmetric log scale)")
    ax.set_ylabel("spread ratio, Phase II / Phase I")
    _title(ax, "How far each sensor moved", "Phase II against Phase I; shaded box: |shift| < 1 SD and spread within 1.5x; "
           f"labelled: top {n_label} by KS")
    ax.legend(loc="lower left")
    save(fig, path)


def monitoring_figure(daily: pd.DataFrame, names: dict[str, str], cp_date, rebase_end_date, path: Path) -> None:
    """Alarms per day for each monitor, with the daily fail count beneath.

    Each monitor is drawn only where it is a genuine forward-looking
    monitor: the re-baselined one after its window, the rolling one once
    its first window is full.
    """
    colors = [MUTED, BLUE, ORANGE]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(WIDE, 5.2), sharex=True, gridspec_kw={"height_ratios": [2, 1], "hspace": 0.35})
    start = daily.date.min() - pd.Timedelta(days=1)
    for ax in (a1, a2):
        ax.axvspan(start, cp_date, color=PHASE_BG, lw=0, zorder=0)
        ax.axvspan(cp_date, rebase_end_date, color="#e6f0fd", lw=0, zorder=0)
    for (col, label), c in zip(names.items(), colors):
        if col == "alarms_rolling":
            v = daily[col].where(daily.covered)
        elif col == "alarms_rebaselined":
            v = daily[col].where(daily.date > rebase_end_date)
        else:
            v = daily[col]
        a1.plot(daily.date, v, color=c, lw=1.4, marker="o", ms=2.2, label=label)
    a1.set_yscale("symlog", linthresh=1)
    a1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    a1.set_ylim(0, None)
    a1.set_ylabel("alarms per day")
    _title(a1, "Alarm burden of three monitors",
           "T\u00b2 or SPE beyond the empirical 99% limit; gray = Phase I, light blue = re-baselining window")
    a1.legend(loc="upper right", ncol=3, bbox_to_anchor=(1, 1.25))
    hf = daily.high_fail_day.fillna(False).to_numpy()
    a2.bar(daily.date[~hf], daily.fails[~hf], color=BLUE_LIGHT, width=0.8, label="other days")
    a2.bar(daily.date[hf], daily.fails[hf], color=ORANGE, width=0.8, label="high-fail day")
    a2.set_ylabel("failed runs")
    a2.legend(loc="upper right", ncol=2, bbox_to_anchor=(1, 1.35))
    _title(a2, "Failed runs per day")
    a2.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
    a2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    a2.set_xlim(start, daily.date.max() + pd.Timedelta(days=1))
    save(fig, path)
