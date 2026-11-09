"""End-to-end analysis: ``python -m secom_spc.pipeline [--fast]``.

The pipeline runs as a sequence of stages. Each stage reads from and
writes to a shared :class:`Context`, saves its own figures and tables,
and records its headline numbers in ``ctx.summary``. At the end the
dashboard exports and ``reports/summary.json`` are written.

Outputs
-------
reports/figures/   every figure in the README
reports/tables/    intermediate tables (CSV)
reports/exports/   dashboard-ready CSVs
reports/summary.json
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import capability, drift, features, modeling, multivariate, plots, spc
from .config import CFG, EXPORT_DIR, FIG_DIR, TABLE_DIR, Config, ensure_dirs
from .data import load, phase_split
from .preprocess import SensorScreen, missingness_profile

log = logging.getLogger("secom_spc")


@dataclass
class Context:
    """State shared between stages."""

    cfg: Config
    fast: bool
    X: pd.DataFrame            # raw sensor matrix
    meta: pd.DataFrame         # run_id, timestamp, fail, date
    y: np.ndarray              # 1 = fail
    phase1: np.ndarray         # first ``phase1_frac`` of runs in time
    ref: np.ndarray            # runs used to estimate control limits
    train: np.ndarray          # first ``1 - test_frac`` of runs in time
    week: np.ndarray           # ISO week label per run
    summary: dict = field(default_factory=dict)
    # filled in by later stages
    Xr: pd.DataFrame | None = None   # screened, with missing values
    Xi: pd.DataFrame | None = None   # screened, median-imputed

    @property
    def phase2(self) -> np.ndarray:
        return ~self.phase1

    @property
    def test(self) -> np.ndarray:
        return ~self.train

    @property
    def boundary(self) -> int:
        """Number of Phase I runs (index of the last Phase I run, 1-based)."""
        return int(self.phase1.sum())


def _pct(mask: np.ndarray, within: np.ndarray) -> float:
    """Percentage of ``within`` runs for which ``mask`` is True."""
    return float(100 * mask[within].mean())


# ----------------------------------------------------------------------
# Stages
# ----------------------------------------------------------------------
def stage_data(cfg: Config, fast: bool) -> Context:
    d = load()
    y = d.meta.fail.to_numpy()
    n = len(y)
    phase1, _ = phase_split(n, cfg.phase1_frac)
    train, _ = phase_split(n, 1 - cfg.test_frac)
    ref = phase1 & (y == 0) if cfg.phase1_pass_only else phase1
    week = d.meta.timestamp.dt.to_period("W").astype(str).to_numpy()
    ctx = Context(cfg, fast, d.X, d.meta, y, phase1, ref, train, week)

    m = d.meta
    ctx.summary["data"] = {
        "runs": n,
        "sensors": d.X.shape[1],
        "fails": int(y.sum()),
        "passes": int((y == 0).sum()),
        "fail_rate": float(y.mean()),
        "start": str(m.timestamp.min()),
        "end": str(m.timestamp.max()),
        "production_days": int(m.date.nunique()),
        "missing_cells_pct": float(100 * d.X.isna().mean().mean()),
        "phase1_runs": ctx.boundary,
        "phase1_fails": int(y[phase1].sum()),
        "phase1_end": str(m.timestamp[ctx.boundary - 1]),
        "phase1_ref_pass_runs": int(ref.sum()),
        "train_runs": int(train.sum()),
        "test_runs": int(ctx.test.sum()),
        "train_fails": int(y[train].sum()),
        "test_fails": int(y[ctx.test].sum()),
        "test_start": str(m.timestamp[int(train.sum())]),
    }
    log.info("loaded %d runs x %d sensors, %d fails", n, d.X.shape[1], y.sum())
    return ctx


def stage_screening(ctx: Context) -> None:
    cfg = ctx.cfg
    miss = missingness_profile(ctx.X)
    miss.to_csv(TABLE_DIR / "missingness.csv")
    plots.missing_figure(miss, FIG_DIR / "01_missingness.png")

    screen = SensorScreen(cfg.max_missing_frac, cfg.min_unique, cfg.min_rel_std, cfg.corr_dedup_threshold)
    screen.fit(ctx.X[ctx.phase1])
    ctx.Xr = screen.transform(ctx.X, impute=False)
    ctx.Xi = screen.transform(ctx.X, impute=True)

    rep = screen.report_
    rep.summary().to_csv(TABLE_DIR / "sensor_screening.csv", index=False)
    (
        pd.Series(rep.dropped_duplicate, name="kept_as")
        .rename_axis("dropped")
        .to_csv(TABLE_DIR / "duplicate_sensors.csv")
    )
    ctx.summary["screening"] = rep.summary().set_index("stage")["count"].to_dict()
    log.info("screening kept %d sensors", len(rep.kept))


def stage_univariate_spc(ctx: Context) -> dict:
    mat = spc.run_rule_matrix(ctx.Xr, ctx.ref, ctx.cfg.sigma_limits, phase1=ctx.phase1)
    summ = mat["summary"]
    summ.to_csv(TABLE_DIR / "spc_sensor_summary.csv")
    n_r1 = mat["rule1"].sum(axis=1).to_numpy()
    plots.alarms_over_time(ctx.meta, n_r1, ctx.boundary, FIG_DIR / "05_alarm_load.png")

    y = ctx.y
    ctx.summary["spc"] = {
        "median_rule1_rate_phase1_pct": float(100 * summ.rule1_rate_phase1.median()),
        "median_rule1_rate_phase2_pct": float(100 * summ.rule1_rate_phase2.median()),
        "sensors_rule1_rate_phase2_gt_5pct": int((summ.rule1_rate_phase2 > 0.05).sum()),
        "runs_with_any_rule1_pct": float(100 * (n_r1 > 0).mean()),
        "median_sensors_in_alarm_per_run": float(np.median(n_r1)),
        "mean_sensors_in_alarm_pass": float(n_r1[y == 0].mean()),
        "mean_sensors_in_alarm_fail": float(n_r1[y == 1].mean()),
    }
    log.info("univariate SPC done")
    return mat


def daily_kpis(meta: pd.DataFrame, phase1: np.ndarray, k: float = 3.0) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Daily yield with p and Laney p' limits, plus a weekly roll-up."""
    m = meta.assign(phase1=phase1)
    daily = (
        m.groupby("date")
        .agg(runs=("run_id", "size"), fails=("fail", "sum"), all_phase1=("phase1", "all"))
        .reset_index()
    )
    daily["passes"] = daily.runs - daily.fails
    daily["yield_pct"] = 100 * daily.passes / daily.runs
    roll = daily.set_index("date")[["runs", "passes"]].rolling("7D").sum()
    daily["yield_7d"] = (100 * roll.passes / roll.runs).to_numpy()

    base = daily.all_phase1.to_numpy()
    pch = spc.p_chart(daily.fails, daily.runs, base, k)
    lan = spc.laney_p_chart(daily.fails, daily.runs, base, k)
    daily["p_ucl"] = pch.ucl.to_numpy()
    daily["p_ooc_high"] = pch.ooc_high.to_numpy()
    daily["laney_ucl"] = lan.ucl.to_numpy()
    daily["laney_ooc_high"] = lan.ooc_high.to_numpy()
    daily.attrs["laney_sigma_z"] = lan.attrs["sigma_z"]

    weekly = (
        m.set_index("timestamp")
        .resample("W-MON", label="left", closed="left")
        .agg({"run_id": "size", "fail": "sum"})
        .rename(columns={"run_id": "runs", "fail": "fails"})
    )
    weekly = weekly[weekly.runs > 0].copy()
    weekly["yield_pct"] = 100 * (1 - weekly.fails / weekly.runs)
    return daily, pch, weekly.reset_index().rename(columns={"timestamp": "week_start"})


def stage_yield(ctx: Context) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily, pch, weekly = daily_kpis(ctx.meta, ctx.phase1, ctx.cfg.sigma_limits)
    daily.to_csv(TABLE_DIR / "daily_yield.csv", index=False)
    weekly.to_csv(TABLE_DIR / "weekly_yield.csv", index=False)
    plots.yield_figure(daily, pch, ctx.meta.date[ctx.boundary], FIG_DIR / "02_yield_p_chart.png")

    y = ctx.y
    ctx.summary["kpi"] = {
        "overall_yield_pct": float(100 * (1 - y.mean())),
        "phase1_yield_pct": float(100 * (1 - y[ctx.phase1].mean())),
        "phase2_yield_pct": float(100 * (1 - y[ctx.phase2].mean())),
        "p_chart_pbar_pct": float(100 * pch.pbar.iloc[0]),
        "p_chart_days_ooc_high": int(daily.p_ooc_high.sum()),
        "laney_sigma_z": float(daily.attrs["laney_sigma_z"]),
        "laney_days_ooc_high": int(daily.laney_ooc_high.sum()),
        "ooc_days": [str(dt.date()) for dt in daily.date[daily.p_ooc_high]],
        "worst_week": weekly.loc[weekly.yield_pct.idxmin()].astype(str).to_dict(),
    }
    log.info("yield KPIs done")
    return daily, weekly


def stage_capability(ctx: Context) -> pd.DataFrame:
    cap = capability.capability_table(ctx.Xr, ctx.phase1, ctx.cfg.spec_k)
    cap.to_csv(TABLE_DIR / "capability.csv")
    plots.capability_figure(cap, FIG_DIR / "07_capability_drift.png")
    ctx.summary["capability"] = {
        "sensors_evaluated": len(cap),
        "spec_k_proxy": ctx.cfg.spec_k,
        "median_Ppk_I": float(cap.Ppk_I.median()),
        "median_Ppk_II": float(cap.Ppk_II.median()),
        "pct_capable_II": float(100 * cap.capable_II.mean()),
        "pct_Ppk_II_below_1": float(100 * (cap.Ppk_II < 1).mean()),
        "pct_normal_at_5pct": float(100 * cap.normal_at_5pct.mean()),
        "worst5": cap.head(5)[["Ppk_I", "Ppk_II", "mean_shift_sigma", "sd_ratio_II_I"]].round(3).to_dict("index"),
    }
    log.info("capability done")
    return cap


def stage_mspc(ctx: Context) -> multivariate.MSPCResult:
    cfg, y, p2, ref = ctx.cfg, ctx.y, ctx.phase2, ctx.ref
    ms = multivariate.fit_mspc(ctx.Xi, ref, cfg.pca_var_explained, cfg.mspc_alpha)
    plots.mspc_figure(ms, ctx.boundary, y, FIG_DIR / "06_hotelling_t2_spe.png")

    t2_par, spe_par = ms.alarms(empirical=False)
    t2_emp, spe_emp = ms.alarms(empirical=True)
    any_emp = t2_emp | spe_emp
    ctx.summary["mspc"] = {
        "pcs_retained": ms.n_components,
        "var_explained_pct": float(100 * ms.explained[: ms.n_components].sum()),
        "t2_ucl": ms.t2_ucl,
        "spe_ucl": ms.spe_ucl,
        "t2_ucl_empirical": ms.t2_ucl_emp,
        "spe_ucl_empirical": ms.spe_ucl_emp,
        "t2_alarm_rate_ref_pct": _pct(t2_par, ref),
        "spe_alarm_rate_ref_pct": _pct(spe_par, ref),
        "t2_alarm_rate_phase2_pct": _pct(t2_par, p2),
        "spe_alarm_rate_phase2_pct": _pct(spe_par, p2),
        "t2_alarm_rate_phase2_empirical_pct": _pct(t2_emp, p2),
        "spe_alarm_rate_phase2_empirical_pct": _pct(spe_emp, p2),
        "fail_rate_given_alarm_phase2_empirical_pct": _pct(y == 1, p2 & any_emp),
        "fail_rate_given_no_alarm_phase2_empirical_pct": _pct(y == 1, p2 & ~any_emp),
    }

    # Contribution analysis for the failing Phase II run with the largest SPE.
    failing_p2 = np.flatnonzero(p2 & (y == 1))
    worst = failing_p2[np.argmax(ms.spe[failing_p2])]
    contrib = multivariate.top_contributors(ms.spe_contrib, worst, k=8)
    run_id = int(ctx.meta.run_id[worst])
    plots.contribution_figure(contrib, run_id, "SPE", FIG_DIR / "06b_spe_contribution.png")
    ctx.summary["mspc"]["example_run"] = run_id
    ctx.summary["mspc"]["example_top_contributors"] = {k: round(float(v), 3) for k, v in contrib}
    log.info("MSPC done (%d PCs)", ms.n_components)
    return ms


def stage_drift(ctx: Context) -> None:
    cfg, Xr, Xi = ctx.cfg, ctx.Xr, ctx.Xi
    ks = drift.ks_drift(Xr, ctx.phase1, ctx.phase2)
    ks.to_csv(TABLE_DIR / "drift_ks.csv")
    adv_auc, adv_imp = drift.adversarial_validation(Xi, ctx.phase2.astype(int), seed=cfg.random_state)
    adv_imp.rename("adversarial_importance").to_csv(TABLE_DIR / "drift_adversarial_importance.csv")
    frozen = drift.frozen_sensors(Xr, pd.Series(ctx.week))
    frozen.to_csv(TABLE_DIR / "frozen_then_live_sensors.csv")

    shown = list(dict.fromkeys([*adv_imp.index[:4], *ks.index[:4]]))[:6]
    plots.drift_figure(Xr, ctx.meta, shown, ctx.meta.timestamp[ctx.boundary], FIG_DIR / "11_drift_sensors.png")

    # Sensitivity: how much of the Phase II SPE alarm rate is explained by
    # the sensors that changed regime?
    rows = []
    for label, drop in (
        ("all retained sensors", []),
        ("minus frozen-then-live sensors", list(frozen.index)),
        ("minus top 20 KS-drift sensors", list(ks.index[:20])),
        ("minus all sensors with KS p < 1e-6", list(ks.index[ks.ks_p < 1e-6])),
    ):
        Xs = Xi.drop(columns=[c for c in drop if c in Xi.columns])
        m = multivariate.fit_mspc(Xs, ctx.ref, cfg.pca_var_explained, cfg.mspc_alpha)
        t2a, spea = m.alarms(empirical=True)
        rows.append(
            {
                "sensor_set": label,
                "n_sensors": Xs.shape[1],
                "pcs": m.n_components,
                "t2_alarm_phase2_pct": _pct(t2a, ctx.phase2),
                "spe_alarm_phase2_pct": _pct(spea, ctx.phase2),
            }
        )
    sens = pd.DataFrame(rows)
    sens.to_csv(TABLE_DIR / "mspc_drift_sensitivity.csv", index=False)

    ctx.summary["drift"] = {
        "adversarial_auc_phase1_vs_phase2": adv_auc,
        "sensors_ks_p_below_1e-6": int((ks.ks_p < 1e-6).sum()),
        "sensors_ks_stat_above_0.5": int((ks.ks_stat > 0.5).sum()),
        "top_ks": ks.head(8)[["ks_stat", "median_shift_in_early_sd", "sd_ratio_late_early"]].round(3).to_dict("index"),
        "top_adversarial": adv_imp.head(8).round(4).to_dict(),
        "frozen_then_live": frozen.to_dict("index"),
        "mspc_sensitivity": sens.round(1).to_dict("records"),
    }
    log.info("drift done (adversarial AUC %.3f)", adv_auc)


def stage_ranking(ctx: Context, spc_mat: dict, cap: pd.DataFrame) -> pd.DataFrame:
    cfg, y, tr, te = ctx.cfg, ctx.y, ctx.train, ctx.test
    n_boot = 30 if ctx.fast else cfg.n_stability_boot
    rank = features.consensus_rank(ctx.Xr[tr], ctx.Xi[tr], y[tr], n_boot=n_boot, C=cfg.stability_C, seed=cfg.random_state)
    lift = features.alarm_lift(spc_mat["rule1"][tr], y[tr])
    rank = (
        rank.join(lift[["n_alarms", "alarm_lift", "fisher_p"]])
        .join(spc_mat["summary"][["rule1_rate_phase2"]])
        .join(cap[["Ppk_I", "Ppk_II"]])
    )
    # Out-of-sample checks: later time period, and within calendar week.
    rank["test_auc"] = features.univariate_tests(ctx.Xr[te], y[te]).auc
    rank["direction_holds_in_test"] = np.sign(rank.auc - 0.5) == np.sign(rank.test_auc - 0.5)
    rank["within_week_auc"] = drift.within_period_auc(ctx.Xr[tr], y[tr], ctx.week[tr])
    rank.to_csv(TABLE_DIR / "sensor_ranking.csv")
    agreement = features.method_agreement(rank)
    agreement.round(3).to_csv(TABLE_DIR / "ranking_method_agreement.csv")

    k = cfg.top_k_sensors
    plots.ranking_figure(rank, k, FIG_DIR / "03_sensor_ranking.png")
    plots.rank_heatmap(rank, k, FIG_DIR / "03b_rank_agreement.png")
    plots.confound_figure(rank, FIG_DIR / "03c_time_confound.png")
    plots.distribution_figure(ctx.Xr[tr], y[tr], list(rank.index[:9]), FIG_DIR / "04_top_sensor_distributions.png")

    top = rank.head(k)
    off_diag = agreement.to_numpy()[np.triu_indices(len(agreement), 1)]
    ctx.summary["ranking"] = {
        "n_sig_fdr_10pct": int((rank.mw_q < 0.10).sum()),
        "n_sig_fdr_5pct": int((rank.mw_q < 0.05).sum()),
        "top10": rank.head(10)[["auc", "within_week_auc", "test_auc", "mw_q", "stability_freq", "alarm_lift", "fisher_p", "direction"]]
        .round(3)
        .to_dict("index"),
        "topk_direction_holds_pct": float(100 * top.direction_holds_in_test.mean()),
        "topk_median_train_auc_effect": float(top.auc_effect.median()),
        "topk_median_test_auc_effect": float((top.test_auc - 0.5).abs().median()),
        "method_spearman_min": float(off_diag.min()),
        "method_spearman_max": float(off_diag.max()),
    }
    log.info("feature ranking done")
    return rank


def stage_sensor_charts(ctx: Context, rank: pd.DataFrame) -> pd.DataFrame:
    """Per-sensor figures for the top sensors and the long-format SPC export."""
    cfg, Xr = ctx.cfg, ctx.Xr
    for i, s in enumerate(rank.index[: cfg.n_charted_sensors], 1):
        ch = spc.imr_chart(Xr[s].to_numpy(), ctx.ref, cfg.sigma_limits, phase1=ctx.phase1)
        plots.imr_figure(ch, s, ctx.boundary, FIG_DIR / f"08_imr_{i:02d}_{s}.png", fails=ctx.y)
        if i <= 2:
            st = spc.xbar_s_chart(Xr[s].to_numpy(), ctx.ref, cfg.subgroup_size, cfg.sigma_limits, phase1=ctx.phase1)
            plots.xbar_s_figure(st, s, cfg.subgroup_size, FIG_DIR / f"09_xbar_s_{i:02d}_{s}.png")
            st.assign(sensor=s).to_csv(TABLE_DIR / f"xbar_s_{s}.csv", index=False)

    frames = []
    for s in rank.index[: cfg.top_k_sensors]:
        x = Xr[s].to_numpy()
        ch = spc.imr_chart(x, ctx.ref, cfg.sigma_limits, phase1=ctx.phase1)
        li = ch.attrs["limits_i"]
        ew = spc.ewma_chart(x, li["center"], li["sigma"], cfg.ewma_lambda, cfg.ewma_L)
        cu = spc.cusum_chart(x, li["center"], li["sigma"], cfg.cusum_k, cfg.cusum_h)
        frames.append(
            pd.DataFrame(
                {
                    "run_id": ctx.meta.run_id,
                    "timestamp": ctx.meta.timestamp,
                    "sensor": s,
                    "consensus_rank": int(rank.at[s, "consensus_rank"]),
                    "value": ch.value,
                    "cl": ch.cl,
                    "lcl": ch.lcl,
                    "ucl": ch.ucl,
                    "z": ch.z.round(4),
                    "moving_range": ch.mr,
                    "mr_ucl": ch.mr_ucl,
                    "ewma": ew.ewma,
                    "ewma_lcl": ew.lcl,
                    "ewma_ucl": ew.ucl,
                    "ewma_ooc": ew.ooc,
                    "cusum_plus": cu.c_plus,
                    "cusum_minus": cu.c_minus,
                    "cusum_h": cu.h,
                    "cusum_ooc": cu.ooc,
                    **{f"nelson_rule{r}": ch[f"rule{r}"] for r in range(1, 9)},
                    "any_rule": ch.any_rule,
                    "phase": ch.phase,
                    "fail": ctx.y,
                }
            )
        )
    log.info("per-sensor charts done")
    return pd.concat(frames, ignore_index=True)


def stage_models(ctx: Context, n_r1: np.ndarray, ms: multivariate.MSPCResult) -> np.ndarray:
    """Fit and evaluate classifiers; return per-run scores of the best model."""
    cfg, Xr, y, tr, te = ctx.cfg, ctx.Xr, ctx.y, ctx.train, ctx.test
    baselines = {
        "SPC baseline: # sensors beyond 3σ": n_r1.astype(float),
        "MSPC baseline: Hotelling T²": ms.t2,
        "MSPC baseline: SPE (Q)": ms.spe,
    }
    ev = modeling.evaluate_models(Xr, y, tr, cfg, extra_scores=baselines)
    ev.table.to_csv(TABLE_DIR / "model_comparison_temporal.csv")
    plots.roc_pr_figure(y[te], ev.test_scores, FIG_DIR / "10_roc_pr_holdout.png")
    cols = ["train_cv_roc_auc", "train_cv_ber", "test_roc_auc", "test_roc_auc_ci95", "test_pr_auc", "test_ber",
            "test_recall_fail", "test_specificity"]
    ctx.summary["models_temporal"] = ev.table[cols].round(3).to_dict("index")

    if not ctx.fast:
        rcv = modeling.random_split_cv(Xr, y, cfg)
        rcv.to_csv(TABLE_DIR / "model_comparison_random_cv10.csv")
        ctx.summary["models_random_cv10"] = rcv.round(3).to_dict("index")

    topk = modeling.topk_model_name(cfg.top_k_sensors)
    wf_models = [topk] if ctx.fast else [topk, modeling.MODEL_L1, modeling.MODEL_GB]
    wf_kw = dict(start_frac=cfg.phase1_frac, block=cfg.walk_forward_block)
    wf, _ = modeling.walk_forward(Xr, y, cfg, models=wf_models, **wf_kw)
    wf_slide, _ = modeling.walk_forward(Xr, y, cfg, models=[topk], window=cfg.walk_forward_window, **wf_kw)
    sliding_label = f"retrained, sliding {cfg.walk_forward_window}-run window"
    wf_slide = wf_slide.xs("retrained", level=1, drop_level=False).rename(index={"retrained": sliding_label}, level=1)
    wf_all = pd.concat([wf, wf_slide])
    wf_all.to_csv(TABLE_DIR / "model_walk_forward.csv")
    plots.walk_forward_figure(wf, FIG_DIR / "10b_walk_forward.png")
    ctx.summary["models_walk_forward"] = {f"{a} | {b}": v for (a, b), v in wf_all.round(3).to_dict("index").items()}

    best = ev.table.drop(index=list(baselines)).test_pr_auc.idxmax()
    ctx.summary["best_model_by_test_pr_auc"] = best
    scores = np.full(len(y), np.nan)
    scores[tr] = ev.oof_scores[best]
    scores[te] = ev.test_scores[best]
    log.info("models done")
    return scores


def stage_exports(
    ctx: Context,
    spc_mat: dict,
    ms: multivariate.MSPCResult,
    scores: np.ndarray,
    daily: pd.DataFrame,
    weekly: pd.DataFrame,
    spc_long: pd.DataFrame,
    rank: pd.DataFrame,
    cap: pd.DataFrame,
) -> None:
    t2_par, spe_par = ms.alarms(empirical=False)
    t2_emp, spe_emp = ms.alarms(empirical=True)
    runs = ctx.meta.drop(columns=["label"]).assign(
        date=ctx.meta.date.dt.date,
        result=np.where(ctx.y == 1, "FAIL", "PASS"),
        spc_phase=np.where(ctx.phase1, "I", "II"),
        model_split=np.where(ctx.train, "train (out-of-fold)", "test (hold-out)"),
        n_sensors_beyond_3sigma=spc_mat["rule1"].sum(axis=1).to_numpy(),
        n_sensors_any_nelson_rule=spc_mat["any_rule"].sum(axis=1).to_numpy(),
        hotelling_t2=ms.t2.round(3),
        hotelling_t2_ucl=round(ms.t2_ucl, 3),
        spe_q=ms.spe.round(3),
        spe_q_ucl=round(ms.spe_ucl, 3),
        t2_alarm=t2_par,
        spe_alarm=spe_par,
        multivariate_alarm=t2_par | spe_par,
        t2_ucl_empirical=round(ms.t2_ucl_emp, 3),
        spe_q_ucl_empirical=round(ms.spe_ucl_emp, 3),
        multivariate_alarm_empirical=t2_emp | spe_emp,
        predicted_fail_prob=np.round(scores, 4),
        pc1=ms.scores.PC1.round(4),
        pc2=ms.scores.PC2.round(4),
    )
    runs.to_csv(EXPORT_DIR / "runs.csv", index=False)
    daily.assign(date=daily.date.dt.date).to_csv(EXPORT_DIR / "daily_kpi.csv", index=False)
    weekly.to_csv(EXPORT_DIR / "weekly_kpi.csv", index=False)
    spc_long.to_csv(EXPORT_DIR / "spc_top_sensors_long.csv", index=False)
    rank.reset_index().to_csv(EXPORT_DIR / "sensor_ranking.csv", index=False)
    cap.reset_index().to_csv(EXPORT_DIR / "capability.csv", index=False)
    log.info("exports written")


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------
def _clear_sensor_specific_outputs() -> None:
    """Remove files whose names depend on the ranking from a previous run."""
    stale = [*FIG_DIR.glob("08_imr_*.png"), *FIG_DIR.glob("09_xbar_s_*.png"), *TABLE_DIR.glob("xbar_s_*.csv")]
    for path in stale:
        path.unlink()


def run(cfg: Config = CFG, fast: bool = False) -> dict:
    """Run every stage and return the summary dictionary."""
    t0 = time.time()
    ensure_dirs()
    _clear_sensor_specific_outputs()
    plots.style()

    ctx = stage_data(cfg, fast)
    stage_screening(ctx)
    spc_mat = stage_univariate_spc(ctx)
    daily, weekly = stage_yield(ctx)
    cap = stage_capability(ctx)
    ms = stage_mspc(ctx)
    stage_drift(ctx)
    rank = stage_ranking(ctx, spc_mat, cap)
    spc_long = stage_sensor_charts(ctx, rank)
    scores = stage_models(ctx, spc_mat["rule1"].sum(axis=1).to_numpy(), ms)
    stage_exports(ctx, spc_mat, ms, scores, daily, weekly, spc_long, rank, cap)

    ctx.summary["runtime_s"] = round(time.time() - t0, 1)
    ctx.summary["config"] = cfg.as_dict()
    (TABLE_DIR.parent / "summary.json").write_text(json.dumps(ctx.summary, indent=2, default=str))
    log.info("done in %.0f s", time.time() - t0)
    return ctx.summary


def main() -> None:
    parser = argparse.ArgumentParser(description="SECOM statistical process control and failure analysis")
    parser.add_argument("--fast", action="store_true", help="fewer bootstrap rounds; skip shuffled CV; one walk-forward model")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    run(fast=args.fast)


if __name__ == "__main__":
    main()
