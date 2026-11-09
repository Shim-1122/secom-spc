"""End-to-end pipeline: ``python -m secom_spc.pipeline``.

Produces every figure in ``reports/figures``, every table in
``reports/tables``, dashboard-ready exports in ``reports/exports`` and a
machine-readable ``reports/summary.json``.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from . import capability as CAP
from . import drift as DR
from . import features as FE
from . import modeling as MO
from . import multivariate as MV
from . import plots as PL
from . import spc as SPC
from .config import CFG, EXPORT_DIR, FIG_DIR, TABLE_DIR, Config, ensure_dirs
from .data import load, phase_split
from .preprocess import SensorScreen, missingness_profile


def log(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)


def daily_kpis(meta: pd.DataFrame, phase1: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    m = meta.assign(phase1=phase1)
    daily = (
        m.groupby("date")
        .agg(runs=("run_id", "size"), fails=("fail", "sum"), all_phase1=("phase1", "all"))
        .reset_index()
    )
    daily["passes"] = daily.runs - daily.fails
    daily["yield_pct"] = 100 * daily.passes / daily.runs
    roll = daily.set_index("date")[["runs", "passes"]].rolling("7D").sum()
    daily["yield_7d"] = (100 * roll.passes / roll.runs).values
    base = daily.all_phase1.values
    pch = SPC.p_chart(daily.fails.values, daily.runs.values, base, CFG.sigma_limits)
    lan = SPC.laney_p_chart(daily.fails.values, daily.runs.values, base, CFG.sigma_limits)
    daily["p_ucl"] = pch.ucl.values
    daily["p_ooc_high"] = pch.ooc_high.values
    daily["laney_ucl"] = lan.ucl.values
    daily["laney_ooc_high"] = lan.ooc_high.values
    daily.attrs["laney_sigma_z"] = lan.attrs["sigma_z"]
    weekly = (
        m.set_index("timestamp")
        .resample("W-MON", label="left", closed="left")
        .agg({"run_id": "size", "fail": "sum"})
        .rename(columns={"run_id": "runs", "fail": "fails"})
    )
    weekly = weekly[weekly.runs > 0]
    weekly["yield_pct"] = 100 * (1 - weekly.fails / weekly.runs)
    return daily, pch, weekly.reset_index().rename(columns={"timestamp": "week_start"})


def run(cfg: Config = CFG, fast: bool = False) -> dict:
    t0 = time.time()
    ensure_dirs()
    for old in list(FIG_DIR.glob("08_imr_*.png")) + list(FIG_DIR.glob("09_xbar_s_*.png")) + list(TABLE_DIR.glob("xbar_s_*.csv")):
        old.unlink()  # sensor-specific outputs; names depend on the ranking
    PL.style()
    S: dict = {}

    # ------------------------------------------------------------ data
    d = load()
    X, meta = d.X, d.meta
    y = meta.fail.values
    n = len(meta)
    phase1, phase2 = phase_split(n, cfg.phase1_frac)
    ref = phase1 & (y == 0) if cfg.phase1_pass_only else phase1
    boundary = int(phase1.sum())
    train, test = phase_split(n, 1 - cfg.test_frac)
    S["data"] = {
        "runs": n,
        "sensors": X.shape[1],
        "fails": int(y.sum()),
        "passes": int((y == 0).sum()),
        "fail_rate": float(y.mean()),
        "start": str(meta.timestamp.min()),
        "end": str(meta.timestamp.max()),
        "production_days": int(meta.date.nunique()),
        "missing_cells_pct": float(100 * X.isna().mean().mean()),
        "phase1_runs": boundary,
        "phase1_end": str(meta.timestamp[boundary - 1]),
        "phase1_ref_pass_runs": int(ref.sum()),
        "train_runs": int(train.sum()),
        "test_runs": int(test.sum()),
        "train_fails": int(y[train].sum()),
        "test_fails": int(y[test].sum()),
        "test_start": str(meta.timestamp[int(train.sum())]),
    }
    log(f"loaded {n} runs x {X.shape[1]} sensors, {y.sum()} fails", t0)

    # ---------------------------------------------------- preprocessing
    miss = missingness_profile(X)
    miss.to_csv(TABLE_DIR / "missingness.csv")
    PL.missing_figure(miss, FIG_DIR / "01_missingness.png")
    screen = SensorScreen(cfg.max_missing_frac, cfg.min_unique, cfg.min_rel_std, cfg.corr_dedup_threshold)
    screen.fit(X[phase1])
    Xr = screen.transform(X, impute=False)
    Xi = screen.transform(X, impute=True)
    rep = screen.report_
    rep.summary().to_csv(TABLE_DIR / "sensor_screening.csv", index=False)
    pd.Series(rep.dropped_duplicate, name="kept_as").rename_axis("dropped").to_csv(TABLE_DIR / "duplicate_sensors.csv")
    S["screening"] = rep.summary().set_index("stage")["count"].to_dict()
    log(f"screening kept {len(rep.kept)} sensors", t0)

    # ---------------------------------------------------- univariate SPC
    mat = SPC.run_rule_matrix(Xr, ref, cfg.sigma_limits)
    spc_sum = mat["summary"]
    n_r1 = mat["rule1"].sum(axis=1).values
    n_any = mat["any_rule"].sum(axis=1).values
    spc_sum.to_csv(TABLE_DIR / "spc_sensor_summary.csv")
    PL.alarms_over_time(meta, n_r1, boundary, FIG_DIR / "05_alarm_load.png")
    S["spc"] = {
        "median_rule1_rate_phase1_pct": float(100 * spc_sum.rule1_rate_phase1.median()),
        "median_rule1_rate_phase2_pct": float(100 * spc_sum.rule1_rate_phase2.median()),
        "sensors_rule1_rate_phase2_gt_5pct": int((spc_sum.rule1_rate_phase2 > 0.05).sum()),
        "runs_with_any_rule1_pct": float(100 * (n_r1 > 0).mean()),
        "median_sensors_in_alarm_per_run": float(np.median(n_r1)),
        "mean_sensors_in_alarm_pass": float(n_r1[y == 0].mean()),
        "mean_sensors_in_alarm_fail": float(n_r1[y == 1].mean()),
    }
    log("univariate SPC done", t0)

    # ---------------------------------------------------- KPI / yield
    daily, pch, weekly = daily_kpis(meta, phase1)
    daily.to_csv(TABLE_DIR / "daily_yield.csv", index=False)
    weekly.to_csv(TABLE_DIR / "weekly_yield.csv", index=False)
    PL.yield_figure(daily, pch, meta.date[boundary], FIG_DIR / "02_yield_p_chart.png")
    S["kpi"] = {
        "overall_yield_pct": float(100 * (1 - y.mean())),
        "phase1_yield_pct": float(100 * (1 - y[phase1].mean())),
        "phase2_yield_pct": float(100 * (1 - y[phase2].mean())),
        "p_chart_pbar_pct": float(100 * pch.pbar.iloc[0]),
        "p_chart_days_ooc_high": int(daily.p_ooc_high.sum()),
        "laney_sigma_z": float(daily.attrs["laney_sigma_z"]),
        "laney_days_ooc_high": int(daily.laney_ooc_high.sum()),
        "ooc_days": [str(dt.date()) for dt in daily.date[daily.p_ooc_high]],
        "worst_week": weekly.loc[weekly.yield_pct.idxmin()].astype(str).to_dict(),
    }
    log("yield KPIs done", t0)

    # ---------------------------------------------------- capability
    cap = CAP.capability_table(Xr, phase1, cfg.spec_k)
    cap.to_csv(TABLE_DIR / "capability.csv")
    PL.capability_figure(cap, FIG_DIR / "07_capability_drift.png")
    S["capability"] = {
        "sensors_evaluated": len(cap),
        "spec_k_proxy": cfg.spec_k,
        "median_Ppk_I": float(cap.Ppk_I.median()),
        "median_Ppk_II": float(cap.Ppk_II.median()),
        "pct_capable_II": float(100 * cap.capable_II.mean()),
        "pct_Ppk_II_below_1": float(100 * (cap.Ppk_II < 1).mean()),
        "pct_normal_at_5pct": float(100 * cap.normal_at_5pct.mean()),
        "worst5": cap.head(5)[["Ppk_I", "Ppk_II", "mean_shift_sigma", "sd_ratio_II_I"]].round(3).to_dict("index"),
    }
    log("capability done", t0)

    # ---------------------------------------------------- multivariate
    ms = MV.fit_mspc(Xi, ref, cfg.pca_var_explained, cfg.mspc_alpha)
    t2_alarm = ms.t2 > ms.t2_ucl
    spe_alarm = ms.spe > ms.spe_ucl
    mv_alarm = t2_alarm | spe_alarm
    PL.mspc_figure(ms.t2, ms.spe, ms.t2_ucl, ms.spe_ucl, boundary, y, ms.t2_ucl_emp, ms.spe_ucl_emp, FIG_DIR / "06_hotelling_t2_spe.png")

    def rate(mask, sel):
        return float(100 * mask[sel].mean())

    S["mspc"] = {
        "pcs_retained": ms.n_components,
        "var_explained_pct": float(100 * ms.explained[: ms.n_components].sum()),
        "t2_ucl": ms.t2_ucl,
        "spe_ucl": ms.spe_ucl,
        "t2_alarm_rate_ref_pct": rate(t2_alarm, ref),
        "spe_alarm_rate_ref_pct": rate(spe_alarm, ref),
        "t2_alarm_rate_phase2_pct": rate(t2_alarm, phase2),
        "spe_alarm_rate_phase2_pct": rate(spe_alarm, phase2),
        "fail_rate_given_mv_alarm_phase2_pct": rate(y == 1, phase2 & mv_alarm),
        "fail_rate_given_no_mv_alarm_phase2_pct": rate(y == 1, phase2 & ~mv_alarm),
        "fails_caught_by_mv_alarm_phase2_pct": rate(mv_alarm, phase2 & (y == 1)),
        "t2_ucl_empirical": ms.t2_ucl_emp,
        "spe_ucl_empirical": ms.spe_ucl_emp,
        "t2_alarm_rate_phase2_empirical_pct": rate(ms.t2 > ms.t2_ucl_emp, phase2),
        "spe_alarm_rate_phase2_empirical_pct": rate(ms.spe > ms.spe_ucl_emp, phase2),
    }
    # contribution plot for the most extreme failing Phase II run
    p2f = np.where(phase2 & (y == 1))[0]
    worst = p2f[np.argmax(ms.spe[p2f] / ms.spe_ucl)]
    contrib = MV.top_contributors(ms.spe_contrib, worst, k=8)
    PL.contribution_figure(contrib, int(meta.run_id[worst]), "SPE", FIG_DIR / "06b_spe_contribution.png")
    S["mspc"]["example_run"] = int(meta.run_id[worst])
    S["mspc"]["example_top_contributors"] = {k: round(float(v), 3) for k, v in contrib}
    log(f"MSPC done ({ms.n_components} PCs)", t0)

    # ---------------------------------------------------- drift
    ks = DR.ks_drift(Xr, phase1, phase2)
    ks.to_csv(TABLE_DIR / "drift_ks.csv")
    adv_auc, adv_imp = DR.adversarial_validation(Xi, phase2.astype(int), seed=cfg.random_state)
    adv_imp.rename("adversarial_importance").to_csv(TABLE_DIR / "drift_adversarial_importance.csv")
    week = meta.timestamp.dt.to_period("W").astype(str)
    frozen = DR.frozen_sensors(Xr, week)
    frozen.to_csv(TABLE_DIR / "frozen_then_live_sensors.csv")
    drift_show = list(dict.fromkeys(list(adv_imp.index[:4]) + list(ks.index[:4])))[:6]
    PL.drift_figure(Xr, meta, drift_show, meta.timestamp[boundary], FIG_DIR / "11_drift_sensors.png")
    S["drift"] = {
        "adversarial_auc_phase1_vs_phase2": adv_auc,
        "sensors_ks_p_below_1e-6": int((ks.ks_p < 1e-6).sum()),
        "sensors_ks_stat_above_0.5": int((ks.ks_stat > 0.5).sum()),
        "top_ks": ks.head(8)[["ks_stat", "median_shift_in_early_sd", "sd_ratio_late_early"]].round(3).to_dict("index"),
        "top_adversarial": adv_imp.head(8).round(4).to_dict(),
        "frozen_then_live": frozen.to_dict("index"),
    }
    # Sensitivity: how much of the Phase II SPE alarm load comes from a
    # handful of sensors that changed regime?
    sens_rows = []
    for label, drop in (
        ("all retained sensors", []),
        ("minus frozen-then-live sensors", list(frozen.index)),
        ("minus top 20 KS-drift sensors", list(ks.index[:20])),
        ("minus all sensors with KS p < 1e-6", list(ks.index[ks.ks_p < 1e-6])),
    ):
        Xs_ = Xi.drop(columns=[c for c in drop if c in Xi.columns])
        m_ = MV.fit_mspc(Xs_, ref, cfg.pca_var_explained, cfg.mspc_alpha)
        sens_rows.append(
            {
                "sensor_set": label,
                "n_sensors": Xs_.shape[1],
                "pcs": m_.n_components,
                "t2_alarm_phase2_pct": 100 * float((m_.t2[phase2] > m_.t2_ucl_emp).mean()),
                "spe_alarm_phase2_pct": 100 * float((m_.spe[phase2] > m_.spe_ucl_emp).mean()),
            }
        )
    sens = pd.DataFrame(sens_rows)
    sens.to_csv(TABLE_DIR / "mspc_drift_sensitivity.csv", index=False)
    S["drift"]["mspc_sensitivity"] = sens.round(1).to_dict("records")
    log(f"drift done (adversarial AUC {adv_auc:.3f})", t0)

    # ---------------------------------------------------- feature ranking
    nb = 30 if fast else cfg.n_stability_boot
    rank = FE.consensus_rank(Xr[train], Xi[train], y[train], n_boot=nb, C=cfg.stability_C, seed=cfg.random_state)
    lift = FE.alarm_lift(mat["rule1"][train], y[train])
    rank = rank.join(lift[["n_alarms", "alarm_lift", "fisher_p"]])
    rank = rank.join(spc_sum[["rule1_rate_phase2"]]).join(cap[["Ppk_I", "Ppk_II"]])
    # out-of-time check: does each sensor's direction and strength hold in the test period?
    test_uni = FE.univariate_tests(Xr[test], y[test])
    rank["test_auc"] = test_uni.auc
    rank["direction_holds_in_test"] = np.sign(rank.auc - 0.5) == np.sign(rank.test_auc - 0.5)
    rank["within_week_auc"] = DR.within_period_auc(Xr[train], y[train], week.values[train])
    rank.to_csv(TABLE_DIR / "sensor_ranking.csv")
    FE.method_agreement(rank).round(3).to_csv(TABLE_DIR / "ranking_method_agreement.csv")
    k = cfg.top_k_sensors
    top = rank.head(k)
    PL.ranking_figure(rank, k, FIG_DIR / "03_sensor_ranking.png")
    PL.rank_heatmap(rank, k, FIG_DIR / "03b_rank_agreement.png")
    PL.confound_figure(rank, FIG_DIR / "03c_time_confound.png")
    PL.distribution_figure(Xr[train], y[train], list(rank.index[:9]), FIG_DIR / "04_top_sensor_distributions.png")
    S["ranking"] = {
        "n_sig_fdr_10pct": int((rank.mw_q < 0.10).sum()),
        "n_sig_fdr_5pct": int((rank.mw_q < 0.05).sum()),
        "top10": rank.head(10)[["auc", "mw_q", "stability_freq", "direction", "test_auc"]].round(3).to_dict("index"),
        "topk_direction_holds_pct": float(100 * top.direction_holds_in_test.mean()),
        "topk_median_train_auc_effect": float(top.auc_effect.median()),
        "topk_median_test_auc_effect": float((top.test_auc - 0.5).abs().median()),
        "top10_within_week_auc": rank.head(10).within_week_auc.round(3).to_dict(),
        "method_spearman_min": float(FE.method_agreement(rank).values[np.triu_indices(5, 1)].min()),
        "method_spearman_max": float(FE.method_agreement(rank).values[np.triu_indices(5, 1)].max()),
    }
    log("feature ranking done", t0)

    # ---------------------------------------------------- per-sensor charts
    charted = list(rank.index[: cfg.n_charted_sensors])
    spc_long = []
    for i, s in enumerate(charted, 1):
        ch = SPC.imr_chart(Xr[s].values, ref, cfg.sigma_limits)
        PL.imr_figure(ch, s, boundary, FIG_DIR / f"08_imr_{i:02d}_{s}.png", fails=y)
        if i <= 2:
            st = SPC.xbar_s_chart(Xr[s].values, ref, cfg.subgroup_size, cfg.sigma_limits)
            PL.xbar_s_figure(st, s, cfg.subgroup_size, FIG_DIR / f"09_xbar_s_{i:02d}_{s}.png")
            st.assign(sensor=s).to_csv(TABLE_DIR / f"xbar_s_{s}.csv", index=False)
    for s in rank.index[:k]:
        ch = SPC.imr_chart(Xr[s].values, ref, cfg.sigma_limits)
        li = ch.attrs["limits_i"]
        ew = SPC.ewma_chart(Xr[s].values, li["center"], li["sigma"], cfg.ewma_lambda, cfg.ewma_L)
        cu = SPC.cusum_chart(Xr[s].values, li["center"], li["sigma"], cfg.cusum_k, cfg.cusum_h)
        spc_long.append(
            pd.DataFrame(
                {
                    "run_id": meta.run_id,
                    "timestamp": meta.timestamp,
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
                    "fail": y,
                }
            )
        )
    log("per-sensor charts done", t0)

    # ---------------------------------------------------- models
    extra = {
        "SPC baseline: # sensors beyond 3σ": n_r1.astype(float),
        "MSPC baseline: Hotelling T²": ms.t2,
        "MSPC baseline: SPE (Q)": ms.spe,
    }
    ev = MO.evaluate_models(Xr, y, train, cfg, extra_scores=extra)
    ev.table.to_csv(TABLE_DIR / "model_comparison_temporal.csv")
    PL.roc_pr_figure(y[test], ev.test_scores, FIG_DIR / "10_roc_pr_holdout.png")
    S["models_temporal"] = ev.table[
        ["train_cv_roc_auc", "train_cv_ber", "test_roc_auc", "test_roc_auc_ci95", "test_pr_auc", "test_ber", "test_recall_fail", "test_specificity"]
    ].round(3).to_dict("index")
    if not fast:
        rcv = MO.random_split_cv(Xr, y, cfg)
        rcv.to_csv(TABLE_DIR / "model_comparison_random_cv10.csv")
        S["models_random_cv10"] = rcv.round(3).to_dict("index")
    wf_models = [f"Logistic (top {cfg.top_k_sensors} by AUC, in-fold)", "L1 logistic (all sensors)", "Gradient boosting (hist, NaN-native)"]
    if fast:
        wf_models = wf_models[:1]
    wf, _ = MO.walk_forward(Xr, y, cfg, start_frac=cfg.phase1_frac, block=100, models=wf_models)
    wf_s, _ = MO.walk_forward(Xr, y, cfg, start_frac=cfg.phase1_frac, block=100, window=500, models=wf_models[:1])
    wf_s = wf_s.rename(index={"retrained": "retrained, sliding 500-run window"}, level=1)
    wf_all = pd.concat([wf, wf_s.xs("retrained, sliding 500-run window", level=1, drop_level=False)])
    wf_all.to_csv(TABLE_DIR / "model_walk_forward.csv")
    PL.walk_forward_figure(wf, FIG_DIR / "10b_walk_forward.png")
    S["models_walk_forward"] = {f"{a} | {b}": v for (a, b), v in wf_all.round(3).to_dict("index").items()}
    log("models done", t0)

    best = ev.table.drop(index=list(extra)).test_pr_auc.idxmax()
    S["best_model_by_test_pr_auc"] = best
    full_score = np.full(n, np.nan)
    full_score[test] = ev.test_scores[best]
    full_score[train] = ev.oof_scores[best]

    # ---------------------------------------------------- exports
    runs = meta.assign(
        timestamp=meta.timestamp,
        date=meta.date.dt.date,
        result=np.where(y == 1, "FAIL", "PASS"),
        spc_phase=np.where(phase1, "I", "II"),
        model_split=np.where(train, "train (out-of-fold)", "test (hold-out)"),
        n_sensors_beyond_3sigma=n_r1,
        n_sensors_any_nelson_rule=n_any,
        hotelling_t2=ms.t2.round(3),
        hotelling_t2_ucl=round(ms.t2_ucl, 3),
        spe_q=ms.spe.round(3),
        spe_q_ucl=round(ms.spe_ucl, 3),
        t2_alarm=t2_alarm,
        spe_alarm=spe_alarm,
        multivariate_alarm=mv_alarm,
        t2_ucl_empirical=round(ms.t2_ucl_emp, 3),
        spe_q_ucl_empirical=round(ms.spe_ucl_emp, 3),
        multivariate_alarm_empirical=(ms.t2 > ms.t2_ucl_emp) | (ms.spe > ms.spe_ucl_emp),
        predicted_fail_prob=np.round(full_score, 4),
        pc1=ms.scores.PC1.round(4),
        pc2=ms.scores.PC2.round(4),
    )
    runs.drop(columns=["label"]).to_csv(EXPORT_DIR / "runs.csv", index=False)
    daily.assign(date=daily.date.dt.date).to_csv(EXPORT_DIR / "daily_kpi.csv", index=False)
    weekly.to_csv(EXPORT_DIR / "weekly_kpi.csv", index=False)
    pd.concat(spc_long).to_csv(EXPORT_DIR / "spc_top_sensors_long.csv", index=False)
    rank.reset_index().to_csv(EXPORT_DIR / "sensor_ranking.csv", index=False)
    cap.reset_index().to_csv(EXPORT_DIR / "capability.csv", index=False)
    log("exports written", t0)

    S["runtime_s"] = round(time.time() - t0, 1)
    S["config"] = {k: v for k, v in cfg.__dict__.items() if k != "extra"}
    (TABLE_DIR.parent / "summary.json").write_text(json.dumps(S, indent=2, default=str))
    log("done", t0)
    return S


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the SECOM SPC + failure analytics pipeline")
    ap.add_argument("--fast", action="store_true", help="fewer bootstrap rounds, skip 10-fold CV")
    args = ap.parse_args()
    run(fast=args.fast)


if __name__ == "__main__":
    main()
