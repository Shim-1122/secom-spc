"""Build the self-contained interactive dashboard (docs/index.html).

Reads the CSV exports and reports/summary.json, embeds the data in a single
HTML file with no external dependencies, and writes it to docs/index.html so
it can be opened locally or served by GitHub Pages.

Usage: python scripts/build_dashboard.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPORTS = ROOT / "reports" / "exports"
TEMPLATE = ROOT / "scripts" / "dashboard_template.html"
OUT = ROOT / "docs" / "index.html"
N_SPC_SENSORS = 8


def build_payload() -> dict:
    summary = json.loads((ROOT / "reports" / "summary.json").read_text())
    runs = pd.read_csv(EXPORTS / "runs.csv")
    daily = pd.read_csv(EXPORTS / "daily_kpi.csv")
    rank = pd.read_csv(EXPORTS / "sensor_ranking.csv")
    spc = pd.read_csv(EXPORTS / "spc_top_sensors_long.csv")
    contrib = pd.read_csv(EXPORTS / "spe_contribution_by_phase.csv")

    alarms = (
        runs.groupby("date")[["alarm_frozen_limits", "alarm_rebaselined", "alarm_rolling"]].sum().reset_index()
    )
    monitoring = runs[runs.monitor_period == "monitoring"]
    burden = {b["monitor"]: b for b in summary["monitoring"]["burden"] if b["period"] == "after re-baselining window"}
    frozen, rebased, rolling = list(burden.values())
    cp_run = int(runs.loc[runs.spc_phase == "II", "run_id"].min())
    cp_date = runs.loc[runs.run_id == cp_run, "date"].iloc[0]
    p1, p2 = runs[runs.spc_phase == "I"], runs[runs.spc_phase == "II"]

    sensors = list(spc.drop_duplicates("sensor").sort_values("consensus_rank").sensor[:N_SPC_SENSORS])
    spc_out = {}
    for s in sensors:
        d = spc[spc.sensor == s].sort_values("run_id")
        spc_out[s] = {
            "run": d.run_id.tolist(),
            "value": d.value.round(4).tolist(),
            "rule": d.any_rule.astype(int).tolist(),
            "cl": float(d.cl.iloc[0]),
            "lcl": float(d.lcl.iloc[0]),
            "ucl": float(d.ucl.iloc[0]),
            "rank": int(d.consensus_rank.iloc[0]),
        }

    top_rank = rank.sort_values("consensus_rank").head(15)
    top_contrib = contrib.head(12)
    return {
        "tiles": {
            "runs": int(len(runs)),
            "sensors": int(summary["drift"]["sensors_evaluated"]),
            "fails": int(runs.fail.sum()),
            "pass_before": round(100 * (1 - p1.fail.mean()), 1),
            "pass_after": round(100 * (1 - p2.fail.mean()), 1),
            "cp_run": cp_run,
            "cp_date": cp_date,
            "alarms_frozen": round(frozen["alarms_per_day"], 1),
            "alarms_rebased": round(rebased["alarms_per_day"], 2),
            "alarms_rolling": round(rolling["alarms_per_day"], 2),
            "monitor_runs": int(len(monitoring)),
            "monitor_days": int(frozen["days"]),
            "monitor_start": monitoring.date.min(),
            "frozen_share_p1": round(summary["mspc"]["spe_share_frozen_then_live_phase1_pct"], 1),
            "frozen_share_p2": round(summary["mspc"]["spe_share_frozen_then_live_phase2_pct"], 1),
        },
        "daily": {
            "date": daily.date.tolist(),
            "runs": daily.runs.tolist(),
            "fails": daily.fails.tolist(),
            "pass": daily.pass_rate_pct.round(2).tolist(),
            "roll": daily.pass_rate_7d.round(2).tolist(),
            "laney": daily.laney_ooc_high.astype(int).tolist(),
        },
        "alarms": {
            "date": alarms.date.tolist(),
            "frozen": alarms.alarm_frozen_limits.astype(int).tolist(),
            "rebased": alarms.alarm_rebaselined.astype(int).tolist(),
            "rolling": alarms.alarm_rolling.astype(int).tolist(),
        },
        "runs": {
            "run": runs.run_id.tolist(),
            "fail": runs.fail.tolist(),
            "t2": runs.hotelling_t2.round(1).tolist(),
            "spe": runs.spe_q.round(1).tolist(),
            "t2_ucl": float(runs.t2_ucl_empirical.iloc[0]),
            "spe_ucl": float(runs.spe_q_ucl_empirical.iloc[0]),
        },
        "rank": {
            "sensor": top_rank.sensor.tolist(),
            "auc": top_rank.auc.round(3).tolist(),
            "test_auc": top_rank.test_auc.round(3).tolist(),
            "within_week_auc": top_rank.within_week_auc.round(3).tolist(),
            "holds": top_rank.direction_holds_in_test.astype(int).tolist(),
        },
        "contrib": {
            "sensor": top_contrib.sensor.tolist(),
            "p1": (100 * top_contrib.share_phase1).round(2).tolist(),
            "p2": (100 * top_contrib.share_phase2).round(2).tolist(),
            "frozen": top_contrib.frozen_then_live.astype(int).tolist(),
        },
        "spc": spc_out,
    }


def _clean(o):
    """Replace NaN (missing sensor readings) with null so the JSON is valid."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    return None if isinstance(o, float) and o != o else o


def main() -> None:
    payload = json.dumps(_clean(build_payload()), separators=(",", ":"), allow_nan=False)
    html = TEMPLATE.read_text().replace("/*__DATA__*/null", payload.replace("</", "<\\/"))
    OUT.write_text(html)
    print(f"wrote {OUT.relative_to(ROOT)} ({len(html) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
