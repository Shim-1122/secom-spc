# Spotfire dashboard build guide

The pipeline writes tidy, analysis-ready CSVs to `reports/exports/`. They are shaped so a TIBCO Spotfire (Spotfire Analyst or the free trial of Spotfire Cloud) dashboard can be assembled with drag and drop, no scripting. This page is the spec for that dashboard.

> The repository ships the data and the spec. Screenshots of the built dashboard belong in `docs/img/` once it has been assembled in Spotfire.

## 1. Data tables

Load each file with **File > Add data > Local file**. Keep column types as detected, then check the ones listed.

| Spotfire table | File | Grain | Key columns |
|---|---|---|---|
| `Runs` | `runs.csv` | one row per production run (1567) | `run_id`, `timestamp` (DateTime), `result`, `spc_phase`, `monitor_period`, `hotelling_t2`, `spe_q`, `*_ucl*`, `alarm_frozen_limits`, `alarm_rebaselined`, `alarm_rolling`, `n_sensors_beyond_3sigma`, `predicted_fail_prob` |
| `DailyKPI` | `daily_kpi.csv` | one row per production day (86) | `date` (Date), `runs`, `fails`, `pass_rate_pct`, `pass_rate_7d`, `p_ucl`, `p_ooc_high`, `laney_ucl` |
| `WeeklyKPI` | `weekly_kpi.csv` | one row per week | `week_start`, `runs`, `fails`, `pass_rate_pct` |
| `SensorSPC` | `spc_top_sensors_long.csv` | run x sensor, top 20 sensors (31 340 rows) | `sensor`, `run_id`, `value`, `cl`, `lcl`, `ucl`, `z`, `nelson_rule1..8`, `any_rule`, `ewma*`, `cusum*`, `phase`, `fail` |
| `SensorRanking` | `sensor_ranking.csv` | one row per retained sensor (321) | `sensor`, `consensus_rank`, `auc`, `within_week_auc`, `test_auc`, `mw_q`, `stability_freq`, `alarm_lift` |
| `ShiftSpread` | `sensor_shift_spread.csv` | one row per sensor | `sensor`, `shift_robust_sd`, `spread_ratio`, `ks_stat`, `ks_p` |

Relations (**Data > Data table properties > Relations**): `SensorSPC.sensor = SensorRanking.sensor = ShiftSpread.sensor` and `SensorSPC.run_id = Runs.run_id`. This lets a marking on one page filter the others.

## 2. Calculated columns

Add these with **Data > Add calculated column**.

```text
# Runs
AlarmStatus     = If([alarm_rebaselined], "ALARM", "OK")
RunDate         = Date([timestamp])

# SensorSPC
PointStatus     = Case when [nelson_rule1] then "Beyond 3 sigma"
                       when [any_rule]     then "Run-rule violation"
                       else "In control" end
SigmaZone       = Case when Abs([z]) > 3 then "> 3s"
                       when Abs([z]) > 2 then "2s to 3s"
                       when Abs([z]) > 1 then "1s to 2s"
                       else "< 1s" end
```

## 3. Pages and visuals

### Page 1: Line-test pass rate

* **KPI tiles** (KPI chart): overall line-test pass rate `100 * (Count([run_id]) - Sum([fail])) / Count([run_id])` on `Runs`; Phase II pass rate (same, filtered `spc_phase = "II"`); days out of control `Sum(If([p_ooc_high],1,0))` on `DailyKPI`.
* **Line chart**: `DailyKPI`, X = `date`, Y = `Avg([pass_rate_pct])` as markers and `Avg([pass_rate_7d])` as a line. Add a horizontal reference line at the Phase I pass rate (90.9 %) via **Lines & Curves > Horizontal line**.
* **p chart**: Line chart, X = `date`, Y = `Sum([fails]) / Sum([runs])`. Add **Lines & Curves > Line from column values** for `p_ucl` (dashed). Color markers by `p_ooc_high` (red = True) and set the marker shape to a cross for True so the alarm does not rely on colour alone.
* Optional rolling pass rate computed inside Spotfire rather than from the CSV: `Sum([passes]) OVER (LastPeriods(7, [date])) / Sum([runs]) OVER (LastPeriods(7, [date]))`.

### Page 2: Sensor control charts

* **Property control** (drop-down list) bound to a document property `SelectedSensor`, values from `SensorRanking.sensor` sorted by `consensus_rank`.
* Limit `SensorSPC` data on the visuals with the expression `[sensor] = "${SelectedSensor}"`.
* **Individuals chart**: Line chart, X = `run_id`, Y = `Avg([value])`. Lines from column values for `cl` (solid), `ucl` and `lcl` (dashed). Color by `PointStatus`; shape by `PointStatus`.
* **Moving range chart** underneath: Y = `Avg([moving_range])` with line `mr_ucl`.
* **EWMA chart**: Y = `Avg([ewma])` with lines `ewma_ucl` and `ewma_lcl`.
* Vertical line at the detected process change (run 737, 7 September 2008) via **Lines & Curves > Vertical line**.
* **Text area** showing the selected sensor's row from `SensorRanking` (AUC, within-week AUC, q value, alarm lift) and `ShiftSpread` (median shift and spread ratio).

### Page 3: Multivariate alarms

* **Scatter plot** `Runs`: X = `run_id`, Y = `hotelling_t2` (log scale), colour by `result`, shape by `AlarmStatus`; horizontal line at `t2_ucl_empirical`.
* Same for `spe_q` with `spe_q_ucl_empirical`.
* **Bar chart**: count of alarms per `RunDate`, stacked by `result`.
* Marking a run here filters page 2 through the `run_id` relation.

### Page 4: Failure drivers

* **Bar chart** `SensorRanking`, top 20 by `consensus_rank`, value = `max([auc], 1 - [auc])`.
* **Scatter plot** `auc` vs `within_week_auc` (the time-confound check), label by `sensor`, with a 45 degree reference line `y = x` from **Lines & Curves > Curve**.
* **Table** of the ranking with conditional formatting on `mw_q` (< 0.05 bold).

### Page 5: Process shift

* **Scatter plot** `ShiftSpread`: X = `shift_robust_sd`, Y = `spread_ratio` (log scale), reference lines at 0 and 1, label the sensors with the largest `ks_stat`.
* **Table** sorted by `ks_stat` descending.
* **Line chart** `Runs`: daily count of `alarm_frozen_limits`, `alarm_rebaselined` and `alarm_rolling`, to show the alarm burden of each limit strategy.

## 4. Alert logic

Spotfire **Data functions** or **Alerting** (Spotfire Automation Services) can watch the same rules the Python pipeline evaluates:

| Alert | Condition | Table |
|---|---|---|
| Yield excursion | `[p_ooc_high] = True` | `DailyKPI` |
| Multivariate excursion | `[alarm_rebaselined] = True` for 3 runs in a row | `Runs` |
| Sensor out of control | `[nelson_rule1] or [nelson_rule2] or [nelson_rule5]` for a top-10 sensor | `SensorSPC` |

## 5. Colour and accessibility conventions

Match the static report: blue for data, orange for the second series (fails), red **plus a cross marker** for out-of-control points, dashed gray for control limits, solid dark gray for centre lines. Every coloured state also carries a shape or a text label, so the dashboard reads in grayscale and for colour-vision-deficient users.
