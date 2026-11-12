"""Build excel/SECOM_SPC.xlsx, the base for SECOM_SPC.xlsm (data + Control sheet) from reports/exports. VBA lives in excel/SECOM_SPC.bas."""

from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

root = Path(__file__).resolve().parents[1]
ex = root / "reports" / "exports"
runs = pd.read_csv(ex / "runs.csv", parse_dates=["timestamp"])
long = pd.read_csv(ex / "spc_top_sensors_long.csv")
wide = long.pivot(index="run_id", columns="sensor", values="value")
order = list(long.drop_duplicates("sensor").sort_values("consensus_rank")["sensor"])
wide = wide[order].reset_index()
sens = runs[["run_id", "timestamp", "fail"]].merge(wide, on="run_id", how="left")

wb = Workbook()
ctl = wb.active
ctl.title = "Control"
bold = Font(bold=True)
inp = PatternFill("solid", fgColor="FFF2CC")
ctl["A1"] = "SECOM SPC in Excel (VBA)"
ctl["A1"].font = Font(bold=True, size=14)
ctl["A2"] = "Yellow cells are inputs. Run macros with Alt+F8 (Option+F8 on Mac): RunSensorChart, RunRebaseline."
rows = [("Sensor", "S028"), ("Phase I last run", 400), ("Change point run", 737), ("Sigma multiplier", 3)]
for i, (k, v) in enumerate(rows, start=4):
    ctl.cell(i, 1, k).font = bold
    c = ctl.cell(i, 2, v)
    c.fill = inp
dv = DataValidation(type="list", formula1='"' + ",".join(order) + '"', allow_blank=False)
ctl.add_data_validation(dv)
dv.add("B4")
ctl["A9"] = "Results"
ctl["A9"].font = bold
for i, k in enumerate(["Center line", "LCL", "UCL", "Mean moving range", "Nelson 1-3 flags", "Alarm days"], start=10):
    ctl.cell(i, 1, k)
ctl["A17"] = "Re-baselining (alarm rule 1, individual value beyond limits)"
ctl["A17"].font = bold
for i, k in enumerate(
    [
        "Runs after change",
        "Alarms with frozen limits",
        "Alarms with re-baselined limits",
        "Days after change",
        "Frozen alarms per day",
        "Re-baselined alarms per day",
    ],
    start=18,
):
    ctl.cell(i, 1, k)
ctl.column_dimensions["A"].width = 52
ctl.column_dimensions["B"].width = 22

ws = wb.create_sheet("Sensors")
ws.append(list(sens.columns))
for r in sens.itertuples(index=False):
    ws.append(
        [None if (isinstance(x, float) and pd.isna(x)) else (x.to_pydatetime() if hasattr(x, "to_pydatetime") else x) for x in r]
    )
for c in ws[1]:
    c.font = bold
ws.freeze_panes = "D2"
ws.column_dimensions["B"].width = 18
for r in range(2, ws.max_row + 1):
    ws.cell(r, 2).number_format = "yyyy-mm-dd hh:mm"

wr = wb.create_sheet("Runs")
keep = [
    "run_id",
    "timestamp",
    "fail",
    "hotelling_t2",
    "hotelling_t2_ucl",
    "spe_q",
    "spe_q_ucl",
    "multivariate_alarm",
    "alarm_frozen_limits",
    "alarm_rebaselined",
]
wr.append(keep)
for r in runs[keep].itertuples(index=False):
    wr.append(
        [x.to_pydatetime() if hasattr(x, "to_pydatetime") else (bool(x) if str(type(x)).find("bool") > 0 else x) for x in r]
    )
for c in wr[1]:
    c.font = bold
wr.freeze_panes = "A2"
(root / "excel").mkdir(exist_ok=True)
wb.save(root / "excel" / "SECOM_SPC.xlsx")
print(sens.shape, order[:5])
