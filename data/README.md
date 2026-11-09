# Data

`raw/` holds an unmodified copy of the **SECOM** dataset from the UCI Machine Learning Repository.

| File | Contents |
|---|---|
| `secom.data` | 1567 rows (one per production run) by 590 space-separated sensor readings; missing values are `NaN` |
| `secom_labels.data` | one row per run: `-1` = pass, `1` = fail, then a quoted `dd/mm/yyyy HH:MM:SS` timestamp |
| `secom.names` | original description from the donors |

Facts verified by loading the files (see `reports/summary.json`):

* 1567 runs, 590 sensors, 104 fails (6.64 %), 1463 passes
* timestamps run from 2008-07-19 11:55 to 2008-10-17 06:07 across 86 production days, already in chronological order
* 4.54 % of all sensor cells are missing
* sensor names are anonymised; there are **no units and no specification limits**

UCI's metadata page lists "591 features". That count includes the label column; the sensor matrix itself is 590 columns wide.

To re-download: `python scripts/download_data.py --force`.

## Citation and license

McCann, M. & Johnston, A. (2008). *SECOM* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C54305

Licensed under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). Redistributed here unchanged with attribution.
