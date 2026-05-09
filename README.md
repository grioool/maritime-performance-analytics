# Maritime Performance Analytics

Maritime Performance Analytics evaluates vessel Speed Over Ground (SOG) under metocean conditions and quantifies the operational impact of wind-assisted propulsion (Flettner rotors), with emphasis on North Sea operations.

## What is implemented

This repository currently includes a complete notebook pipeline:

1. Data alignment and preprocessing (AIS + ERA5 + Copernicus)
2. Leakage-aware feature engineering
3. Voyage-based model training and evaluation
4. Rotor what-if scenario simulation
5. Weather-window comparison (baseline vs rotor-assisted)
6. Four extension analyses (energy/fuel/CO₂, route segmentation, route optimisation, rotor gain detection)

## Current repository structure

```text
maritime-performance-analytics/
├── data/
│   ├── MasterSet.csv                        # AIS + ERA5 + Copernicus merged (not committed — large)
│   ├── MasterSet_features.csv               # 57-column engineered feature set (not committed — large)
│   ├── PositionReport_cleaned.csv
│   └── PositionReport_final.csv
├── eda/
│   ├── EDA.ipynb
│   └── feature_engineering.ipynb
├── extensions/
│   ├── 01_energy_fuel_co2.ipynb             # Fuel consumption & CO₂ savings from rotor operation
│   ├── 02_route_segmentation.ipynb          # Heading-based regime segmentation + per-regime models
│   ├── 03_route_optimization.ipynb          # Dijkstra routing to maximise rotor gain
│   └── 04_rotor_gain_detection.ipynb        # Wind-angle analysis & high-gain classifier
├── ml/
│   └── baseline_model.ipynb
├── poc/
│   ├── data_preprocessing.ipynb
│   ├── interpolation.ipynb
│   └── test_ERA5.ipynb
├── results/                                 # PNG outputs from all notebooks
├── specification/
└── README.md
```

## Modeling goal

Predict SOG as a function of forecast-available conditions and observed trajectory context.

Primary target:
- `Speed_kn` (SOG)

Primary questions:
- How strongly do weather and sea-state conditions influence SOG?
- Under what conditions does rotor assistance improve operational performance?
- Do rotors expand useful weather-window coverage?

## Essential engineering updates

Recent updates implemented in `eda/feature_engineering.ipynb` and consumed in `ml/baseline_model.ipynb`:

- Lag features (voyage-grouped and target-safe):
	- `Speed_kn_lag1`, `Speed_kn_lag2`, `Speed_kn_mean3`
	- `Speed_kn_mean3` is shifted first, so it uses only previous observations and never includes the current target row
- Apparent wind features:
	- `Apparent_Wind_Speed_ms`, apparent wind direction, apparent relative angle
- Circular angle encoding:
	- `alpha_sin`, `alpha_cos`, `app_alpha_sin`, `app_alpha_cos`, wind/current directional sin-cos encodings
- Current-effect features:
	- current magnitude, direction, along-course and cross-course components
- Interaction features:
	- wind-angle, apparent-wind-angle, wind-wave, nonlinear wind/wave terms

## Validation rules

The workflow now enforces:

- Voyage-based chronological split:
	- train and test use different voyages
	- no temporal leakage between train/test voyage groups
- No direct target leakage features:
	- do not use `STW`, `AWS`, `AWA` as model features
- Model goals are separated:
	- weather-only explanatory model excludes target-history features
	- lagged nowcast model uses previous observed SOG and apparent-wind context for short-horizon prediction
- Rotor contribution is applied post-prediction (scenario step), not as a training feature

## Models and current results

From the latest full run of `ml/baseline_model.ipynb`:

Dataset split:
- Train: 88,346 rows across 422 voyages
- Test: 35,758 rows across 106 voyages
- Temporal leakage check: `True`

Model performance:
- B1 Constant mean:
	- MAE 1.388
	- RMSE 1.893
	- R2 -0.002
	- within +/-1 kn: 49.3%
- B2 Linear (wind + wave + encoded alpha):
	- MAE 1.377
	- RMSE 1.860
	- R2 0.034
	- within +/-1 kn: 49.5%
- M1 XGBoost (weather-only):
	- MAE 1.468
	- RMSE 1.883
	- R2 0.010
	- within +/-1 kn: 41.7%
- M2 XGBoost (lagged nowcast):
	- MAE 0.406
	- RMSE 0.758
	- R2 0.839
	- within +/-1 kn: 91.7%

Top feature importances (latest run):
- M1 weather-only:
	- `Course_deg`
	- `current_x_course_help`
	- `wind_speed_sq`
	- wind-angle/current/wave features
- M2 lagged nowcast:
	- `Speed_kn_lag1`
	- `Speed_kn_mean3`
	- followed by `Course_deg`, `Speed_kn_lag2`, and apparent-wind features

Interpretation note:
- The strongest predictive signal in the nowcast model comes from lagged speed terms.
- Weather-only features alone explain little test-set variance under the current formulation, so weather-impact interpretation should be treated separately from short-horizon prediction.

## Rotor what-if and weather windows

Rotor scenario (latest run):
- Rotor active share: 98.1%
- Mean rotor power (active): 41.2 kW
- Mean delta SOG (active): 0.206 kn
- Mean delta SOG (overall): 0.202 kn

Weather-window definition used:
- `SOG >= 10 kn` and `H_s <= 3 m`

Weather-window results:
- Without rotor:
	- windows: 866
	- points in window: 27,167
	- coverage: 76.0%
- With rotor:
	- windows: 808
	- points in window: 27,696
	- coverage: 77.5%
- Delta:
	- windows: -58
	- points: +529
	- coverage: +1.5%

Note:
- Fewer windows with higher coverage suggests many windows become longer/merged when rotor uplift is applied.

## Extensions

Four optional extension notebooks in `extensions/` build on the baseline pipeline:

| Notebook | Topic | Key output |
|---|---|---|
| `01_energy_fuel_co2.ipynb` | Fuel & CO₂ savings | Annual fuel reduction and CII delta from rotor operation |
| `02_route_segmentation.ipynb` | Route segmentation | Per-regime (N/S/E/W) XGBoost models and rotor benefit by heading |
| `03_route_optimization.ipynb` | Route optimisation | Dijkstra routing on 0.5° grid to maximise integrated rotor gain |
| `04_rotor_gain_detection.ipynb` | Gain detection | Wind-angle polar analysis and binary classifier for high-gain moments |

All extensions read from `data/MasterSet_features.csv` and use the same rotor polar diagram and voyage-based split as the baseline model.

## Known caveat

Current exported feature file `data/MasterSet_features.csv` includes `Course_deg` but does not currently export `course_sin` and `course_cos`. The model currently uses `Course_deg` directly.

## How to run

Recommended notebook order:

1. `poc/data_preprocessing.ipynb`
2. `poc/interpolation.ipynb`
3. `poc/test_ERA5.ipynb`
4. `eda/feature_engineering.ipynb`
5. `ml/baseline_model.ipynb`
6. `extensions/01_energy_fuel_co2.ipynb` *(optional)*
7. `extensions/02_route_segmentation.ipynb` *(optional)*
8. `extensions/03_route_optimization.ipynb` *(optional)*
9. `extensions/04_rotor_gain_detection.ipynb` *(optional)*

Environment notes:
- Python libraries used include: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`, `scipy`, `openpyxl`
- On macOS, XGBoost may require OpenMP runtime (`libomp`)

## Data and security notice

This repository should contain only code, notebooks, and reproducible workflows.

Do not commit:
- proprietary/raw restricted datasets
- credentials or tokens
- private infrastructure paths

Use approved institutional/local storage for restricted inputs.
