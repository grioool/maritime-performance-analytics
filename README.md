# Maritime Performance Analytics

Maritime Performance Analytics evaluates vessel Speed Over Ground (SOG) under metocean conditions and quantifies the operational impact of wind-assisted propulsion (Flettner rotors), with emphasis on North Sea operations.

## What Is Implemented

This repository currently includes a complete notebook pipeline:

1. Data alignment and preprocessing (AIS + ERA5 + Copernicus)
2. Leakage-aware feature engineering
3. Voyage-based model training and evaluation
4. Rotor what-if scenario simulation
5. Weather-window comparison (baseline vs rotor-assisted)

## Current Repository Structure

```text
maritime-performance-analytics/
├── data/
│   ├── MasterSet.csv
│   ├── MasterSet_features.csv
│   ├── PositionReport_cleaned.csv
│   └── PositionReport_final.csv
├── eda/
│   ├── EDA.ipynb
│   └── feature_engineering.ipynb
├── ml/
│   └── baseline_model.ipynb
├── poc/
│   ├── data_preprocessing.ipynb
│   ├── interpolation.ipynb
│   └── test_ERA5.ipynb
├── specification/
└── README.md
```

## Modeling Goal

Predict SOG as a function of forecast-available conditions and observed trajectory context.

Primary target:
- `Speed_kn` (SOG)

Primary questions:
- How strongly do weather and sea-state conditions influence SOG?
- Under what conditions does rotor assistance improve operational performance?
- Do rotors expand useful weather-window coverage?

## Key Engineering Updates

Recent updates implemented in `eda/feature_engineering.ipynb` and consumed in `ml/baseline_model.ipynb`:

- Lag features (voyage-grouped):
	- `Speed_kn_lag1`, `Speed_kn_lag2`, `Speed_kn_mean3`
- Apparent wind features:
	- `Apparent_Wind_Speed_ms`, apparent wind direction, apparent relative angle
- Circular angle encoding:
	- `alpha_sin`, `alpha_cos`, `app_alpha_sin`, `app_alpha_cos`, wind/current directional sin-cos encodings
- Current-effect features:
	- current magnitude, direction, along-course and cross-course components
- Interaction features:
	- wind-angle, apparent-wind-angle, wind-wave, nonlinear wind/wave terms

## Leakage and Validation Rules

The workflow now enforces:

- Voyage-based chronological split:
	- train and test use different voyages
	- no temporal leakage between train/test voyage groups
- No direct target leakage features:
	- do not use `STW`, `AWS`, `AWA` as model features
- Rotor contribution is applied post-prediction (scenario step), not as a training feature

## Models and Current Results

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
- M1 XGBoost (extended features):
	- MAE 0.111
	- RMSE 0.162
	- R2 0.993
	- within +/-1 kn: 99.9%

Top feature importances (latest run):
- `Speed_kn_mean3`
- `Speed_kn_lag1`
- `Speed_kn_lag2`
- followed by `Course_deg` and interaction/current features

Interpretation note:
- The strongest predictive signal currently comes from lagged speed terms (short-horizon trajectory memory).

## Rotor What-If and Weather Windows

Rotor scenario (latest run):
- Rotor active share: 98.1%
- Mean rotor power (active): 41.2 kW
- Mean delta SOG (active): 0.206 kn
- Mean delta SOG (overall): 0.202 kn

Weather-window definition used:
- `SOG >= 10 kn` and `H_s <= 3 m`

Weather-window results:
- Without rotor:
	- windows: 754
	- points in window: 27,155
	- coverage: 75.9%
- With rotor:
	- windows: 711
	- points in window: 27,734
	- coverage: 77.6%
- Delta:
	- windows: -43
	- points: +579
	- coverage: +1.6%

Note:
- Fewer windows with higher coverage suggests many windows become longer/merged when rotor uplift is applied.

## Known Caveat

Current exported feature file `data/MasterSet_features.csv` includes `Course_deg` but does not currently export `course_sin` and `course_cos`. The model currently uses `Course_deg` directly.

## How To Run

Recommended notebook order:

1. `poc/data_preprocessing.ipynb`
2. `poc/interpolation.ipynb`
3. `poc/test_ERA5.ipynb`
4. `eda/feature_engineering.ipynb`
5. `ml/baseline_model.ipynb`

Environment notes:
- Python libraries used include: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`, `scipy`, `openpyxl`
- On macOS, XGBoost may require OpenMP runtime (`libomp`)

## Data and Security Notice

This repository should contain only code, notebooks, and reproducible workflows.

Do not commit:
- proprietary/raw restricted datasets
- credentials or tokens
- private infrastructure paths

Use approved institutional/local storage for restricted inputs.
