# Maritime Performance Analytics

Data-driven analysis of wind-assisted shipping to evaluate voyage performance, operational efficiency, and weather-related sailing opportunities.

## Overview

This project explores how environmental conditions influence vessel performance in shipping operations supported by wind-assisted propulsion technologies. The main objective is to analyze how weather and sea-state conditions affect vessel speed, transit time, and operational weather windows, and to compare baseline and assisted-operating scenarios.

## Project Objective

The project builds a data science workflow that combines vessel movement data with environmental data to model the relationship between operating conditions and voyage performance.

Key questions include:
- How do environmental conditions affect vessel speed and transit time?
- Under which conditions does wind-assisted propulsion improve performance?
- Can assisted propulsion expand operational weather windows?

## Scope

### Core scope
- Combine vessel trajectory data with environmental observations or model outputs
- Model vessel speed and/or segment transit time as a function of operating conditions
- Compare baseline and assisted scenarios
- Evaluate the effect of assisted propulsion on operational weather windows

### Extended scope
- Include additional environmental variables such as currents or tides
- Compute derived features such as relative wind angle
- Add a simplified physics-based or surrogate model for propulsion support
- Estimate uncertainty and validate across voyage groups

## Data

This repository is designed to work with:
- Vessel trajectory data
- Environmental and ocean-condition data
- Optional vessel and propulsion metadata

## Data Handling Notice

This repository does **not** include:
- raw proprietary datasets
- restricted operational records
- confidential vessel information
- private API keys, credentials, or access tokens
- internal data locations or private storage paths

Only code, documentation, and reproducible processing workflows should be stored here. Any restricted datasets should be accessed separately through approved local or institutional storage.

## Methodology

### 1. Data integration
- Align trajectory and environmental data in time and space
- Prepare cleaned analytical datasets for modeling

### 2. Feature engineering
Potential features include:
- wind conditions
- wave conditions
- vessel course and speed
- relative environmental angles
- aggregated segment-level statistics

### 3. Modeling
Example modeling targets:
- vessel speed
- segment transit time
- operational threshold compliance

### 4. Scenario comparison
- **Baseline scenario:** standard vessel operations
- **Assisted scenario:** operations with wind-assisted propulsion support

### 5. Evaluation
Possible evaluation metrics:
- MAE
- RMSE
- transit time error
- optional goodness-of-fit metrics

Validation should be performed using voyage-based or time-aware splits.

## Weather Windows

A weather window is defined here as a time interval in which operating conditions satisfy selected voyage criteria, such as minimum speed or acceptable sea state.

Example comparisons:
- number of valid windows
- average duration of valid windows
- difference between baseline and assisted cases

## Expected Outputs

- reproducible codebase
- processed analytical outputs
- model evaluation results
- figures and visualizations
- summary report and presentation materials

## Proposed Repository Structure

```text
maritime-performance-analytics/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── notebooks/
├── src/
│   ├── data_ingestion/
│   ├── preprocessing/
│   ├── features/
│   ├── modeling/
│   ├── evaluation/
│   └── visualization/
├── reports/
│   ├── figures/
│   └── outputs/
├── configs/
├── requirements.txt
├── .gitignore
└── README.md
