<div align="center">

# ⚡ Panama Electricity Load Forecasting using Explainable Machine Learning

### Day-Ahead National Grid Demand Forecasting, Benchmarked Against the Real Grid Operator's Own Forecast

*Built on real hourly grid data from Panama's national transmission system operator (CND) — from raw data to a benchmark beat against a real published forecast*

![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.x-013243?logo=numpy&logoColor=white)
![Scikit--learn](https://img.shields.io/badge/Scikit--learn-1.x-F7931E?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.x-006400?logo=xgboost&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-4.x-9ACD32?logo=lightgbm&logoColor=white)
![CatBoost](https://img.shields.io/badge/CatBoost-1.x-FFCC00?logo=catboost&logoColor=black)
![Optuna](https://img.shields.io/badge/Optuna-Hyperparameter_Tuning-0078D4)
![Statsmodels](https://img.shields.io/badge/Statsmodels-SARIMAX-8A2BE2)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-FF4B4B)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Live Demo](https://img.shields.io/badge/Live_Demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)

[**🚀 Live Dashboard**](https://panama-electricity-load-forecasting-v57n8npkidx7vzuda6iqbx.streamlit.app/) &nbsp;•&nbsp; [**LinkedIn**](#) &nbsp;•&nbsp; [**Resume**](#) &nbsp;•&nbsp; [**Portfolio**](#)

</div>

---

<p align="center">
  <img src="assets/dashboard_overview.png" alt="Streamlit Dashboard Overview" width="850">
  <br>
  <em>Live dashboard — KPI summary and week-view forecast vs. CND's real published forecast</em>
</p>

---

## Table of Contents

- [Executive Summary](#executive-summary)
- [Business Problem](#business-problem)
- [Dataset Overview](#dataset-overview)
- [Screenshots](#screenshots)
- [Project Objectives](#project-objectives)
- [End-to-End Workflow](#end-to-end-workflow)
- [Folder Structure](#folder-structure)
- [Technology Stack](#technology-stack)
- [Methodology](#methodology)
- [Exploratory Data Analysis Highlights](#exploratory-data-analysis-highlights)
- [Feature Engineering Summary](#feature-engineering-summary)
- [Baseline Models](#baseline-models)
- [Machine Learning Models](#machine-learning-models)
- [Classical Time Series Model](#classical-time-series-model)
- [Model Comparison and Selection](#model-comparison-and-selection)
- [Benchmark: Beating the Real Grid Operator's Forecast](#benchmark-beating-the-real-grid-operators-forecast)
- [Explainability (SHAP)](#explainability-shap)
- [Error Analysis](#error-analysis)
- [Stress Test: Performance Under a Structural Break](#stress-test-performance-under-a-structural-break)
- [Business Recommendations](#business-recommendations)
- [Key Project Results](#key-project-results)
- [Known Limitations](#known-limitations)
- [Future Improvements](#future-improvements)
- [Model Persistence for Deployment](#model-persistence-for-deployment)
- [Installation Guide](#installation-guide)
- [How to Run Locally](#how-to-run-locally)
- [References](#references)
- [License](#license)

---

## Executive Summary

Electricity cannot be stored economically at grid scale — whatever a country consumes in a given hour must be generated in that same hour. Panama's national grid operator (**CND — Centro Nacional de Despacho**) publishes a weekly pre-dispatch schedule built on a load forecast; get that forecast wrong and the cost is asymmetric — under-forecast and the operator buys emergency power at a premium, over-forecast and fuel is burned generating electricity nobody used.

This project builds a day-ahead (24-hour horizon) machine learning forecasting system for Panama's national grid load, using 5.5 years of hourly demand, weather, and calendar data. Five machine learning models, four statistical baselines, and a SARIMAX classical benchmark were trained and evaluated using strict chronological validation and walk-forward `TimeSeriesSplit` — never a random split, which would leak future information into lag features.

The selected model — **LightGBM, tuned with Optuna** — achieved a test MAPE of **3.81%**, a **36.8% improvement** over the strongest naive baseline. More importantly, it was benchmarked against CND's own real, published pre-dispatch forecast for the same period — the forecast Panama's grid actually operates on — and **beat it by 23.9% (MAE)**.

This is not a Kaggle notebook. Every modeling decision — from excluding the single most predictive feature (`lag_1`) because it would leak future information at prediction time, to deliberately holding out the COVID-19 lockdown period and using it as a stress test rather than letting it silently corrupt the test metric, to choosing the cross-validated model over the model that happened to score marginally better on one test split — is documented and defensible under direct technical questioning.

## Business Problem

**Problem statement:** a national grid operator must schedule generation a week in advance, hour by hour, based on a load forecast. Because electricity cannot be stored at scale, forecast error translates directly into operational cost: under-forecasting risks load shedding or emergency power purchases at spot prices; over-forecasting means burning fuel for capacity that goes unused.

**Business goals:**
- Forecast day-ahead hourly national demand accurately enough to support real generation scheduling
- Benchmark against the grid operator's own real forecast, not just a synthetic baseline
- Identify *where* the model is weakest (which hours, which conditions) so operational safety margins can be applied intelligently rather than uniformly
- Quantify how the model behaves under a genuine structural shock, not just under normal conditions

**Why this matters to companies:** short-term load forecasting sits at the center of energy trading, generation scheduling, and capacity planning — a core analytics function for utilities and grid operators, and a standard vertical within the energy & utilities practices of major consulting and analytics firms.

## Dataset Overview

**Source:** [Short-term Electricity Load Forecasting (Panama)](https://www.kaggle.com/datasets/ernestojaguilar/shortterm-electricity-load-forecasting-panama) — Kaggle, sourced from CND operational reports, Panama's Ministry of Education (school calendar), and Earthdata (weather reanalysis).

| Property | Value |
|---|---|
| Frequency | Hourly |
| Raw coverage | January 2015 – June 2020 |
| Modelling period (pre-COVID) | 2015-01-18 → 2020-02-29 |
| Final engineered dataset | **44,856 rows**, 64 engineered features |
| Target variable | `nat_demand` — national grid load (MW) |
| Exogenous features | Weather (3 cities × 4 variables), holiday calendar, school-term flag |
| Also used | CND's own weekly pre-dispatch forecast (for benchmarking, not training) |

The March 2020 COVID-19 lockdown caused a sharp, sustained drop in demand — a genuine structural break. This period was deliberately excluded from training and testing, and instead used as a dedicated stress test (see [Stress Test](#stress-test-performance-under-a-structural-break)).

## Screenshots

### 📓 Notebook Highlights

| | |
|---|---|
| <img src="assets/eda_hourly_profile.png" width="380"><br>*Average demand by hour of day* | <img src="assets/eda_weekly_pattern.png" width="380"><br>*Weekday vs. weekend demand profile* |
| <img src="assets/model_comparison_chart.png" width="380"><br>*Ranked test-set model comparison* | <img src="assets/benchmark_vs_operator.png" width="380"><br>*Model vs. CND's real published forecast* |
| <img src="assets/shap_global_importance.png" width="380"><br>*SHAP global feature importance* | <img src="assets/shap_beeswarm.png" width="380"><br>*SHAP beeswarm — direction and magnitude* |
| <img src="assets/error_by_hour.png" width="380"><br>*Forecast error by hour of day* | <img src="assets/covid_stress_test.png" width="380"><br>*Model behaviour under the COVID-19 structural break* |

### 🖥️ Streamlit Dashboard

| | |
|---|---|
| <img src="assets/dashboard_overview.png" width="380"><br>*KPI summary and week-view chart with the CND-forecast toggle* | <img src="assets/dashboard_hour_explainer.png" width="380"><br>*Live SHAP waterfall explaining a selected hour's forecast* |
| <img src="assets/dashboard_model_card.png" width="380"><br>*Model card and known limitations, expanded* | |

**🔗 Live Demo:** [panama-electricity-load-forecasting-v57n8npkidx7vzuda6iqbx.streamlit.app](https://panama-electricity-load-forecasting-v57n8npkidx7vzuda6iqbx.streamlit.app/)

---

## Project Objectives

1. Build a real, defensible day-ahead load forecasting pipeline — not a toy Kaggle notebook
2. Compare baseline, machine learning, and classical statistical approaches under identical, leakage-safe validation
3. Benchmark against the grid operator's own real forecast, not only a self-constructed baseline
4. Explain every prediction with SHAP, and confirm the model has learned physically sensible relationships
5. Diagnose where and why the model fails — peak hours, holidays, and a genuine regime change
6. Persist the model and its exact feature recipe for future deployment
7. Produce work suitable for technical discussion in Data Analyst / Data Scientist / ML Engineer interviews

## End-to-End Workflow

```
Raw Panama Grid Data (Kaggle)
│
▼
[1] Data Loading, Quality Assessment, COVID Structural-Break Handling
│
▼
[2] Exploratory Data Analysis (hourly/weekly/monthly/holiday/weather patterns)
│
▼
[3] Time Series Analysis (STL decomposition, ADF/KPSS, ACF/PACF)
│
▼
[4] Feature Engineering (lags ≥24h, rolling stats, EMA, cyclical encoding, cooling-degree hours)
│
▼
[5] Chronological Train/Validation/Test Split + Walk-Forward TimeSeriesSplit
│
▼
[6] Baseline Models (Naive, Seasonal Naive, Hourly Mean, Hour×Day-Type Mean)
│
▼
[7] Classical Model (SARIMAX, benchmarked on a recent window)
│
▼
[8] Machine Learning Models (Ridge, Random Forest, XGBoost, LightGBM, CatBoost)
│
▼
[9] Hyperparameter Tuning (Optuna, TPE sampler, walk-forward CV objective)
│
▼
[10] Final Evaluation + Benchmark vs. CND's Real Pre-Dispatch Forecast
│
▼
[11] SHAP Explainability + Error Analysis + COVID Stress Test
│
▼
[12] Business Recommendations + Model Persistence for Deployment
```

## Folder Structure

```
panama-electricity-load-forecasting/
│
├── data/
│   ├── raw/                      # Kaggle download (not tracked in git)
│   └── README.md                 # where to get the data
│
├── notebooks/
│   └── Panama_Electricity_Load_Forecasting.ipynb
│
├── model_artifacts/
│   ├── final_model.joblib
│   ├── feature_config.json
│   ├── model_card.json
│   ├── feature_engineering.py
│   └── warm_start_history.csv
│
├── app/                          # Streamlit dashboard
│   ├── app.py
│   └── requirements.txt
│
├── assets/                       # README images/screenshots
│
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

## Technology Stack

| Category | Tools |
|---|---|
| Data manipulation | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Classical time series | Statsmodels (STL, ADF, KPSS, SARIMAX) |
| Machine learning | Scikit-learn (Ridge, Random Forest), XGBoost, LightGBM, CatBoost |
| Hyperparameter tuning | Optuna (TPE sampler) |
| Explainability | SHAP (TreeExplainer) |
| Model persistence | joblib |
| Deployment | Streamlit |

## Methodology

**Chronological validation only.** A random train/test split would leak the future into the past — and with lag features, it's worse than usual, since a row's features literally contain earlier rows' target values. The dataset is split strictly chronologically into train (through 2018), validation (H1 2019), and test (H2 2019 onward), with `TimeSeriesSplit` walk-forward folds used for every cross-validated decision.

**No look-ahead leakage in features.** This is a day-ahead (24-hour horizon) forecast, so `lag_1` — the single most predictive feature available — is deliberately excluded, because it would not exist at real prediction time. Every lag and rolling window starts at 24 hours, and rolling windows are computed on a series pre-shifted by the full forecast horizon.

**Model selection happens before the test set is touched.** The final model (LightGBM, Optuna-tuned) was selected using cross-validated training performance, then tuned with a 25-trial Optuna search scored under walk-forward CV. The test set is used exactly once, to report — not to select — the final model's performance.

## Exploratory Data Analysis Highlights

- **Daily cycle dominates:** demand swings by several hundred MW between the overnight trough (~04:00) and the afternoon peak (~13:00).
- **Weekends are a different shape, not just a lower level** — commercial/industrial daytime load largely disappears, which is why the model needs hour × day-type interactions, not just separate flat effects.
- **Temperature drives a monotonic cooling response with no heating leg** — Panama is tropical, so air-conditioning load rises with temperature with no cold-weather heating demand to offset it.
- **A genuine structural break in March 2020** (COVID-19 lockdown) causes a sharp, sustained drop in demand — handled explicitly rather than left to silently corrupt the test metric.

## Feature Engineering Summary

| Feature group | Examples | Why it helps |
|---|---|---|
| Calendar | hour, day-of-week, month, is_working_day | Captures daily/weekly activity cycles |
| Cyclical encoding | hour_sin/cos, dayofweek_sin/cos | Restores adjacency (hour 23 → hour 0) |
| Lag features (≥24h) | lag_24, lag_168, lag_336 | Chosen directly from ACF spikes at daily/weekly cycles |
| Rolling statistics | roll_mean/std/min/max, EMA | Recent demand level and volatility, leakage-safe (shifted by the full horizon) |
| Weather | cooling_degree_hours, temp_national | Air-conditioning load response |

64 features in total, all justified by a specific EDA or ACF finding — nothing included speculatively.

## Baseline Models

| Baseline | Logic |
|---|---|
| Naive (t-24h) | Same hour, previous day |
| Seasonal Naive (t-168h) | Same hour, same weekday, previous week |
| Hourly Mean Profile | Historical average by hour, from training data |
| Hour × Day-Type Mean | Historical average by hour and weekday/weekend |

## Machine Learning Models

Ridge Regression, Random Forest, XGBoost, LightGBM, and CatBoost were trained and compared on identical features, first on a single validation split, then on 5-fold walk-forward cross-validation. LightGBM was selected for Optuna tuning based on cross-validated MAE. CatBoost was additionally tested with its features explicitly declared categorical, to isolate the effect of its native ordered target-statistic encoding versus treating it as a plain numeric booster.

## Classical Time Series Model

SARIMAX(1,0,1)(1,1,1,24) was fit as an honest benchmark (not a competitor) on a recent 60-day window, with order chosen directly from ACF/PACF diagnostics rather than a blind grid search — full seasonal ARIMA at period 24 across the full ~35,000-hour training set is computationally prohibitive on standard hardware.

## Model Comparison and Selection

| Model | MAE (MW) | RMSE (MW) | MAPE | R² |
|---|---|---|---|---|
| Random Forest | 47.1 | 61.8 | 3.74% | 0.8935 |
| **LightGBM (Optuna-tuned) — selected** | **48.4** | **63.1** | **3.81%** | **0.8888** |
| LightGBM (default) | 49.1 | 64.2 | 3.87% | 0.8850 |
| XGBoost | 49.3 | 64.6 | 3.88% | 0.8837 |
| Ridge Regression | 50.2 | 66.7 | 4.13% | 0.8761 |
| Seasonal Naive (t-168h) | 74.8 | 103.1 | 6.03% | 0.7034 |
| Naive (t-24h) | 82.8 | 122.1 | 6.51% | 0.5844 |

**Why LightGBM was selected despite Random Forest scoring marginally lower MAE on this specific test window:** model selection was locked in using cross-validated training performance *before* the test set was opened. Random Forest's 2.9% edge here is within ordinary model-selection variance — switching to it after seeing test results would mean selecting a model using the test set, the same leakage this project avoids everywhere else.

## Benchmark: Beating the Real Grid Operator's Forecast

The most important comparison in this project is not against a statistical baseline — it's against **CND's own real, published weekly pre-dispatch forecast**, evaluated on the exact same 5,856 test hours (100% overlap):

| | MAE (MW) | RMSE (MW) | MAPE | R² |
|---|---|---|---|---|
| **This model** | **48.4** | **63.1** | **3.81%** | **0.8888** |
| CND official pre-dispatch forecast | 63.6 | 82.2 | 5.18% | 0.8114 |

**This model beats Panama's actual operational forecast by 23.9% (MAE).**

## Explainability (SHAP)

Top 5 forecast drivers by mean absolute SHAP contribution:

1. `lag_168` (same hour, last week) — 47.97 MW
2. `lag_336` (same hour, 2 weeks ago) — 36.49 MW
3. `lag_24` (same hour, yesterday) — 21.06 MW
4. `is_working_day` — 16.92 MW
5. `time_index` (long-run trend) — 16.39 MW

The model's learned drivers match physical intuition — weekly and daily demand memory dominate, followed by working-day status and long-run growth — confirming it has learned genuine structure rather than spurious correlations.

## Error Analysis

| | n | MAE (MW) | MAPE | Bias |
|---|---|---|---|---|
| Normal hours | 5,270 | 43.9 | 3.61% | −15.4 |
| **Peak hours (top 10%, ≥1,524 MW)** | 586 | **88.7** | **5.56%** | **−88.7 (under-forecast)** |

The model **under-forecasts peak hours** by a meaningful margin — the operationally expensive direction, since under-scheduling at peak risks emergency generation. This is the single most valuable finding for an operational deployment and the clearest next area for improvement (e.g., asymmetric loss weighting toward high-demand hours).

Other findings: holidays are *not* a particular weak point (47 MW MAE vs. 49 MW on normal days); weather sensitivity is approximately +58 MW per +1°C national temperature.

## Stress Test: Performance Under a Structural Break

The COVID-19 lockdown period was deliberately excluded from training and held out as a stress test:

| | MAE (MW) | MAPE | R² |
|---|---|---|---|
| Normal test period | 48.4 | 3.81% | 0.8888 |
| COVID-19 regime | 62.2 | 5.51% | 0.6457 |
| **Degradation** | — | **+44.6%** | — |

The model systematically over-forecasts once lockdown begins, because its lag features carry forward a pre-lockdown demand level that no longer applies. This is a data-distribution problem, not a model-quality problem — the correct operational response is drift monitoring and scheduled retraining, not a more complex architecture.

## Business Recommendations

1. **Deploy for day-ahead dispatch planning**, publishing the stated MAPE (3.81%) alongside every forecast so planners size reserve margin against a known error distribution.
2. **Apply an asymmetric safety margin at peak hours** — the model under-forecasts exactly the hours where under-scheduling is most expensive; a uniform margin over-provisions cheap overnight hours and under-provisions the costly ones.
3. **Treat temperature as a leading operational indicator** — a forecast heatwave converts directly into an expected MW increment (~58 MW/°C) that can be scheduled for in advance.
4. **Implement drift monitoring with a defined retraining trigger** — e.g., rolling 7-day MAPE exceeding twice the normal-conditions level — since the COVID stress test showed no model architecture prevents degradation under a genuine regime change; only detection and retraining do.

## Key Project Results

- **44,856 hours** of real Panamanian national grid data processed and modelled
- **5 ML models, 4 baselines, and a SARIMAX classical benchmark** implemented and rigorously compared under walk-forward validation
- **36.8%** MAPE improvement over the strongest naive baseline
- **23.9%** MAE improvement over CND's own real, currently-used pre-dispatch forecast
- **Zero look-ahead leakage** — every lag and rolling feature respects the 24-hour forecast horizon by construction
- **A documented structural-break stress test** (COVID-19), with a diagnosed cause and a stated operational mitigation
- **Full model persistence** for deployment — model, feature recipe, and warm-start history all saved and round-trip verified

## Known Limitations

Stated plainly, because a project that claims no limitations is not credible:

- Weather is used at its actual value, not a forecast — reported accuracy is mildly optimistic relative to a true live deployment.
- Point forecasts only; no prediction intervals.
- Trained and evaluated on the pre-COVID regime by design; behaviour under structural change is measured, not solved.
- Weather is drawn from three cities only; a finer spatial grid could improve the temperature signal.

## Future Improvements

1. Quantile regression for prediction intervals (P10/P50/P90), a small change with large operational value
2. Multi-horizon evaluation across 1–168 hours, not just 24
3. Automated drift detection and retraining, the single most valuable robustness improvement
4. Asymmetric loss weighting toward peak hours to directly address the under-forecast bias
5. Streamlit dashboard surfacing the day-ahead forecast, error distribution, and SHAP explanation

## Model Persistence for Deployment

The trained model, its exact feature configuration, and a model card documenting known limitations are saved to `model_artifacts/` — along with the feature-engineering code itself (not a hand-copied reimplementation) and a warm-start history window, since the longest feature requires 336 hours of prior data to compute. Save/load round-trip verified to produce bit-identical predictions.

## Installation Guide

```bash
git clone https://github.com/singhrajveervns-gif/panama-electricity-load-forecasting.git
cd panama-electricity-load-forecasting
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## How to Run Locally

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/ernestojaguilar/shortterm-electricity-load-forecasting-panama) into `data/raw/` (see `data/README.md`).
2. Run `notebooks/Panama_Electricity_Load_Forecasting.ipynb` top to bottom in Jupyter or Google Colab.
3. Model artifacts are saved to `model_artifacts/` by the notebook's final section.

## References

- [Short-term Electricity Load Forecasting (Panama) — Kaggle](https://www.kaggle.com/datasets/ernestojaguilar/shortterm-electricity-load-forecasting-panama)
- Hyndman, R.J., & Athanasopoulos, G. — *Forecasting: Principles and Practice*
- Lundberg, S.M., & Lee, S.I. — *A Unified Approach to Interpreting Model Predictions* (SHAP)
- Akiba, T., et al. — *Optuna: A Next-generation Hyperparameter Optimization Framework*

## License

This project is licensed under the MIT License — see [`LICENSE`](LICENSE) for details.

---

## Author

**Rajveer Singh**
M.Sc. Statistics, Banaras Hindu University

GitHub: [@singhrajveervns-gif](https://github.com/singhrajveervns-gif)
