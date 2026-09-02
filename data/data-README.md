# Data

Raw data is not committed to this repository (see `.gitignore`). Download it yourself:

**Source:** [Short-term Electricity Load Forecasting (Panama) — Kaggle](https://www.kaggle.com/datasets/ernestojaguilar/shortterm-electricity-load-forecasting-panama)

**Files needed**, placed directly in this `data/raw/` folder:

- `continuous dataset.csv` — the main hourly dataset (load, weather, calendar)
- `weekly pre-dispatch forecast.csv` — CND's own official forecast, used in the notebook to benchmark the model against the real forecast Panama's grid operator actually uses

After downloading:

```
data/
└── raw/
    ├── continuous dataset.csv
    └── weekly pre-dispatch forecast.csv
```

Then run `notebooks/Panama_Electricity_Load_Forecasting.ipynb` top to bottom.
