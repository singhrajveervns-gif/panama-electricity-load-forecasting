"""
Panama Electricity Load Forecasting — Streamlit Dashboard

Single-page app: KPI summary, a week-view forecast chart with a toggle for
CND's real published forecast, and a live SHAP explanation for any selected
hour. Reads only the small artifacts saved by the training notebook's
persistence section — no raw data, no retraining, no GPU required.
"""
import json
from pathlib import Path

import joblib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

st.set_page_config(
    page_title="Panama Electricity Load Forecasting",
    page_icon="⚡",
    layout="wide",
)

def _find_artifact_dir() -> Path:
    """Locate model_artifacts/ regardless of whether this script is deployed
    at the repo root (app.py) or inside an app/ subfolder (app/app.py) - the
    correct relative depth depends on exactly how the repo was pushed, and
    guessing wrong silently points at a directory one level off. Check the
    plausible candidates instead of hardcoding one assumption."""
    here = Path(__file__).resolve().parent
    for candidate in (here, here.parent, here.parent.parent):
        maybe = candidate / "model_artifacts"
        if maybe.exists():
            return maybe
    # Fall back to the original assumption so the error message below is
    # still informative about where it looked.
    return here.parent / "model_artifacts"


ARTIFACT_DIR = _find_artifact_dir()


# --------------------------------------------------------------------------- loaders
@st.cache_resource
def load_model():
    path = ARTIFACT_DIR / "final_model.joblib"
    if not path.exists():
        st.error(f"Model file not found at `{path}`. Run the notebook's "
                 f"persistence section first.")
        st.stop()
    return joblib.load(path)


@st.cache_resource
def load_explainer(_model):
    return shap.TreeExplainer(_model)


@st.cache_data
def load_json(name):
    with open(ARTIFACT_DIR / name) as f:
        return json.load(f)


@st.cache_data
def load_predictions():
    path = ARTIFACT_DIR / "test_predictions.csv"
    if not path.exists():
        st.error(f"`{path.name}` not found. Run the dashboard-export cell in "
                 f"the notebook first.")
        st.stop()
    return pd.read_csv(path, parse_dates=["datetime"], index_col="datetime")


@st.cache_data
def load_features():
    path = ARTIFACT_DIR / "test_features.csv"
    if not path.exists():
        st.error(f"`{path.name}` not found. Run the dashboard-export cell in "
                 f"the notebook first.")
        st.stop()
    return pd.read_csv(path, parse_dates=["datetime"], index_col="datetime")


model = load_model()
explainer = load_explainer(model)
config = load_json("feature_config.json")
card = load_json("model_card.json")
predictions = load_predictions()
features = load_features()


# --------------------------------------------------------------------------- header
st.title("⚡ Panama Electricity Load Forecasting")
st.caption(
    "Day-ahead national grid demand forecasting — "
    f"**{card['model_name']}**, benchmarked against Panama's real "
    "grid-operator forecast (CND)."
)

# --------------------------------------------------------------------------- KPI row
op = predictions.dropna(subset=["operator_forecast"])
beat_pct = None
if len(op) > 0:
    operator_mae = (op["operator_forecast"] - op["actual"]).abs().mean()
    our_mae_matched = (op["model_forecast"] - op["actual"]).abs().mean()
    beat_pct = 100 * (operator_mae - our_mae_matched) / operator_mae

col1, col2, col3, col4 = st.columns(4)
col1.metric("Test MAPE", f"{card['test_mape_pct']:.2f}%")
col2.metric("Test MAE", f"{card['test_mae_mw']:.1f} MW")
col3.metric("vs. Seasonal-Naive Baseline",
            f"-{card['improvement_vs_seasonal_naive_pct']:.1f}%")
if beat_pct is not None:
    col4.metric("Beats CND's Own Forecast", f"{beat_pct:.1f}%",
                help="MAE improvement over Panama's actual published "
                     "pre-dispatch forecast, on matched hours.")
else:
    col4.metric("Beats CND's Own Forecast", "n/a")

st.divider()

# --------------------------------------------------------------------------- date picker
available_dates = sorted(predictions.index.normalize().unique())

if "date_idx" not in st.session_state:
    st.session_state.date_idx = len(available_dates) // 2

selected_date = st.select_slider(
    "Select a date in the test period",
    options=available_dates,
    value=available_dates[st.session_state.date_idx],
    format_func=lambda d: d.strftime("%Y-%m-%d (%a)"),
)

show_operator = st.checkbox("Show CND's official forecast", value=True)

# --------------------------------------------------------------------------- week chart
week_start = selected_date - pd.Timedelta(days=3)
week_end = selected_date + pd.Timedelta(days=4) - pd.Timedelta(hours=1)
week = predictions.loc[week_start:week_end]

st.subheader(f"Forecast vs. actual — week centred on {selected_date:%b %d, %Y}")

fig, ax = plt.subplots(figsize=(12, 4.2))
ax.plot(week.index, week["actual"], color="black", linewidth=2, label="Actual")
ax.plot(week.index, week["model_forecast"], color="seagreen", linewidth=1.8,
        label=card["model_name"])
if show_operator and week["operator_forecast"].notna().any():
    ax.plot(week.index, week["operator_forecast"], color="crimson",
            linewidth=1.6, linestyle="--", label="CND official forecast")
ax.axvspan(selected_date, selected_date + pd.Timedelta(days=1),
          color="steelblue", alpha=0.08, label="Selected day")
ax.set_ylabel("MW")
ax.legend(loc="upper right", fontsize=9)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d"))
fig.tight_layout()
st.pyplot(fig)

st.divider()

# --------------------------------------------------------------------------- hour explainer
st.subheader("Explain a specific hour")

day_hours = predictions.loc[
    selected_date: selected_date + pd.Timedelta(days=1) - pd.Timedelta(hours=1)
].index

if len(day_hours) == 0:
    st.info("No data available for the selected day.")
else:
    selected_hour = st.select_slider(
        "Hour of day", options=list(day_hours),
        value=day_hours[len(day_hours) // 2],
        format_func=lambda t: t.strftime("%H:%M"),
    )

    actual_val = predictions.loc[selected_hour, "actual"]
    pred_val = predictions.loc[selected_hour, "model_forecast"]
    op_val = predictions.loc[selected_hour, "operator_forecast"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Actual demand", f"{actual_val:,.0f} MW")
    c2.metric("Our forecast", f"{pred_val:,.0f} MW",
              delta=f"{pred_val - actual_val:+.0f} MW")
    if pd.notna(op_val):
        c3.metric("CND forecast", f"{op_val:,.0f} MW",
                  delta=f"{op_val - actual_val:+.0f} MW")
    else:
        c3.metric("CND forecast", "n/a")

    if selected_hour in features.index:
        row = features.loc[[selected_hour]]
        shap_vals = explainer.shap_values(row)

        st.write("**Why the model predicted this value:**")
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_vals[0],
                base_values=explainer.expected_value,
                data=row.iloc[0],
                feature_names=list(row.columns),
            ),
            max_display=10,
            show=False,
        )
        fig2.tight_layout()
        st.pyplot(fig2)
    else:
        st.warning("No feature row available for this hour.")

st.divider()

# --------------------------------------------------------------------------- model card
with st.expander("Model card & known limitations"):
    st.markdown(f"**Model:** {card['model_name']}")
    st.markdown(f"**Training period:** {card['training_period']}")
    st.markdown(f"**Test period:** {card['test_period']}")
    st.markdown(
        f"**Test metrics:** MAE {card['test_mae_mw']} MW · "
        f"RMSE {card['test_rmse_mw']} MW · MAPE {card['test_mape_pct']}%"
    )
    st.markdown("**Known limitations:**")
    for item in card["known_limitations"]:
        st.markdown(f"- {item}")

st.caption(
    "Built by Rajveer Singh · M.Sc. Statistics, Banaras Hindu University · "
    "[GitHub](https://github.com/singhrajveervns-gif)"
)
