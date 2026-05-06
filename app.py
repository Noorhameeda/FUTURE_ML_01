import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression


st.set_page_config(
    page_title="Sales Forecasting",
    layout="wide",
    initial_sidebar_state="expanded",
)


DEFAULT_DATA = pd.DataFrame(
    {
        "Date": [
            "2023-01-01",
            "2023-02-01",
            "2023-03-01",
            "2023-04-01",
            "2023-05-01",
            "2023-06-01",
            "2023-07-01",
            "2023-08-01",
            "2023-09-01",
            "2023-10-01",
            "2023-11-01",
            "2023-12-01",
        ],
        "Sales": [200, 245, 310, 360, 405, 430, 480, 510, 545, 610, 665, 720],
    }
)


st.markdown(
    """
    <style>
    #MainMenu, footer, header {visibility: hidden;}

    .stApp {
        background: #f6f8fb;
        color: #172033;
    }

    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 2.5rem;
        max-width: 1220px;
    }

    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e4e9f2;
    }

    h1, h2, h3 {
        color: #132238;
        letter-spacing: 0;
    }

    h1 {
        font-size: 2.35rem;
        line-height: 1.1;
        margin-bottom: .35rem;
    }

    h2, h3 {
        font-size: 1.2rem;
    }

    .subtitle {
        color: #62708a;
        font-size: 1rem;
        margin-bottom: 1.3rem;
    }

    .metric-card {
        background: #ffffff;
        border: 1px solid #e4e9f2;
        border-radius: 8px;
        box-shadow: 0 8px 24px rgba(18, 34, 56, 0.06);
        padding: 1rem 1.1rem;
        min-height: 112px;
    }

    .metric-label {
        color: #64748b;
        font-size: .86rem;
        font-weight: 700;
        text-transform: uppercase;
    }

    .metric-value {
        color: #111827;
        font-size: 1.75rem;
        font-weight: 800;
        margin-top: .45rem;
    }

    .metric-note {
        color: #64748b;
        font-size: .9rem;
        margin-top: .25rem;
    }

    .stButton>button, .stDownloadButton>button {
        border-radius: 7px;
        border: 0;
        background: #126c5b;
        color: #ffffff;
        font-weight: 700;
        min-height: 2.7rem;
    }

    .stButton>button:hover, .stDownloadButton>button:hover {
        background: #0f5b4d;
        color: #ffffff;
    }

    div[data-testid="stDataFrame"], div[data-testid="stDataEditor"] {
        border: 1px solid #e4e9f2;
        border-radius: 8px;
        overflow: hidden;
    }

    @media (max-width: 700px) {
        h1 {font-size: 1.85rem;}
        .metric-value {font-size: 1.35rem;}
        .block-container {padding-left: 1rem; padding-right: 1rem;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def clean_sales_data(raw_df):
    df = raw_df.copy()
    df.columns = [str(column).strip() for column in df.columns]

    if "Date" not in df.columns or "Sales" not in df.columns:
        raise ValueError("Your data needs Date and Sales columns.")

    df = df[["Date", "Sales"]]
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")
    df = df.dropna(subset=["Date", "Sales"])
    df = df.sort_values("Date").drop_duplicates(subset=["Date"], keep="last")
    return df.reset_index(drop=True)


def add_time_features(df):
    featured = df.copy()
    featured["Month"] = featured["Date"].dt.month
    featured["Year"] = featured["Date"].dt.year
    featured["TimeIndex"] = range(len(featured))
    return featured


def fit_model(df):
    featured = add_time_features(df)
    X = featured[["TimeIndex", "Month", "Year"]]
    y = featured["Sales"]
    model = LinearRegression()
    model.fit(X, y)
    return model


def build_forecast(df, periods):
    model = fit_model(df)
    featured = add_time_features(df)
    last_date = df["Date"].max()
    future_dates = pd.date_range(
        last_date + pd.offsets.MonthBegin(1), periods=periods, freq="MS"
    )
    future = pd.DataFrame({"Date": future_dates})
    future["Month"] = future["Date"].dt.month
    future["Year"] = future["Date"].dt.year
    future["TimeIndex"] = range(len(featured), len(featured) + periods)
    future["Forecast"] = model.predict(future[["TimeIndex", "Month", "Year"]])

    fitted = model.predict(featured[["TimeIndex", "Month", "Year"]])
    residual_std = float((featured["Sales"] - fitted).std() or 0)
    future["Low"] = future["Forecast"] - residual_std
    future["High"] = future["Forecast"] + residual_std
    return future


def metric_card(label, value, note=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


with st.sidebar:
    st.header("Controls")
    uploaded_file = st.file_uploader("Upload sales CSV", type=["csv"])
    forecast_months = st.slider("Forecast months", min_value=1, max_value=24, value=6)
    show_editor = st.checkbox("Edit data in app", value=True)
    show_band = st.checkbox("Show forecast range", value=True)
    chart_style = st.selectbox("Chart style", ["Line", "Bar + Line"])

    st.caption("CSV format: Date, Sales")
    st.download_button(
        "Download sample CSV",
        DEFAULT_DATA.to_csv(index=False).encode("utf-8"),
        file_name="sample_sales.csv",
        mime="text/csv",
        use_container_width=True,
    )


st.title("Sales Forecasting Dashboard")
st.markdown(
    '<div class="subtitle">Forecast monthly sales, compare momentum, and export the next plan from one clean workspace.</div>',
    unsafe_allow_html=True,
)


try:
    source_df = pd.read_csv(uploaded_file) if uploaded_file else DEFAULT_DATA
    df = clean_sales_data(source_df)
except Exception as exc:
    st.error(f"Could not load the sales data. {exc}")
    st.stop()

if show_editor:
    edited_df = st.data_editor(
        df,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
            "Sales": st.column_config.NumberColumn("Sales", min_value=0, step=10),
        },
    )
    try:
        df = clean_sales_data(edited_df)
    except Exception as exc:
        st.error(f"Please fix the edited table. {exc}")
        st.stop()

if len(df) < 3:
    st.warning("Add at least 3 valid sales rows to create a forecast.")
    st.stop()

forecast = build_forecast(df, forecast_months)
last_sales = float(df["Sales"].iloc[-1])
next_forecast = float(forecast["Forecast"].iloc[0])
total_forecast = float(forecast["Forecast"].sum())
growth = ((next_forecast - last_sales) / last_sales) * 100 if last_sales else 0
best_month = df.loc[df["Sales"].idxmax()]

metric_cols = st.columns(4)
with metric_cols[0]:
    metric_card("Last Actual", f"{last_sales:,.0f}", df["Date"].iloc[-1].strftime("%b %Y"))
with metric_cols[1]:
    metric_card("Next Forecast", f"{next_forecast:,.0f}", f"{growth:+.1f}% vs last actual")
with metric_cols[2]:
    metric_card(f"{forecast_months}-Month Plan", f"{total_forecast:,.0f}", "Projected total sales")
with metric_cols[3]:
    metric_card("Best Month", f"{best_month['Sales']:,.0f}", best_month["Date"].strftime("%b %Y"))


left, right = st.columns([2, 1])

with left:
    st.subheader("Sales Trend")
    fig, ax = plt.subplots(figsize=(11, 5.8))
    fig.patch.set_facecolor("#f6f8fb")
    ax.set_facecolor("#ffffff")

    if chart_style == "Bar + Line":
        ax.bar(df["Date"], df["Sales"], width=20, color="#b6d7d1", label="Actual Sales")
    ax.plot(
        df["Date"],
        df["Sales"],
        marker="o",
        linewidth=2.6,
        color="#126c5b",
        label="Actual Sales",
    )
    ax.plot(
        forecast["Date"],
        forecast["Forecast"],
        marker="o",
        linestyle="--",
        linewidth=2.4,
        color="#cf5c36",
        label="Forecast",
    )

    if show_band:
        ax.fill_between(
            forecast["Date"],
            forecast["Low"],
            forecast["High"],
            color="#cf5c36",
            alpha=0.14,
            label="Forecast range",
        )

    ax.set_xlabel("Month")
    ax.set_ylabel("Sales")
    ax.set_title("Actual Sales and Forecast")
    ax.grid(True, color="#e7ecf3", linewidth=0.8)
    ax.legend(frameon=False)
    for spine in ax.spines.values():
        spine.set_color("#dbe3ee")

    st.pyplot(fig, use_container_width=True)

with right:
    st.subheader("Forecast Table")
    display_forecast = forecast[["Date", "Forecast", "Low", "High"]].copy()
    display_forecast["Date"] = display_forecast["Date"].dt.strftime("%Y-%m-%d")
    display_forecast[["Forecast", "Low", "High"]] = display_forecast[
        ["Forecast", "Low", "High"]
    ].round(2)
    st.dataframe(display_forecast, use_container_width=True, hide_index=True)

    st.download_button(
        "Download forecast CSV",
        display_forecast.to_csv(index=False).encode("utf-8"),
        file_name="sales_forecast.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.subheader("Historical Data")
st.dataframe(df, use_container_width=True, hide_index=True)
