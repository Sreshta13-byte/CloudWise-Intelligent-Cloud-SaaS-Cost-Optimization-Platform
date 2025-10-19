# frontend/app.py
import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="CloudWise — Intelligent Cloud & SaaS Cost Optimization Platform", layout="wide")
st.markdown(
    """
    <h1 style='text-align: center; color: #2E86C1; margin-bottom: 4px;'>CloudWise — Intelligent Cloud & SaaS Cost Optimization Platform</h1>
    <p style='text-align: center; color: #9AA3AD; margin-top:0; margin-bottom:10px;'>
    "Where Cloud Costs Meet Clarity.." — Vision4 
    </p>
    <hr style='opacity:0.2'>
    """,
    unsafe_allow_html=True,
)

# helper - local fallback to ease testing
DEFAULT_DATA_PATH = Path(__file__).parent.parent / "data" / "usage_sample.csv"

uploaded = st.file_uploader("Upload your usage history CSV file", type="csv")

# if the user doesn't upload, try using default dataset (convenience)
use_df = None
if uploaded is not None:
    use_df = pd.read_csv(uploaded, parse_dates=["timestamp"])
else:
    if DEFAULT_DATA_PATH.exists():
        try:
            use_df = pd.read_csv(DEFAULT_DATA_PATH, parse_dates=["timestamp"])
            st.info("No file uploaded — using local default dataset (usage_sample.csv).")
        except Exception:
            use_df = None

if use_df is None:
    st.info("Upload usage_sample.csv to start predictions and bidding, or put the file in the `data/` folder.")
    st.stop()

# prepare dataframe
df = use_df.sort_values("timestamp").reset_index(drop=True)
df["date"] = pd.to_datetime(df["timestamp"])

# Top summary metrics
st.markdown("### Usage Summary")
mcol1, mcol2, mcol3, mcol4 = st.columns(4)
mcol1.metric("Average CPU", f"{df['cpu'].mean():.2f}")
mcol2.metric("Peak CPU", f"{df['cpu'].max():.2f}")
mcol3.metric("Min CPU", f"{df['cpu'].min():.2f}")
mcol4.metric("Data points", f"{len(df)}")

st.markdown("---")

# Tabs for layout separation
tab_usage, tab_market, tab_bids = st.tabs(["Usage", "Marketplace", "Bidding & Savings"])

with tab_usage:
    st.subheader("Historical Resource Usage Trend")
    fig = px.line(
        df,
        x="date",
        y="cpu",
        title="CPU Usage Over Time",
        template="plotly_dark",
        markers=False,
        color_discrete_sequence=["#1E90FF"],
    )
    fig.update_layout(height=520, title_x=0.5, margin=dict(l=40, r=40, t=50, b=30))
    fig.update_xaxes(nticks=10)
    st.plotly_chart(fig, use_container_width=True)

    # show distribution and hourly average
    st.markdown("#### Hourly average by hour of day")
    df["hour"] = df["date"].dt.hour
    hourly = df.groupby("hour")["cpu"].mean().reset_index()
    fig2 = px.bar(hourly, x="hour", y="cpu", template="plotly_dark", labels={"cpu":"Avg CPU", "hour":"Hour of Day"})
    fig2.update_layout(height=300, title_x=0.5)
    st.plotly_chart(fig2, use_container_width=True)

with tab_market:
    st.subheader("Marketplace Offers")
    # call backend /market endpoint
    try:
        offers = requests.get("http://localhost:8000/market", timeout=3).json()
        offers_df = pd.DataFrame(offers)
    except Exception:
        offers_df = pd.DataFrame()  # empty if backend unreachable
        st.error("Could not reach backend /market endpoint. Is the backend running on port 8000?")

    if not offers_df.empty:
        # styled table
        st.dataframe(offers_df, use_container_width=True, height=320)

        # price comparison chart (grouped by provider)
        st.markdown("#### Marketplace Price Comparison")
        fig3 = px.bar(
            offers_df,
            x="offer_id",
            y="price_per_hour",
            color="provider",
            hover_data=["resource_type", "vCPU", "mem_GB"],
            template="plotly_dark",
        )
        fig3.update_layout(height=420, title_x=0.5, margin=dict(l=30, r=30, t=40, b=30))
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("No marketplace offers to display. Start backend or add `marketplace_offers.csv` in data/")

with tab_bids:
    st.subheader("Auto-Bid Simulation & Savings")

    # compute lags
    last = df.iloc[-1]
    lag1 = float(last["cpu"])
    lag24 = float(df.iloc[-24]["cpu"]) if len(df) >= 24 else lag1

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("### Predict Next Hour")
        if st.button("Predict Next Hour Demand"):
            try:
                r = requests.post("http://localhost:8000/predict", json={"lag1": lag1, "lag24": lag24}, timeout=5)
                pred = r.json().get("prediction", None)
                if pred is None:
                    st.error("Backend returned no prediction.")
                else:
                    pred = float(pred)
                    st.success(f"Predicted next hour usage: {pred:.2f} CPU units")
                    est_cost_on_demand = pred * 0.25
                    st.info(f"Estimated cost (on-demand): ${est_cost_on_demand:.2f}")
                    # write values to session_state for later use
                    st.session_state["last_prediction"] = pred
                    st.session_state["est_cost_on_demand"] = est_cost_on_demand
            except Exception as e:
                st.error(f"Prediction request failed: {e}")

        # show last N predictions stored in bidding_history if file exists
        hist_path = Path(__file__).parent.parent / "data" / "bidding_history.csv"
        if hist_path.exists():
            bh = pd.read_csv(hist_path, parse_dates=["timestamp"])
            st.markdown("#### Recent Bidding History")
            st.dataframe(bh.tail(10), use_container_width=True, height=240)

    with col2:
        st.markdown("### Place Auto-Bid")
        offer_id = st.text_input("Offer ID", "offer_1")
        bid_price = st.number_input("Bid price ($/unit)", min_value=0.005, max_value=1.0, value=0.03, step=0.005)
        if st.button("Place Auto-Bid"):
            try:
                resp = requests.post("http://localhost:8000/bid", params={"offer_id": offer_id, "bid_price": float(bid_price)}, timeout=5)
                bid = resp.json()
                if bid.get("accepted"):
                    st.success(f"Bid accepted for {bid['offer_id']} at ${bid['bid_price']}/unit")
                    # compute savings if prediction exists
                    if "last_prediction" in st.session_state:
                        pred = st.session_state["last_prediction"]
                        est_cost_on_demand = st.session_state.get("est_cost_on_demand", pred * 0.25)
                        bid_cost = pred * float(bid["bid_price"])
                        saved = est_cost_on_demand - bid_cost
                        st.info(f"Estimated savings: ${saved:.2f}")
                    # append to local bidding_history.csv
                    try:
                        row = {
                            "timestamp": pd.Timestamp.now(),
                            "predicted_usage": st.session_state.get("last_prediction", ""),
                            "offer_id": bid["offer_id"],
                            "bid_price_per_unit": bid["bid_price"],
                            "accepted": True,
                            "original_cost": st.session_state.get("est_cost_on_demand", ""),
                            "bid_cost": pred * float(bid["bid_price"]),
                            "cost_saved": saved
                        }
                        bh_path = hist_path
                        if bh_path.exists():
                            bh_df = pd.read_csv(bh_path)
                            bh_df = bh_df.append(row, ignore_index=True)
                        else:
                            bh_df = pd.DataFrame([row])
                        bh_df.to_csv(bh_path, index=False)
                    except Exception:
                        # non-fatal
                        pass
                else:
                    st.error(f"Bid rejected for {offer_id}")
            except Exception as e:
                st.error(f"Bid request failed: {e}")

    st.markdown("---")
    # cumulative savings chart if bidding_history exists
    if hist_path.exists():
        bh = pd.read_csv(hist_path, parse_dates=["timestamp"])
        bh["timestamp"] = pd.to_datetime(bh["timestamp"])
        bh_sorted = bh.sort_values("timestamp")
        bh_sorted["cum_savings"] = bh_sorted["cost_saved"].cumsum()
        st.markdown("#### Cumulative Savings Over Time")
        fig_s = px.line(bh_sorted, x="timestamp", y="cum_savings", template="plotly_dark")
        fig_s.update_layout(height=350, title_x=0.5)
        st.plotly_chart(fig_s, use_container_width=True)

st.markdown("<hr style='opacity:0.2'>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color: #9AA3AD;'>Built by Team Vision4 for SuperHack 2025 | Powered by FastAPI & Streamlit</p>", unsafe_allow_html=True)
