import logging
import requests
import re
import time
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timezone

# --- Thalex API Configuration ---
THALEX_BASE_URL = "https://thalex.com/api/v2"
REQUEST_TIMEOUT = 15

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

# --- API Functions ---
def get_all_instruments():
    url = f"{THALEX_BASE_URL}/public/instruments"
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json().get("result", [])
    except requests.RequestException as e:
        logging.error(f"API Error (get_all_instruments): {e}")
        st.error(f"Failed to fetch instruments: {e}")
        return []

def get_instrument_ticker(instrument_name: str):
    url = f"{THALEX_BASE_URL}/public/ticker"
    params = {"instrument_name": instrument_name}
    try:
        response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json().get("result", {})
    except requests.RequestException as e:
        logging.error(f"API Error (get_instrument_ticker): {e}")
        return None

# --- Plotting Function ---
def plot_market_directionality(sel_coin: str, sel_exp: str, all_instruments: list) -> None:
    st.header(f"Market Directionality and Intent: {sel_coin}-{sel_exp}")
    with st.spinner("Fetching market directionality data..."):
        # Filter options for the selected expiry
        date_pattern = re.compile(rf"^{re.escape(sel_coin.upper())}-{sel_exp}-(.*)")
        option_instruments = [instr for instr in all_instruments if instr.get('type') == 'option' and date_pattern.match(instr.get('instrument_name', ''))]

        if not option_instruments:
            st.warning(f"No option instruments found for {sel_coin}-{sel_exp}.")
            return

        # Fetch ticker data for all options
        option_data = []
        for instr in option_instruments:
            ticker = get_instrument_ticker(instr['instrument_name'])
            if ticker and all(k in ticker for k in ['mark_price', 'bid_price', 'ask_price', 'iv', 'open_interest']):
                option_data.append({
                    'instrument': instr['instrument_name'],
                    'type': instr['instrument_name'].split('-')[3],
                    'mark_price': float(ticker['mark_price']),
                    'bid_price': float(ticker['bid_price']),
                    'ask_price': float(ticker['ask_price']),
                    'iv': float(ticker['iv']),
                    'open_interest': float(ticker['open_interest'])
                })
                time.sleep(0.02)  # Respect API rate limits

        if not option_data:
            st.warning("No valid ticker data retrieved for options.")
            return

        df_options = pd.DataFrame(option_data)
        if df_options.empty:
            st.warning("Option data DataFrame is empty.")
            return

        # Get spot price from perpetual
        perpetual_name = f"{sel_coin}-PERPETUAL"
        perp_ticker = get_instrument_ticker(perpetual_name)
        spot_price = float(perp_ticker['mark_price']) if perp_ticker and 'mark_price' in perp_ticker else np.nan
        if pd.isna(spot_price):
            st.warning(f"Spot price unavailable for {perpetual_name}.")
            spot_price = 0

        # Aggregate notional over time
        current_time = datetime.now(timezone.utc)
        df_options['timestamp'] = current_time
        df_options['notional_ask'] = df_options['ask_price'] * df_options['open_interest']
        df_options['notional_bid'] = df_options['bid_price'] * df_options['open_interest']

        # Filter puts and calls
        df_puts = df_options[df_options['type'] == 'P'].copy()
        df_calls = df_options[df_options['type'] == 'C'].copy()

        if df_puts.empty:
            st.warning("No put option data available.")
            return

        # Group by timestamp
        df_puts_ask = df_puts.groupby('timestamp').agg({'notional_ask': 'sum'}).reset_index()
        df_puts_bid = df_puts.groupby('timestamp').agg({'notional_bid': 'sum'}).reset_index()
        df_calls_ask = df_calls.groupby('timestamp').agg({'notional_ask': 'sum'}).reset_index() if not df_calls.empty else pd.DataFrame({'timestamp': [current_time], 'notional_ask': [0]})

        # Calculate metrics
        avg_iv = df_puts['iv'].mean() * 100
        total_notional_ask = df_puts['notional_ask'].sum()
        total_notional_bid = df_puts['notional_bid'].sum()
        call_notional_ask = df_calls_ask['notional_ask'].sum() if not df_calls.empty else 0
        put_notional_ask = df_puts['notional_ask'].sum()
        bullish_bearish_ratio = (call_notional_ask / put_notional_ask) if put_notional_ask > 0 else float('inf')

        # Create plots
        fig_ask = go.Figure()
        fig_ask.add_trace(go.Scatter(
            x=df_puts_ask['timestamp'], y=df_puts_ask['notional_ask'],
            mode='markers', marker=dict(size=10, color='red', opacity=0.6),
            name='Puts @ Ask Notional'
        ))
        fig_ask.add_trace(go.Scatter(
            x=[current_time], y=[spot_price],
            mode='lines', line=dict(color='yellow', width=2),
            name='Spot Price'
        ))
        fig_ask.update_layout(
            title=f"Notional vs Time: Puts @ Ask",
            xaxis_title="Time",
            yaxis_title="Spot / Notional ($)",
            template="plotly_dark",
            height=300
        )
        fig_ask.add_annotation(
            x=current_time, y=total_notional_ask,
            text=f"${total_notional_ask:,.0f}",
            showarrow=True, arrowhead=1, ax=20, ay=-30, bgcolor="red", font=dict(color="white")
        )
        fig_ask.add_annotation(
            x=current_time, y=avg_iv,
            text=f"Avg IV {avg_iv:.2f}%",
            showarrow=True, arrowhead=1, ax=-20, ay=-30, bgcolor="gray", font=dict(color="white")
        )

        fig_bid = go.Figure()
        fig_bid.add_trace(go.Scatter(
            x=df_puts_bid['timestamp'], y=df_puts_bid['notional_bid'],
            mode='markers', marker=dict(size=10, color='blue', opacity=0.6),
            name='Puts @ Bid Notional'
        ))
        fig_bid.add_trace(go.Scatter(
            x=[current_time], y=[spot_price],
            mode='lines', line=dict(color='yellow', width=2),
            name='Spot Price'
        ))
        fig_bid.update_layout(
            title=f"Notional vs Time: Puts @ Bid",
            xaxis_title="Time",
            yaxis_title="Spot / Notional ($)",
            template="plotly_dark",
            height=300
        )
        fig_bid.add_annotation(
            x=current_time, y=total_notional_bid,
            text=f"${total_notional_bid:,.0f}",
            showarrow=True, arrowhead=1, ax=20, ay=-30, bgcolor="blue", font=dict(color="white")
        )
        fig_bid.add_annotation(
            x=current_time, y=avg_iv,
            text=f"Avg IV {avg_iv:.2f}%",
            showarrow=True, arrowhead=1, ax=-20, ay=-30, bgcolor="gray", font=dict(color="white")
        )

        # Display plots and metrics
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_ask, use_container_width=True)
        with col2:
            st.plotly_chart(fig_bid, use_container_width=True)

        st.metric("Total Notional @ Ask", f"${total_notional_ask:,.0f}")
        st.metric("Total Notional @ Bid", f"${total_notional_bid:,.0f}")
        st.metric("Average Implied Volatility", f"{avg_iv:.2f}%")
        st.metric("Bullish:Bearish Ratio", f"{bullish_bearish_ratio:.2f}" if bullish_bearish_ratio != float('inf') else "N/A")

# --- Main Streamlit App ---
st.set_page_config(layout="wide", page_title="CNO Options Analyzer")
st.title("CNO Options Analyzer")

# Fetch instruments
all_instruments = get_all_instruments()
if not all_instruments:
    st.error("Failed to load instruments. Please try again later.")
    st.stop()

# Derive coins and expiries dynamically
coins = sorted(set(
    instr['instrument_name'].split('-')[0]
    for instr in all_instruments
    if instr.get('type') == 'option'
))
expiries = sorted(set(
    instr['instrument_name'].split('-')[1]
    for instr in all_instruments
    if instr.get('type') == 'option' and '-' in instr['instrument_name']
))

# Sidebar for user input
st.sidebar.header("Select Options")
sel_coin = st.sidebar.selectbox("Select Coin", coins if coins else ["No coins available"])
sel_exp = st.sidebar.selectbox("Select Expiry", expiries if expiries else ["No expiries available"])

# Plot data if valid selections
if coins and expiries and sel_coin != "No coins available" and sel_exp != "No expiries available":
    plot_market_directionality(sel_coin, sel_exp, all_instruments)
else:
    st.warning("No valid coins or expiries available. Please check API data.")
