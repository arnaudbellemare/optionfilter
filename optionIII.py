import logging
import requests
import re
import time
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import ccxt.async_support as ccxt
from datetime import datetime, timezone
from scipy.stats import norm
import asyncio

# --- Thalex API Configuration ---
THALEX_BASE_URL = "https://thalex.com/api/v2"
REQUEST_TIMEOUT = 15

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

# --- Thalex API Functions ---
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
        result = response.json().get("result", {})
        if not result:
            logging.warning(f"No ticker data for {instrument_name}")
        return result
    except requests.RequestException as e:
        logging.error(f"API Error for {instrument_name}: {e}")
        return None

# --- CCXT Kraken Functions ---
async def fetch_historical_ohlcv(exchange, symbol, timeframe='1d', limit=100):
    try:
        ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        logging.error(f"Kraken OHLCV error for {symbol}: {e}")
        st.error(f"Failed to fetch historical prices: {e}")
        return pd.DataFrame()

async def fetch_funding_rate(exchange, symbol):
    try:
        funding = await exchange.fetch_funding_rate(symbol)
        return funding.get('fundingRate', None)
    except Exception as e:
        logging.error(f"Kraken funding rate error for {symbol}: {e}")
        return None

# --- Greeks Calculation (Simplified Black-Scholes) ---
def calculate_greeks(spot, strike, iv, expiry_date, option_type, risk_free_rate=0.01):
    try:
        T = (pd.to_datetime(expiry_date) - datetime.now(timezone.utc)).days / 365.0
        if T <= 0:
            return None, None
        S = spot
        K = strike
        r = risk_free_rate
        sigma = iv

        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == 'C':
            delta = norm.cdf(d1)
            vega = S * norm.pdf(d1) * np.sqrt(T) / 100  # Vega per 1% change in IV
        else:  # Put
            delta = norm.cdf(d1) - 1
            vega = S * norm.pdf(d1) * np.sqrt(T) / 100

        return delta, vega
    except Exception as e:
        logging.error(f"Greeks calculation error: {e}")
        return None, None

# --- Modified Plotting Function ---
async def plot_market_directionality(sel_coin: str, sel_exp: str, all_instruments: list, kraken_exchange):
    st.header(f"Market Directionality and Intent: {sel_coin}-{sel_exp}")

    # Initialize Kraken symbol for perpetual
    kraken_perp_symbol = f"PI_{sel_coin}USD"  # E.g., PI_XBTUSD for BTC
    kraken_spot_symbol = f"{sel_coin}/USD"    # E.g., BTC/USD

    # Fetch historical OHLCV from Kraken
    with st.spinner("Fetching historical prices from Kraken..."):
        ohlcv_df = await fetch_historical_ohlcv(kraken_exchange, kraken_spot_symbol, timeframe='1d', limit=30)
        if not ohlcv_df.empty:
            st.subheader("Historical Prices (Kraken)")
            fig_price = go.Figure()
            fig_price.add_trace(go.Candlestick(
                x=ohlcv_df['timestamp'],
                open=ohlcv_df['open'],
                high=ohlcv_df['high'],
                low=ohlcv_df['low'],
                close=ohlcv_df['close'],
                name='OHLC'
            ))
            fig_price.update_layout(title=f"{sel_coin}/USD Historical Prices", xaxis_title="Date", yaxis_title="Price (USD)", template="plotly_dark")
            st.plotly_chart(fig_price, use_container_width=True)

    # Fetch funding rate for perpetual
    with st.spinner("Fetching funding rate from Kraken..."):
        funding_rate = await fetch_funding_rate(kraken_exchange, kraken_perp_symbol)
        if funding_rate is not None:
            st.metric("Current Funding Rate (Kraken Perpetual)", f"{funding_rate * 100:.4f}%")

    # Thalex options data
    with st.spinner("Fetching market directionality data from Thalex..."):
        date_pattern = re.compile(rf"^{re.escape(sel_coin.upper())}-{sel_exp}-(.*)")
        option_instruments = [instr for instr in all_instruments if instr.get('type') == 'option' and date_pattern.match(instr.get('instrument_name', ''))]
        logging.info(f"Found {len(option_instruments)} option instruments for {sel_coin}-{sel_exp}")

        if not option_instruments:
            st.warning(f"No option instruments found for {sel_coin}-{sel_exp}.")
            return

        option_data = []
        for instr in option_instruments:
            ticker = get_instrument_ticker(instr['instrument_name'])
            if ticker is None or not all(k in ticker for k in ['mark_price', 'bid_price', 'ask_price', 'iv', 'open_interest']):
                logging.warning(f"Invalid ticker for {instr['instrument_name']}: {ticker}")
                continue
            try:
                strike = float(instr['instrument_name'].split('-')[2])
                option_data.append({
                    'instrument': instr['instrument_name'],
                    'type': instr['instrument_name'].split('-')[3],
                    'strike': strike,
                    'mark_price': float(ticker['mark_price']),
                    'bid_price': float(ticker['bid_price']),
                    'ask_price': float(ticker['ask_price']),
                    'iv': float(ticker['iv']),
                    'open_interest': float(ticker['open_interest'])
                })
            except (ValueError, TypeError) as e:
                logging.error(f"Data conversion error for {instr['instrument_name']}: {e}")
                continue
            time.sleep(0.05)

        if not option_data:
            st.warning(f"No valid ticker data retrieved for {sel_coin}-{sel_exp}. Check API or instrument availability.")
            return

        df_options = pd.DataFrame(option_data)
        if df_options.empty:
            st.warning("Option data DataFrame is empty.")
            return

        # Get spot price from Kraken (more reliable than Thalex perpetual)
        spot_price = ohlcv_df['close'].iloc[-1] if not ohlcv_df.empty else np.nan
        if pd.isna(spot_price):
            st.warning("Spot price unavailable.")
            spot_price = 0

        # Calculate Greeks
        df_options['delta'] = None
        df_options['vega'] = None
        for idx, row in df_options.iterrows():
            delta, vega = calculate_greeks(spot_price, row['strike'], row['iv'], sel_exp, row['type'])
            df_options.at[idx, 'delta'] = delta
            df_options.at[idx, 'vega'] = vega

        # Aggregate notional
        current_time = datetime.now(timezone.utc)
        df_options['timestamp'] = current_time
        df_options['notional_ask'] = df_options['ask_price'] * df_options['open_interest']
        df_options['notional_bid'] = df_options['bid_price'] * df_options['open_interest']

        df_puts = df_options[df_options['type'] == 'P'].copy()
        df_calls = df_options[df_options['type'] == 'C'].copy()

        if df_puts.empty:
            st.warning("No put option data available.")
            return

        df_puts_ask = df_puts.groupby('timestamp').agg({'notional_ask': 'sum'}).reset_index()
        df_puts_bid = df_puts.groupby('timestamp').agg({'notional_bid': 'sum'}).reset_index()
        df_calls_ask = df_calls.groupby('timestamp').agg({'notional_ask': 'sum'}).reset_index() if not df_calls.empty else pd.DataFrame({'timestamp': [current_time], 'notional_ask': [0]})

        avg_iv = df_puts['iv'].mean() * 100
        total_notional_ask = df_puts['notional_ask'].sum()
        total_notional_bid = df_puts['notional_bid'].sum()
        call_notional_ask = df_calls_ask['notional_ask'].sum() if not df_calls.empty else 0
        put_notional_ask = df_puts['notional_ask'].sum()
        bullish_bearish_ratio = (call_notional_ask / put_notional_ask) if put_notional_ask > 0 else float('inf')

        # Plot notional
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

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_ask, use_container_width=True)
        with col2:
            st.plotly_chart(fig_bid, use_container_width=True)

        # Display metrics
        st.metric("Total Notional @ Ask", f"${total_notional_ask:,.0f}")
        st.metric("Total Notional @ Bid", f"${total_notional_bid:,.0f}")
        st.metric("Average Implied Volatility", f"{avg_iv:.2f}%")
        st.metric("Bullish:Bearish Ratio", f"{bullish_bearish_ratio:.2f}" if bullish_bearish_ratio != float('inf') else "N/A")

        # Display Greeks
        st.subheader("Option Greeks (Thalex)")
        st.dataframe(df_options[['instrument', 'type', 'strike', 'delta', 'vega']])

# --- Main Streamlit App ---
async def main():
    st.set_page_config(layout="wide", page_title="CNO Options Analyzer")
    st.title("CNO Options Analyzer")

    # Initialize Kraken exchange
    kraken_exchange = ccxt.kraken({
        'enableRateLimit': True,
        'asyncio_loop': asyncio.get_event_loop()
    })

    # Fetch Thalex instruments
    all_instruments = get_all_instruments()
    if not all_instruments:
        st.error("Failed to load Thalex instruments.")
        st.stop()

    # Derive coins and expiries
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

    # Sidebar
    st.sidebar.header("Select Options")
    sel_coin = st.sidebar.selectbox("Select Coin", coins if coins else ["No coins available"])
    sel_exp = st.sidebar.selectbox("Select Expiry", expiries if expiries else ["No expiries available"])

    if coins and expiries and sel_coin != "No coins available" and sel_exp != "No expiries available":
        await plot_market_directionality(sel_coin, sel_exp, all_instruments, kraken_exchange)
    else:
        st.warning("No valid coins or expiries available.")

    # Clean up
    await kraken_exchange.close()

if __name__ == "__main__":
    asyncio.run(main())
