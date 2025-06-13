import streamlit as st
import ccxt.async_support as ccxt
import asyncio
import logging
import pandas as pd
import numpy as np
from scipy.stats import norm  # Retained for existing calculations
from datetime import datetime

# Configure logging (matches your log format)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)

# Streamlit page setup
st.set_page_config(page_title="Crypto Options Analysis", layout="wide")
st.title("Crypto Options Analysis with Deribit")

# Sidebar for user inputs
st.sidebar.header("Configuration")
available_cash = st.sidebar.number_input(
    "Available Cash (USD)", min_value=0.0, value=25000.0, step=1000.0, format="%.2f"
)
calculation_end_date = st.sidebar.date_input(
    "Calculation End Date", value=datetime.now()
)

# Initialize Deribit exchange
async def initialize_deribit(testnet=False):
    exchange = ccxt.deribit({
        'enableRateLimit': True,  # Built-in rate limiting
        'asyncio_loop': asyncio.get_event_loop(),
        'urls': {'api': 'https://test.deribit.com/api/v2'} if testnet else {},
    })
    try:
        await exchange.load_markets()
        logging.info("Deribit markets loaded successfully")
        return exchange
    except Exception as e:
        logging.error(f"Failed to initialize Deribit: {e}")
        st.error("Failed to connect to Deribit. Check logs.")
        return None

# Fetch available BTC options tickers for specific expirations
async def get_btc_options_tickers(exchange, expirations=['13JUN25', '27JUN25']):
    if not exchange:
        return []
    
    tickers = []
    try:
        markets = exchange.markets
        current_price = (await exchange.fetch_ticker('BTC/USD:BTC')).get('last', 100000)  # Approx BTC price (~$103,842 from logs)
        strike_range = range(int(current_price * 0.8), int(current_price * 1.2), 500)  # ±20% of spot, 500 increment
        
        for expiry in expirations:
            for strike in strike_range:
                for option_type in ['C', 'P']:
                    ticker = f"BTC-{expiry}-{strike}-{option_type}"
                    if ticker in markets and markets[ticker].get('active', False):
                        tickers.append(ticker)
        
        logging.info(f"Found {len(tickers)} BTC options tickers for expirations: {', '.join(expirations)}")
        return tickers[:100]  # Limit to 100 to avoid rate limit issues
    except Exception as e:
        logging.error(f"Error fetching tickers: {e}")
        st.error("Failed to fetch tickers.")
        return []

# Ticker validation (relaxed to reduce warnings)
def is_valid_ticker(ticker_data):
    if ticker_data is None:
        return False
    mark_price = ticker_data.get('mark_price', 0)
    if mark_price < 1e-10:  # Allow small prices for deep OTM options
        logging.warning(f"Low mark price: {mark_price}")
        return False
    # Allow zero volume/open interest for now
    return True

# Fetch single ticker data
async def fetch_deribit_option_ticker(exchange, ticker):
    try:
        ticker_data = await exchange.fetch_ticker(ticker)
        normalized_data = {
            'mark_price': ticker_data.get('last', 0.0),
            'mark_timestamp': ticker_data.get('timestamp', 0) / 1000,
            'iv': ticker_data.get('info', {}).get('implied_volatility', 0.0),
            'delta': ticker_data.get('info', {}).get('delta', 0.0),
            'volume_24h': ticker_data.get('info', {}).get('volume', 0.0),
            'value_24h': ticker_data.get('info', {}).get('volume_usd', 0.0),
            'index': ticker_data.get('info', {}).get('underlying_price', 0.0),
            'forward': ticker_data.get('info', {}).get('forward_price', 0.0),
            'open_interest': ticker_data.get('info', {}).get('open_interest', 0.0),
        }
        if is_valid_ticker(normalized_data):
            return normalized_data
        else:
            logging.warning(f"Invalid ticker for {ticker}: {normalized_data}")
            return None
    except ccxt.RateLimitExceeded:
        logging.warning(f"Rate limit exceeded for {ticker}, retrying after 1s")
        await asyncio.sleep(1)
        return await fetch_deribit_option_ticker(exchange, ticker)
    except ccxt.NetworkError as e:
        logging.error(f"Network error for {ticker}: {e}")
        return None
    except Exception as e:
        logging.error(f"Error fetching {ticker}: {e}")
        return None

# Fetch multiple tickers in batches
async def fetch_deribit_option_tickers(exchange, tickers):
    results = {}
    batch_size = 20  # Deribit: ~20 req/sec
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        tasks = [fetch_deribit_option_ticker(exchange, ticker) for ticker in batch]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        for ticker, result in zip(batch, batch_results):
            if result is not None and not isinstance(result, Exception):
                results[ticker] = result
        await asyncio.sleep(0.1)  # 100ms delay between batches
    return results

# Fetch funding rate
async def fetch_deribit_funding_rate(exchange, symbol='BTC-PERPETUAL'):
    try:
        funding_rate = await exchange.fetch_funding_rate(symbol)
        logging.info(f"Funding rate for {symbol}: {funding_rate.get('fundingRate')}")
        return funding_rate.get('fundingRate', None)
    except ccxt.NotSupported:
        logging.error(f"Funding rate not supported for {symbol}")
        return None
    except ccxt.RateLimitExceeded:
        logging.warning(f"Rate limit exceeded for {symbol}, retrying after 1s")
        await asyncio.sleep(1)
        return await fetch_deribit_funding_rate(exchange, symbol)
    except Exception as e:
        logging.error(f"Error fetching funding rate for {symbol}: {e}")
        return None

# Main async function
async def main():
    # Initialize Deribit (use testnet for development)
    exchange = await initialize_deribit(testnet=True)  # Switch to False for production
    if not exchange:
        return
    
    # Fetch tickers dynamically
    expirations = ['13JUN25', '27JUN25']  # From logs
    tickers = await get_btc_options_tickers(exchange, expirations)
    if not tickers:
        st.error("No tickers found.")
        await exchange.close()
        return
    
    # Fetch options data
    with st.spinner("Fetching options data..."):
        options_data = await fetch_deribit_option_tickers(exchange, tickers)
    
    # Display options data
    if options_data:
        df = pd.DataFrame.from_dict(options_data, orient='index')
        df = df[['mark_price', 'iv', 'delta', 'volume_24h', 'open_interest']].round(4)
        st.subheader("Bitcoin Options Data")
        st.dataframe(df, use_container_width=True)
        st.write(f"**Fetched {len(options_data)} valid tickers**")
    else:
        st.warning("No valid options data retrieved.")
    
    # Fetch funding rate
    with st.spinner("Fetching funding rate..."):
        funding_rate = await fetch_deribit_funding_rate(exchange)
        st.subheader("Funding Rate")
        st.write(f"BTC-PERPETUAL Funding Rate: {funding_rate if funding_rate is not None else 'N/A'}")
    
    # Placeholder for existing logic (e.g., Varadi MinCorr)
    st.subheader("Portfolio Analysis")
    st.write("Add your Varadi MinCorr or other calculations here, using `norm.cdf` if needed.")
    
    await exchange.close()

# Run the app
if __name__ == "__main__":
    asyncio.run(main())
