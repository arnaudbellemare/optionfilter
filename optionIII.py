import streamlit as st
import ccxt.async_support as ccxt
import asyncio
import logging
import pandas as pd
import numpy as np
from scipy.stats import norm  # Retained for existing calculations (e.g., Varadi MinCorr)
from datetime import datetime

# Configure logging to match your log format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger('optionIII')

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
testnet = st.sidebar.checkbox("Use Deribit TestNet", value=True)  # User toggle for TestNet

# Initialize Deribit exchange
async def initialize_deribit(testnet=False):
    config = {
        'enableRateLimit': True,
        'asyncio_loop': asyncio.get_event_loop(),
    }
    if testnet:
        config['urls'] = {'api': 'https://test.deribit.com/api/v2'}
        logger.info("Initializing Deribit TestNet")
    else:
        logger.info("Initializing Deribit Production")
    
    try:
        exchange = ccxt.deribit(config)
        # Test a simple API call to verify connectivity
        await exchange.fetch_time()
        logger.info("Deribit API connectivity verified")
        
        # Load markets with detailed error handling
        markets = await exchange.load_markets(reload=True)
        if not markets:
            raise ValueError("No markets returned from Deribit")
        
        # Log market count to verify data
        logger.info(f"Loaded {len(markets)} markets from Deribit")
        
        # Basic validation of markets structure
        sample_ticker = next(iter(markets), None)
        if sample_ticker and not isinstance(markets[sample_ticker], dict):
            raise ValueError(f"Invalid market data structure for {sample_ticker}")
        
        return exchange
    except TypeError as e:
        logger.error(f"TypeError during Deribit initialization: {e}", exc_info=True)
        st.error(f"TypeError initializing Deribit: {e}. Check logs.")
        return None
    except ccxt.NetworkError as e:
        logger.error(f"Network error during Deribit initialization: {e}", exc_info=True)
        st.error(f"Network error: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during Deribit initialization: {e}", exc_info=True)
        st.error(f"Failed to connect to Deribit: {e}. Check logs.")
        return None

# Fetch available BTC options tickers for specific expirations
async def get_btc_options_tickers(exchange, expirations=None):
    if not exchange:
        return []
    
    try:
        markets = exchange.markets
        if not markets:
            logger.error("No markets available in exchange")
            return []
        
        # Default to expirations from logs if not specified
        if expirations is None:
            expirations = ['13JUN25', '27JUN25']
            logger.info(f"Using default expirations: {', '.join(expirations)}")
        
        # Fetch current BTC price for strike filtering
        btc_ticker = await exchange.fetch_ticker('BTC/USD:BTC')
        current_price = btc_ticker.get('last', 100000)  # Fallback ~$103,842 from logs
        logger.info(f"Current BTC price: ${current_price:,.2f}")
        
        # Generate strikes within ±20% of spot
        strike_range = range(int(current_price * 0.8), int(current_price * 1.2), 1000)
        
        tickers = []
        for expiry in expirations:
            for strike in strike_range:
                for option_type in ['C', 'P']:
                    ticker = f"BTC-{expiry}-{strike}-{option_type}"
                    if ticker in markets and markets[ticker].get('active', False):
                        tickers.append(ticker)
        
        logger.info(f"Found {len(tickers)} active BTC options tickers")
        return tickers[:100]  # Limit to avoid rate limits
    except Exception as e:
        logger.error(f"Error fetching tickers: {e}", exc_info=True)
        st.error("Failed to fetch tickers.")
        return []

# Ticker validation (relaxed to reduce warnings)
def is_valid_ticker(ticker_data):
    if ticker_data is None:
        return False
    mark_price = ticker_data.get('mark_price', 0)
    if mark_price < 1e-10:
        logger.warning(f"Low mark price: {mark_price}")
        return False
    # Allow zero volume/open interest
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
            logger.warning(f"Invalid ticker for {ticker}: {normalized_data}")
            return None
    except ccxt.RateLimitExceeded:
        logger.warning(f"Rate limit exceeded for {ticker}, retrying after 1s")
        await asyncio.sleep(1)
        return await fetch_deribit_option_ticker(exchange, ticker)
    except ccxt.NetworkError as e:
        logger.error(f"Network error for {ticker}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error fetching {ticker}: {e}")
        return None

# Fetch multiple tickers in batches
async def fetch_deribit_option_tickers(exchange, tickers):
    results = {}
    batch_size = 10  # Reduced for stability
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        tasks = [fetch_deribit_option_ticker(exchange, ticker) for ticker in batch]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        for ticker, result in zip(batch, batch_results):
            if result is not None and not isinstance(result, Exception):
                results[ticker] = result
        await asyncio.sleep(0.2)  # Increased delay for stability
    logger.info(f"Fetched {len(results)} valid tickers")
    return results

# Fetch funding rate
async def fetch_deribit_funding_rate(exchange, symbol='BTC-PERPETUAL'):
    try:
        funding_rate = await exchange.fetch_funding_rate(symbol)
        logger.info(f"Funding rate for {symbol}: {funding_rate.get('fundingRate')}")
        return funding_rate.get('fundingRate', None)
    except ccxt.NotSupported:
        logger.error(f"Funding rate not supported for {symbol}")
        return None
    except ccxt.RateLimitExceeded:
        logger.warning(f"Rate limit exceeded for {symbol}, retrying after 1s")
        await asyncio.sleep(1)
        return await fetch_deribit_funding_rate(exchange, symbol)
    except Exception as e:
        logger.error(f"Error fetching funding rate for {symbol}: {e}")
        return None

# Main async function
async def main():
    # Initialize Deribit
    exchange = await initialize_deribit(testnet=testnet)
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
