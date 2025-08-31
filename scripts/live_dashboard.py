import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import asyncio
from datetime import datetime, timedelta
import json
import aiohttp
from typing import Dict, Optional

from src.data.live_data import LiveDataFetcher
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Initialize session state
if 'data_fetcher' not in st.session_state:
    symbols = ["RELIANCE.NS", "INFY.NS", "TCS.NS", "HDFCBANK.NS"]  # Add your symbols
    st.session_state.data_fetcher = LiveDataFetcher(symbols)

if 'predictions' not in st.session_state:
    st.session_state.predictions = {}

async def get_prediction(symbol: str, horizon: int = 1) -> Optional[float]:
    """Get volatility prediction from API"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                'http://localhost:8013/predict',
                json={'symbol': symbol.replace('.NS', ''), 'horizon': horizon}
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get('forecast')
    except Exception as e:
        logger.error(f"Failed to get prediction: {e}")
    return None

def create_candlestick_chart(df: pd.DataFrame, predictions: Dict[str, float], symbol: str):
    """Create candlestick chart with predictions"""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                       vertical_spacing=0.03, subplot_titles=(f'{symbol} Price', 'Volatility'),
                       row_heights=[0.7, 0.3])

    # Candlestick chart
    fig.add_trace(
        go.Candlestick(x=df.index,
                       open=df['Open'],
                       high=df['High'],
                       low=df['Low'],
                       close=df['Close'],
                       name='OHLC'),
        row=1, col=1
    )

    # Add predictions
    if symbol in predictions:
        pred_val = predictions[symbol]
        fig.add_trace(
            go.Scatter(x=[df.index[-1]], y=[pred_val],
                      mode='markers+text',
                      name='Predicted Volatility',
                      text=[f'{pred_val:.2f}%'],
                      textposition='top center',
                      marker=dict(size=10, symbol='star')),
            row=2, col=1
        )

    # Historical volatility (20-day rolling)
    returns = df['Close'].pct_change()
    vol = returns.rolling(window=20).std() * (252 ** 0.5) * 100  # Annualized
    fig.add_trace(
        go.Scatter(x=df.index, y=vol,
                  mode='lines',
                  name='Historical Vol',
                  line=dict(width=1)),
        row=2, col=1
    )

    # Layout updates
    fig.update_layout(
        height=800,
        xaxis_rangeslider_visible=False,
        title_text=f"{symbol} Live Chart with Volatility Predictions"
    )

    return fig

async def app():
    st.title("Live Stock Volatility Predictions")
    
    # Sidebar controls
    symbol = st.sidebar.selectbox(
        "Select Stock",
        ["RELIANCE.NS", "INFY.NS", "TCS.NS", "HDFCBANK.NS"]
    )
    
    update_interval = st.sidebar.slider(
        "Update Interval (seconds)",
        min_value=5,
        max_value=60,
        value=10
    )

    # Main content
    chart_placeholder = st.empty()
    metrics_placeholder = st.empty()
    
    while True:
        try:
            # Get live data
            df = await st.session_state.data_fetcher.get_ohlcv_data(symbol)
            if df is not None:
                # Get new prediction
                pred = await get_prediction(symbol)
                if pred is not None:
                    st.session_state.predictions[symbol] = pred

                # Update chart
                fig = create_candlestick_chart(df, st.session_state.predictions, symbol)
                chart_placeholder.plotly_chart(fig, use_container_width=True)

                # Update metrics
                with metrics_placeholder.container():
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Current Price", f"₹{df['Close'].iloc[-1]:.2f}")
                    with col2:
                        st.metric("Predicted Volatility", 
                                f"{st.session_state.predictions.get(symbol, 0):.2f}%")
                    with col3:
                        returns = df['Close'].pct_change()
                        current_vol = returns.rolling(20).std().iloc[-1] * (252 ** 0.5) * 100
                        st.metric("Current Volatility", f"{current_vol:.2f}%")

            await asyncio.sleep(update_interval)
            
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            await asyncio.sleep(5)  # Back off on error

if __name__ == "__main__":
    asyncio.run(app())
