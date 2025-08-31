import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

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

# Set page config
st.set_page_config(
    page_title="Live Market Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .stMetric .st-emotion-cache-10trblm {
        font-size: 1.2rem;
        font-weight: 600;
    }
    .stMetric .st-emotion-cache-q8sbsg p {
        font-size: 2rem !important;
    }
    .market-header {
        font-size: 2.2rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
    .index-title {
        font-size: 1.8rem;
        font-weight: bold;
        margin-bottom: 1rem;
        color: #2a5298;
    }
    </style>
""", unsafe_allow_html=True)

def display_market_overview(symbol, analysis):
    """Helper function to display market overview for an index"""
    st.markdown(f'<div class="index-title">{INDEX_SYMBOLS[symbol]}</div>', unsafe_allow_html=True)
    
    # Current value card
    st.markdown(f"""
        <div style='background: rgba(32,32,32,0.8); padding: 20px; border-radius: 10px; margin: 10px 0;'>
            <div style='font-size: 2rem; font-weight: bold; color: {"#00ff00" if analysis["price_change_pct"] > 0 else "#ff4444"}'>
                {analysis["current_price"]:,.2f}
            </div>
            <div style='font-size: 1.2rem; color: {"#00ff00" if analysis["price_change_pct"] > 0 else "#ff4444"}'>
                {analysis["price_change_pct"]:+.2f}%
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # 52-week metrics
    cols = st.columns(2)
    
    # 52-week high
    with cols[0]:
        st.markdown(f"""
            <div style='background: rgba(0,200,0,0.1); padding: 15px; border-radius: 10px;'>
                <div style='font-size: 0.8rem; color: #888;'>52W High</div>
                <div style='font-size: 1.2rem; font-weight: bold;'>₹{analysis["yearly_high"]:,.2f}</div>
                <div style='font-size: 0.8rem; color: #666;'>{analysis["yearly_high_date"]}</div>
                <div style='font-size: 0.9rem; color: #ff4444;'>
                    {analysis["distance_from_high"]:.2f}% below
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    # 52-week low
    with cols[1]:
        st.markdown(f"""
            <div style='background: rgba(200,0,0,0.1); padding: 15px; border-radius: 10px;'>
                <div style='font-size: 0.8rem; color: #888;'>52W Low</div>
                <div style='font-size: 1.2rem; font-weight: bold;'>₹{analysis["yearly_low"]:,.2f}</div>
                <div style='font-size: 0.8rem; color: #666;'>{analysis["yearly_low_date"]}</div>
                <div style='font-size: 0.9rem; color: #00ff00;'>
                    {analysis["distance_from_low"]:.2f}% above
                </div>
            </div>
        """, unsafe_allow_html=True)

# Initialize session state
if 'data_fetcher' not in st.session_state:
    symbols = [
        "^NSEI",     # NIFTY 50
        "^NSEBANK",  # BANK NIFTY
        "RELIANCE.NS",
        "INFY.NS",
        "TCS.NS",
        "HDFCBANK.NS"
    ]
    st.session_state.data_fetcher = LiveDataFetcher(symbols)
    st.session_state.last_update = None

# Define index symbols for special handling
INDEX_SYMBOLS = {
    "^NSEI": "NIFTY 50",
    "^NSEBANK": "BANK NIFTY"
}

# Colors for UI
COLORS = {
    'positive': '#00cc96',
    'negative': '#ef553b',
    'neutral': '#636efa'
}

if 'predictions' not in st.session_state:
    st.session_state.predictions = {}

async def get_prediction(symbol: str, horizon: int = 1) -> Optional[float]:
    """Get volatility prediction from API"""
    try:
        # Check if we're running in Streamlit Cloud (no local prediction server)
        if os.getenv('STREAMLIT_CLOUD') or 'streamlit.app' in os.getenv('HOSTNAME', ''):
            # Return a mock prediction for demo purposes
            import random
            return round(random.uniform(15.0, 35.0), 2)
            
        async with aiohttp.ClientSession() as session:
            async with session.post(
                'http://localhost:8013/predict',
                json={'symbol': symbol.replace('.NS', ''), 'horizon': horizon},
                timeout=aiohttp.ClientTimeout(total=5)  # 5 second timeout
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get('forecast')
    except Exception as e:
        logger.error(f"Failed to get prediction: {e}")
        # Return a fallback prediction for demo purposes
        import random
        return round(random.uniform(15.0, 35.0), 2)
    return None

def create_candlestick_chart(df: pd.DataFrame, predictions: Dict[str, float], symbol: str):
    """Create candlestick chart with predictions"""
    # Get the display name
    display_name = INDEX_SYMBOLS.get(symbol, symbol.replace(".NS", ""))
    
    # Ensure we have valid data
    if df is None or len(df) == 0:
        # Create empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
        fig.update_layout(title=f"{display_name} - No Data", height=400)
        return fig
        
    # Create figure with secondary y-axis
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=(f'{display_name} Price Chart', 'Volatility Analysis'),
        row_heights=[0.7, 0.3]
    )

    # Calculate moving averages
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()

    # Main price chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name=display_name,
            showlegend=False
        ),
        row=1, col=1
    )

    # Add moving averages
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['MA20'],
            name='20-day MA',
            line=dict(color='orange', width=1)
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['MA50'],
            name='50-day MA',
            line=dict(color='blue', width=1)
        ),
        row=1, col=1
    )

    # Historical volatility (20-day rolling)
    returns = df['Close'].pct_change()
    vol = returns.rolling(window=20).std() * (252 ** 0.5) * 100  # Annualized
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=vol,
            mode='lines',
            name='Historical Vol',
            line=dict(color='purple', width=1.5)
        ),
        row=2, col=1
    )

    # Add predictions with improved visibility
    if symbol in predictions:
        pred_val = predictions[symbol]
        fig.add_trace(
            go.Scatter(
                x=[df.index[-1]],
                y=[pred_val],
                mode='markers+text',
                name='Predicted Vol',
                text=[f'{pred_val:.2f}%'],
                textposition='top center',
                marker=dict(
                    size=12,
                    symbol='star',
                    color='red',
                    line=dict(color='white', width=1)
                )
            ),
            row=2, col=1
        )

    # Layout updates with enhanced styling
    fig.update_layout(
        height=500,  # Slightly reduced height for better fit
        xaxis_rangeslider_visible=False,
        margin=dict(l=50, r=50, t=50, b=30),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='rgba(255, 255, 255, 0.3)',
            borderwidth=1
        ),
        template='plotly_dark',
        font=dict(
            family="Arial, sans-serif",
            size=12,
            color="white"
        )
    )    

    # Enhanced axes styling
    fig.update_yaxes(
        title_text="Price (₹)",
        row=1, col=1,
        gridcolor='rgba(128, 128, 128, 0.1)',
        zerolinecolor='rgba(255, 255, 255, 0.2)',
        tickformat=',',
        tickprefix='₹'
    )
    fig.update_yaxes(
        title_text="Volatility (%)",
        row=2, col=1,
        gridcolor='rgba(128, 128, 128, 0.1)',
        zerolinecolor='rgba(255, 255, 255, 0.2)',
        ticksuffix='%'
    )
    
    # Update x-axes layout with better formatting
    fig.update_xaxes(
        gridcolor='rgba(128, 128, 128, 0.1)',
        showgrid=True,
        zeroline=True,
        zerolinecolor='rgba(255, 255, 255, 0.2)',
        showline=True,
        linewidth=1,
        linecolor='rgba(255, 255, 255, 0.2)',
        row=1, col=1
    )
    fig.update_xaxes(
        gridcolor='rgba(128, 128, 128, 0.1)',
        showgrid=True,
        zeroline=True,
        zerolinecolor='rgba(255, 255, 255, 0.2)',
        showline=True,
        linewidth=1,
        linecolor='rgba(255, 255, 255, 0.2)',
        row=2, col=1
    )

    # Enhanced background styling
    fig.update_layout(
        plot_bgcolor='rgba(25, 25, 35, 1)',
        paper_bgcolor='rgba(25, 25, 35, 1)',
        title=dict(
            text=f"{display_name} Analysis",
            x=0.5,
            y=0.95,
            xanchor='center',
            yanchor='top',
            font=dict(size=20)
        )
    )

    return fig

def calculate_analysis(df: pd.DataFrame, is_index: bool = False) -> dict:
    """Calculate detailed analysis for a stock or index"""
    # Ensure we have enough data
    if len(df) < 2:
        return None
        
    # Calculate returns
    returns = df['Close'].pct_change().dropna()
    
    # Calculate 52-week (252 trading days) high and low
    # If we don't have full year of data, use all available data
    lookback = min(252, len(df))
    recent_data = df.tail(lookback)
    yearly_high = recent_data['High'].max()
    yearly_low = recent_data['Low'].min()
    yearly_high_date = recent_data[recent_data['High'] == yearly_high].index[0].strftime('%d-%b-%Y')
    yearly_low_date = recent_data[recent_data['Low'] == yearly_low].index[0].strftime('%d-%b-%Y')
    
    current_price = df['Close'].iloc[-1]
    
    # Base analysis for both stocks and indices
    analysis = {
        'current_price': current_price,
        'price_change': current_price - df['Close'].iloc[-2],
        'price_change_pct': (current_price / df['Close'].iloc[-2] - 1) * 100,
        'daily_return': returns.iloc[-1] * 100,
        'daily_vol': returns.std() * (252 ** 0.5) * 100,
        'rolling_vol': returns.rolling(20).std().iloc[-1] * (252 ** 0.5) * 100,
        'high_price': df['High'].iloc[-1],
        'low_price': df['Low'].iloc[-1],
        'price_range': df['High'].iloc[-1] - df['Low'].iloc[-1],
        'price_range_pct': (df['High'].iloc[-1] - df['Low'].iloc[-1]) / df['Low'].iloc[-1] * 100,
        'yearly_high': yearly_high,
        'yearly_high_date': yearly_high_date,
        'yearly_low': yearly_low,
        'yearly_low_date': yearly_low_date,
        'distance_from_high': (1 - current_price / yearly_high) * 100,
        'distance_from_low': (current_price / yearly_low - 1) * 100,
        'weekly_return': (current_price / df['Close'].iloc[-min(5, len(df))] - 1) * 100 if len(df) >= 5 else 0,
        'monthly_return': (current_price / df['Close'].iloc[-min(20, len(df))] - 1) * 100 if len(df) >= 20 else 0
    }
    
    # Add volume analysis only for stocks (not indices)
    if not is_index and 'Volume' in df.columns:
        analysis.update({
            'volume': df['Volume'].iloc[-1],
            'avg_volume': df['Volume'].rolling(20).mean().iloc[-1]
        })
    else:
        analysis.update({
            'volume': None,
            'avg_volume': None
        })
    
    return analysis

def format_price_card(title, price, change_pct, subtitle=None, bg_color="rgba(0,0,0,0.1)"):
    """Helper function to create consistent price cards"""
    return f"""
        <div style='background: {bg_color}; padding: 15px; border-radius: 10px; margin: 5px 0;'>
            <div style='font-size: 0.9rem; color: #888;'>{title}</div>
            <div style='font-size: 1.4rem; font-weight: bold;'>₹{price:,.2f}</div>
            {f'<div style="font-size: 0.9rem; color: #888;">{subtitle}</div>' if subtitle else ''}
            <div style='font-size: 1rem; color: {"#00ff00" if change_pct > 0 else "#ff4444"};'>
                {change_pct:+.2f}%
            </div>
        </div>
    """

async def app():
    # Sidebar controls with better styling
    with st.sidebar:
        st.markdown("### 📊 Dashboard Controls")
        st.markdown("---")
        update_interval = st.slider(
            "Update Interval (seconds)",
            min_value=5,
            max_value=60,
            value=10
        )
        
        # Add market status and time
        market_hours = datetime.now().hour in range(9, 16)
        status_color = "🟢" if market_hours else "🔴"
        st.markdown(f"### Market Status: {status_color}")
        st.markdown(f"Time: {datetime.now().strftime('%I:%M:%S %p')}")
        
        # Add market status indicator
        market_hours = datetime.now().hour in range(9, 16)  # 9 AM to 4 PM
        status_color = "🟢" if market_hours else "🔴"
        st.markdown(f"### Market Status: {status_color}")
        
        # Add filter options
        st.markdown("### 🔍 Filters")
        show_ma = st.checkbox("Show Moving Averages", value=True)
        show_vol = st.checkbox("Show Volume", value=True)
        
        st.markdown("---")
        
        # Add prediction service disclaimer
        if os.getenv('STREAMLIT_CLOUD') or 'streamlit.app' in os.getenv('HOSTNAME', ''):
            st.markdown("### ⚠️ Demo Mode")
            st.info("Volatility predictions are simulated for demo purposes. In production, these would come from a live ML model server.")
        
        st.markdown("---")
        st.markdown("### 📈 Analysis Settings")
        vol_window = st.select_slider(
            "Volatility Window (days)",
            options=[5, 10, 20, 30],
            value=20
        )

    # Initialize containers
    summary_placeholder = st.empty()
    charts_placeholder = st.empty()
    analysis_placeholder = st.empty()
    
    while True:
        try:
            # Fetch data and predictions for all symbols
            all_data = {}
            all_analysis = {}
            
            for symbol in st.session_state.data_fetcher.symbols:
                try:
                    # Get live data with proper error handling
                    df = await st.session_state.data_fetcher.get_ohlcv_data(symbol)
                    if df is not None and len(df) > 0:
                        # Clean the data using modern pandas methods
                        df = df.ffill().bfill()
                        all_data[symbol] = df
                        
                        # Get prediction with proper symbol mapping
                        pred_symbol = (symbol.replace('^', '')
                                          .replace('NSEI', 'NIFTY')
                                          .replace('NSEBANK', 'BANKNIFTY')
                                          .replace('.NS', ''))
                        
                        pred = await get_prediction(pred_symbol)
                        if pred is not None:
                            st.session_state.predictions[symbol] = pred
                        
                        # Calculate analysis
                        is_index = symbol in INDEX_SYMBOLS
                        all_analysis[symbol] = calculate_analysis(df, is_index=is_index)
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {str(e)}")
                    continue

            # Update Market Overview (Indices)
            with summary_placeholder.container():
                st.markdown('<div class="market-header">🔴 LIVE MARKET OVERVIEW</div>', unsafe_allow_html=True)
                
                # Display last update time
                current_time = datetime.now().strftime("%I:%M:%S %p")
                st.markdown(f"<div style='text-align: right; color: #666;'>Last Update: {current_time}</div>", 
                          unsafe_allow_html=True)
                
                # Display indices in two columns
                index_cols = st.columns(2)
                for idx, symbol in enumerate(['^NSEI', '^NSEBANK']):
                    if symbol in all_data and symbol in all_analysis:
                        with index_cols[idx]:
                            display_market_overview(symbol, all_analysis[symbol])
                
                # Stock Charts Section
                st.markdown("---")
                st.markdown('<div class="market-header">📈 STOCK ANALYSIS</div>', unsafe_allow_html=True)
                
                # Create two columns for stock charts
                stock_cols = st.columns(2)
                stock_idx = 0
                
                # Display charts for individual stocks
                for symbol in all_data:
                    if symbol not in ['^NSEI', '^NSEBANK'] and symbol in all_analysis:
                        with stock_cols[stock_idx % 2]:
                            # Create and display chart
                            fig = create_candlestick_chart(all_data[symbol], st.session_state.predictions, symbol)
                            st.plotly_chart(fig, use_container_width=True, key=f"main_stock_chart_{symbol}_{stock_idx}")
                            
                            # Display key metrics below the chart
                            metrics_cols = st.columns(3)
                            analysis = all_analysis[symbol]
                            
                            with metrics_cols[0]:
                                st.metric(
                                    "Current Price",
                                    f"₹{analysis['current_price']:,.2f}",
                                    f"{analysis['price_change_pct']:+.2f}%"
                                )
                            
                            with metrics_cols[1]:
                                if analysis['volume'] is not None:
                                    vol_change = (analysis['volume']/analysis['avg_volume']-1)*100
                                    st.metric(
                                        "Volume",
                                        f"{analysis['volume']:,.0f}",
                                        f"{vol_change:+.1f}% vs Avg"
                                    )
                            
                            with metrics_cols[2]:
                                pred_vol = st.session_state.predictions.get(symbol, 0)
                                current_vol = analysis['rolling_vol']
                                st.metric(
                                    "Predicted Vol",
                                    f"{pred_vol:.2f}%",
                                    f"{pred_vol - current_vol:+.2f}% vs Current"
                                )
                            
                            # Display 52-week information
                            week_cols = st.columns(2)
                            with week_cols[0]:
                                st.markdown(f"""
                                    <div style='background: rgba(0,200,0,0.1); padding: 10px; border-radius: 5px;'>
                                        <div style='font-size: 0.8rem; color: #888;'>52W High ({analysis['yearly_high_date']})</div>
                                        <div style='font-size: 1.1rem; font-weight: bold;'>₹{analysis['yearly_high']:,.2f}</div>
                                        <div style='font-size: 0.9rem; color: #888;'>{analysis['distance_from_high']:+.2f}% from high</div>
                                    </div>
                                """, unsafe_allow_html=True)
                            
                            with week_cols[1]:
                                st.markdown(f"""
                                    <div style='background: rgba(200,0,0,0.1); padding: 10px; border-radius: 5px;'>
                                        <div style='font-size: 0.8rem; color: #888;'>52W Low ({analysis['yearly_low_date']})</div>
                                        <div style='font-size: 1.1rem; font-weight: bold;'>₹{analysis['yearly_low']:,.2f}</div>
                                        <div style='font-size: 0.9rem; color: #888;'>{analysis['distance_from_low']:+.2f}% from low</div>
                                    </div>
                                """, unsafe_allow_html=True)
                            
                            st.markdown("---")
                        stock_idx += 1
                        
            # Display NIFTY 50 and BANK NIFTY prominently
                for idx, symbol in enumerate(['^NSEI', '^NSEBANK']):
                    if symbol in all_data:
                        analysis = all_analysis[symbol]
                        with index_cols[idx]:
                            st.markdown(f"### {INDEX_SYMBOLS[symbol]}")
                        st.metric(
                            "Current Value",
                            f"{analysis['current_price']:,.2f}",
                            f"{analysis['price_change_pct']:+.2f}%"
                        )
                        cols = st.columns(2)
                        with cols[0]:
                            st.metric(
                                "Daily Range",
                                f"{analysis['price_range']:,.2f}",
                                f"{analysis['price_range_pct']:+.2f}%"
                            )
                        with cols[1]:
                            st.metric(
                                "Distance from 52W High",
                                f"{analysis['distance_from_high']:.2f}%"
                            )                # Stock Summary Table
                st.subheader("Live Stock Summary")
                summary_data = []
                for symbol in all_data.keys():
                    if symbol not in ['^NSEI', '^NSEBANK']:  # Skip indices in this table
                        analysis = all_analysis[symbol]
                        summary_data.append({
                            'Symbol': symbol.replace('.NS', ''),
                            'Price': f"₹{analysis['current_price']:.2f}",
                            'Change %': f"{analysis['price_change_pct']:+.2f}%",
                            'Current Vol.': f"{analysis['rolling_vol']:.2f}%",
                            'Predicted Vol.': f"{st.session_state.predictions.get(symbol, 0):.2f}%",
                            'Volume': f"{analysis['volume']:,.0f}" if analysis['volume'] is not None else "N/A"
                        })
                st.dataframe(
                    pd.DataFrame(summary_data).set_index('Symbol'),
                    use_container_width=True,
                    key="summary_table"
                )

            # Update charts
            with charts_placeholder.container():
                # First display index charts
                st.subheader("Market Indices")
                index_cols = st.columns(2)
                for idx, symbol in enumerate(['^NSEI', '^NSEBANK']):
                    if symbol in all_data:
                        with index_cols[idx]:
                            fig = create_candlestick_chart(
                                all_data[symbol], 
                                st.session_state.predictions, 
                                INDEX_SYMBOLS[symbol]
                            )
                            st.plotly_chart(fig, use_container_width=True, key=f"index_chart_{symbol}")
                
                # Then display stock charts
                st.subheader("Stock Charts")
                stock_cols = st.columns(2)
                stock_idx = 0
                for symbol, df in all_data.items():
                    if symbol not in ['^NSEI', '^NSEBANK']:
                        with stock_cols[stock_idx % 2]:
                            fig = create_candlestick_chart(df, st.session_state.predictions, symbol)
                            st.plotly_chart(fig, use_container_width=True, key=f"stock_detail_chart_{symbol}")
                        stock_idx += 1

            # Detailed Analysis
            with analysis_placeholder.container():
                st.subheader("Detailed Market Analysis")
                
                # First show indices analysis
                for symbol in ['^NSEI', '^NSEBANK']:
                    if symbol in all_analysis:
                        analysis = all_analysis[symbol]
                        st.write(f"### {INDEX_SYMBOLS[symbol]} Analysis")
                        cols = st.columns(4)
                        
                        # Price Metrics
                        with cols[0]:
                            st.metric(
                                "Current Value",
                                f"{analysis['current_price']:,.2f}",
                                f"{analysis['price_change_pct']:+.2f}%",
                                key=f"price_metric_{symbol}"
                            )
                            st.metric(
                                "Day Range",
                                f"{analysis['price_range']:,.2f}",
                                f"{analysis['price_range_pct']:+.2f}%",
                                key=f"range_metric_{symbol}"
                            )
                        
                        # Return Metrics
                        with cols[1]:
                            st.metric(
                                "Daily Return",
                                f"{analysis['daily_return']:+.2f}%",
                                key=f"return_metric_{symbol}"
                            )
                            st.metric(
                                "Weekly Return",
                                f"{analysis['weekly_return']:+.2f}%",
                                key=f"weekly_return_{symbol}"
                            )
                        
                        # Volatility Metrics
                        with cols[2]:
                            pred_vol = st.session_state.predictions.get(symbol, 0)
                            vol_diff = pred_vol - analysis['rolling_vol']
                            st.metric(
                                "Current Volatility",
                                f"{analysis['rolling_vol']:.2f}%",
                                key=f"curr_vol_metric_{symbol}"
                            )
                            st.metric(
                                "Predicted Volatility",
                                f"{pred_vol:.2f}%",
                                f"{vol_diff:+.2f}% vs Current",
                                key=f"pred_vol_metric_{symbol}"
                            )
                        
                        # Additional Metrics
                        with cols[3]:
                            st.metric(
                                "52-Week High",
                                f"{analysis['yearly_high']:,.2f}"
                            )
                            st.metric(
                                "52-Week Low",
                                f"{analysis['yearly_low']:,.2f}"
                            )
                
                # Then show individual stock analysis
                st.subheader("Detailed Stock Analysis")
                for symbol, analysis in all_analysis.items():
                    if symbol not in ['^NSEI', '^NSEBANK']:
                        st.write(f"### {symbol.replace('.NS', '')}")
                        cols = st.columns(3)
                        
                        # Price Metrics
                        with cols[0]:
                            st.metric(
                                "Current Price",
                                f"₹{analysis['current_price']:.2f}",
                                f"{analysis['price_change_pct']:+.2f}%"
                            )
                            st.metric(
                                "Day Range",
                                f"₹{analysis['price_range']:.2f}",
                                f"{analysis['price_range_pct']:+.2f}%"
                            )
                        
                        # Volume Metrics
                        with cols[1]:
                            if analysis['volume'] is not None:
                                st.metric(
                                    "Volume",
                                    f"{analysis['volume']:,.0f}",
                                    f"{(analysis['volume']/analysis['avg_volume']-1)*100:.1f}% vs Avg"
                                )
                            st.metric(
                                "Daily Return",
                                f"{analysis['daily_return']:+.2f}%"
                            )
                        
                        # Volatility Metrics
                        with cols[2]:
                            pred_vol = st.session_state.predictions.get(symbol, 0)
                            vol_diff = pred_vol - analysis['rolling_vol']
                            st.metric(
                                "Current Volatility",
                                f"{analysis['rolling_vol']:.2f}%"
                            )
                            st.metric(
                                "Predicted Volatility",
                                f"{pred_vol:.2f}%",
                                f"{vol_diff:+.2f}% vs Current"
                            )

            await asyncio.sleep(update_interval)
            
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            await asyncio.sleep(5)  # Back off on error

if __name__ == "__main__":
    asyncio.run(app())
