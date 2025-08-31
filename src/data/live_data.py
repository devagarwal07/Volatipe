from __future__ import annotations
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import asyncio
from typing import Dict, Optional
from ..utils.logging import get_logger

logger = get_logger(__name__)

class LiveDataFetcher:
    def __init__(self, symbols: list[str], interval: str = "1m"):
        """
        Initialize live data fetcher
        Args:
            symbols: List of stock symbols (e.g., ["RELIANCE.NS", "INFY.NS"])
            interval: Data interval ("1m", "5m", "15m", "30m", "60m")
        """
        self.symbols = symbols
        self.interval = interval
        self._cache: Dict[str, pd.DataFrame] = {}
        self._last_update: Dict[str, datetime] = {}
        
    async def get_live_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get latest stock data for symbol"""
        now = datetime.now()
        
        # Check cache freshness (refresh if older than interval)
        if symbol in self._last_update:
            age = now - self._last_update[symbol]
            if age < timedelta(minutes=1):  # Minimum cache time
                return self._cache.get(symbol)
        
        try:
            # Get data from Yahoo Finance
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="1d", interval=self.interval)
            
            if df.empty:
                logger.warning(f"No data received for {symbol}")
                return None
                
            # Update cache
            self._cache[symbol] = df
            self._last_update[symbol] = now
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch live data for {symbol}: {e}")
            return None
            
    async def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get just the latest price for a symbol"""
        df = await self.get_live_data(symbol)
        if df is not None and not df.empty:
            return df['Close'].iloc[-1]
        return None

    async def get_ohlcv_data(self, symbol: str, lookback: int = 60) -> Optional[pd.DataFrame]:
        """Get OHLCV data for the last n intervals"""
        df = await self.get_live_data(symbol)
        if df is not None and not df.empty:
            return df.tail(lookback)
        return None
