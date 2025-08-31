from __future__ import annotations
"""Live data fetching and forecasting support.

This module maintains a lightweight in-memory price cache for a selected set of
symbols and provides helper functions to fetch the latest price and model
forecasts suitable for a polling-based frontend dashboard.

Design goals:
  * Do not refit models (expensive) – use existing loaded models for now.
  * Provide hooks where incremental / online updating can later be added.
  * Keep dependencies minimal (yfinance already present).
"""
import threading
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List

import pandas as pd
import yfinance as yf

from ..utils.logging import get_logger
from ..models.predict import predict_garch, predict_har

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Symbol handling
# ---------------------------------------------------------------------------
INDEX_SYMBOLS = {"^NSEI", "^NSEBANK", "^CNXIT", "^CNXPHARMA"}

def to_yahoo_symbol(symbol: str) -> str:
    """Map internal symbol (e.g. RELIANCE) to Yahoo Finance symbol.

    Assumes Indian equities (NSE) without suffix internally; adds '.NS'.
    Leaves indices (starting with ^) untouched and passes through if already suffixed.
    """
    if symbol.upper() in INDEX_SYMBOLS or symbol.startswith('^'):
        return symbol
    if symbol.endswith('.NS'):
        return symbol
    return f"{symbol}.NS"

# ---------------------------------------------------------------------------
# In-memory caches
# ---------------------------------------------------------------------------
class LiveCache:
    def __init__(self):
        self._lock = threading.Lock()
        self.prices: Dict[str, pd.DataFrame] = {}  # symbol -> intraday df (DatetimeIndex)
        self.last_fetch: Dict[str, datetime] = {}
        self.last_forecasts: Dict[str, Dict[str, Any]] = {}

    def update_price(self, symbol: str, df: pd.DataFrame):
        with self._lock:
            if symbol in self.prices:
                existing = self.prices[symbol]
                combined = pd.concat([existing, df]).sort_index()
                combined = combined[~combined.index.duplicated(keep='last')]
                self.prices[symbol] = combined
            else:
                self.prices[symbol] = df
            self.last_fetch[symbol] = datetime.now(timezone.utc)

    def set_forecast(self, symbol: str, forecast: Dict[str, Any]):
        with self._lock:
            self.last_forecasts[symbol] = forecast

    def get_snapshot(self, symbol: str) -> Dict[str, Any]:
        with self._lock:
            price_df = self.prices.get(symbol)
            forecast = self.last_forecasts.get(symbol)
            last_price = None
            last_time = None
            if price_df is not None and not price_df.empty:
                last_row = price_df.iloc[-1]
                last_price = float(last_row['close']) if 'close' in last_row else None
                last_time = price_df.index[-1].isoformat()
            return {
                'symbol': symbol,
                'last_price': last_price,
                'last_price_time': last_time,
                'last_fetch_utc': self.last_fetch.get(symbol).isoformat() if symbol in self.last_fetch else None,
                'forecast': forecast,
            }

LIVE_CACHE = LiveCache()

# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------
def fetch_intraday(symbol: str, interval: str = '1m', lookback_minutes: int = 120) -> Optional[pd.DataFrame]:
    yahoo_sym = to_yahoo_symbol(symbol)
    try:
        # yfinance: period must cover interval size; use '2d' to be safe for 1m
        period = '1d' if interval != '1m' else '2d'
        data = yf.download(yahoo_sym, period=period, interval=interval, progress=False, auto_adjust=False)
        if data.empty:
            return None
        data = data.tail(lookback_minutes)  # approximate trimming
        data = data.rename(columns={
            'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'
        })[['open', 'high', 'low', 'close', 'volume']]
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)
        return data
    except Exception as e:
        logger.warning(f"Intraday fetch failed for {symbol}: {e}")
        return None


def compute_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out['ret'] = out['close'].pct_change()
    return out


def build_live_forecast(models: Dict[str, Any], symbol: str, horizon: int = 1) -> Dict[str, Any]:
    """Generate a composite live forecast using existing models.

    Currently calls the existing predict_* functions (no online update)."""
    result: Dict[str, Any] = {}
    try:
        garch_out = predict_garch(models, horizon=horizon)
        if 'ensemble' in garch_out:
            result['garch'] = float(garch_out['ensemble'])
            result['garch_components'] = garch_out
    except Exception as e:
        result['garch_error'] = str(e)
    try:
        har_out = predict_har(models, horizon=horizon, symbol=symbol)
        if isinstance(har_out, dict) and 'ensemble' in har_out:
            result['har_rv'] = float(har_out['ensemble'])
            result['har_components'] = har_out
    except Exception as e:
        result['har_error'] = str(e)
    # Simple blend if both available
    if 'garch' in result and 'har_rv' in result:
        result['blend_simple'] = (result['garch'] + result['har_rv']) / 2.0
    result['generated_at_utc'] = datetime.now(timezone.utc).isoformat()
    result['symbol'] = symbol
    return result


# ---------------------------------------------------------------------------
# Background thread
# ---------------------------------------------------------------------------
class LiveUpdater(threading.Thread):
    def __init__(self, symbols: List[str], models_provider, interval_sec: int = 60):
        super().__init__(daemon=True)
        self.symbols = symbols
        self.interval_sec = interval_sec
        self.models_provider = models_provider
        self._stop = threading.Event()

    def run(self):
        logger.info(f"Starting LiveUpdater for {len(self.symbols)} symbols interval={self.interval_sec}s")
        while not self._stop.is_set():
            models = None
            try:
                models = self.models_provider()
            except Exception:
                pass
            for sym in self.symbols:
                df = fetch_intraday(sym)
                if df is not None:
                    LIVE_CACHE.update_price(sym, df)
                    if models:
                        forecast = build_live_forecast(models, symbol=sym, horizon=1)
                        LIVE_CACHE.set_forecast(sym, forecast)
            self._stop.wait(self.interval_sec)
        logger.info("LiveUpdater stopped")

    def stop(self):
        self._stop.set()


_live_updater: Optional[LiveUpdater] = None

def start_live_updater(models_provider, symbols: Optional[List[str]] = None, interval_sec: int = 60):
    global _live_updater
    if _live_updater and _live_updater.is_alive():
        return
    symbols = symbols or ["RELIANCE", "HDFCBANK", "INFY"]
    _live_updater = LiveUpdater(symbols=symbols, models_provider=models_provider, interval_sec=interval_sec)
    _live_updater.start()


def stop_live_updater():
    global _live_updater
    if _live_updater:
        _live_updater.stop()


# ---------------------------------------------------------------------------
# FastAPI router
# ---------------------------------------------------------------------------
from fastapi import APIRouter, Depends

live_router = APIRouter(prefix="/live", tags=["live"])


@live_router.on_event("startup")
def _startup():
    # Delay import of get_models to avoid circular
    from .routes import get_models  # type: ignore
    start_live_updater(models_provider=get_models, symbols=["RELIANCE", "HDFCBANK", "INFY"], interval_sec=120)


@live_router.get('/price/{symbol}')
def live_price(symbol: str):
    return LIVE_CACHE.get_snapshot(symbol.upper())


@live_router.get('/forecast/{symbol}')
def live_forecast(symbol: str):
    # Simply return snapshot (contains forecast)
    return LIVE_CACHE.get_snapshot(symbol.upper())


@live_router.get('/status')
def live_status():
    return {'symbols': list(LIVE_CACHE.prices.keys()), 'running': True}
