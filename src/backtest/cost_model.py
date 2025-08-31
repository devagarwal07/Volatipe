from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from ..utils.logging import get_logger

logger = get_logger(__name__)

@dataclass
class TransactionCost:
    """Indian market transaction costs."""
    brokerage_pct: float = 0.0025  # 0.25%
    stt_pct: float = 0.00125  # Securities Transaction Tax
    stamp_duty_pct: float = 0.00015
    gst_pct: float = 0.18  # GST on brokerage
    
    def calculate_cost(self, trade_value: float) -> float:
        """Calculate total transaction cost."""
        brokerage = trade_value * self.brokerage_pct
        stt = trade_value * self.stt_pct
        stamp_duty = trade_value * self.stamp_duty_pct
        gst = brokerage * self.gst_pct
        return brokerage + stt + stamp_duty + gst

@dataclass 
class Trade:
    """Individual trade record."""
    timestamp: pd.Timestamp
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    price: float
    cost: float
    
    @property
    def value(self) -> float:
        return self.quantity * self.price

class CostModel:
    """Transaction cost calculator for Indian markets."""
    
    def __init__(self, cost_config: Optional[Dict] = None):
        if cost_config:
            self.costs = TransactionCost(**cost_config)
        else:
            self.costs = TransactionCost()
        
    def calculate_trade_cost(self, symbol: str, quantity: float, price: float) -> float:
        """Calculate cost for a single trade."""
        trade_value = abs(quantity * price)
        return self.costs.calculate_cost(trade_value)
    
    def calculate_round_trip_cost(self, symbol: str, quantity: float, entry_price: float, exit_price: float) -> float:
        """Calculate round-trip trading cost."""
        entry_cost = self.calculate_trade_cost(symbol, quantity, entry_price)
        exit_cost = self.calculate_trade_cost(symbol, quantity, exit_price)
        return entry_cost + exit_cost
