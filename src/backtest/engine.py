from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from .cost_model import CostModel, Trade
from ..utils.logging import get_logger

logger = get_logger(__name__)

@dataclass
class BacktestConfig:
    """Backtest configuration."""
    initial_capital: float = 1_000_000
    start_date: str = "2020-01-01"
    end_date: str = "2024-12-31"
    rebalance_freq: str = "monthly"  # daily, weekly, monthly
    
@dataclass
class PortfolioMetrics:
    """Portfolio performance metrics."""
    total_return: float
    annual_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float

class BacktestEngine:
    """Walk-forward backtesting engine for volatility strategies."""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.cost_model = CostModel()
        self.trades: List[Trade] = []
        self.portfolio_values: List[float] = []
        self.positions: Dict[str, float] = {}
        
    def run_backtest(self, 
                    data: pd.DataFrame, 
                    strategy_func, 
                    model_predictions: pd.DataFrame) -> Dict:
        """Run backtest with given strategy and predictions."""
        logger.info(f"Starting backtest from {self.config.start_date} to {self.config.end_date}")
        
        # Filter data by date range
        start_dt = pd.to_datetime(self.config.start_date)
        end_dt = pd.to_datetime(self.config.end_date)
        data = data[(data.index >= start_dt) & (data.index <= end_dt)]
        
        # Initialize portfolio
        cash = self.config.initial_capital
        portfolio_value = cash
        
        results = {
            'daily_returns': [],
            'portfolio_values': [],
            'trades': [],
            'positions': []
        }
        
        for date, row in data.iterrows():
            # Get model prediction for this date
            if date in model_predictions.index:
                pred = model_predictions.loc[date]
                
                # Execute strategy
                signals = strategy_func(row, pred, self.positions)
                
                # Process trades
                for symbol, target_weight in signals.items():
                    current_weight = self.positions.get(symbol, 0.0)
                    
                    if abs(target_weight - current_weight) > 0.001:  # Minimum trade threshold
                        trade_value = (target_weight - current_weight) * portfolio_value
                        
                        if symbol in data.columns:
                            price = row[symbol] if not pd.isna(row[symbol]) else data[symbol].fillna(method='ffill').loc[date]
                            quantity = trade_value / price
                            
                            # Calculate transaction cost
                            cost = self.cost_model.calculate_trade_cost(symbol, quantity, price)
                            
                            # Execute trade
                            trade = Trade(
                                timestamp=date,
                                symbol=symbol,
                                side='buy' if quantity > 0 else 'sell',
                                quantity=abs(quantity),
                                price=price,
                                cost=cost
                            )
                            
                            self.trades.append(trade)
                            cash -= trade_value + cost
                            self.positions[symbol] = target_weight
            
            # Calculate portfolio value
            portfolio_value = cash
            for symbol, weight in self.positions.items():
                if symbol in data.columns and not pd.isna(row[symbol]):
                    portfolio_value += weight * self.config.initial_capital * (row[symbol] / data[symbol].iloc[0])
            
            results['portfolio_values'].append(portfolio_value)
            if len(results['portfolio_values']) > 1:
                daily_ret = (portfolio_value / results['portfolio_values'][-2]) - 1
                results['daily_returns'].append(daily_ret)
            
        results['trades'] = self.trades
        results['final_value'] = portfolio_value
        
        logger.info(f"Backtest completed. Final portfolio value: {portfolio_value:,.0f}")
        return results
    
    def calculate_metrics(self, results: Dict) -> PortfolioMetrics:
        """Calculate performance metrics from backtest results."""
        returns = pd.Series(results['daily_returns'])
        portfolio_values = pd.Series(results['portfolio_values'])
        
        # Total and annualized returns
        total_return = (results['final_value'] / self.config.initial_capital) - 1
        periods_per_year = 252  # Trading days
        years = len(returns) / periods_per_year
        annual_return = (1 + total_return) ** (1/years) - 1
        
        # Volatility (annualized)
        volatility = returns.std() * np.sqrt(periods_per_year)
        
        # Sharpe ratio (assuming 6% risk-free rate for India)
        risk_free_rate = 0.06
        sharpe_ratio = (annual_return - risk_free_rate) / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        peak = portfolio_values.expanding().max()
        drawdown = (portfolio_values - peak) / peak
        max_drawdown = drawdown.min()
        
        # Calmar ratio
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Win rate
        winning_trades = len([r for r in returns if r > 0])
        win_rate = winning_trades / len(returns) if len(returns) > 0 else 0
        
        # Profit factor
        gross_profit = sum([r for r in returns if r > 0])
        gross_loss = abs(sum([r for r in returns if r < 0]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
        
        return PortfolioMetrics(
            total_return=total_return,
            annual_return=annual_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
            win_rate=win_rate,
            profit_factor=profit_factor
        )
