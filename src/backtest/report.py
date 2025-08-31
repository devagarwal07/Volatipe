from __future__ import annotations
from typing import Dict, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from .engine import PortfolioMetrics
from ..utils.logging import get_logger

logger = get_logger(__name__)

class BacktestReporter:
    """Generate comprehensive backtest reports."""
    
    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def generate_report(self, results: Dict, metrics: PortfolioMetrics, strategy_name: str) -> None:
        """Generate complete backtest report."""
        logger.info(f"Generating backtest report for {strategy_name}")
        
        # Create strategy-specific directory
        strategy_dir = self.output_dir / strategy_name
        strategy_dir.mkdir(exist_ok=True)
        
        # Generate plots
        self._plot_equity_curve(results, strategy_dir)
        self._plot_drawdown(results, strategy_dir)
        self._plot_monthly_returns(results, strategy_dir)
        self._plot_trade_analysis(results, strategy_dir)
        
        # Generate summary report
        self._generate_summary_report(metrics, strategy_dir)
        
        logger.info(f"Report saved to {strategy_dir}")
    
    def _plot_equity_curve(self, results: Dict, output_dir: Path) -> None:
        """Plot portfolio equity curve."""
        portfolio_values = pd.Series(results['portfolio_values'])
        
        plt.figure(figsize=(12, 6))
        plt.plot(portfolio_values.index, portfolio_values.values)
        plt.title('Portfolio Equity Curve')
        plt.xlabel('Trading Days')
        plt.ylabel('Portfolio Value (INR)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'equity_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_drawdown(self, results: Dict, output_dir: Path) -> None:
        """Plot drawdown chart."""
        portfolio_values = pd.Series(results['portfolio_values'])
        peak = portfolio_values.expanding().max()
        drawdown = (portfolio_values - peak) / peak * 100
        
        plt.figure(figsize=(12, 6))
        plt.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
        plt.plot(drawdown.index, drawdown.values, color='red', linewidth=1)
        plt.title('Portfolio Drawdown')
        plt.xlabel('Trading Days')
        plt.ylabel('Drawdown (%)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'drawdown.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_monthly_returns(self, results: Dict, output_dir: Path) -> None:
        """Plot monthly returns heatmap."""
        if not results['daily_returns']:
            return
        
        returns = pd.Series(results['daily_returns'])
        # Assuming daily data, resample to monthly
        monthly_returns = (1 + returns).resample('M').prod() - 1
        
        # Create year-month matrix
        monthly_returns.index = pd.to_datetime(monthly_returns.index)
        monthly_df = monthly_returns.to_frame('returns')
        monthly_df['year'] = monthly_df.index.year
        monthly_df['month'] = monthly_df.index.month
        
        pivot_table = monthly_df.pivot_table(values='returns', index='year', columns='month')
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_table * 100, annot=True, fmt='.1f', cmap='RdYlGn', center=0)
        plt.title('Monthly Returns Heatmap (%)')
        plt.xlabel('Month')
        plt.ylabel('Year')
        plt.tight_layout()
        plt.savefig(output_dir / 'monthly_returns.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_trade_analysis(self, results: Dict, output_dir: Path) -> None:
        """Plot trade analysis."""
        trades = results.get('trades', [])
        if not trades:
            return
        
        # Trade P&L distribution
        trade_pnl = [t.value for t in trades]
        
        plt.figure(figsize=(12, 6))
        plt.hist(trade_pnl, bins=30, alpha=0.7, edgecolor='black')
        plt.title('Trade P&L Distribution')
        plt.xlabel('Trade P&L (INR)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'trade_pnl_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_summary_report(self, metrics: PortfolioMetrics, output_dir: Path) -> None:
        """Generate text summary report."""
        report = f"""
BACKTEST SUMMARY REPORT
=======================

Performance Metrics:
-------------------
Total Return: {metrics.total_return:.2%}
Annual Return: {metrics.annual_return:.2%}
Volatility: {metrics.volatility:.2%}
Sharpe Ratio: {metrics.sharpe_ratio:.2f}
Maximum Drawdown: {metrics.max_drawdown:.2%}
Calmar Ratio: {metrics.calmar_ratio:.2f}
Win Rate: {metrics.win_rate:.2%}
Profit Factor: {metrics.profit_factor:.2f}

Risk Metrics:
------------
Max Drawdown: {metrics.max_drawdown:.2%}
Volatility: {metrics.volatility:.2%}

Trade Statistics:
----------------
Win Rate: {metrics.win_rate:.2%}
Profit Factor: {metrics.profit_factor:.2f}

Notes:
------
- All returns are calculated using daily data
- Transaction costs include brokerage, STT, stamp duty, and GST
- Indian market specific costs are applied
- Sharpe ratio assumes 6% risk-free rate (typical for India)
"""
        
        with open(output_dir / 'summary_report.txt', 'w') as f:
            f.write(report)
    
    def compare_strategies(self, strategy_results: Dict[str, Dict]) -> pd.DataFrame:
        """Compare multiple strategies."""
        comparison_data = []
        
        for strategy_name, results in strategy_results.items():
            metrics = results['metrics']
            comparison_data.append({
                'Strategy': strategy_name,
                'Total Return': f"{metrics.total_return:.2%}",
                'Annual Return': f"{metrics.annual_return:.2%}",
                'Volatility': f"{metrics.volatility:.2%}",
                'Sharpe Ratio': f"{metrics.sharpe_ratio:.2f}",
                'Max Drawdown': f"{metrics.max_drawdown:.2%}",
                'Calmar Ratio': f"{metrics.calmar_ratio:.2f}",
                'Win Rate': f"{metrics.win_rate:.2%}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_csv(self.output_dir / 'strategy_comparison.csv', index=False)
        
        return comparison_df
