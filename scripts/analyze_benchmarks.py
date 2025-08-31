#!/usr/bin/env python
"""
This script analyzes the performance of volatility forecasting models
across multiple stocks and generates a comprehensive benchmark report.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import seaborn as sns
from datetime import datetime

# Set up
results_dir = Path("results/benchmarks")
results_dir.mkdir(parents=True, exist_ok=True)
output_file = results_dir / f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

def load_evaluation_results():
    """Load all evaluation JSON files from the benchmarks directory"""
    eval_files = list(results_dir.glob("*_evaluation.json"))
    if not eval_files:
        print("No evaluation files found!")
        return None
    
    all_results = []
    for file in eval_files:
        try:
            with open(file, 'r') as f:
                data = json.load(f)
                all_results.append(data)
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    # Convert to DataFrame
    if all_results:
        df = pd.DataFrame(all_results)
        df = df.set_index('symbol')
        return df
    return None

def analyze_model_performance(df):
    """Analyze model performance across different stocks"""
    metrics = {
        'MSE': ['garch_mse', 'har_mse'],
        'MAE': ['garch_mae', 'har_mae'],
        'MAPE': ['garch_mape', 'har_mape']
    }
    
    results = {}
    
    # Calculate average metrics
    for metric_name, columns in metrics.items():
        avgs = {}
        for col in columns:
            if col in df.columns:
                model = col.split('_')[0].upper()
                avgs[model] = df[col].mean()
        results[metric_name] = avgs
    
    # Determine best model per stock
    best_models = {}
    for symbol in df.index:
        row = df.loc[symbol]
        if 'garch_mse' in row and 'har_mse' in row:
            if not pd.isna(row['garch_mse']) and not pd.isna(row['har_mse']):
                best_model = 'GARCH' if row['garch_mse'] < row['har_mse'] else 'HAR-RV'
                best_models[symbol] = best_model
    
    # Count model wins
    model_counts = {}
    for model in best_models.values():
        model_counts[model] = model_counts.get(model, 0) + 1
    
    results['best_models'] = best_models
    results['model_counts'] = model_counts
    
    return results

def create_visualizations(df, save_dir=None):
    """Create visualizations of model performance"""
    if save_dir is None:
        save_dir = results_dir
    
    # Ensure directory exists
    if not isinstance(save_dir, Path):
        save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Set Seaborn style
    sns.set(style="whitegrid")
    
    # Figure 1: MSE Comparison
    if 'garch_mse' in df.columns and 'har_mse' in df.columns:
        plt.figure(figsize=(14, 7))
        comparison = df[['garch_mse', 'har_mse']].copy()
        comparison.columns = ['GARCH', 'HAR-RV']
        ax = comparison.plot(kind='bar', title='MSE Comparison Across Stocks')
        ax.set_ylabel('Mean Squared Error')
        ax.set_xlabel('Stock Symbol')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(save_dir / 'mse_comparison.png', dpi=300, bbox_inches='tight')
    
    # Figure 2: MAPE Comparison
    if 'garch_mape' in df.columns and 'har_mape' in df.columns:
        plt.figure(figsize=(14, 7))
        comparison = df[['garch_mape', 'har_mape']].copy()
        comparison.columns = ['GARCH', 'HAR-RV']
        ax = comparison.plot(kind='bar', title='MAPE Comparison Across Stocks')
        ax.set_ylabel('Mean Absolute Percentage Error (%)')
        ax.set_xlabel('Stock Symbol')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(save_dir / 'mape_comparison.png', dpi=300, bbox_inches='tight')
    
    # Figure 3: Average Metrics Comparison
    metrics = ['mse', 'mae', 'mape']
    avg_metrics = {'GARCH': [], 'HAR-RV': []}
    
    for metric in metrics:
        garch_col = f'garch_{metric}'
        har_col = f'har_{metric}'
        
        if garch_col in df.columns:
            avg_metrics['GARCH'].append(df[garch_col].mean())
        else:
            avg_metrics['GARCH'].append(np.nan)
            
        if har_col in df.columns:
            avg_metrics['HAR-RV'].append(df[har_col].mean())
        else:
            avg_metrics['HAR-RV'].append(np.nan)
    
    plt.figure(figsize=(10, 6))
    index = ['MSE', 'MAE', 'MAPE']
    avg_df = pd.DataFrame(avg_metrics, index=index)
    ax = avg_df.plot(kind='bar', title='Average Model Performance Across All Stocks')
    ax.set_ylabel('Error Value')
    ax.set_xlabel('Metric')
    
    # Add percentage improvement annotations
    for i, metric in enumerate(index):
        if not np.isnan(avg_df.loc[metric, 'GARCH']) and not np.isnan(avg_df.loc[metric, 'HAR-RV']):
            garch_val = avg_df.loc[metric, 'GARCH']
            har_val = avg_df.loc[metric, 'HAR-RV']
            better_model = 'GARCH' if garch_val < har_val else 'HAR-RV'
            pct_diff = abs((garch_val - har_val) / max(garch_val, har_val) * 100)
            plt.text(i, max(garch_val, har_val) * 1.05, 
                    f"{better_model} better by {pct_diff:.1f}%",
                    ha='center')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'average_metrics.png', dpi=300, bbox_inches='tight')
    
    # Figure 4: Model Win Distribution
    if 'garch_mse' in df.columns and 'har_mse' in df.columns:
        best_models = []
        for symbol in df.index:
            row = df.loc[symbol]
            if not pd.isna(row['garch_mse']) and not pd.isna(row['har_mse']):
                best = 'GARCH' if row['garch_mse'] < row['har_mse'] else 'HAR-RV'
                best_models.append(best)
        
        model_counts = {'GARCH': 0, 'HAR-RV': 0}
        for model in best_models:
            model_counts[model] += 1
        
        plt.figure(figsize=(8, 8))
        plt.pie(model_counts.values(), labels=model_counts.keys(), autopct='%1.1f%%',
                startangle=90, shadow=True, explode=(0.05, 0.05))
        plt.axis('equal')
        plt.title('Distribution of Best Performing Model by Stock')
        plt.tight_layout()
        plt.savefig(save_dir / 'model_win_distribution.png', dpi=300, bbox_inches='tight')
    
    print(f"Visualizations saved to {save_dir}")
    return [f.name for f in save_dir.glob('*.png')]

def generate_report(df, analysis_results, viz_files):
    """Generate a comprehensive Markdown report"""
    with open(output_file, 'w') as f:
        # Header
        f.write("# Volatility Model Benchmark Report\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("This report provides a comprehensive evaluation of volatility forecasting models ")
        f.write("across multiple Indian stocks. The models evaluated include GARCH ensemble and HAR-RV ensemble.\n\n")
        
        # Summary Statistics
        f.write("## Summary Statistics\n\n")
        
        # Average metrics table
        f.write("### Average Metrics\n\n")
        f.write("| Metric | GARCH | HAR-RV |\n")
        f.write("|--------|-------|--------|\n")
        
        for metric, values in analysis_results.items():
            if metric in ['MSE', 'MAE', 'MAPE']:
                garch_val = values.get('GARCH', 'N/A')
                harv_val = values.get('HAR-RV', 'N/A')
                garch_fmt = f"{garch_val:.6f}" if isinstance(garch_val, (int, float)) else "N/A"
                harv_fmt = f"{harv_val:.6f}" if isinstance(harv_val, (int, float)) else "N/A"
                f.write(f"| {metric} | {garch_fmt} | {harv_fmt} |\n")
        
        # Model win counts
        f.write("\n### Model Performance Summary\n\n")
        win_counts = analysis_results.get('model_counts', {})
        f.write(f"- GARCH wins: {win_counts.get('GARCH', 0)} stocks\n")
        f.write(f"- HAR-RV wins: {win_counts.get('HAR-RV', 0)} stocks\n\n")
        
        # Overall winner determination
        if win_counts.get('GARCH', 0) > win_counts.get('HAR-RV', 0):
            overall_winner = "GARCH"
        elif win_counts.get('HAR-RV', 0) > win_counts.get('GARCH', 0):
            overall_winner = "HAR-RV"
        else:
            overall_winner = "Tie between GARCH and HAR-RV"
        
        f.write(f"**Overall Winner: {overall_winner}**\n\n")
        
        # Detailed Results
        f.write("## Detailed Results by Stock\n\n")
        f.write("| Symbol | GARCH MSE | GARCH MAE | GARCH MAPE | HAR-RV MSE | HAR-RV MAE | HAR-RV MAPE | Best Model |\n")
        f.write("|--------|-----------|-----------|------------|------------|------------|------------|------------|\n")
        
        best_models = analysis_results.get('best_models', {})
        for symbol in sorted(df.index):
            row = df.loc[symbol]
            garch_mse = f"{row.get('garch_mse', 'N/A'):.6f}" if 'garch_mse' in row and not pd.isna(row['garch_mse']) else 'N/A'
            garch_mae = f"{row.get('garch_mae', 'N/A'):.6f}" if 'garch_mae' in row and not pd.isna(row['garch_mae']) else 'N/A'
            garch_mape = f"{row.get('garch_mape', 'N/A'):.6f}" if 'garch_mape' in row and not pd.isna(row['garch_mape']) else 'N/A'
            har_mse = f"{row.get('har_mse', 'N/A'):.6f}" if 'har_mse' in row and not pd.isna(row['har_mse']) else 'N/A'
            har_mae = f"{row.get('har_mae', 'N/A'):.6f}" if 'har_mae' in row and not pd.isna(row['har_mae']) else 'N/A'
            har_mape = f"{row.get('har_mape', 'N/A'):.6f}" if 'har_mape' in row and not pd.isna(row['har_mape']) else 'N/A'
            
            best_model = best_models.get(symbol, 'N/A')
            
            f.write(f"| {symbol} | {garch_mse} | {garch_mae} | {garch_mape} | {har_mse} | {har_mae} | {har_mape} | {best_model} |\n")
        
        # Visualizations
        f.write("\n## Visualizations\n\n")
        
        for viz_file in viz_files:
            f.write(f"![{viz_file}](./benchmarks/{viz_file})\n\n")
        
        # Conclusions
        f.write("## Conclusions\n\n")
        
        # Automatic conclusions based on results
        mse_diff = analysis_results.get('MSE', {}).get('GARCH', float('inf')) - analysis_results.get('MSE', {}).get('HAR-RV', float('inf'))
        mape_diff = analysis_results.get('MAPE', {}).get('GARCH', float('inf')) - analysis_results.get('MAPE', {}).get('HAR-RV', float('inf'))
        
        f.write("Based on the benchmark evaluation across multiple Indian stocks:\n\n")
        
        if mse_diff < 0:
            f.write(f"- GARCH models show better accuracy in terms of MSE, with {abs(mse_diff/analysis_results.get('MSE', {}).get('HAR-RV', 1)*100):.2f}% lower error on average.\n")
        elif mse_diff > 0:
            f.write(f"- HAR-RV models show better accuracy in terms of MSE, with {abs(mse_diff/analysis_results.get('MSE', {}).get('GARCH', 1)*100):.2f}% lower error on average.\n")
        
        if mape_diff < 0:
            f.write(f"- GARCH models provide better percentage accuracy, with {abs(mape_diff/analysis_results.get('MAPE', {}).get('HAR-RV', 1)*100):.2f}% lower percentage error.\n")
        elif mape_diff > 0:
            f.write(f"- HAR-RV models provide better percentage accuracy, with {abs(mape_diff/analysis_results.get('MAPE', {}).get('GARCH', 1)*100):.2f}% lower percentage error.\n")
        
        garch_wins = win_counts.get('GARCH', 0)
        har_wins = win_counts.get('HAR-RV', 0)
        total_stocks = garch_wins + har_wins
        
        if total_stocks > 0:
            f.write(f"- GARCH models performed better on {garch_wins} stocks ({garch_wins/total_stocks*100:.1f}% of evaluated stocks).\n")
            f.write(f"- HAR-RV models performed better on {har_wins} stocks ({har_wins/total_stocks*100:.1f}% of evaluated stocks).\n\n")
        
        # Final recommendation
        f.write("### Recommendation\n\n")
        
        if overall_winner == "GARCH":
            f.write("Based on the benchmark results, the **GARCH ensemble** is recommended as the primary forecasting model for Indian stock volatility prediction. ")
            f.write("However, considering HAR-RV's better performance on some stocks, a combined approach using both models might provide more robust predictions across different market conditions.\n")
        elif overall_winner == "HAR-RV":
            f.write("Based on the benchmark results, the **HAR-RV ensemble** is recommended as the primary forecasting model for Indian stock volatility prediction. ")
            f.write("However, considering GARCH's better performance on some stocks, a combined approach using both models might provide more robust predictions across different market conditions.\n")
        else:
            f.write("Based on the benchmark results, both models show equivalent performance overall. ")
            f.write("This suggests that a **combined approach** using both GARCH and HAR-RV models would provide the most robust predictions across different stocks and market conditions.\n")
    
    print(f"Report generated: {output_file}")
    return str(output_file)

def main():
    # Load evaluation results
    results_df = load_evaluation_results()
    if results_df is None:
        print("No results to analyze. Run training and evaluation first.")
        return
    
    # Analyze performance
    analysis_results = analyze_model_performance(results_df)
    
    # Create visualizations
    viz_files = create_visualizations(results_df)
    
    # Generate report
    report_path = generate_report(results_df, analysis_results, viz_files)
    
    print(f"Analysis complete. Report generated at: {report_path}")

if __name__ == "__main__":
    main()
