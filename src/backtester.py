"""
Strategy backtesting module for Task 5
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class Backtester:
    """Class for backtesting portfolio strategies"""
    
    def __init__(self, returns_data, prices_data, start_date, end_date):
        """
        Initialize backtester
        
        Parameters:
        -----------
        returns_data: DataFrame with daily returns
        prices_data: DataFrame with prices
        start_date: Backtest start date
        end_date: Backtest end date
        """
        self.returns_data = returns_data.loc[start_date:end_date]
        self.prices_data = prices_data.loc[start_date:end_date]
        self.start_date = start_date
        self.end_date = end_date
        
    def calculate_metrics(self, portfolio_returns, benchmark_returns=None, risk_free_rate=0.02):
        """Calculate performance metrics"""
        # Calculate cumulative returns
        cumulative_returns = (1 + portfolio_returns).cumprod()
        
        # Total return
        total_return = cumulative_returns.iloc[-1] - 1
        
        # Annualized return
        days = len(portfolio_returns)
        annualized_return = (1 + total_return) ** (252 / days) - 1
        
        # Sharpe ratio (annualized)
        excess_returns = portfolio_returns - risk_free_rate / 252
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / portfolio_returns.std()
        
        # Maximum drawdown
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Volatility (annualized)
        volatility = portfolio_returns.std() * np.sqrt(252)
        
        metrics = {
            'Total_Return': total_return,
            'Annualized_Return': annualized_return,
            'Sharpe_Ratio': sharpe_ratio,
            'Max_Drawdown': max_drawdown,
            'Volatility': volatility,
            'Cumulative_Returns': cumulative_returns
        }
        
        # Calculate alpha and beta if benchmark is provided
        if benchmark_returns is not None:
            covariance = portfolio_returns.cov(benchmark_returns)
            benchmark_variance = benchmark_returns.var()
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
            
            # Alpha (Jensen's Alpha)
            alpha = annualized_return - (risk_free_rate + beta * (benchmark_returns.mean() * 252 - risk_free_rate))
            
            metrics.update({
                'Alpha': alpha,
                'Beta': beta,
                'Information_Ratio': (annualized_return - benchmark_returns.mean() * 252) / \
                                   (np.std(portfolio_returns - benchmark_returns) * np.sqrt(252))
            })
            
        return metrics
    
    def simulate_strategy(self, initial_weights, rebalance_freq='M'):
        """
        Simulate strategy with given initial weights
        
        Parameters:
        -----------
        initial_weights: Dictionary of initial weights
        rebalance_freq: Rebalancing frequency ('M' for monthly, 'Q' for quarterly, None for no rebalancing)
        """
        # Convert weights to array in correct order
        assets = list(initial_weights.keys())
        weights = np.array([initial_weights[asset] for asset in assets])
        
        # Initialize portfolio value
        initial_value = 10000  # Start with $10,000
        portfolio_value = initial_value
        portfolio_values = [portfolio_value]
        
        # Get returns for backtest period
        returns_subset = self.returns_data[assets]
        
        # Rebalancing logic
        if rebalance_freq is None:
            # No rebalancing - let weights drift
            for i in range(len(returns_subset)):
                daily_return = np.dot(weights, returns_subset.iloc[i])
                portfolio_value *= (1 + daily_return)
                portfolio_values.append(portfolio_value)
                
                # Update weights based on performance
                asset_values = weights * portfolio_value * (1 + returns_subset.iloc[i])
                weights = asset_values / asset_values.sum()
        else:
            # With rebalancing
            portfolio_values = [initial_value]
            current_weights = weights.copy()
            
            # Group by rebalancing period
            if rebalance_freq == 'M':
                groups = returns_subset.groupby(pd.Grouper(freq='M'))
            elif rebalance_freq == 'Q':
                groups = returns_subset.groupby(pd.Grouper(freq='Q'))
            else:
                groups = [(None, returns_subset)]  # Single period
            
            for period, period_returns in groups:
                if len(period_returns) == 0:
                    continue
                    
                # Apply current weights for the period
                for i in range(len(period_returns)):
                    daily_return = np.dot(current_weights, period_returns.iloc[i])
                    portfolio_value *= (1 + daily_return)
                    portfolio_values.append(portfolio_value)
                
                # Rebalance at end of period
                current_weights = weights.copy()
        
        # Calculate portfolio returns
        portfolio_values_series = pd.Series(portfolio_values[1:], index=returns_subset.index)
        portfolio_returns = portfolio_values_series.pct_change().fillna(0)
        
        return portfolio_returns, portfolio_values_series
    
    def create_benchmark(self, spy_weight=0.6, bnd_weight=0.4):
        """Create 60/40 SPY/BND benchmark portfolio"""
        if 'SPY' not in self.returns_data.columns or 'BND' not in self.returns_data.columns:
            raise ValueError("SPY and BND must be in returns data")
        
        benchmark_returns = (
            spy_weight * self.returns_data['SPY'] +
            bnd_weight * self.returns_data['BND']
        )
        
        return benchmark_returns
    
    def plot_comparison(self, strategy_returns, benchmark_returns, save_path=None):
        """Plot cumulative returns comparison"""
        # Calculate cumulative returns
        strategy_cumulative = (1 + strategy_returns).cumprod()
        benchmark_cumulative = (1 + benchmark_returns).cumprod()
        
        plt.figure(figsize=(14, 8))
        
        # Plot cumulative returns
        plt.plot(strategy_cumulative.index, strategy_cumulative.values,
                label='Optimized Portfolio', linewidth=2.5, color='darkblue')
        plt.plot(benchmark_cumulative.index, benchmark_cumulative.values,
                label='60/40 SPY/BND Benchmark', linewidth=2.5, color='darkred', alpha=0.8)
        
        # Fill between for visual contrast
        plt.fill_between(strategy_cumulative.index,
                        strategy_cumulative.values,
                        benchmark_cumulative.values,
                        where=(strategy_cumulative.values > benchmark_cumulative.values),
                        alpha=0.2, color='green', label='Outperformance')
        
        plt.fill_between(strategy_cumulative.index,
                        strategy_cumulative.values,
                        benchmark_cumulative.values,
                        where=(strategy_cumulative.values <= benchmark_cumulative.values),
                        alpha=0.2, color='red', label='Underperformance')
        
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Cumulative Return (Normalized to 1)', fontsize=12)
        plt.title('Strategy vs Benchmark Performance (Jan 2025 - Jan 2026)', 
                 fontsize=16, fontweight='bold')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        # Add final value annotations
        final_strategy = strategy_cumulative.iloc[-1]
        final_benchmark = benchmark_cumulative.iloc[-1]
        
        plt.annotate(f'Strategy: {final_strategy:.2f}x',
                    xy=(strategy_cumulative.index[-1], final_strategy),
                    xytext=(-100, 20),
                    textcoords='offset points',
                    fontsize=10,
                    fontweight='bold',
                    color='darkblue')
        
        plt.annotate(f'Benchmark: {final_benchmark:.2f}x',
                    xy=(benchmark_cumulative.index[-1], final_benchmark),
                    xytext=(-100, -20),
                    textcoords='offset points',
                    fontsize=10,
                    fontweight='bold',
                    color='darkred')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_performance_report(self, strategy_metrics, benchmark_metrics):
        """Generate comprehensive performance report"""
        report_data = {
            'Metric': ['Total Return', 'Annualized Return', 'Sharpe Ratio', 
                      'Max Drawdown', 'Volatility', 'Alpha', 'Beta'],
            'Strategy': [
                f"{strategy_metrics['Total_Return']:.2%}",
                f"{strategy_metrics['Annualized_Return']:.2%}",
                f"{strategy_metrics['Sharpe_Ratio']:.2f}",
                f"{strategy_metrics['Max_Drawdown']:.2%}",
                f"{strategy_metrics['Volatility']:.2%}",
                f"{strategy_metrics.get('Alpha', 'N/A')}",
                f"{strategy_metrics.get('Beta', 'N/A')}"
            ],
            'Benchmark': [
                f"{benchmark_metrics['Total_Return']:.2%}",
                f"{benchmark_metrics['Annualized_Return']:.2%}",
                f"{benchmark_metrics['Sharpe_Ratio']:.2f}",
                f"{benchmark_metrics['Max_Drawdown']:.2%}",
                f"{benchmark_metrics['Volatility']:.2%}",
                'N/A',
                'N/A'
            ]
        }
        
        report_df = pd.DataFrame(report_data)
        
        # Calculate outperformance
        strategy_win = 0
        benchmark_win = 0
        for metric in ['Total_Return', 'Annualized_Return', 'Sharpe_Ratio']:
            if metric in strategy_metrics and metric in benchmark_metrics:
                if strategy_metrics[metric] > benchmark_metrics[metric]:
                    strategy_win += 1
                else:
                    benchmark_win += 1
        
        conclusion = f"""
        Performance Summary:
        - Strategy outperforms benchmark in {strategy_win} out of 3 key metrics
        - Benchmark outperforms strategy in {benchmark_win} out of 3 key metrics
        
        Key Insights:
        1. {'✅ Strategy shows better risk-adjusted returns (Sharpe Ratio)' 
            if strategy_metrics['Sharpe_Ratio'] > benchmark_metrics['Sharpe_Ratio'] 
            else '⚠️ Benchmark has better risk-adjusted returns'}
        2. {'✅ Strategy has higher total returns' 
            if strategy_metrics['Total_Return'] > benchmark_metrics['Total_Return'] 
            else '⚠️ Benchmark has higher total returns'}
        3. {'✅ Strategy manages drawdowns better' 
            if abs(strategy_metrics['Max_Drawdown']) < abs(benchmark_metrics['Max_Drawdown']) 
            else '⚠️ Benchmark has lower maximum drawdown'}
        
        Recommendation:
        {'The optimized portfolio strategy appears viable and worth further consideration.' 
        if strategy_win >= 2 else 
        'The benchmark portfolio may be more suitable for risk-averse investors.'}
        """
        
        return report_df, conclusion