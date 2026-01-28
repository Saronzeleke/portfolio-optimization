"""
Unit tests for the portfolio optimization models
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import DataLoader
from src.portfolio_optimizer import PortfolioOptimizer
from src.backtester import Backtester

class TestDataLoader(unittest.TestCase):
    """Test DataLoader class"""
    
    def setUp(self):
        self.loader = DataLoader()
        
    def test_data_loader_initialization(self):
        self.assertEqual(self.loader.tickers, ['TSLA', 'BND', 'SPY'])
        self.assertEqual(self.loader.start_date, '2015-01-01')
        
    def test_risk_metrics_calculation(self):
        # Create test data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        prices = np.random.normal(100, 10, 100).cumsum()
        df = pd.DataFrame({
            'Adj Close': prices,
            'Daily_Return': np.random.normal(0, 0.01, 100)
        }, index=dates)
        
        metrics = self.loader.calculate_risk_metrics(df)
        
        self.assertIn('Mean_Return', metrics)
        self.assertIn('Volatility', metrics)
        self.assertIn('Sharpe_Ratio', metrics)
        self.assertIn('VaR_Historical', metrics)
        
class TestPortfolioOptimizer(unittest.TestCase):
    """Test PortfolioOptimizer class"""
    
    def setUp(self):
        # Create synthetic returns data
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        returns_data = pd.DataFrame({
            'TSLA': np.random.normal(0.001, 0.02, 100),
            'SPY': np.random.normal(0.0005, 0.01, 100),
            'BND': np.random.normal(0.0002, 0.005, 100)
        }, index=dates)
        
        self.optimizer = PortfolioOptimizer(returns_data)
        
    def test_covariance_matrix_shape(self):
        cov_matrix = self.optimizer.cov_matrix
        self.assertEqual(cov_matrix.shape, (3, 3))
        
    def test_portfolio_stats_calculation(self):
        weights = [0.4, 0.4, 0.2]
        stats = self.optimizer.calculate_portfolio_stats(weights)
        
        self.assertIn('return', stats)
        self.assertIn('volatility', stats)
        self.assertIn('sharpe_ratio', stats)
        
    def test_optimize_portfolio(self):
        weights, stats = self.optimizer.optimize_portfolio()
        
        self.assertAlmostEqual(np.sum(weights), 1.0, places=5)
        self.assertGreaterEqual(stats['return'], 0)
        self.assertGreaterEqual(stats['volatility'], 0)
        
class TestBacktester(unittest.TestCase):
    """Test Backtester class"""
    
    def setUp(self):
        # Create synthetic data
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        
        returns_data = pd.DataFrame({
            'TSLA': np.random.normal(0.001, 0.02, 100),
            'SPY': np.random.normal(0.0005, 0.01, 100),
            'BND': np.random.normal(0.0002, 0.005, 100)
        }, index=dates)
        
        prices_data = pd.DataFrame({
            'TSLA': 100 * (1 + returns_data['TSLA']).cumprod(),
            'SPY': 100 * (1 + returns_data['SPY']).cumprod(),
            'BND': 100 * (1 + returns_data['BND']).cumprod()
        }, index=dates)
        
        self.backtester = Backtester(
            returns_data, 
            prices_data,
            start_date='2024-01-01',
            end_date='2024-04-10'
        )
        
    def test_benchmark_creation(self):
        benchmark_returns = self.backtester.create_benchmark(spy_weight=0.6, bnd_weight=0.4)
        self.assertEqual(len(benchmark_returns), len(self.backtester.returns_data))
        
    def test_metrics_calculation(self):
        test_returns = pd.Series(np.random.normal(0.0005, 0.01, 100))
        metrics = self.backtester.calculate_metrics(test_returns)
        
        self.assertIn('Total_Return', metrics)
        self.assertIn('Sharpe_Ratio', metrics)
        self.assertIn('Max_Drawdown', metrics)
        
    def test_strategy_simulation(self):
        initial_weights = {'TSLA': 0.5, 'SPY': 0.3, 'BND': 0.2}
        strategy_returns, strategy_values = self.backtester.simulate_strategy(
            initial_weights,
            rebalance_freq=None
        )
        
        self.assertEqual(len(strategy_returns), len(self.backtester.returns_data))


if __name__ == '__main__':
    unittest.main()