"""
Portfolio Optimization Package

A comprehensive solution for time series forecasting and portfolio optimization
"""

__version__ = '1.0.0'
__author__ = 'Data Science Team'

from .data_loader import DataLoader
from .eda import EDA
from .forecasting import TimeSeriesForecaster
from .portfolio_optimizer import PortfolioOptimizer
from .backtester import Backtester

__all__ = [
    'DataLoader',
    'EDA',
    'TimeSeriesForecaster',
    'PortfolioOptimizer',
    'Backtester'
]