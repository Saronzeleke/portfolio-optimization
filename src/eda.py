"""
Exploratory Data Analysis module for Task 1
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class EDA:
    """Class for performing exploratory data analysis"""
    
    def __init__(self, data_dict):
        self.data_dict = data_dict
        
    def plot_price_series(self, save_path=None):
        """Plot price series for all assets"""
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        
        for idx, (ticker, df) in enumerate(self.data_dict.items()):
            ax = axes[idx]
            ax.plot(df.index, df['Adj Close'], linewidth=2)
            ax.set_title(f'{ticker} - Adjusted Closing Price', fontsize=14, fontweight='bold')
            ax.set_ylabel('Price ($)', fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # Add rolling mean
            ax.plot(df.index, df['Rolling_Mean_20'], 'r--', alpha=0.7, label='20-day MA')
            ax.legend()
            
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_returns_distribution(self, save_path=None):
        """Plot distribution of returns"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for idx, (ticker, df) in enumerate(self.data_dict.items()):
            ax = axes[idx]
            returns = df['Daily_Return'].dropna()
            
            # Histogram with KDE
            sns.histplot(returns, bins=50, kde=True, ax=ax)
            ax.axvline(x=returns.mean(), color='r', linestyle='--', label=f'Mean: {returns.mean():.4f}')
            ax.axvline(x=returns.mean() + returns.std(), color='g', linestyle=':', alpha=0.7)
            ax.axvline(x=returns.mean() - returns.std(), color='g', linestyle=':', alpha=0.7)
            
            ax.set_title(f'{ticker} - Daily Returns Distribution', fontsize=12, fontweight='bold')
            ax.set_xlabel('Daily Return')
            ax.set_ylabel('Frequency')
            ax.legend()
            
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_volatility(self, save_path=None):
        """Plot volatility measures"""
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Focus on TSLA for volatility analysis
        tsla_df = self.data_dict['TSLA']
        
        # Rolling volatility
        rolling_vol = tsla_df['Daily_Return'].rolling(window=20).std() * np.sqrt(252)
        axes[0].plot(tsla_df.index, rolling_vol, linewidth=2, color='darkred')
        axes[0].set_title('TSLA - 20-day Rolling Annualized Volatility', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Volatility')
        axes[0].grid(True, alpha=0.3)
        
        # Bollinger Bands
        axes[1].plot(tsla_df.index, tsla_df['Adj Close'], label='Price', linewidth=2)
        axes[1].plot(tsla_df.index, tsla_df['Bollinger_High'], 'r--', alpha=0.7, label='Upper Band')
        axes[1].plot(tsla_df.index, tsla_df['Rolling_Mean_20'], 'g--', alpha=0.7, label='20-day MA')
        axes[1].plot(tsla_df.index, tsla_df['Bollinger_Low'], 'r--', alpha=0.7, label='Lower Band')
        axes[1].fill_between(tsla_df.index, tsla_df['Bollinger_Low'], tsla_df['Bollinger_High'], alpha=0.1)
        axes[1].set_title('TSLA - Price with Bollinger Bands', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Price ($)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def perform_stationarity_test(self):
        """Perform Augmented Dickey-Fuller test"""
        results = {}
        
        for ticker, df in self.data_dict.items():
            # Test on price
            price_test = adfuller(df['Adj Close'].dropna())
            # Test on returns
            return_test = adfuller(df['Daily_Return'].dropna())
            
            results[ticker] = {
                'Price_ADF_Statistic': price_test[0],
                'Price_p_value': price_test[1],
                'Price_Stationary': price_test[1] < 0.05,
                'Return_ADF_Statistic': return_test[0],
                'Return_p_value': return_test[1],
                'Return_Stationary': return_test[1] < 0.05
            }
            
        results_df = pd.DataFrame(results).T
        return results_df
    
    def detect_outliers(self, threshold=3):
        """Detect outliers using z-score method"""
        outliers_summary = {}
        
        for ticker, df in self.data_dict.items():
            returns = df['Daily_Return'].dropna()
            z_scores = np.abs((returns - returns.mean()) / returns.std())
            outliers = returns[z_scores > threshold]
            
            outliers_summary[ticker] = {
                'total_returns': len(returns),
                'outlier_count': len(outliers),
                'outlier_percentage': len(outliers) / len(returns) * 100,
                'largest_positive': returns.max(),
                'largest_negative': returns.min(),
                'outlier_dates': outliers.index.tolist() if len(outliers) > 0 else []
            }
            
        return outliers_summary
    
    def calculate_correlation_matrix(self):
        """Calculate correlation between assets"""
        # Combine returns
        returns_data = {}
        for ticker, df in self.data_dict.items():
            returns_data[ticker] = df['Daily_Return']
        
        returns_df = pd.DataFrame(returns_data).dropna()
        correlation_matrix = returns_df.corr()
        
        # Plot heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Correlation Matrix of Daily Returns', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        return correlation_matrix
    
    def generate_summary_statistics(self):
        """Generate summary statistics for all assets"""
        summary_stats = {}
        
        for ticker, df in self.data_dict.items():
            price = df['Price']
            returns = df['Return'].dropna()
            
            stats = {
                'Start_Date': df.index.min(),
                'End_Date': df.index.max(),
                'Total_Days': len(df),
                'Initial_Price': price.iloc[0],
                'Final_Price': price.iloc[-1],
                'Total_Return': (price.iloc[-1] - price.iloc[0]) / price.iloc[0] * 100,
                'Annualized_Return': ((price.iloc[-1] / price.iloc[0]) ** (252/len(df)) - 1) * 100,
                'Mean_Daily_Return': returns.mean() * 100,
                'Std_Daily_Return': returns.std() * 100,
                'Skewness': returns.skew(),
                'Kurtosis': returns.kurtosis(),
                'Min_Daily_Return': returns.min() * 100,
                'Max_Daily_Return': returns.max() * 100
            }
            
            summary_stats[ticker] = stats
            
        return pd.DataFrame(summary_stats).T