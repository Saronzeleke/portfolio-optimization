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
    
    def __init__(self, cleaned_data):
        """
        cleaned_data: dict of DataFrames from DataLoader.clean_data()
        Each DataFrame contains:
            - 'Adj Close' (price)
            - 'Daily_Return' (pct change)
            - 'Log_Return'
            - Rolling stats and Bollinger Bands
        """
        self.data_dict = cleaned_data
        
    def plot_price_series(self, save_path=None):
        """Plot price series for all assets"""
        fig, axes = plt.subplots(len(self.data_dict), 1, figsize=(14, 4*len(self.data_dict)))
        
        for idx, (ticker, df) in enumerate(self.data_dict.items()):
            ax = axes[idx] if len(self.data_dict) > 1 else axes
            price_col = next((c for c in df.columns if 'Adj Close' in c), None)
            if price_col is None:
               price_col = next((c for c in df.columns if 'Close' in c), None)

            ax.plot(df.index, df[price_col], linewidth=2, label='Price')

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
        plt.close()

    def plot_returns_distribution(self, save_path=None):
        """Plot distribution of returns"""
        fig, axes = plt.subplots(1, len(self.data_dict), figsize=(5*len(self.data_dict), 5))
        
        for idx, (ticker, df) in enumerate(self.data_dict.items()):
            ax = axes[idx] if len(self.data_dict) > 1 else axes
            returns = df['Daily_Return'].dropna()
            
            sns.histplot(returns, bins=50, kde=True, ax=ax)
            ax.axvline(returns.mean(), color='r', linestyle='--', label=f'Mean: {returns.mean():.4f}')
            ax.axvline(returns.mean() + returns.std(), color='g', linestyle=':', alpha=0.7)
            ax.axvline(returns.mean() - returns.std(), color='g', linestyle=':', alpha=0.7)
            
            ax.set_title(f'{ticker} - Daily Returns Distribution', fontsize=12, fontweight='bold')
            ax.set_xlabel('Daily Return')
            ax.set_ylabel('Frequency')
            ax.legend()
            
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

    def plot_volatility(self, save_path=r'C:\Users\admin\portfolio-optimization\reports'):
        """Plot volatility measures"""
        import os
        os.makedirs(save_path, exist_ok=True)
        for ticker, df in self.data_dict.items():
            # Rolling volatility
            rolling_vol = df['Daily_Return'].rolling(window=20).std() * np.sqrt(252)
            plt.figure(figsize=(14,4))
            plt.plot(df.index, rolling_vol, linewidth=2, color='darkred')
            plt.title(f'{ticker} - 20-day Rolling Annualized Volatility', fontsize=14, fontweight='bold')
            plt.ylabel('Volatility')
            plt.grid(True, alpha=0.3)
            file_path = os.path.join(save_path, 'volatility.png')
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            plt.show()
            plt.close()

            # Bollinger Bands
            plt.figure(figsize=(14,4))
            price_col = next((c for c in df.columns if 'Adj Close' in c), None)
            if price_col is None:
              price_col = next((c for c in df.columns if 'Close' in c), None)
            plt.plot(df.index, df[price_col], label='Price', linewidth=2)
            price_test = adfuller(df[price_col].dropna())
            plt.plot(df.index, df['Bollinger_High'], 'r--', alpha=0.7, label='Upper Band')
            plt.plot(df.index, df['Rolling_Mean_20'], 'g--', alpha=0.7, label='20-day MA')
            plt.plot(df.index, df['Bollinger_Low'], 'r--', alpha=0.7, label='Lower Band')
            plt.fill_between(df.index, df['Bollinger_Low'], df['Bollinger_High'], alpha=0.1)
            plt.title(f'{ticker} - Price with Bollinger Bands', fontsize=14, fontweight='bold')
            plt.ylabel('Price ($)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            file_path = os.path.join(save_path, 'Price_with_Bollinger_Bands.png')
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            plt.show()
            plt.close()

    def perform_stationarity_test(self):
        """Perform Augmented Dickey-Fuller test"""
        results = {}
        
        for ticker, df in self.data_dict.items():
            price_col = next((c for c in df.columns if 'Adj Close' in c), None)
            if price_col is None:
              price_col = next((c for c in df.columns if 'Close' in c), None)

            price_test = adfuller(df[price_col].dropna())

            return_test = adfuller(df['Daily_Return'].dropna())
            
            results[ticker] = {
                'Price_ADF_Statistic': price_test[0],
                'Price_p_value': price_test[1],
                'Price_Stationary': price_test[1] < 0.05,
                'Return_ADF_Statistic': return_test[0],
                'Return_p_value': return_test[1],
                'Return_Stationary': return_test[1] < 0.05
            }
            
        return pd.DataFrame(results).T
    
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
        returns_data = {ticker: df['Daily_Return'] for ticker, df in self.data_dict.items()}
        returns_df = pd.DataFrame(returns_data).dropna()
        correlation_matrix = returns_df.corr()
        
        plt.figure(figsize=(10,8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                    square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Correlation Matrix of Daily Returns', fontsize=16, fontweight='bold')
        plt.tight_layout()
        import os 
        save_path=r'C:\Users\admin\portfolio-optimization\reports'
        file_path = os.path.join(save_path, 'Correlation_Matrix.png')
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()

        return correlation_matrix
    
    def generate_summary_statistics(self):
        """Generate summary statistics for all assets"""
        summary_stats = {}
        
        for ticker, df in self.data_dict.items():
            price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
            price = df[price_col]

            returns = df['Daily_Return'].dropna()
            
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
