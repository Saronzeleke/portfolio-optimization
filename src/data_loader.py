import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class DataLoader:
    """Class to handle data extraction and preprocessing for portfolio optimization pipeline"""

    def __init__(self):
        self.tickers = ['TSLA', 'BND', 'SPY']
        self.start_date = '2015-01-01'
        self.end_date = '2026-01-15'
        self.max_retries = 3  # Retry downloads if they fail

    def download_data(self):
        """Download data from Yahoo Finance with retries and safe handling"""
        print("Downloading data from Yahoo Finance...")
        data_dict = {}

        for ticker in self.tickers:
            attempts = 0
            while attempts < self.max_retries:
                try:
                    stock = yf.download(ticker, start=self.start_date, end=self.end_date)

                    # Check if download returned a valid DataFrame
                    if stock is None or stock.empty:
                        raise ValueError(f"No data downloaded for {ticker}")

                    # Flatten MultiIndex columns if exist
                    if (stock.columns, pd.MultiIndex):
                        stock.columns = [col[0] for col in stock.columns]

                    stock['Ticker'] = ticker
                    stock['Date'] = stock.index
                    data_dict[ticker] = stock
                    print(f"Downloaded {ticker}: {len(stock)} records")
                    break  # success, exit retry loop
                except Exception as e:
                    attempts += 1
                    print(f"Attempt {attempts} failed for {ticker}: {e}")
                    if attempts == self.max_retries:
                        print(f"Skipping {ticker} after {self.max_retries} failed attempts.")

        if not data_dict:
            raise ValueError("No tickers were successfully downloaded. Cannot proceed.")

        return data_dict

    def clean_data(self, data_dict):
        """Clean and preprocess data, handling MultiIndex columns"""
        cleaned_data = {}

        for ticker, df in data_dict.items():
            if df.empty:
                print(f"Skipping {ticker}: no data")
                continue

            df_clean = df.copy()
            missing_before = df_clean.isnull().sum().sum()

            # Flatten columns if needed
            if isinstance(df_clean.columns, pd.MultiIndex):
                df_clean.columns = [col[0] for col in df_clean.columns]

            # Use 'Adj Close' if exists, else 'Close'
            adj_close_col = 'Adj Close' if 'Adj Close' in df_clean.columns else 'Close'

            # Fill minor gaps
            df_clean = df_clean.ffill().bfill()

            # Calculate returns
            df_clean['Daily_Return'] = df_clean[adj_close_col].pct_change()
            df_clean['Log_Return'] = np.log(df_clean[adj_close_col] / df_clean[adj_close_col].shift(1))

            # Rolling statistics for volatility & Bollinger Bands
            df_clean['Rolling_Mean_20'] = df_clean[adj_close_col].rolling(window=20).mean()
            df_clean['Rolling_Std_20'] = df_clean[adj_close_col].rolling(window=20).std()
            df_clean['Bollinger_High'] = df_clean['Rolling_Mean_20'] + 2 * df_clean['Rolling_Std_20']
            df_clean['Bollinger_Low'] = df_clean['Rolling_Mean_20'] - 2 * df_clean['Rolling_Std_20']

            # Drop initial NaNs
            df_clean = df_clean.dropna()
            cleaned_data[ticker] = df_clean
            print(f"Cleaned {ticker}: Removed {missing_before} missing values, {len(df_clean)} rows remaining")

        if not cleaned_data:
            raise ValueError("No valid data available after cleaning. Cannot proceed.")

        return cleaned_data

    def calculate_risk_metrics(self, df, confidence_level=0.95):
        """Calculate risk metrics for a single asset"""
        adj_close_col = next((c for c in df.columns if 'Adj Close' in c), None)
        if adj_close_col is None:
          adj_close_col = next((c for c in df.columns if 'Close' in c), None)
        returns = df['Daily_Return'].dropna()

        # Historical Value at Risk (VaR)
        var_hist = np.percentile(returns, (1 - confidence_level) * 100)

        # Parametric VaR assuming normal distribution
        var_param = returns.mean() + returns.std() * np.percentile(np.random.randn(10000),
                                                                  (1 - confidence_level) * 100)

        # Annualized Sharpe Ratio
        risk_free_rate = 0.02  # 2% assumed annual
        excess_returns = returns - risk_free_rate / 252
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / returns.std() if returns.std() > 0 else 0

        metrics = {
            'Mean_Return': returns.mean() * 252,  # Annualized
            'Volatility': returns.std() * np.sqrt(252),  # Annualized
            'VaR_Historical': var_hist,
            'VaR_Parametric': var_param,
            'Sharpe_Ratio': sharpe_ratio,
            'Max_Drawdown': self.calculate_max_drawdown(df[returns.name.replace('Daily_Return', adj_close_col)] if 'Daily_Return' in df.columns else df.columns[0])
        }
        return metrics

    def calculate_max_drawdown(self, prices):
        """Calculate maximum drawdown"""
        peak = prices.expanding(min_periods=1).max()
        drawdown = (prices - peak) / peak
        return drawdown.min()

    def get_combined_data(self, cleaned_data):
        """Combine all ticker data into a single DataFrame for portfolio analysis"""
        combined = []
        for ticker, df in cleaned_data.items():
            if df.empty:
                continue

            adj_close_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
            df_temp = df[['Date', adj_close_col, 'Daily_Return']].copy()
            df_temp['Ticker'] = ticker
            df_temp = df_temp.rename(columns={adj_close_col: 'Price', 'Daily_Return': 'Return'})
            combined.append(df_temp)

        if not combined:
            raise ValueError("No data available to combine for portfolio analysis.")

        combined_df = pd.concat(combined, ignore_index=True)
        return combined_df


# import yfinance as yf
# import pandas as pd
# import numpy as np
# from datetime import datetime, timedelta
# import warnings
# warnings.filterwarnings('ignore')

# class DataLoader:
#     """Class to handle data extraction and preprocessing"""
    
#     def __init__(self):
#         self.tickers = ['TSLA', 'BND', 'SPY']
#         self.start_date = '2015-01-01'
#         self.end_date = '2026-01-15'
        
#     def download_data(self):
#         """Download data from Yahoo Finance"""
#         print("Downloading data from Yahoo Finance...")
        
#         data_dict = {}
#         for ticker in self.tickers:
#             try:
#                 stock = yf.download(ticker, start=self.start_date, end=self.end_date)
#                 stock['Ticker'] = ticker
#                 stock['Date'] = stock.index
#                 data_dict[ticker] = stock
#                 print(f"Downloaded {ticker}: {len(stock)} records")
#             except Exception as e:
#                 print(f"Error downloading {ticker}: {e}")
                
#         return data_dict
    
#     def clean_data(self, data_dict):
#         """Clean and preprocess data"""
#         cleaned_data = {}
        
#         for ticker, df in data_dict.items():
#             # Create a copy
#             df_clean = df.copy()
            
#             # Check for missing values
#             missing_before = df_clean.isnull().sum().sum()
            
#             # Forward fill for minor gaps, then backward fill
#             df_clean = df_clean.ffill().bfill()
            
#             # Calculate returns
#             df_clean['Daily_Return'] = df_clean['Adj Close'].pct_change()
#             df_clean['Log_Return'] = np.log(df_clean['Adj Close'] / df_clean['Adj Close'].shift(1))
            
#             # Calculate rolling statistics
#             df_clean['Rolling_Mean_20'] = df_clean['Adj Close'].rolling(window=20).mean()
#             df_clean['Rolling_Std_20'] = df_clean['Adj Close'].rolling(window=20).std()
#             df_clean['Bollinger_High'] = df_clean['Rolling_Mean_20'] + (df_clean['Rolling_Std_20'] * 2)
#             df_clean['Bollinger_Low'] = df_clean['Rolling_Mean_20'] - (df_clean['Rolling_Std_20'] * 2)
            
#             # Remove first row with NaN returns
#             df_clean = df_clean.dropna()
            
#             cleaned_data[ticker] = df_clean
#             print(f"Cleaned {ticker}: Removed {missing_before} missing values")
            
#         return cleaned_data
    
#     def calculate_risk_metrics(self, df, confidence_level=0.95):
#         """Calculate risk metrics for a single asset"""
#         returns = df['Daily_Return'].dropna()
        
#         # Value at Risk (Historical)
#         var_hist = np.percentile(returns, (1 - confidence_level) * 100)
        
#         # Parametric VaR (assuming normal distribution)
#         var_param = returns.mean() + returns.std() * np.percentile(np.random.randn(10000), (1 - confidence_level) * 100)
        
#         # Sharpe Ratio (annualized)
#         risk_free_rate = 0.02  # Assume 2% risk-free rate
#         excess_returns = returns - risk_free_rate/252  # Daily risk-free rate
#         sharpe_ratio = np.sqrt(252) * excess_returns.mean() / returns.std() if returns.std() > 0 else 0
        
#         metrics = {
#             'Mean_Return': returns.mean() * 252,  # Annualized
#             'Volatility': returns.std() * np.sqrt(252),  # Annualized
#             'VaR_Historical': var_hist,
#             'VaR_Parametric': var_param,
#             'Sharpe_Ratio': sharpe_ratio,
#             'Max_Drawdown': self.calculate_max_drawdown(df['Adj Close'])
#         }
        
#         return metrics
    
#     def calculate_max_drawdown(self, prices):
#         """Calculate maximum drawdown"""
#         peak = prices.expanding(min_periods=1).max()
#         drawdown = (prices - peak) / peak
#         return drawdown.min()
    
#     def get_combined_data(self, cleaned_data):
#         """Combine all ticker data into a single DataFrame"""
#         combined = []
#         for ticker, df in cleaned_data.items():
#             df_temp = df[['Date', 'Adj Close', 'Daily_Return']].copy()
#             df_temp['Ticker'] = ticker
#             df_temp = df_temp.rename(columns={'Adj Close': 'Price', 'Daily_Return': 'Return'})
#             combined.append(df_temp)
        
#         combined_df = pd.concat(combined, ignore_index=True)
#         return combined_df