import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta

from src.data_loader import DataLoader
from src.eda import EDA
from src.forecasting import TimeSeriesForecaster
from src.portfolio_optimizer import PortfolioOptimizer
from src.backtester import Backtester

def main():
    """Main execution function for the complete pipeline"""
    print("=" * 80)
    print("PORTFOLIO OPTIMIZATION PIPELINE")
    print("=" * 80)
    
    # -----------------------
    # Task 1: Data Preprocessing & EDA
    # -----------------------
    print("\nTASK 1: DATA PREPROCESSING AND EDA")
    
    data_loader = DataLoader()
    raw_data = data_loader.download_data()
    cleaned_data = data_loader.clean_data(raw_data)
    
    # Ensure at least TSLA is available for forecasting
    if 'TSLA' not in cleaned_data:
        raise ValueError("TSLA data is missing. Cannot proceed with forecasting.")
    
    # EDA
    eda = EDA(cleaned_data)
    summary_stats = eda.generate_summary_statistics()
    print("\nSummary Statistics:")
    print(summary_stats.round(4))
    
    # Risk metrics
    risk_metrics = {}
    for ticker, df in cleaned_data.items():
        metrics = data_loader.calculate_risk_metrics(df)
        risk_metrics[ticker] = metrics
    risk_df = pd.DataFrame(risk_metrics).T
    print("\nRisk Metrics:")
    print(risk_df.round(4))
    
    # Stationarity tests
    stationarity_results = eda.perform_stationarity_test()
    print("\nStationarity Test Results (ADF):")
    print(stationarity_results)
    
    # Visualizations
    eda.plot_price_series(save_path=r'C:/Users/admin/portfolio-optimization/data/processed/price_series.png')
    eda.plot_returns_distribution(save_path=r'C:/Users/admin/portfolio-optimization/data/processed/returns_distribution.png')
    eda.plot_volatility(save_path=r'C:/Users/admin/portfolio-optimization/data/processed/volatility_analysis.png')
    eda.calculate_correlation_matrix()
    
    # -----------------------
    # Task 2: Time Series Forecasting
    # -----------------------
    print("\nTASK 2: TIME SERIES FORECASTING")
    
    tsla_data = cleaned_data['TSLA']
    forecaster = TimeSeriesForecaster(tsla_data)
    
    # Prepare data
    train_scaled, test_scaled, train_raw, test_raw = forecaster.prepare_data(test_size=0.2)
    
    # ARIMA
    arima_model = forecaster.build_arima_model(train_scaled.flatten(), seasonal=False)
    arima_forecast = forecaster.forecast_arima(arima_model, steps=len(test_scaled))
    arima_forecast_rescaled = forecaster.scaler.inverse_transform(arima_forecast.reshape(-1,1)).flatten()
    arima_metrics = forecaster.calculate_metrics(test_raw.flatten(), arima_forecast_rescaled)
    
    # LSTM
    sequence_length = 60
    X_train, y_train = forecaster.prepare_sequences(train_scaled, sequence_length)
    X_test, y_test = forecaster.prepare_sequences(test_scaled, sequence_length)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1],1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1],1))
    lstm_model, history = forecaster.train_lstm(X_train, y_train, X_val=X_test, y_val=y_test, epochs=50, batch_size=32)
    
    last_sequence = train_scaled[-sequence_length:].flatten()
    lstm_forecast = forecaster.forecast_lstm(lstm_model, last_sequence, steps=len(test_scaled))
    lstm_forecast_rescaled = forecaster.scaler.inverse_transform(lstm_forecast.reshape(-1,1)).flatten()
    lstm_metrics = forecaster.calculate_metrics(test_raw.flatten(), lstm_forecast_rescaled)
    
    # Compare models
    best_model = 'ARIMA' if arima_metrics['RMSE'] < lstm_metrics['RMSE'] else 'LSTM'
    print(f"Best performing model: {best_model}")
    
    # -----------------------
    # Task 3: Forecast Future Market Trends (12 months)
    # -----------------------
    print("\nTASK 3: FUTURE MARKET TRENDS FORECASTING")
    
    if best_model == 'ARIMA':
        full_data = np.concatenate([train_scaled, test_scaled])
        final_model = forecaster.build_arima_model(full_data.flatten(), seasonal=False)
        future_forecast = forecaster.forecast_arima(final_model, steps=252)
    else:
        full_data_scaled = np.concatenate([train_scaled, test_scaled])
        last_seq_full = full_data_scaled[-sequence_length:].flatten()
        future_forecast = forecaster.forecast_lstm(lstm_model, last_seq_full, steps=252)
    
    future_forecast_rescaled = forecaster.scaler.inverse_transform(future_forecast.reshape(-1,1)).flatten()
    last_date = tsla_data.index[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=252, freq='B')
    
    # Forecast confidence intervals
    forecast_std = np.std(future_forecast_rescaled)
    upper_bound = future_forecast_rescaled + 1.96*forecast_std
    lower_bound = future_forecast_rescaled - 1.96*forecast_std
    
    # Plot
    plt.figure(figsize=(14,8))
    historical_dates = tsla_data.index[-504:]
    historical_prices = tsla_data.loc[historical_dates,'Adj Close']
    plt.plot(historical_dates, historical_prices, label='Historical', color='blue', linewidth=2)
    plt.plot(future_dates, future_forecast_rescaled, label='Forecast', color='red', linewidth=2)
    plt.fill_between(future_dates, lower_bound, upper_bound, color='red', alpha=0.2, label='95% CI')
    plt.axvline(x=last_date, color='gray', linestyle='--', alpha=0.7)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title('TSLA 12-Month Forecast')
    plt.tight_layout()
    plt.savefig(r'C:/Users/admin/portfolio-optimization/data/processed/forecast_with_ci.png', dpi=300)
    plt.show()
    
    # -----------------------
    # Task 4: Portfolio Optimization
    # -----------------------
    print("\nTASK 4: PORTFOLIO OPTIMIZATION")
    
    returns_list = []
    for ticker, df in cleaned_data.items():
        returns_list.append(pd.DataFrame({'Date': df.index, 'Ticker': ticker, 'Return': df['Daily_Return']}))
    
    all_returns = pd.concat(returns_list, ignore_index=True)
    returns_matrix = all_returns.pivot(index='Date', columns='Ticker', values='Return').dropna()
    
    tsla_forecast_return = (future_forecast_rescaled[-1] - future_forecast_rescaled[0]) / future_forecast_rescaled[0]
    tsla_annualized_return = (1 + tsla_forecast_return)**1 - 1
    expected_returns = {'TSLA': tsla_annualized_return}
    
    for t in ['BND','SPY']:
        if t in returns_matrix.columns:
            expected_returns[t] = returns_matrix[t].mean()*252
    
    optimizer = PortfolioOptimizer(returns_matrix, expected_returns)
    results_df, max_sharpe, min_vol = optimizer.generate_efficient_frontier(num_portfolios=10000)
    optimizer.plot_efficient_frontier(results_df, max_sharpe, min_vol, save_path=r'C:/Users/admin/portfolio-optimization/data/processed/efficient_frontier.png')
    optimizer.plot_covariance_heatmap(save_path=r'C:/Users/admin/portfolio-optimization/data/processed/covariance_heatmap.png')
    
    portfolio_summary = optimizer.recommend_portfolio(risk_tolerance='medium')
    optimal_weights = dict(zip(portfolio_summary['Weights']['Asset'], portfolio_summary['Weights']['Weight']))
    
    print("\nRecommended Portfolio:")
    print(portfolio_summary)
    
    # -----------------------
    # Task 5: Strategy Backtesting
    # -----------------------
    print("\nTASK 5: STRATEGY BACKTESTING")
    
    backtest_start = '2025-01-01'
    backtest_end = '2026-01-15'
    
    prices_list = []
    for ticker, df in cleaned_data.items():
        prices_list.append(pd.DataFrame({'Date': df.index, 'Ticker': ticker, 'Price': df['Adj Close']}))
    all_prices = pd.concat(prices_list, ignore_index=True)
    prices_matrix = all_prices.pivot(index='Date', columns='Ticker', values='Price').dropna()
    
    backtester = Backtester(returns_matrix, prices_matrix, backtest_start, backtest_end)
    strategy_returns, strategy_values = backtester.simulate_strategy(optimal_weights, rebalance_freq='M')
    benchmark_returns = backtester.create_benchmark(spy_weight=0.6, bnd_weight=0.4)
    
    strategy_metrics = backtester.calculate_metrics(strategy_returns, benchmark_returns)
    benchmark_metrics = backtester.calculate_metrics(benchmark_returns)
    report_df, conclusion = backtester.generate_performance_report(strategy_metrics, benchmark_metrics)
    
    print("\nPerformance Comparison:")
    print(report_df.to_string(index=False))
    print("\n" + conclusion)
    
    backtester.plot_comparison(strategy_returns, benchmark_returns, save_path=r'C:/Users/admin/portfolio-optimization/data/processed/backtest_comparison.png')
    
    # -----------------------
    # Save Results Summary
    # -----------------------
    results_summary = {
        'Task_1': {'Tickers': list(cleaned_data.keys())},
        'Task_2': {'Best_Model': best_model},
        'Task_3': {'TSLA_Forecast_Return': tsla_forecast_return},
        'Task_4': {'Portfolio_Summary': portfolio_summary},
        'Task_5': {'Strategy_Conclusion': conclusion}
    }
    summary_df = pd.DataFrame(results_summary).T
    summary_df.to_csv(r'C:/Users/admin/portfolio-optimization/data/processed/results_summary.csv')
    print("\nResults saved to data/processed/results_summary.csv")
    
    print("\nPIPELINE EXECUTION COMPLETE")
    return results_summary

if __name__ == "__main__":
    main()

# import sys
# import os
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from datetime import datetime, timedelta

# from src.data_loader import DataLoader
# from src.eda import EDA
# from src.forecasting import TimeSeriesForecaster
# from src.portfolio_optimizer import PortfolioOptimizer
# from src.backtester import Backtester

# def main():
#     """Main execution function for the complete pipeline"""
#     print("=" * 80)
#     print("PORTFOLIO OPTIMIZATION PIPELINE")
#     print("=" * 80)
    
#     # Task 1: Data Preprocessing and EDA
#     print("\n" + "=" * 80)
#     print("TASK 1: DATA PREPROCESSING AND EXPLORATORY DATA ANALYSIS")
#     print("=" * 80)
    
#     # Load data
#     print("\n1. Loading and preprocessing data...")
#     data_loader = DataLoader()
#     raw_data = data_loader.download_data()
#     cleaned_data = data_loader.clean_data(raw_data)
    
#     # Perform EDA
#     print("\n2. Performing exploratory data analysis...")
#     eda = EDA(cleaned_data)
    
#     # Generate summary statistics
#     print("\n3. Generating summary statistics...")
#     summary_stats = eda.generate_summary_statistics()
#     print("\nSummary Statistics:")
#     print(summary_stats.round(4))
    
#     # Calculate risk metrics
#     print("\n4. Calculating risk metrics...")
#     risk_metrics = {}
#     for ticker, df in cleaned_data.items():
#         metrics = data_loader.calculate_risk_metrics(df)
#         risk_metrics[ticker] = metrics
    
#     risk_df = pd.DataFrame(risk_metrics).T
#     print("\nRisk Metrics:")
#     print(risk_df.round(4))
    
#     # Stationarity test
#     print("\n5. Performing stationarity tests...")
#     stationarity_results = eda.perform_stationarity_test()
#     print("\nStationarity Test Results (ADF):")
#     print(stationarity_results)
    
#     # Outlier detection
#     print("\n6. Detecting outliers...")
#     outliers = eda.detect_outliers(threshold=3)
#     for ticker, outlier_info in outliers.items():
#         print(f"\n{ticker}: {outlier_info['outlier_count']} outliers "
#               f"({outlier_info['outlier_percentage']:.2f}% of data)")
    
#     # Visualizations
#     print("\n7. Creating visualizations...")
#     eda.plot_price_series(save_path='../data/processed/price_series.png')
#     eda.plot_returns_distribution(save_path='../data/processed/returns_distribution.png')
#     eda.plot_volatility(save_path='../data/processed/volatility_analysis.png')
#     eda.calculate_correlation_matrix()
    
#     # Task 2: Time Series Forecasting
#     print("\n" + "=" * 80)
#     print("TASK 2: TIME SERIES FORECASTING MODELS")
#     print("=" * 80)
    
#     # Focus on TSLA for forecasting
#     print("\n1. Preparing TSLA data for forecasting...")
#     tsla_data = cleaned_data['TSLA']
#     forecaster = TimeSeriesForecaster(tsla_data)
    
#     # Prepare data
#     train_scaled, test_scaled, train_raw, test_raw = forecaster.prepare_data(test_size=0.2)
    
#     # ARIMA model
#     print("\n2. Building and training ARIMA model...")
#     arima_model = forecaster.build_arima_model(train_scaled.flatten(), seasonal=False)
    
#     # Generate forecasts
#     arima_forecast = forecaster.forecast_arima(arima_model, steps=len(test_scaled))
#     arima_forecast_rescaled = forecaster.scaler.inverse_transform(
#         arima_forecast.reshape(-1, 1)
#     ).flatten()
    
#     # Calculate ARIMA metrics
#     arima_metrics = forecaster.calculate_metrics(test_raw.flatten(), arima_forecast_rescaled)
#     print(f"\nARIMA Model Metrics:")
#     for metric, value in arima_metrics.items():
#         print(f"{metric}: {value:.4f}")
    
#     # LSTM model
#     print("\n3. Building and training LSTM model...")
#     sequence_length = 60
    
#     # Prepare sequences for LSTM
#     X_train, y_train = forecaster.prepare_sequences(train_scaled, sequence_length)
#     X_test, y_test = forecaster.prepare_sequences(test_scaled, sequence_length)
    
#     # Reshape for LSTM
#     X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
#     X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
#     # Train LSTM
#     lstm_model, history = forecaster.train_lstm(
#         X_train, y_train, 
#         X_val=X_test, y_val=y_test,
#         epochs=50, 
#         batch_size=32
#     )
    
#     # Generate LSTM forecasts
#     last_sequence = train_scaled[-sequence_length:].flatten()
#     lstm_forecast = forecaster.forecast_lstm(lstm_model, last_sequence, steps=len(test_scaled))
#     lstm_forecast_rescaled = forecaster.scaler.inverse_transform(
#         lstm_forecast.reshape(-1, 1)
#     ).flatten()
    
#     # Calculate LSTM metrics
#     lstm_metrics = forecaster.calculate_metrics(test_raw.flatten(), lstm_forecast_rescaled)
#     print(f"\nLSTM Model Metrics:")
#     for metric, value in lstm_metrics.items():
#         print(f"{metric}: {value:.4f}")
    
#     # Compare models
#     print("\n4. Comparing model performance...")
#     models_results = {
#         'ARIMA': arima_metrics,
#         'LSTM': lstm_metrics
#     }
#     comparison_df = forecaster.compare_models(models_results)
    
#     # Select best model
#     best_model = 'ARIMA' if arima_metrics['RMSE'] < lstm_metrics['RMSE'] else 'LSTM'
#     print(f"\nBest performing model: {best_model}")
    
#     # Task 3: Future Market Trends Forecasting
#     print("\n" + "=" * 80)
#     print("TASK 3: FUTURE MARKET TRENDS FORECASTING")
#     print("=" * 80)
    
#     print("\n1. Generating 12-month future forecasts...")
    
#     # Use best model for forecasting
#     if best_model == 'ARIMA':
#         # Retrain on full data for future forecasting
#         full_data = np.concatenate([train_scaled, test_scaled])
#         final_arima_model = forecaster.build_arima_model(full_data.flatten(), seasonal=False)
        
#         # Generate 12-month forecast (252 trading days)
#         future_forecast = forecaster.forecast_arima(final_arima_model, steps=252)
#         future_forecast_rescaled = forecaster.scaler.inverse_transform(
#             future_forecast.reshape(-1, 1)
#         ).flatten()
#     else:
#         # LSTM forecasting
#         full_data_scaled = np.concatenate([train_scaled, test_scaled])
#         last_sequence_full = full_data_scaled[-sequence_length:].flatten()
#         future_forecast = forecaster.forecast_lstm(lstm_model, last_sequence_full, steps=252)
#         future_forecast_rescaled = forecaster.scaler.inverse_transform(
#             future_forecast.reshape(-1, 1)
#         ).flatten()
    
#     # Create forecast dates
#     last_date = tsla_data.index[-1]
#     future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=252, freq='B')
    
#     print(f"\n2. Forecast Summary (12 months):")
#     print(f"  Initial forecast price: ${future_forecast_rescaled[0]:.2f}")
#     print(f"  Final forecast price: ${future_forecast_rescaled[-1]:.2f}")
#     print(f"  Total forecast return: {(future_forecast_rescaled[-1] - future_forecast_rescaled[0]) / future_forecast_rescaled[0] * 100:.2f}%")
    
#     # Calculate confidence intervals (simplified)
#     forecast_std = np.std(future_forecast_rescaled)
#     upper_bound = future_forecast_rescaled + 1.96 * forecast_std
#     lower_bound = future_forecast_rescaled - 1.96 * forecast_std
    
#     # Plot forecasts with confidence intervals
#     plt.figure(figsize=(14, 8))
    
#     # Historical data (last 2 years)
#     historical_dates = tsla_data.index[-504:]  # Last 2 years
#     historical_prices = tsla_data.loc[historical_dates, 'Adj Close']
    
#     plt.plot(historical_dates, historical_prices, label='Historical Prices', linewidth=2, color='blue')
#     plt.plot(future_dates, future_forecast_rescaled, label='Forecast', linewidth=2, color='red')
#     plt.fill_between(future_dates, lower_bound, upper_bound, alpha=0.2, color='red', label='95% Confidence Interval')
    
#     plt.axvline(x=last_date, color='gray', linestyle='--', alpha=0.7, label='Forecast Start')
#     plt.xlabel('Date', fontsize=12)
#     plt.ylabel('Price ($)', fontsize=12)
#     plt.title('TSLA 12-Month Price Forecast with Confidence Intervals', fontsize=16, fontweight='bold')
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.savefig('../data/processed/forecast_with_ci.png', dpi=300, bbox_inches='tight')
#     plt.show()
    
#     print("\n3. Trend Analysis:")
#     print("""
#     Based on the forecast analysis:
    
#     Opportunities:
#     1. The forecast suggests a generally upward trend for TSLA over the next 12 months
#     2. Confidence intervals indicate reasonable certainty in the short-term forecast (first 3 months)
#     3. Volatility appears manageable within historical ranges
    
#     Risks:
#     1. Confidence intervals widen significantly beyond 6 months, indicating higher uncertainty
#     2. Potential for market corrections or external shocks not captured by the model
#     3. Dependence on Tesla's execution of growth plans and market conditions
    
#     Forecast Reliability:
#     - High reliability in the first 3 months (narrow confidence intervals)
#     - Moderate reliability from 3-6 months
#     - Lower reliability beyond 6 months (wide confidence intervals suggest higher uncertainty)
#     """)
    
#     # Task 4: Portfolio Optimization
#     print("\n" + "=" * 80)
#     print("TASK 4: PORTFOLIO OPTIMIZATION")
#     print("=" * 80)
    
#     print("\n1. Preparing data for portfolio optimization...")
    
#     # Combine returns data for all assets
#     returns_list = []
#     for ticker, df in cleaned_data.items():
#         returns_df = pd.DataFrame({
#             'Date': df.index,
#             'Ticker': ticker,
#             'Return': df['Daily_Return']
#         })
#         returns_list.append(returns_df)
    
#     all_returns = pd.concat(returns_list, ignore_index=True)
    
#     # Pivot to get returns matrix
#     returns_matrix = all_returns.pivot(index='Date', columns='Ticker', values='Return')
#     returns_matrix = returns_matrix.dropna()
    
#     print(f"\nReturns data shape: {returns_matrix.shape}")
#     print(f"Date range: {returns_matrix.index.min()} to {returns_matrix.index.max()}")
    
#     # Calculate expected returns
#     # For TSLA, use forecasted returns
#     tsla_forecast_return = (future_forecast_rescaled[-1] - future_forecast_rescaled[0]) / future_forecast_rescaled[0]
#     tsla_annualized_return = (1 + tsla_forecast_return) ** (1) - 1  # 1-year forecast
    
#     # For BND and SPY, use historical returns
#     bnd_historical_return = returns_matrix['BND'].mean() * 252
#     spy_historical_return = returns_matrix['SPY'].mean() * 252
    
#     expected_returns = {
#         'TSLA': tsla_annualized_return,
#         'BND': bnd_historical_return,
#         'SPY': spy_historical_return
#     }
    
#     print("\n2. Expected Returns (Annualized):")
#     for asset, ret in expected_returns.items():
#         print(f"  {asset}: {ret:.2%}")
    
#     # Initialize portfolio optimizer
#     print("\n3. Optimizing portfolio...")
#     optimizer = PortfolioOptimizer(returns_matrix, expected_returns)
    
#     # Generate efficient frontier
#     results_df, max_sharpe_port, min_vol_port = optimizer.generate_efficient_frontier(num_portfolios=10000)
    
#     # Plot efficient frontier
#     optimizer.plot_efficient_frontier(results_df, max_sharpe_port, min_vol_port,
#                                      save_path='../data/processed/efficient_frontier.png')
    
#     # Plot covariance matrix
#     optimizer.plot_covariance_heatmap(save_path='../data/processed/covariance_heatmap.png')
    
#     # Recommend portfolio
#     print("\n4. Portfolio Recommendation:")
#     portfolio_summary = optimizer.recommend_portfolio(risk_tolerance='medium')
    
#     print(f"\nRecommended Portfolio ({portfolio_summary['Recommendation']}):")
#     print(f"Expected Annual Return: {portfolio_summary['Total_Expected_Return']:.2%}")
#     print(f"Expected Volatility: {portfolio_summary['Total_Volatility']:.2%}")
#     print(f"Sharpe Ratio: {portfolio_summary['Sharpe_Ratio']:.2f}")
    
#     print("\nOptimal Weights:")
#     for _, row in portfolio_summary['Weights'].iterrows():
#         print(f"  {row['Asset']}: {row['Weight']:.2%}")
    
#     # Extract optimal weights for backtesting
#     optimal_weights = dict(zip(portfolio_summary['Weights']['Asset'], 
#                               portfolio_summary['Weights']['Weight']))
    
#     # Task 5: Strategy Backtesting
#     print("\n" + "=" * 80)
#     print("TASK 5: STRATEGY BACKTESTING")
#     print("=" * 80)
    
#     print("\n1. Setting up backtesting...")
    
#     # Define backtesting period (Jan 2025 - Jan 2026)
#     backtest_start = '2025-01-01'
#     backtest_end = '2026-01-15'
    
#     # Get price data for backtesting
#     prices_list = []
#     for ticker, df in cleaned_data.items():
#         price_df = pd.DataFrame({
#             'Date': df.index,
#             'Ticker': ticker,
#             'Price': df['Adj Close']
#         })
#         prices_list.append(price_df)
    
#     all_prices = pd.concat(prices_list, ignore_index=True)
#     prices_matrix = all_prices.pivot(index='Date', columns='Ticker', values='Price')
    
#     # Initialize backtester
#     backtester = Backtester(returns_matrix, prices_matrix, backtest_start, backtest_end)
    
#     print(f"\nBacktesting period: {backtest_start} to {backtest_end}")
#     print(f"Initial portfolio weights: {optimal_weights}")
    
#     # Simulate strategy
#     print("\n2. Simulating optimized portfolio strategy...")
#     strategy_returns, strategy_values = backtester.simulate_strategy(
#         optimal_weights, 
#         rebalance_freq='M'  # Monthly rebalancing
#     )
    
#     # Create benchmark
#     print("\n3. Creating 60/40 SPY/BND benchmark...")
#     benchmark_returns = backtester.create_benchmark(spy_weight=0.6, bnd_weight=0.4)
    
#     # Calculate metrics
#     print("\n4. Calculating performance metrics...")
#     strategy_metrics = backtester.calculate_metrics(strategy_returns, benchmark_returns)
#     benchmark_metrics = backtester.calculate_metrics(benchmark_returns)
    
#     # Generate performance report
#     report_df, conclusion = backtester.generate_performance_report(strategy_metrics, benchmark_metrics)
    
#     print("\nPerformance Comparison:")
#     print(report_df.to_string(index=False))
    
#     print("\n" + conclusion)
    
#     # Plot comparison
#     print("\n5. Visualizing performance comparison...")
#     backtester.plot_comparison(strategy_returns, benchmark_returns,
#                               save_path='../data/processed/backtest_comparison.png')
    
#     # Plot drawdown comparison
#     plt.figure(figsize=(12, 6))
    
#     # Calculate drawdowns
#     strategy_cumulative = (1 + strategy_returns).cumprod()
#     benchmark_cumulative = (1 + benchmark_returns).cumprod()
    
#     strategy_drawdown = (strategy_cumulative - strategy_cumulative.expanding().max()) / strategy_cumulative.expanding().max()
#     benchmark_drawdown = (benchmark_cumulative - benchmark_cumulative.expanding().max()) / benchmark_cumulative.expanding().max()
    
#     plt.plot(strategy_drawdown.index, strategy_drawdown.values * 100, 
#              label='Optimized Portfolio', linewidth=2, color='darkblue')
#     plt.plot(benchmark_drawdown.index, benchmark_drawdown.values * 100,
#              label='60/40 Benchmark', linewidth=2, color='darkred', alpha=0.8)
    
#     plt.fill_between(strategy_drawdown.index, strategy_drawdown.values * 100, 0,
#                      alpha=0.2, color='darkblue')
#     plt.fill_between(benchmark_drawdown.index, benchmark_drawdown.values * 100, 0,
#                      alpha=0.2, color='darkred')
    
#     plt.xlabel('Date', fontsize=12)
#     plt.ylabel('Drawdown (%)', fontsize=12)
#     plt.title('Maximum Drawdown Comparison', fontsize=16, fontweight='bold')
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.savefig('../data/processed/drawdown_comparison.png', dpi=300, bbox_inches='tight')
#     plt.show()
    
#     print("\n" + "=" * 80)
#     print("PIPELINE EXECUTION COMPLETE")
#     print("=" * 80)
    
#     # Save results
#     print("\nSaving results...")
    
#     # Create results summary
#     results_summary = {
#         'Task_1': {
#             'Data_Period': f"{tsla_data.index.min().date()} to {tsla_data.index.max().date()}",
#             'TSLA_Annualized_Return': f"{risk_metrics['TSLA']['Mean_Return']:.2%}",
#             'TSLA_Volatility': f"{risk_metrics['TSLA']['Volatility']:.2%}",
#             'TSLA_Sharpe_Ratio': f"{risk_metrics['TSLA']['Sharpe_Ratio']:.2f}",
#             'Stationary_Returns': stationarity_results.loc['TSLA', 'Return_Stationary']
#         },
#         'Task_2': {
#             'Best_Model': best_model,
#             'Best_Model_RMSE': f"{models_results[best_model]['RMSE']:.4f}",
#             'Best_Model_MAPE': f"{models_results[best_model]['MAPE']:.2f}%"
#         },
#         'Task_3': {
#             'TSLA_Forecast_Return': f"{tsla_forecast_return:.2%}",
#             'Forecast_Horizon': '12 months',
#             'Confidence_Level': '95%'
#         },
#         'Task_4': {
#             'Optimal_Portfolio_Return': f"{portfolio_summary['Total_Expected_Return']:.2%}",
#             'Optimal_Portfolio_Volatility': f"{portfolio_summary['Total_Volatility']:.2%}",
#             'Optimal_Portfolio_Sharpe': f"{portfolio_summary['Sharpe_Ratio']:.2f}",
#             'TSLA_Weight': f"{optimal_weights['TSLA']:.2%}",
#             'SPY_Weight': f"{optimal_weights['SPY']:.2%}",
#             'BND_Weight': f"{optimal_weights['BND']:.2%}"
#         },
#         'Task_5': {
#             'Strategy_Total_Return': f"{strategy_metrics['Total_Return']:.2%}",
#             'Benchmark_Total_Return': f"{benchmark_metrics['Total_Return']:.2%}",
#             'Strategy_Sharpe': f"{strategy_metrics['Sharpe_Ratio']:.2f}",
#             'Benchmark_Sharpe': f"{benchmark_metrics['Sharpe_Ratio']:.2f}",
#             'Outperformed_Benchmark': strategy_metrics['Total_Return'] > benchmark_metrics['Total_Return']
#         }
#     }
    
#     # Save summary to CSV
#     summary_df = pd.DataFrame(results_summary).T
#     summary_df.to_csv('../data/processed/results_summary.csv')
    
#     print("\nResults saved to data/processed/")
#     print("Summary of results saved to results_summary.csv")
    
#     return results_summary

# if __name__ == "__main__":
#     results = main()