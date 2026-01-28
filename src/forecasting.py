"""
Time series forecasting models for Task 2
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ARIMA/SARIMA
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
import pmdarima as pm

# LSTM
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

import warnings
warnings.filterwarnings('ignore')

class TimeSeriesForecaster:
    """Class for time series forecasting models"""
    
    def __init__(self, tsla_data):
        self.tsla_data = tsla_data
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
    def prepare_data(self, test_size=0.2):
        """Prepare data for time series forecasting"""
        # Use adjusted close prices
        price_col = next((c for c in self.tsla_data.columns if 'Adj Close' in c), None)
        if price_col is None:
           price_col = next((c for c in self.tsla_data.columns if 'Close' in c), None)

        prices = self.tsla_data[price_col].values.reshape(-1, 1)

        
        # Split chronologically
        split_idx = int(len(prices) * (1 - test_size))
        train_data = prices[:split_idx]
        test_data = prices[split_idx:]
        
        # Scale the data
        train_scaled = self.scaler.fit_transform(train_data)
        test_scaled = self.scaler.transform(test_data)
        
        return train_scaled, test_scaled, train_data, test_data
    
    def build_arima_model(self, train_data, seasonal=False):
        """Build and train ARIMA/SARIMA model"""
        print("Building ARIMA model...")
        
        if seasonal:
            # Use auto_arima to find best SARIMA parameters
            print("Searching for optimal SARIMA parameters...")
            model = auto_arima(
                train_data,
                start_p=0, start_q=0,
                max_p=5, max_q=5,
                start_P=0, start_Q=0,
                max_P=2, max_Q=2,
                m=12,  # Monthly seasonality
                seasonal=True,
                d=None, D=None,
                trace=True,
                error_action='ignore',
                suppress_warnings=True,
                stepwise=True
            )
            
            print(f"Best SARIMA parameters: {model.order} x {model.seasonal_order}")
            best_model = model
        else:
            # Use auto_arima for non-seasonal ARIMA
            print("Searching for optimal ARIMA parameters...")
            model = auto_arima(
                train_data,
                start_p=0, start_q=0,
                max_p=5, max_q=5,
                seasonal=False,
                trace=True,
                error_action='ignore',
                suppress_warnings=True,
                stepwise=True
            )
            
            print(f"Best ARIMA parameters: {model.order}")
            best_model = model
            
        return best_model
    
    def forecast_arima(self, model, steps):
        """Generate forecasts using ARIMA model"""
        forecast = model.predict(n_periods=steps)
        return forecast
    
    def prepare_sequences(self, data, sequence_length=60):
        """Prepare sequences for LSTM"""
        X, y = [], []
        for i in range(sequence_length, len(data)):
            X.append(data[i-sequence_length:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)
    
    def build_lstm_model(self, sequence_length=60, units=50, dropout=0.2):
        """Build LSTM model architecture"""
        model = Sequential([
            LSTM(units=units, return_sequences=True, input_shape=(sequence_length, 1)),
            Dropout(dropout),
            LSTM(units=units, return_sequences=False),
            Dropout(dropout),
            Dense(units=25, activation='relu'),
            Dense(units=1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    
    def train_lstm(self, X_train, y_train, X_val=None, y_val=None, epochs=50, batch_size=32):
        """Train LSTM model"""
        model = self.build_lstm_model(sequence_length=X_train.shape[1])
        
        early_stopping = EarlyStopping(
            monitor='val_loss' if X_val is not None else 'loss',
            patience=10,
            restore_best_weights=True
        )
        
        if X_val is not None:
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stopping],
                verbose=1
            )
        else:
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stopping],
                verbose=1
            )
            
        return model, history
    
    def forecast_lstm(self, model, last_sequence, steps):
        """Generate multi-step forecast using LSTM"""
        forecasts = []
        current_sequence = last_sequence.copy()
        
        for _ in range(steps):
            # Reshape for prediction
            current_input = current_sequence.reshape(1, -1, 1)
            
            # Predict next value
            next_pred = model.predict(current_input, verbose=0)[0, 0]
            forecasts.append(next_pred)
            
            # Update sequence
            current_sequence = np.roll(current_sequence, -1)
            current_sequence[-1] = next_pred
            
        return np.array(forecasts)
    
    def calculate_metrics(self, y_true, y_pred):
        """Calculate evaluation metrics"""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape
        }
    
    def compare_models(self, models_results):
        """Compare performance of different models"""
        comparison_df = pd.DataFrame(models_results).T
        
        # Plot comparison
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        metrics = ['MAE', 'RMSE', 'MAPE']
        for idx, metric in enumerate(metrics):
            axes[idx].bar(comparison_df.index, comparison_df[metric])
            axes[idx].set_title(f'{metric} Comparison')
            axes[idx].set_ylabel(metric)
            axes[idx].tick_params(axis='x', rotation=45)
            
            # Add value labels
            for i, v in enumerate(comparison_df[metric]):
                axes[idx].text(i, v, f'{v:.4f}', ha='center', va='bottom')
                
        plt.tight_layout()
        plt.show()
        
        return comparison_df