# LSTM-stock-price-forecasting
Time series forecasting of IBM and Facebook stock prices using LSTM, including data preprocessing, windowing, baseline modeling, and architecture modification.

# LSTM Stock Price Forecasting

This project applies Long Short-Term Memory (LSTM) networks for time series forecasting of IBM and Facebook stock prices.

## Project Overview

The project covers:
- Time series data exploration and preprocessing
- MinMaxScaler normalization
- Time series train-test splitting
- Windowing with window size = 5 and horizon = 1
- Baseline LSTM architecture
- Modified LSTM architecture
- Model evaluation using RMSE, MAE, and MAPE

## Model Architecture

### Baseline
- LSTM: 50 units
- Activation: ReLU
- Dense: 1 unit
- Optimizer: SGD
- Loss: MSE

### Modified
The baseline architecture was modified to improve forecasting performance by adjusting the LSTM architecture and training configuration.

## Evaluation Metrics

The models are evaluated using:
- RMSE
- MAE
- MAPE
