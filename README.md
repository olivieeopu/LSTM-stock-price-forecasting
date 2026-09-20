# LSTM-stock-price-forecasting
Time series forecasting of IBM and Facebook stock prices using LSTM, including data preprocessing, windowing, baseline modeling, and architecture modification.

# LSTM Stock Price Forecasting

This project applies Long Short-Term Memory (LSTM) networks for time series forecasting of IBM and Facebook stock prices.

## Background

Financial market data is inherently sequential, making stock price forecasting a suitable 
application of time series modeling. Unlike traditional machine learning models, Long 
Short-Term Memory (LSTM) networks are designed to capture temporal dependencies within 
sequential data.

This project explores the application of LSTM networks for forecasting IBM and Facebook 
stock closing prices. Baseline and modified architectures are developed and compared to 
evaluate how architectural changes affect forecasting performance.

## Objectives

- Explore and preprocess historical IBM and Facebook stock price data.
- Transform sequential data into supervised learning samples using time series windowing.
- Develop baseline LSTM models for stock price forecasting.
- Improve the baseline models through architectural and hyperparameter modifications.
- Evaluate forecasting performance using RMSE, MAE, and MAPE.
- Compare baseline and modified models to determine the most effective architecture.

## Dataset

The project uses historical stock market data from:

- IBM
- Facebook (Meta)

The primary variables used for forecasting are:
- Date
- Close Price

The data is treated as time series data to preserve its chronological structure.

## Methodology

Data Collection

      ↓
      
Exploratory Data Analysis

      ↓
      
Data Preprocessing

      ↓
      
MinMax Normalization

      ↓
      
Time Series Train-Test Split

      ↓
      
Sliding Window Transformation

      ↓
      
Baseline LSTM

      ↓
      
Modified LSTM

      ↓
      
Model Evaluation

      ↓
      
Model Comparison


### Data Preprocessing

The dataset was sorted chronologically and normalized using MinMaxScaler to transform 
stock prices into a consistent range suitable for LSTM training.

The data was split chronologically rather than randomly to prevent future information 
from leaking into the training data.

### Time Series Windowing

A sliding window approach was used with:

- Window size: 5
- Forecast horizon: 1

Five consecutive trading-day observations are therefore used as input to predict the 
next observation. Overlapping windows are used to generate sequential training samples.

## Model Architecture

### Baseline LSTM

The baseline architecture consists of:

- LSTM layer: 50 units
- Activation: ReLU
- Output layer: Dense (1 unit)
- Optimizer: SGD
- Loss function: Mean Squared Error (MSE)
- Evaluation metric: MAE

### Modified LSTM

The baseline architecture was further modified by adjusting the network architecture, 
activation functions, regularization, and optimization strategy to improve forecasting 
performance and model generalization.

## Results and Findings

The models are evaluated using:
- RMSE
- MAE
- MAPE

The experiments show that architectural modification does not automatically improve 
forecasting performance.

For IBM, Modified 1 achieved the best overall performance with an RMSE of 2.71, 
MAE of 1.76, and MAPE of 1.35%, outperforming both the baseline and Modified 2 models.

A similar pattern was observed for Facebook. Modified 1 reduced the RMSE from 8.28 
to 5.01 and the MAPE from 3.75% to 2.16%, indicating a substantial improvement over 
the baseline model.

Modified 2 introduced additional model complexity and regularization but produced 
higher prediction errors on both datasets. This suggests that increasing model 
complexity does not necessarily lead to better forecasting accuracy.

Overall, Modified 1 provided the best balance between prediction accuracy and 
generalization across both datasets.

## Project Overview

The project covers:
- Time series data exploration and preprocessing
- MinMaxScaler normalization
- Time series train-test splitting
- Windowing with window size = 5 and horizon = 1
- Baseline LSTM architecture
- Modified LSTM architecture
- Model evaluation using RMSE, MAE, and MAPE

