
# Performance Summary for morefeatures2

## Trading Strategy Performance

### Returns
- **Average Daily Return**: 0.0515%
- **Annualized Return**: 13.84%
- **Total Return**: 1.03%
- **Median Daily Return**: 0.0628%

### Risk Metrics
- **Daily Volatility**: 0.1086%
- **Annualized Volatility**: 1.72%
- **Sharpe Ratio**: 7.52
- **Maximum Drawdown**: -0.39%

## Prediction Performance

### Error Metrics
- **Mean Squared Error (MSE)**: 0.001524
- **Mean Absolute Error (MAE)**: 0.033921
- **R² Score**: -2.8866

## Quintile Returns

- **Q1**: 0.0186%
- **Q2**: 0.0420%
- **Q3**: 0.0489%
- **Q4**: 0.0491%
- **Q5**: 0.0701%

## Sample Size
- **Days Evaluated**: 20
- **Trading Days with Valid Long-Short Returns**: 20

## Model Characteristics
- The model uses Reddit sentiment data to predict next-day stock returns
- Stocks are ranked daily based on predicted returns and divided into quintiles
- The trading strategy goes long the highest quintile (Q5) and shorts the lowest quintile (Q1)

## Visualizations
See the accompanying PNG files for detailed visualizations of model performance.
