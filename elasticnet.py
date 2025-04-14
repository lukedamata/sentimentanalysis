"""
group_number_prediction_function.py

Final Project Deliverable #2: Sentiment-Based Return Prediction
------------------------------------------------------------------
This module implements a prediction framework for next-day stock returns based on Reddit
sentiment data. The approach involves extensive feature engineering (including lagged and
rolling statistics), systematic hyperparameter tuning via time-series cross-validation,
and model selection among multiple candidates (ElasticNet, Ridge, Lasso, and LinearRegression).

The output of the training phase is a standardized model_info dictionary containing the best
trained model pipeline and metadata. The predict_returns function accepts today's sentiment data
and a stock universe to generate predictions along with a signal rank.

Usage:
------
The module contains two primary functions:
    • train_model(sentiment_data, return_data)
    • predict_returns(model, sentiment_data_today, stock_universe_today, historical_data=None)

A test case at the bottom demonstrates a simple training/prediction cycle using sample data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
from sklearn.linear_model import ElasticNet, Ridge, Lasso, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

# =============================================================================
# Helper Functions for Preprocessing and Feature Engineering
# =============================================================================

def preprocess_sentiment_data(sentiment_data):
    """
    Preprocess the raw sentiment data:
      - Convert 'Received_Time' (assumed in UTC) to a timezone-aware datetime.
      - Convert UTC to US/Eastern, and if a post's time (EST) is after 4pm, shift its date to the next day.
      - Ensure numeric fields are properly converted and fill missing values with 0.
      - Ensure Ticker field is uppercase.
    
    Parameters
    ----------
    sentiment_data : DataFrame
        Raw sentiment data with columns such as 'Received_Time', 'Ticker', 'Sentiment', etc.
    
    Returns
    -------
    df : DataFrame
        Preprocessed sentiment data.
    """
    df = sentiment_data.copy()
    
    # Convert 'Received_Time' to datetime (UTC) and then to US/Eastern
    df['Received_Time'] = pd.to_datetime(df['Received_Time'], utc=True)
    df['Received_Time_EST'] = df['Received_Time'].dt.tz_convert('America/New_York')
    
    # Shift posts after 4:00 PM EST to the next day
    cutoff = time(16, 0)
    df['local_date'] = df['Received_Time_EST'].dt.date
    df['Date'] = np.where(
        df['Received_Time_EST'].dt.time > cutoff,
        pd.to_datetime(df['local_date']) + pd.Timedelta(days=1),
        pd.to_datetime(df['local_date'])
    )
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Convert key numeric columns and fill missing values with 0
    required_num_cols = ['Sentiment', 'Prob_POS', 'Prob_NTR', 'Prob_NEG', 
                         'TopicWeight', 'SourceWeight', 'Confidence', 'Novelty']
    for col in required_num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Fill missing Author values if applicable
    if 'Author' in df.columns:
        df['Author'] = df['Author'].fillna('unknown')
    
    # Ensure Ticker is uppercase and fill missing with a default value
    if 'Ticker' in df.columns:
        df['Ticker'] = df['Ticker'].str.upper().fillna('UNKNOWN')
    
    # Fill any remaining missing values with 0
    df.fillna(0, inplace=True)
    
    return df

def preprocess_return_data(return_data):
    """
    Preprocess the return data:
      - Convert 'Date' to datetime.
      - Convert the 'Return' field from string percentage (e.g., "2.13%") to float (e.g., 0.0213).
      - Ensure Ticker is uppercase and fill missing values with 0.
    
    Parameters
    ----------
    return_data : DataFrame
        Raw return data with columns ['Date', 'Ticker', 'Return'].
    
    Returns
    -------
    df : DataFrame
        Preprocessed return data.
    """
    df = return_data.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    
    def convert_return(ret):
        if isinstance(ret, str):
            if '%' in ret:
                return float(ret.replace('%', '')) / 100
            else:
                return float(ret)
        return ret
    
    df['Return'] = df['Return'].apply(convert_return)
    
    if 'Ticker' in df.columns:
        df['Ticker'] = df['Ticker'].str.upper().fillna('UNKNOWN')
    df.fillna(0, inplace=True)
    
    return df

def create_features(sentiment_data):
    """
    Generate engineered features from preprocessed sentiment data.
    The function groups the data by Ticker and Date and calculates statistics including:
        - Mean, sum, standard deviation, and skew of Sentiment.
        - Average probability measures (Prob_POS, Prob_NTR, Prob_NEG).
        - Average Topic and Source weights.
        - Average confidence.
        - Count of unique authors.
        - Maximum novelty.
    Additionally, lagged features (previous day's sentiment) and rolling averages
    (3-day and 7-day rolling mean of sentiment_mean) are computed, as well as the daily
    change in sentiment sum.
    
    Parameters
    ----------
    sentiment_data : DataFrame
        Preprocessed sentiment data with a 'Date' column.
    
    Returns
    -------
    features : DataFrame
        DataFrame with aggregated and engineered features per Ticker and Date.
    """
    # Aggregate basic statistics per Ticker and Date
    agg_funcs = {
        'Sentiment': ['mean', 'sum', 'std', lambda x: x.skew()],
        'Prob_POS': 'mean',
        'Prob_NTR': 'mean',
        'Prob_NEG': 'mean',
        'TopicWeight': 'mean',
        'SourceWeight': 'mean',
        'Confidence': 'mean',
        'Author': pd.Series.nunique,
        'Novelty': 'max'
    }
    features = sentiment_data.groupby(['Ticker', 'Date']).agg(agg_funcs)
    features.columns = ['sentiment_mean', 'sentiment_sum', 'sentiment_std', 'sentiment_skew',
                        'avg_prob_pos', 'avg_prob_ntr', 'avg_prob_neg',
                        'avg_topic_weight', 'avg_source_weight', 'avg_confidence',
                        'unique_authors', 'max_novelty']
    features = features.reset_index()
    
    # Sort by Ticker and Date
    features = features.sort_values(['Ticker', 'Date'])
    
    # Create lagged features (previous day's values)
    features['lag_sentiment_sum'] = features.groupby('Ticker')['sentiment_sum'].shift(1).fillna(0)
    features['lag_sentiment_mean'] = features.groupby('Ticker')['sentiment_mean'].shift(1).fillna(0)
    
    # Create rolling features: 3-day and 7-day rolling mean of sentiment_mean
    features['roll3_sentiment_mean'] = features.groupby('Ticker')['sentiment_mean']\
                                               .rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)
    features['roll7_sentiment_mean'] = features.groupby('Ticker')['sentiment_mean']\
                                               .rolling(window=7, min_periods=1).mean().reset_index(level=0, drop=True)
    # Compute day-over-day sentiment change (difference in sentiment_sum)
    features['sentiment_change'] = features.groupby('Ticker')['sentiment_sum'].diff().fillna(0)
    
    # Fill any remaining missing values with 0
    features.fillna(0, inplace=True)
    
    return features

def portfolio_performance_score(valid_df, quantile=0.8):
    """
    Compute a simple portfolio performance metric on the validation set.
    For each Date in the validation data, stocks with predicted return above the given quantile
    are “selected” into the portfolio, and the average actual return of these stocks is computed.
    The final score is the average of these daily portfolio returns.
    
    Parameters
    ----------
    valid_df : DataFrame
        DataFrame with columns: 'Date', 'Return', 'Predicted_Return'.
    quantile : float, default 0.8
        The quantile threshold for stock selection (e.g., top 20%).
    
    Returns
    -------
    score : float
        The average portfolio return over all dates in the validation set.
    """
    performances = []
    for date, group in valid_df.groupby('Date'):
        threshold = group['Predicted_Return'].quantile(quantile)
        portfolio = group[group['Predicted_Return'] >= threshold]
        if not portfolio.empty:
            performances.append(portfolio['Return'].mean())
    if performances:
        return np.mean(performances)
    else:
        return -np.inf

# =============================================================================
# Model Training and Prediction Functions
# =============================================================================

def train_model(sentiment_data, return_data):
    """
    Train a model using sentiment features to predict next-day stock returns.
    
    This function performs the following steps:
      1. Preprocess sentiment and return data.
      2. Create engineered features (including lagged and rolling features).
      3. Merge the features with the return data on Ticker and Date.
      4. Split the merged data into a training and validation set based on date.
         (If the split results in an insufficient training set, the entire dataset is used.)
      5. Build pipelines for candidate models:
            - ElasticNet, Ridge, Lasso (with hyperparameter grid search via TimeSeriesSplit)
            - LinearRegression (as a benchmark)
      6. Tune the hyperparameters using cross-validation and evaluate on the hold-out
         validation set using a portfolio performance metric.
      7. Select and return the best model along with its metadata.
    
    Parameters
    ----------
    sentiment_data : DataFrame
        The Reddit sentiment data for training (e.g., sentiment_train_2017_2021.csv).
    return_data : DataFrame
        The stock return data for training (e.g., return_train_2017_2021.csv).
    
    Returns
    -------
    model_info : dict
        Dictionary containing:
            - 'model': The best-trained model pipeline.
            - 'feature_columns': List of feature column names.
            - 'validation_score': Portfolio performance score on the validation set.
    """
    # Preprocess the raw data
    sentiment_processed = preprocess_sentiment_data(sentiment_data)
    returns_processed = preprocess_return_data(return_data)
    
    # Create engineered features from sentiment data
    features = create_features(sentiment_processed)
    
    # Merge features with return data (inner join on Date and Ticker)
    model_data = pd.merge(features, returns_processed[['Date', 'Ticker', 'Return']], 
                          on=['Date', 'Ticker'], how='inner')
    model_data = model_data.sort_values('Date')
    
    # Split data into training and validation sets (time-based split: last 20% dates as validation)
    unique_dates = model_data['Date'].sort_values().unique()
    if len(unique_dates) < 2:
        print("Warning: Not enough unique dates in the dataset; using entire data for training and validation.")
        train_data = model_data.copy()
        valid_data = model_data.copy()
    else:
        split_date = unique_dates[int(0.8 * len(unique_dates))]
        train_data = model_data[model_data['Date'] < split_date]
        valid_data = model_data[model_data['Date'] >= split_date]
        if train_data.empty:
            print("Warning: Training data split is empty; using entire dataset for training.")
            train_data = model_data.copy()
            valid_data = model_data.copy()
    
    # Define feature columns (all columns except Ticker, Date, and Return)
    feature_columns = [col for col in model_data.columns if col not in ['Ticker', 'Date', 'Return']]
    X_train = train_data[feature_columns]
    y_train = train_data['Return']
    X_valid = valid_data[feature_columns]
    y_valid = valid_data['Return']
    
    # Adjust cross-validation strategy if training samples are very few
    if X_train.shape[0] < 5:
        print("Warning: Insufficient training samples for 5-fold time series cross-validation; using cv=2.")
        cv = 2
    else:
        cv = TimeSeriesSplit(n_splits=5)
    
    # Set up candidate models with pipelines and hyperparameter grids
    candidate_models = {}
    
    # ElasticNet Pipeline and Grid
    enet_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', ElasticNet(max_iter=10000))
    ])
    enet_params = {
        'model__alpha': np.logspace(-4, 1, 10),
        'model__l1_ratio': [0.1, 0.5, 0.9]
    }
    candidate_models['ElasticNet'] = (enet_pipe, enet_params)
    
    # Ridge Pipeline and Grid
    ridge_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', Ridge())
    ])
    ridge_params = {
        'model__alpha': np.logspace(-4, 1, 10)
    }
    candidate_models['Ridge'] = (ridge_pipe, ridge_params)
    
    # Lasso Pipeline and Grid
    lasso_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', Lasso(max_iter=10000))
    ])
    lasso_params = {
        'model__alpha': np.logspace(-4, 1, 10)
    }
    candidate_models['Lasso'] = (lasso_pipe, lasso_params)
    
    # Linear Regression (no hyperparameters)
    lr_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LinearRegression())
    ])
    candidate_models['LinearRegression'] = (lr_pipe, {})
    
    best_score = -np.inf
    best_model_name = None
    best_pipeline = None
    
    # Train each candidate model with cross-validation or fit directly if no grid search is needed.
    for name, (pipeline, param_grid) in candidate_models.items():
        print(f"Training candidate model: {name}")
        
        if param_grid:
            grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)
            grid.fit(X_train, y_train)
            candidate_pipeline = grid.best_estimator_
            print(f"Best params for {name}: {grid.best_params_}")
        else:
            candidate_pipeline = pipeline.fit(X_train, y_train)
        
        # Predict on the validation set and compute portfolio performance metric
        valid_preds = candidate_pipeline.predict(X_valid)
        valid_eval_df = pd.DataFrame({
            'Date': valid_data['Date'],
            'Return': y_valid,
            'Predicted_Return': valid_preds
        })
        score = portfolio_performance_score(valid_eval_df, quantile=0.8)
        print(f"Validation portfolio performance for {name}: {score:.6f}")
        
        if score > best_score:
            best_score = score
            best_model_name = name
            best_pipeline = candidate_pipeline
    
    print(f"Selected best model: {best_model_name} with validation portfolio return of {best_score:.6f}")
    
    model_info = {
        'model': best_pipeline,
        'feature_columns': feature_columns,
        'validation_score': best_score
    }
    
    return model_info

def predict_returns(model, sentiment_data_today, stock_universe_today, historical_data=None):
    """
    Generate predictions of next-day stock returns for all stocks in the universe.
    
    The function performs the following:
      1. Preprocess today's sentiment data.
      2. Create engineered features using the same process as during training.
      3. For tickers in the stock universe that are not present in today's sentiment data,
         default feature values of 0 are assigned.
      4. Use the trained model pipeline to predict next-day returns.
      5. Compute a signal rank (percentile rank) for the predicted returns among the universe.
    
    Parameters
    ----------
    model : dict
        Dictionary containing the trained model pipeline and metadata ('model' and 'feature_columns').
    sentiment_data_today : DataFrame
        Sentiment data for the current day.
    stock_universe_today : list
        List of stock tickers available for trading today.
    historical_data : dict, optional
        Dictionary containing historical sentiment/return data (not used in this implementation).
    
    Returns
    -------
    predictions : DataFrame
        DataFrame with columns ['Ticker', 'Predicted_Return', 'Signal_Rank'].
    """
    # Preprocess today's sentiment data
    processed = preprocess_sentiment_data(sentiment_data_today)
    
    # Create features from today's sentiment data
    features_today = create_features(processed)
    
    # For tickers in the stock universe not present in today's features, create default rows with feature values 0.
    missing_tickers = set(stock_universe_today) - set(features_today['Ticker'])
    if missing_tickers:
        default_features = {col: 0 for col in model['feature_columns']}
        default_rows = []
        current_date = processed['Date'].max()
        for ticker in missing_tickers:
            row = {'Ticker': ticker, 'Date': current_date}
            row.update(default_features)
            default_rows.append(row)
        missing_df = pd.DataFrame(default_rows)
        features_today = pd.concat([features_today, missing_df], ignore_index=True)
    
    # Keep only tickers in the provided stock universe
    features_today = features_today[features_today['Ticker'].isin(stock_universe_today)]
    
    # Prepare feature matrix for prediction
    X_today = features_today[model['feature_columns']]
    
    # Generate predictions using the trained model pipeline
    features_today = features_today.copy()
    features_today['Predicted_Return'] = model['model'].predict(X_today)
    
    # Compute signal rank as the percentile rank of the predicted return among the universe.
    features_today['Signal_Rank'] = features_today['Predicted_Return'].rank(pct=True)
    
    predictions = features_today[['Ticker', 'Predicted_Return', 'Signal_Rank']].copy()
    
    return predictions

# =============================================================================
# Test Case (Executed when the script is run directly)
# =============================================================================

if __name__ == "__main__":
    # For testing, we create sample sentiment and return data via StringIO.
    # In a real scenario, these would be loaded from CSV files.
    from io import StringIO

    # Sample sentiment data CSV (includes times that exercise the cutoff logic)
    sentiment_csv = """StoryID,Received_Time,Ticker,Sentiment,Prob_POS,Prob_NTR,Prob_NEG,TopicWeight,SourceWeight,Confidence,Author,Novelty
R1,2021-05-31 15:00:00.000,GME,1,0.8,0.1,0.1,0.5,0.6,0.7,user1,1
R2,2021-05-31 17:00:00.000,GME,-1,0.2,0.3,0.5,0.2,0.4,0.6,user2,1
R3,2021-05-31 14:30:00.000,AMC,1,0.7,0.2,0.1,0.4,0.5,0.8,user3,1
R4,2021-05-31 16:30:00.000,AMC,1,0.9,0.05,0.05,0.3,0.7,0.9,user4,1
R5,2021-06-01 15:30:00.000,GME,1,0.85,0.1,0.05,0.55,0.65,0.75,user1,1
R6,2021-06-01 15:45:00.000,AMC,-1,0.3,0.4,0.3,0.35,0.45,0.65,user5,1
"""
    # Sample return data CSV; note overlapping dates are required for merging.
    return_csv = """Date,Ticker,Return
2021-06-01,GME,2.5%
2021-06-01,AMC,1.5%
2021-06-02,GME,3.0%
2021-06-02,AMC,-0.5%
"""
    sentiment_data = pd.read_csv(StringIO(sentiment_csv))
    return_data = pd.read_csv(StringIO(return_csv))
    
    print("===== Training Model =====")
    model_info = train_model(sentiment_data, return_data)
    print("\nTrained Model Information:")
    print("Feature Columns:", model_info['feature_columns'])
    print("Validation Portfolio Score:", model_info['validation_score'])
    
    # Simulate a prediction scenario for an unseen day (e.g., 2021-06-02)
    sentiment_today_csv = """StoryID,Received_Time,Ticker,Sentiment,Prob_POS,Prob_NTR,Prob_NEG,TopicWeight,SourceWeight,Confidence,Author,Novelty
R7,2021-06-02 15:00:00.000,GME,1,0.8,0.1,0.1,0.5,0.6,0.7,user1,1
R8,2021-06-02 16:30:00.000,AMC,1,0.9,0.05,0.05,0.3,0.7,0.9,user4,1
"""
    sentiment_today = pd.read_csv(StringIO(sentiment_today_csv))
    
    # Define the stock universe for prediction (include tickers even if missing from today's sentiment)
    stock_universe_today = ['GME', 'AMC']
    
    print("\n===== Generating Predictions =====")
    predictions = predict_returns(model_info, sentiment_today, stock_universe_today)
    print("\nPredictions:")
    print(predictions)
