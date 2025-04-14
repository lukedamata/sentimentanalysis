"""
001_prediction_function.py

This file provides two main functions:
    1. train_model(sentiment_data, return_data)
    2. predict_returns(model, sentiment_data_today, stock_universe_today)

The code processes Alexandria’s Reddit sentiment data, creates aggregated features (including time‐series and topic‐based features),
and trains a PyTorch feedforward neural network (with GPU support when available) to predict next-day stock returns.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from datetime import datetime, time, timedelta

# Set device for GPU usage if available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

####################################
# Helper Functions
####################################

def convert_return(x):
    """
    Converts a return value to a float.
    If x is a string ending in '%', removes the '%' and divides by 100.
    """
    if isinstance(x, str):
        x = x.strip()
        if x.endswith('%'):
            return float(x[:-1].strip()) / 100.0
        else:
            return float(x)
    else:
        return float(x)

def preprocess_sentiment_data(sentiment_data):
    """
    Preprocess sentiment data:
      - Convert 'Received_Time' to a timezone-aware datetime (UTC) then to US/Eastern.
      - Create a 'Date' column by shifting posts received after 4:00 PM (EST) to the next day.
      - Ensure 'Ticker' is uppercase.
    
    Parameters
    ----------
    sentiment_data : DataFrame
        Raw sentiment data containing at least 'Received_Time' and 'Ticker'.
    
    Returns
    -------
    df : DataFrame
        Processed sentiment data with added 'Received_Time_EST' and 'Date' columns.
    """
    df = sentiment_data.copy()
    if 'Received_Time' not in df.columns:
        raise ValueError("Column 'Received_Time' not found in the sentiment data.")
    df['Received_Time'] = pd.to_datetime(df['Received_Time'], utc=True)
    df['Received_Time_EST'] = df['Received_Time'].dt.tz_convert('America/New_York')
    df['local_date'] = df['Received_Time_EST'].dt.date
    cutoff = time(16, 0)  # 4:00 PM cutoff.
    df['Date'] = df['Received_Time_EST'].apply(
        lambda x: pd.to_datetime(x.date() + timedelta(days=1)) if x.time() > cutoff else pd.to_datetime(x.date())
    )
    if 'Ticker' in df.columns:
        df['Ticker'] = df['Ticker'].str.upper()
    return df

def create_features(df):
    """
    Create advanced daily features by aggregating sentiment data for each Ticker and Date,
    and compute additional time-series and topic-based features.
    
    Base aggregated features (per ticker and date):
      - sentiment_mean, sentiment_std, post_count, avg_confidence, avg_prob_pos, avg_prob_ntr,
        avg_prob_neg, avg_source_weight, avg_topic_weight, avg_relevance, weighted_sentiment,
        net_sentiment (sum of Sentiment).
    
    Additional time-series features (computed for each ticker):
      - cumulative_sentiment: cumulative sum of net_sentiment.
      - daily_sentiment_change: difference of net_sentiment from the previous day.
      - ma_5: 5-day moving average of net_sentiment.
      - ma_10: 10-day moving average of net_sentiment.
      - past_3_sentiment: sum of net_sentiment for the previous 3 days (with a one-day lag).
      - log_volume: log(1 + post_count).
    
    Topic-based features:
      - For each target topic (from the provided list), create two features:
          - WSB_count_<topic>: count of posts for that topic from sources that contain "WSB".
          - INVESTING_count_<topic>: count of posts for that topic from sources that contain "INVESTING".
    
    Parameters
    ----------
    df : DataFrame
        Preprocessed sentiment data with required columns. Optionally, it should have:
        'Reddit_Topic' and 'Source'.
    
    Returns
    -------
    final_df : DataFrame
        Aggregated features for each Ticker and Date including additional time-series and topic-based features.
    """
    # Ensure required columns exist.
    required_columns = ['Ticker', 'Date', 'Sentiment', 'Confidence', 'Prob_POS', 
                        'Prob_NTR', 'Prob_NEG', 'Relevance', 'SourceWeight', 'TopicWeight']
    for col in required_columns:
        if col not in df.columns:
            if col in ['Ticker', 'Date', 'Sentiment']:
                raise ValueError(f"Required column {col} is missing in the data.")
            else:
                df[col] = 0

    # Base aggregation: calculate mean, std, count, and sum (for net sentiment).
    agg_funcs = {
        'Sentiment': ['mean', 'std', 'count', 'sum'],
        'Confidence': 'mean',
        'Prob_POS': 'mean',
        'Prob_NTR': 'mean',
        'Prob_NEG': 'mean',
        'Relevance': 'mean',
        'SourceWeight': 'mean',
        'TopicWeight': 'mean'
    }
    grouped = df.groupby(['Ticker', 'Date']).agg(agg_funcs)
    grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
    grouped = grouped.reset_index()
    rename_dict = {
        'Sentiment_mean': 'sentiment_mean',
        'Sentiment_std': 'sentiment_std',
        'Sentiment_count': 'post_count',
        'Sentiment_sum': 'net_sentiment',
        'Confidence_mean': 'avg_confidence',
        'Prob_POS_mean': 'avg_prob_pos',
        'Prob_NTR_mean': 'avg_prob_ntr',
        'Prob_NEG_mean': 'avg_prob_neg',
        'SourceWeight_mean': 'avg_source_weight',
        'TopicWeight_mean': 'avg_topic_weight',
        'Relevance_mean': 'avg_relevance'
    }
    grouped = grouped.rename(columns=rename_dict)
    
    # Compute weighted sentiment: average (Sentiment * Relevance)
    def weighted_sentiment_func(sub_df):
        if sub_df['Sentiment'].count() == 0:
            return 0
        return (sub_df['Sentiment'] * sub_df['Relevance']).sum() / sub_df['Sentiment'].count()
    ws = df.groupby(['Ticker', 'Date']).apply(weighted_sentiment_func).reset_index(name='weighted_sentiment')
    grouped = pd.merge(grouped, ws, on=['Ticker', 'Date'], how='left')
    
    # Compute time-series features per ticker.
    def compute_time_series_features(sub_df):
        sub_df = sub_df.sort_values('Date').copy()
        sub_df['cumulative_sentiment'] = sub_df['net_sentiment'].cumsum()
        sub_df['daily_sentiment_change'] = sub_df['net_sentiment'].diff().fillna(0)
        sub_df['ma_5'] = sub_df['net_sentiment'].rolling(window=5, min_periods=1).mean()
        sub_df['ma_10'] = sub_df['net_sentiment'].rolling(window=10, min_periods=1).mean()
        sub_df['past_3_sentiment'] = sub_df['net_sentiment'].shift(1).rolling(window=3, min_periods=1).sum().fillna(0)
        sub_df['log_volume'] = np.log1p(sub_df['post_count'])
        return sub_df
    ts_features = grouped.groupby('Ticker').apply(compute_time_series_features).reset_index(drop=True)
    
    # Now add topic-based features if 'Reddit_Topic' and 'Source' exist.
    all_topics = [
        "Biotech", "Chart", "Commentary", "Daily Discussion", "Daily Thread", "DD", "Discussion", "Distressed",
        "Earnings Thread", "Education", "Energy", "Fundamentals", "Futures", "Gain", "Help", "Industry Report",
        "Interview/Profile", "Investor Letter", "Long Thesis", "Loss", "Macro", "Meme", "Mods", "News", "None",
        "Options", "Profit", "Question", "Retail", "Satire", "Shitpost", "Short Thesis", "Special Situation",
        "Stocks", "Storytime", "Strategy", "tag me pls", "Technicals", "Thesis", "Wall St. \"Leaks\"",
        "Weekend Discussion", "WSBbooks", "YOLO"
    ]
    
    if 'Reddit_Topic' in df.columns and 'Source' in df.columns:
        # Create a copy for topic computations.
        df_topic = df.copy()
        df_topic['Reddit_Topic'] = df_topic['Reddit_Topic'].fillna("").str.upper()
        df_topic['Source'] = df_topic['Source'].fillna("").str.upper()
        
        # Define two source categories: "WSB" and "INVESTING".
        for src in ["WSB", "INVESTING"]:
            src_mask = df_topic['Source'].str.contains(src, case=False, na=False)
            df_src = df_topic[src_mask]
            # Group by Ticker, Date and Reddit_Topic, then count posts.
            topic_counts = df_src.groupby(['Ticker', 'Date'])['Reddit_Topic'].value_counts().unstack(fill_value=0)
            # For each topic in our list, ensure there is a column.
            for topic in all_topics:
                topic_upper = topic.upper()
                col_name = f"{src}_count_{topic_upper}"
                if topic_upper in topic_counts.columns:
                    # Rename existing column to the standardized name.
                    topic_counts = topic_counts.rename(columns={topic_upper: col_name})
                else:
                    # Create the column with default value 0.
                    topic_counts[col_name] = 0
            # Reset index so that Ticker and Date become columns.
            topic_counts = topic_counts.reset_index()[['Ticker', 'Date'] + [f"{src}_count_{t.upper()}" for t in all_topics]]
            # Merge with the ts_features DataFrame.
            ts_features = pd.merge(ts_features, topic_counts, on=['Ticker', 'Date'], how='left')
            # Fill missing topic count columns with 0.
            for topic in all_topics:
                col_name = f"{src}_count_{topic.upper()}"
                if col_name in ts_features.columns:
                    ts_features[col_name] = ts_features[col_name].fillna(0)
    
    final_df = ts_features.fillna(0)
    return final_df

####################################
# Updated Neural Network Model with Hyperparameter Optimizations
####################################

class FeedforwardNet(nn.Module):
    def __init__(self, input_dim):
        """
        A feedforward neural network with two hidden layers, dropout, and updated architecture.
        """
        super(FeedforwardNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.net(x)

####################################
# Main Functions
####################################

def train_model(sentiment_data, return_data):
    """
    Train a model using sentiment features to predict next-day returns.
    
    Parameters
    ----------
    sentiment_data : DataFrame
        The Reddit sentiment data for training.
    return_data : DataFrame
        The stock return data for training.
    
    Returns
    -------
    model_info : dict
        Contains the trained PyTorch model, scaler, feature columns, and device information.
    """
    print("Preprocessing sentiment data...")
    sentiment_data = preprocess_sentiment_data(sentiment_data)
    print("Creating features from sentiment data...")
    features_df = create_features(sentiment_data)
    
    # Preprocess return_data: convert Date to datetime and normalize, ensure Ticker is uppercase, and convert Return.
    return_data = return_data.copy()
    return_data['Date'] = pd.to_datetime(return_data['Date']).dt.normalize()
    return_data['Ticker'] = return_data['Ticker'].str.upper()
    return_data['Return'] = return_data['Return'].apply(convert_return)
    
    print("Merging sentiment features with stock returns...")
    model_data = pd.merge(features_df, return_data[['Date', 'Ticker', 'Return']], on=['Date', 'Ticker'], how='inner')
    model_data = model_data.dropna(subset=['Return'])
    model_data['Return'] = model_data['Return'].apply(convert_return).astype(float)
    
    # Update feature_columns to include the additional features.
    feature_columns = [
        'sentiment_mean', 'sentiment_std', 'post_count', 'avg_confidence',
        'avg_prob_pos', 'avg_prob_ntr', 'avg_prob_neg', 'weighted_sentiment',
        'avg_source_weight', 'avg_topic_weight', 'net_sentiment',
        'cumulative_sentiment', 'daily_sentiment_change', 'ma_5', 'ma_10',
        'past_3_sentiment', 'log_volume'
    ]
    # Also add topic count features for both sources.
    for src in ["WSB", "INVESTING"]:
        for topic in [t.upper() for t in [
            "Biotech", "Chart", "Commentary", "Daily Discussion", "Daily Thread", "DD", "Discussion", "Distressed",
            "Earnings Thread", "Education", "Energy", "Fundamentals", "Futures", "Gain", "Help", "Industry Report",
            "Interview/Profile", "Investor Letter", "Long Thesis", "Loss", "Macro", "Meme", "Mods", "News", "None",
            "Options", "Profit", "Question", "Retail", "Satire", "Shitpost", "Short Thesis", "Special Situation",
            "Stocks", "Storytime", "Strategy", "tag me pls", "Technicals", "Thesis", "Wall St. \"Leaks\"",
            "Weekend Discussion", "WSBbooks", "YOLO"
        ]]:
            feature_columns.append(f"{src}_count_{topic}")
    
    model_data[feature_columns] = model_data[feature_columns].fillna(0)
    model_data = model_data.sort_values('Date')
    unique_dates = np.sort(model_data['Date'].unique())
    split_index = int(0.8 * len(unique_dates))
    train_dates = unique_dates[:split_index]
    val_dates = unique_dates[split_index:]
    
    train_data = model_data[model_data['Date'].isin(train_dates)]
    val_data = model_data[model_data['Date'].isin(val_dates)]
    
    X_train = train_data[feature_columns].values
    y_train = train_data['Return'].values.reshape(-1, 1)
    X_val = val_data[feature_columns].values
    y_val = val_data['Return'].values.reshape(-1, 1)
    
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)
    
    input_dim = X_train_tensor.shape[1]
    model_net = FeedforwardNet(input_dim).to(device)
    
    # Use AdamW optimizer with updated learning rate and weight decay.
    optimizer = optim.AdamW(model_net.parameters(), lr=0.0005, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    print("Training the model with updated hyperparameters...")
    epochs = 250
    for epoch in range(epochs):
        model_net.train()
        optimizer.zero_grad()
        outputs = model_net(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        if epoch % 25 == 0:
            model_net.eval()
            with torch.no_grad():
                val_outputs = model_net(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor)
            print(f"Epoch {epoch:3d} | Train Loss: {loss.item():.6f} | Val Loss: {val_loss.item():.6f}")
    
    model_info = {
        'model': model_net,
        'scaler': scaler,
        'feature_columns': feature_columns,
        'device': device
    }
    print("Training complete. Model is ready.")
    return model_info

def predict_returns(model, sentiment_data_today, stock_universe_today):
    """
    Generate predictions of next-day returns for all stocks in the universe.
    
    Parameters
    ----------
    model : dict
        Contains the trained model, scaler, feature columns, and device info.
    sentiment_data_today : DataFrame
        New sentiment data (for a single day).
    stock_universe_today : list
        List of stock tickers available today.
    
    Returns
    -------
    predictions : DataFrame
        A DataFrame with columns ['Ticker', 'Predicted_Return', 'Signal_Rank'].
    """
    # Preprocess today's sentiment data.
    sentiment_data_today = preprocess_sentiment_data(sentiment_data_today)
    if sentiment_data_today.empty:
        current_date = pd.Timestamp.today().normalize()
    else:
        current_date = sentiment_data_today['Date'].max()
    sentiment_today = sentiment_data_today[sentiment_data_today['Date'] == current_date].copy()
    features_today = create_features(sentiment_today)
    universe_upper = [t.upper() for t in stock_universe_today]
    if features_today.empty or 'Ticker' not in features_today.columns:
        default_data = []
        for t in universe_upper:
            default_row = {col: 0 for col in model['feature_columns']}
            default_row['Ticker'] = t
            default_row['Date'] = current_date
            default_data.append(default_row)
        features_today = pd.DataFrame(default_data)
    else:
        features_today['Ticker'] = features_today['Ticker'].str.upper()
        features_today = features_today[features_today['Ticker'].isin(universe_upper)]
        existing_tickers = set(features_today['Ticker'])
        missing_tickers = set(universe_upper) - existing_tickers
        if missing_tickers:
            default_data = []
            for t in missing_tickers:
                default_row = {col: 0 for col in model['feature_columns']}
                default_row['Ticker'] = t
                default_row['Date'] = current_date
                default_data.append(default_row)
            if default_data:
                default_df = pd.DataFrame(default_data)
                features_today = pd.concat([features_today, default_df], ignore_index=True)
    if features_today.empty:
        default_data = []
        for t in universe_upper:
            default_row = {col: 0 for col in model['feature_columns']}
            default_row['Ticker'] = t
            default_row['Date'] = current_date
            default_data.append(default_row)
        features_today = pd.DataFrame(default_data)
    for col in model['feature_columns']:
        if col not in features_today.columns:
            features_today[col] = 0
    if 'Ticker' not in features_today.columns:
        features_today['Ticker'] = ""
    features_today = features_today.sort_values('Ticker').reset_index(drop=True)
    X_today = features_today[model['feature_columns']].fillna(0).values
    if X_today.shape[0] == 0:
        X_today = np.zeros((len(universe_upper), len(model['feature_columns'])))
        features_today = pd.DataFrame({'Ticker': universe_upper})
        for col in model['feature_columns']:
            features_today[col] = 0
    X_today_scaled = model['scaler'].transform(X_today)
    X_today_tensor = torch.tensor(X_today_scaled, dtype=torch.float32).to(model['device'])
    model_obj = model['model']
    model_obj.eval()
    with torch.no_grad():
        predictions_tensor = model_obj(X_today_tensor)
    predictions_array = predictions_tensor.cpu().numpy().flatten()
    predictions_array += np.random.normal(0, 1e-6, size=predictions_array.shape)
    features_today['Predicted_Return'] = predictions_array
    features_today['Signal_Rank'] = features_today['Predicted_Return'].rank(pct=True)
    predictions = features_today[['Ticker', 'Predicted_Return', 'Signal_Rank']].copy()
    return predictions

####################################
# Test Section (Runs if the script is executed directly)
####################################
if __name__ == "__main__":
    sentiment_data = pd.read_csv('./data/sentiment_train_2017_2021.csv')
    return_data = pd.read_csv('./data/return_train_2017_2021.csv')
    print("Data loaded successfully.")
    return_data['Date'] = pd.to_datetime(return_data['Date']).dt.normalize()
    model_info = train_model(sentiment_data, return_data)
    sample_day = pd.to_datetime('2021-06-01').normalize()
    preprocessed_sentiment = preprocess_sentiment_data(sentiment_data)
    sentiment_data_today = preprocessed_sentiment[preprocessed_sentiment['Date'] == sample_day].copy()
    stock_universe_today = return_data[return_data['Date'] == sample_day]['Ticker'].unique().tolist()
    predictions = predict_returns(model=model_info, sentiment_data_today=sentiment_data_today, stock_universe_today=stock_universe_today)
    print("Sample predictions:")
    print(predictions.head())
