"""
001_prediction_function.py

This file provides two main functions:
    1. train_model(sentiment_data, return_data)
    2. predict_returns(model, sentiment_data_today, stock_universe_today)

The code processes Alexandria’s Reddit sentiment data, creates aggregated features, and trains a PyTorch feedforward neural network 
(with GPU support when available) to predict next-day stock returns.
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
    Create advanced daily features by aggregating sentiment data for each Ticker and Date.
    
    Features include:
      - sentiment_mean: Average sentiment.
      - sentiment_std: Standard deviation of sentiment.
      - post_count: Count of posts.
      - avg_confidence: Average confidence.
      - avg_prob_pos, avg_prob_ntr, avg_prob_neg: Averages of probabilities.
      - avg_source_weight, avg_topic_weight: Averages from respective columns.
      - weighted_sentiment: Average (Sentiment * Relevance) per post.
    
    Parameters
    ----------
    df : DataFrame
        Preprocessed sentiment data with columns including 
        ['Ticker', 'Date', 'Sentiment', 'Confidence', 'Prob_POS', 'Prob_NTR', 'Prob_NEG',
         'Relevance', 'SourceWeight', 'TopicWeight'].
    
    Returns
    -------
    grouped : DataFrame
        Aggregated features for each Ticker and Date.
    """
    required_columns = ['Ticker', 'Date', 'Sentiment', 'Confidence', 'Prob_POS', 
                        'Prob_NTR', 'Prob_NEG', 'Relevance', 'SourceWeight', 'TopicWeight']
    for col in required_columns:
        if col not in df.columns:
            if col in ['Ticker', 'Date', 'Sentiment']:
                raise ValueError(f"Required column {col} is missing in the data.")
            else:
                df[col] = 0

    agg_funcs = {
        'Sentiment': ['mean', 'std', 'count'],
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
    grouped = grouped.rename(columns={
        'Sentiment_mean': 'sentiment_mean',
        'Sentiment_std': 'sentiment_std',
        'Sentiment_count': 'post_count',
        'Confidence_mean': 'avg_confidence',
        'Prob_POS_mean': 'avg_prob_pos',
        'Prob_NTR_mean': 'avg_prob_ntr',
        'Prob_NEG_mean': 'avg_prob_neg',
        'SourceWeight_mean': 'avg_source_weight',
        'TopicWeight_mean': 'avg_topic_weight',
        'Relevance_mean': 'avg_relevance'
    })
    
    def weighted_sentiment_func(sub_df):
        if sub_df['Sentiment'].count() == 0:
            return 0
        return (sub_df['Sentiment'] * sub_df['Relevance']).sum() / sub_df['Sentiment'].count()
    
    ws = df.groupby(['Ticker', 'Date']).apply(weighted_sentiment_func).reset_index(name='weighted_sentiment')
    grouped = pd.merge(grouped, ws, on=['Ticker', 'Date'], how='left')
    return grouped

####################################
# PyTorch Neural Network Model
####################################

class FeedforwardNet(nn.Module):
    def __init__(self, input_dim):
        """
        A simple feedforward neural network with two hidden layers.
        """
        super(FeedforwardNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
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
    
    # Preprocess return_data: convert Date to datetime and normalize, Ticker to uppercase, and convert Return.
    return_data = return_data.copy()
    return_data['Date'] = pd.to_datetime(return_data['Date']).dt.normalize()
    return_data['Ticker'] = return_data['Ticker'].str.upper()
    return_data['Return'] = return_data['Return'].apply(convert_return)
    
    print("Merging sentiment features with stock returns...")
    model_data = pd.merge(features_df, return_data[['Date', 'Ticker', 'Return']], on=['Date', 'Ticker'], how='inner')
    model_data = model_data.dropna(subset=['Return'])
    
    model_data['Return'] = model_data['Return'].apply(convert_return).astype(float)
    feature_columns = ['sentiment_mean', 'sentiment_std', 'post_count', 'avg_confidence', 
                         'avg_prob_pos', 'avg_prob_ntr', 'avg_prob_neg', 'weighted_sentiment', 
                         'avg_source_weight', 'avg_topic_weight']
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
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model_net.parameters(), lr=0.001)
    
    print("Training the model...")
    epochs = 200
    for epoch in range(epochs):
        model_net.train()
        optimizer.zero_grad()
        outputs = model_net(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
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
    
    # Define current_date: if no sentiment data is present, use today's normalized date;
    # otherwise, use the maximum date.
    if sentiment_data_today.empty:
        current_date = pd.Timestamp.today().normalize()
    else:
        current_date = sentiment_data_today['Date'].max()
    
    # Filter sentiment_data_today to only include rows for current_date.
    sentiment_today = sentiment_data_today[sentiment_data_today['Date'] == current_date].copy()
    features_today = create_features(sentiment_today)
    
    # Convert provided stock universe tickers to uppercase.
    universe_upper = [t.upper() for t in stock_universe_today]
    
    # If features_today is empty or missing the 'Ticker' column, create default rows.
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
        # Add missing tickers if necessary.
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
    
    # Final safeguard: if features_today is still empty, force a default DataFrame.
    if features_today.empty:
        default_data = []
        for t in universe_upper:
            default_row = {col: 0 for col in model['feature_columns']}
            default_row['Ticker'] = t
            default_row['Date'] = current_date
            default_data.append(default_row)
        features_today = pd.DataFrame(default_data)
    
    # Ensure that all required feature columns exist.
    for col in model['feature_columns']:
        if col not in features_today.columns:
            features_today[col] = 0
    
    if 'Ticker' not in features_today.columns:
        features_today['Ticker'] = ""
    
    features_today = features_today.sort_values('Ticker').reset_index(drop=True)
    
    # Prepare features for prediction.
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
    try:
        sentiment_data = pd.read_csv('./data/sentiment_train_2017_2021.csv')
        return_data = pd.read_csv('./data/return_train_2017_2021.csv')
        print("Data loaded successfully.")
    except FileNotFoundError:
        print("Data files not found. Please check file paths.")
        sentiment_data = None
        return_data = None

    if sentiment_data is not None and return_data is not None:
        # Normalize the return_data Date column.
        return_data['Date'] = pd.to_datetime(return_data['Date']).dt.normalize()
        
        model_info = train_model(sentiment_data, return_data)
        
        sample_day = pd.to_datetime('2021-06-01').normalize()
        preprocessed_sentiment = preprocess_sentiment_data(sentiment_data)
        sentiment_data_today = preprocessed_sentiment[preprocessed_sentiment['Date'] == sample_day].copy()
        
        stock_universe_today = return_data[return_data['Date'] == sample_day]['Ticker'].unique().tolist()
        
        predictions = predict_returns(model=model_info, sentiment_data_today=sentiment_data_today, stock_universe_today=stock_universe_today)
        print("Sample predictions:")
        print(predictions.head())
