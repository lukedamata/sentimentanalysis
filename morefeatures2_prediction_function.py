import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from datetime import datetime, time, timedelta, timezone
from scipy import stats

# Set device for GPU usage if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

####################################
# Helper Functions
####################################

def convert_return(x):
    if isinstance(x, str):
        x = x.strip()
        if x.endswith('%'):
            return float(x[:-1].strip()) / 100.0
        else:
            return float(x)
    else:
        return float(x)

def preprocess_sentiment_data(sentiment_data):
    df = sentiment_data.copy()
    if 'Received_Time' not in df.columns:
        raise ValueError("Column 'Received_Time' not found in the sentiment data.")
    df['Received_Time'] = pd.to_datetime(df['Received_Time'], utc=True)
    df['Received_Time_EST'] = df['Received_Time'].dt.tz_convert('America/New_York')
    df['local_date'] = df['Received_Time_EST'].dt.date
    cutoff = time(16, 0)  # 4:00 PM cutoff
    df['Date'] = df['Received_Time_EST'].apply(
        lambda x: pd.to_datetime(x.date() + timedelta(days=1)) if x.time() > cutoff else pd.to_datetime(x.date())
    )
    if 'Ticker' in df.columns:
        df['Ticker'] = df['Ticker'].str.upper()
    return df

def handle_outliers(df, columns, method='winsorize', limits=(0.01, 0.01)):
    """
    Handle outliers in specified columns using winsorization or z-score method
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe
    columns : list
        List of columns to process
    method : str, optional
        Method to handle outliers ('winsorize' or 'zscore')
    limits : tuple, optional
        Lower and upper percentiles for winsorization
        
    Returns:
    --------
    DataFrame with outliers handled
    """
    result = df.copy()
    for col in columns:
        if col in result.columns:
            if method == 'winsorize':
                # Winsorize the column values (clip at specified percentiles)
                lower = np.nanpercentile(result[col], limits[0] * 100)
                upper = np.nanpercentile(result[col], (1 - limits[1]) * 100)
                result[col] = result[col].clip(lower=lower, upper=upper)
            elif method == 'zscore':
                # Remove outliers based on z-score
                z_scores = stats.zscore(result[col], nan_policy='omit')
                abs_z_scores = np.abs(z_scores)
                result = result[(abs_z_scores < 3) | np.isnan(abs_z_scores)]
    return result

def compute_sentiment_decay(df, decay_factor=0.5, max_days=7):
    """
    Apply time decay to sentiment values - more recent posts have higher weight
    
    Parameters:
    -----------
    df : DataFrame
        Sentiment data with 'Received_Time_EST' and 'Sentiment' columns
    decay_factor : float
        Controls decay rate of sentiment influence
    max_days : int
        Maximum number of days to consider for decay
        
    Returns:
    --------
    DataFrame with time-decayed sentiment column added
    """
    result = df.copy()
    now = pd.Timestamp.now(tz=timezone.utc).tz_convert('America/New_York')
    
    # Calculate time difference in days (as float)
    result['time_diff_days'] = (now - result['Received_Time_EST']).dt.total_seconds() / (24 * 3600)
    
    # Clip time difference to max_days
    result['time_diff_days'] = result['time_diff_days'].clip(upper=max_days)
    
    # Calculate decay factor
    result['decay_weight'] = np.exp(-decay_factor * result['time_diff_days'])
    
    # Apply decay to sentiment
    result['decayed_sentiment'] = result['Sentiment'] * result['decay_weight']
    
    return result

def calculate_topic_sentiment(df):
    """
    Calculate sentiment metrics for each topic
    
    Parameters:
    -----------
    df : DataFrame
        Sentiment data with 'Reddit_Topic' and 'Sentiment' columns
        
    Returns:
    --------
    DataFrame with topic-specific sentiment features
    """
    if 'Reddit_Topic' not in df.columns or 'Sentiment' not in df.columns:
        return df
    
    result = df.copy()
    
    # Create an aggregation for topic sentiment
    topic_sentiment = result.groupby(['Ticker', 'Date', 'Reddit_Topic']).agg({
        'Sentiment': ['mean', 'count', 'sum'],
        'decayed_sentiment': 'sum'
    })
    
    topic_sentiment.columns = ['_'.join(col).strip() for col in topic_sentiment.columns.values]
    topic_sentiment = topic_sentiment.reset_index()
    
    # Pivot to create features for each topic
    pivot_mean = topic_sentiment.pivot_table(
        index=['Ticker', 'Date'], 
        columns='Reddit_Topic', 
        values='Sentiment_mean',
        fill_value=0
    ).add_prefix('topic_sentiment_')
    
    pivot_sum = topic_sentiment.pivot_table(
        index=['Ticker', 'Date'], 
        columns='Reddit_Topic', 
        values='decayed_sentiment_sum',
        fill_value=0
    ).add_prefix('topic_decayed_sentiment_')
    
    # Reset index to merge
    pivot_mean = pivot_mean.reset_index()
    pivot_sum = pivot_sum.reset_index()
    
    # Merge all features
    result_df = pd.merge(pivot_mean, pivot_sum, on=['Ticker', 'Date'], how='outer')
    
    return result_df

def create_features(df):
    """
    Create advanced daily features by aggregating sentiment data
    """
    # Apply sentiment decay
    df = compute_sentiment_decay(df)
    
    # Handle outliers for sentiment values
    sentiment_columns = ['Sentiment', 'decayed_sentiment']
    df = handle_outliers(df, sentiment_columns, method='winsorize')
    
    # Ensure required columns exist
    required_columns = ['Ticker', 'Date', 'Sentiment', 'Confidence', 'Prob_POS', 
                        'Prob_NTR', 'Prob_NEG', 'Relevance', 'SourceWeight', 'TopicWeight']
    for col in required_columns:
        if col not in df.columns:
            if col in ['Ticker', 'Date', 'Sentiment']:
                raise ValueError(f"Required column {col} is missing in the data.")
            else:
                df[col] = 0

    # Base aggregation: calculate mean, std, count, and sum (for net sentiment)
    agg_funcs = {
        'Sentiment': ['mean', 'std', 'count', 'sum'],
        'decayed_sentiment': ['sum', 'mean'],
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
        'decayed_sentiment_sum': 'net_decayed_sentiment',
        'decayed_sentiment_mean': 'avg_decayed_sentiment',
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
    
    # Calculate topic-specific sentiment if topic data exists
    if 'Reddit_Topic' in df.columns:
        topic_sentiment_df = calculate_topic_sentiment(df)
        grouped = pd.merge(grouped, topic_sentiment_df, on=['Ticker', 'Date'], how='left')
    
    # Compute time-series features per ticker
    def compute_time_series_features(sub_df):
        sub_df = sub_df.sort_values('Date').copy()
        sub_df['cumulative_sentiment'] = sub_df['net_sentiment'].cumsum()
        sub_df['cumulative_decayed_sentiment'] = sub_df['net_decayed_sentiment'].cumsum()
        sub_df['daily_sentiment_change'] = sub_df['net_sentiment'].diff().fillna(0)
        sub_df['ma_5'] = sub_df['net_sentiment'].rolling(window=5, min_periods=1).mean()
        sub_df['ma_10'] = sub_df['net_sentiment'].rolling(window=10, min_periods=1).mean()
        sub_df['past_3_sentiment'] = sub_df['net_sentiment'].shift(1).rolling(window=3, min_periods=1).sum().fillna(0)
        sub_df['log_volume'] = np.log1p(sub_df['post_count'])
        
        # Add volatility measures
        sub_df['sentiment_volatility_7d'] = sub_df['net_sentiment'].rolling(window=7, min_periods=2).std().fillna(0)
        sub_df['sentiment_volatility_14d'] = sub_df['net_sentiment'].rolling(window=14, min_periods=2).std().fillna(0)
        
        return sub_df
    
    ts_features = grouped.groupby('Ticker').apply(compute_time_series_features).reset_index(drop=True)
    
    # Add topic count features like in the original code
    if 'Reddit_Topic' in df.columns and 'Source' in df.columns:
        # Create a copy for topic computations
        df_topic = df.copy()
        df_topic['Reddit_Topic'] = df_topic['Reddit_Topic'].fillna("").str.upper()
        df_topic['Source'] = df_topic['Source'].fillna("").str.upper()
        
        # Define two source categories: "WSB" and "INVESTING"
        for src in ["WSB", "INVESTING"]:
            src_mask = df_topic['Source'].str.contains(src, case=False, na=False)
            df_src = df_topic[src_mask]
            # Group by Ticker, Date and Reddit_Topic, then count posts
            topic_counts = df_src.groupby(['Ticker', 'Date'])['Reddit_Topic'].value_counts().unstack(fill_value=0)
            
            all_topics = [
                "Biotech", "Chart", "Commentary", "Daily Discussion", "Daily Thread", "DD", "Discussion", "Distressed",
                "Earnings Thread", "Education", "Energy", "Fundamentals", "Futures", "Gain", "Help", "Industry Report",
                "Interview/Profile", "Investor Letter", "Long Thesis", "Loss", "Macro", "Meme", "Mods", "News", "None",
                "Options", "Profit", "Question", "Retail", "Satire", "Shitpost", "Short Thesis", "Special Situation",
                "Stocks", "Storytime", "Strategy", "tag me pls", "Technicals", "Thesis", "Wall St. \"Leaks\"",
                "Weekend Discussion", "WSBbooks", "YOLO"
            ]
            
            # For each topic in our list, ensure there is a column
            for topic in all_topics:
                topic_upper = topic.upper()
                col_name = f"{src}_count_{topic_upper}"
                if topic_upper in topic_counts.columns:
                    # Rename existing column to the standardized name
                    topic_counts = topic_counts.rename(columns={topic_upper: col_name})
                else:
                    # Create the column with default value 0
                    topic_counts[col_name] = 0
            
            # Reset index so that Ticker and Date become columns
            topic_counts = topic_counts.reset_index()[['Ticker', 'Date'] + [f"{src}_count_{t.upper()}" for t in all_topics]]
            
            # Merge with the ts_features DataFrame
            ts_features = pd.merge(ts_features, topic_counts, on=['Ticker', 'Date'], how='left')
            
            # Fill missing topic count columns with 0
            for topic in all_topics:
                col_name = f"{src}_count_{topic.upper()}"
                if col_name in ts_features.columns:
                    ts_features[col_name] = ts_features[col_name].fillna(0)
    
    final_df = ts_features.fillna(0)
    return final_df

####################################
# Updated Neural Network Model
####################################

class FeedforwardNet(nn.Module):
    def __init__(self, input_dim):
        """
        A feedforward neural network with two hidden layers, dropout, and updated architecture
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
# Custom Loss Functions
####################################

class WeightedMSELoss(nn.Module):
    """
    MSE loss that weights samples based on trading volume or post count
    """
    def __init__(self):
        super(WeightedMSELoss, self).__init__()
        
    def forward(self, pred, target, weights=None):
        if weights is None:
            return nn.MSELoss()(pred, target)
        
        # Normalize weights
        weights = weights / weights.sum()
        # Calculate squared error
        squared_error = (pred - target) ** 2
        # Apply weights
        weighted_squared_error = weights * squared_error
        # Return mean
        return weighted_squared_error.mean()

####################################
# Main Functions
####################################

def train_model(sentiment_data, return_data):
    """
    Train a model using sentiment features to predict next-day returns
    """
    print("Preprocessing sentiment data...")
    sentiment_data = preprocess_sentiment_data(sentiment_data)
    print("Creating features from sentiment data...")
    features_df = create_features(sentiment_data)
    
    # Preprocess return_data
    return_data = return_data.copy()
    return_data['Date'] = pd.to_datetime(return_data['Date']).dt.normalize()
    return_data['Ticker'] = return_data['Ticker'].str.upper()
    return_data['Return'] = return_data['Return'].apply(convert_return)
    
    print("Merging sentiment features with stock returns...")
    model_data = pd.merge(features_df, return_data[['Date', 'Ticker', 'Return']], on=['Date', 'Ticker'], how='inner')
    model_data = model_data.dropna(subset=['Return'])
    model_data['Return'] = model_data['Return'].apply(convert_return).astype(float)
    
    # Get feature columns
    base_features = [
        'sentiment_mean', 'sentiment_std', 'post_count', 'avg_confidence',
        'avg_prob_pos', 'avg_prob_ntr', 'avg_prob_neg', 'weighted_sentiment',
        'avg_source_weight', 'avg_topic_weight', 'net_sentiment'
    ]
    
    # Add new features
    new_features = [
        'net_decayed_sentiment', 'avg_decayed_sentiment', 'cumulative_decayed_sentiment',
        'sentiment_volatility_7d', 'sentiment_volatility_14d'
    ]
    
    time_series_features = [
        'cumulative_sentiment', 'daily_sentiment_change', 'ma_5', 'ma_10',
        'past_3_sentiment', 'log_volume'
    ]
    
    # Collect all feature columns
    feature_columns = base_features + new_features + time_series_features
    
    # Add topic-specific sentiment columns if they exist
    topic_cols = [col for col in model_data.columns if col.startswith('topic_sentiment_') or 
                  col.startswith('topic_decayed_sentiment_')]
    feature_columns.extend(topic_cols)
    
    # Add topic count features for both sources
    for src in ["WSB", "INVESTING"]:
        all_topics = [
            "Biotech", "Chart", "Commentary", "Daily Discussion", "Daily Thread", "DD", "Discussion", "Distressed",
            "Earnings Thread", "Education", "Energy", "Fundamentals", "Futures", "Gain", "Help", "Industry Report",
            "Interview/Profile", "Investor Letter", "Long Thesis", "Loss", "Macro", "Meme", "Mods", "News", "None",
            "Options", "Profit", "Question", "Retail", "Satire", "Shitpost", "Short Thesis", "Special Situation",
            "Stocks", "Storytime", "Strategy", "tag me pls", "Technicals", "Thesis", "Wall St. \"Leaks\"",
            "Weekend Discussion", "WSBbooks", "YOLO"
        ]
        for topic in [t.upper() for t in all_topics]:
            col_name = f"{src}_count_{topic}"
            if col_name in model_data.columns:
                feature_columns.append(col_name)
    
    # Make sure all feature columns exist in the dataframe
    for col in feature_columns:
        if col not in model_data.columns:
            model_data[col] = 0
    
    model_data[feature_columns] = model_data[feature_columns].fillna(0)
    model_data = model_data.sort_values('Date')
    
    # Handle outliers in target variable
    model_data = handle_outliers(model_data, ['Return'], method='winsorize')
    
    # Time-based train/validation split
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
    
    # Create weights for weighted loss function based on post count
    weights_train = train_data['post_count'].values.reshape(-1, 1)
    weights_val = val_data['post_count'].values.reshape(-1, 1)
    
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    weights_train_tensor = torch.tensor(weights_train, dtype=torch.float32).to(device)
    
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)
    weights_val_tensor = torch.tensor(weights_val, dtype=torch.float32).to(device)
    
    input_dim = X_train_tensor.shape[1]
    model_net = FeedforwardNet(input_dim).to(device)
    
    # Use AdamW optimizer with updated learning rate and weight decay
    optimizer = optim.AdamW(model_net.parameters(), lr=0.0005, weight_decay=1e-4)
    criterion = WeightedMSELoss()
    
    print("Training the model with updated features and weighted loss...")
    epochs = 250
    best_val_loss = float('inf')
    best_model_state = None
    patience = 25
    no_improve_count = 0
    
    for epoch in range(epochs):
        model_net.train()
        optimizer.zero_grad()
        outputs = model_net(X_train_tensor)
        loss = criterion(outputs, y_train_tensor, weights_train_tensor)
        loss.backward()
        optimizer.step()
        
        # Validation
        model_net.eval()
        with torch.no_grad():
            val_outputs = model_net(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor, weights_val_tensor)
            val_loss_value = val_loss.item()
        
        # Early stopping check
        if val_loss_value < best_val_loss:
            best_val_loss = val_loss_value
            best_model_state = model_net.state_dict().copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        if epoch % 25 == 0:
            print(f"Epoch {epoch:3d} | Train Loss: {loss.item():.6f} | Val Loss: {val_loss_value:.6f}")
        
        # Stop if no improvement for 'patience' epochs
        if no_improve_count >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    # Load best model if we have one
    if best_model_state is not None:
        model_net.load_state_dict(best_model_state)
    
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
    Generate predictions of next-day returns for all stocks in the universe
    """
    # Preprocess today's sentiment data
    sentiment_data_today = preprocess_sentiment_data(sentiment_data_today)
    if sentiment_data_today.empty:
        current_date = pd.Timestamp.today().normalize()
    else:
        current_date = sentiment_data_today['Date'].max()
    
    sentiment_today = sentiment_data_today[sentiment_data_today['Date'] == current_date].copy()
    features_today = create_features(sentiment_today)
    
    universe_upper = [t.upper() for t in stock_universe_today]
    
    # Handle empty features or missing tickers
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
        
        # Add missing tickers
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
    
    # Add missing feature columns
    for col in model['feature_columns']:
        if col not in features_today.columns:
            features_today[col] = 0
    
    features_today = features_today.sort_values('Ticker').reset_index(drop=True)
    
    X_today = features_today[model['feature_columns']].fillna(0).values
    X_today_scaled = model['scaler'].transform(X_today)
    X_today_tensor = torch.tensor(X_today_scaled, dtype=torch.float32).to(model['device'])
    
    model_obj = model['model']
    model_obj.eval()
    with torch.no_grad():
        predictions_tensor = model_obj(X_today_tensor)
    
    predictions_array = predictions_tensor.cpu().numpy().flatten()
    
    # Add small random noise to break ties
    predictions_array += np.random.normal(0, 1e-6, size=predictions_array.shape)
    
    features_today['Predicted_Return'] = predictions_array
    features_today['Signal_Rank'] = features_today['Predicted_Return'].rank(pct=True)
    
    predictions = features_today[['Ticker', 'Predicted_Return', 'Signal_Rank']].copy()
    return predictions