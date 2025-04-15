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
    result = df.copy()
    for col in columns:
        if col in result.columns:
            if method == 'winsorize':
                lower = np.nanpercentile(result[col], limits[0]*100)
                upper = np.nanpercentile(result[col], (1-limits[1])*100)
                result[col] = result[col].clip(lower=lower, upper=upper)
            elif method == 'zscore':
                z_scores = stats.zscore(result[col], nan_policy='omit')
                abs_z_scores = np.abs(z_scores)
                result = result[(abs_z_scores < 3) | np.isnan(abs_z_scores)]
    return result

def compute_sentiment_decay(df, decay_factor=0.5, max_days=7):
    result = df.copy()
    now = pd.Timestamp.now(tz=timezone.utc).tz_convert('America/New_York')
    result['time_diff_days'] = (now - result['Received_Time_EST']).dt.total_seconds() / (24 * 3600)
    result['time_diff_days'] = result['time_diff_days'].clip(upper=max_days)
    result['decay_weight'] = np.exp(-decay_factor * result['time_diff_days'])
    result['decayed_sentiment'] = result['Sentiment'] * result['decay_weight']
    return result

def calculate_topic_sentiment(df):
    if 'Reddit_Topic' not in df.columns or 'Sentiment' not in df.columns:
        return df
    result = df.copy()
    topic_sentiment = result.groupby(['Ticker', 'Date', 'Reddit_Topic']).agg({
        'Sentiment': ['mean', 'count', 'sum'],
        'decayed_sentiment': 'sum'
    })
    topic_sentiment.columns = ['_'.join(col).strip() for col in topic_sentiment.columns.values]
    topic_sentiment = topic_sentiment.reset_index()
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
    pivot_mean = pivot_mean.reset_index()
    pivot_sum = pivot_sum.reset_index()
    result_df = pd.merge(pivot_mean, pivot_sum, on=['Ticker', 'Date'], how='outer')
    return result_df

def create_features(df):
    df = compute_sentiment_decay(df)
    sentiment_columns = ['Sentiment', 'decayed_sentiment']
    df = handle_outliers(df, sentiment_columns, method='winsorize')
    required_columns = ['Ticker', 'Date', 'Sentiment', 'Confidence', 'Prob_POS', 
                        'Prob_NTR', 'Prob_NEG', 'Relevance', 'SourceWeight', 'TopicWeight']
    for col in required_columns:
        if col not in df.columns:
            if col in ['Ticker', 'Date', 'Sentiment']:
                raise ValueError(f"Required column {col} is missing in the data.")
            else:
                df[col] = 0
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
    grouped = df.groupby(['Ticker','Date']).agg(agg_funcs)
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
    
    def weighted_sentiment_func(sub_df):
        if sub_df['Sentiment'].count() == 0:
            return 0
        return (sub_df['Sentiment'] * sub_df['Relevance']).sum() / sub_df['Sentiment'].count()
    
    ws = df.groupby(['Ticker', 'Date'], include_groups=False).apply(weighted_sentiment_func).reset_index(name='weighted_sentiment')
    grouped = pd.merge(grouped, ws, on=['Ticker','Date'], how='left')
    
    if 'Reddit_Topic' in df.columns:
        topic_sentiment_df = calculate_topic_sentiment(df)
        grouped = pd.merge(grouped, topic_sentiment_df, on=['Ticker','Date'], how='left')
    
    def compute_time_series_features(sub_df):
        sub_df = sub_df.sort_values('Date').copy()
        sub_df['cumulative_sentiment'] = sub_df['net_sentiment'].cumsum()
        sub_df['cumulative_decayed_sentiment'] = sub_df['net_decayed_sentiment'].cumsum()
        sub_df['daily_sentiment_change'] = sub_df['net_sentiment'].diff().fillna(0)
        sub_df['ma_5'] = sub_df['net_sentiment'].rolling(window=5, min_periods=1).mean()
        sub_df['ma_10'] = sub_df['net_sentiment'].rolling(window=10, min_periods=1).mean()
        sub_df['past_3_sentiment'] = sub_df['net_sentiment'].shift(1).rolling(window=3, min_periods=1).sum().fillna(0)
        sub_df['log_volume'] = np.log1p(sub_df['post_count'])
        sub_df['sentiment_volatility_7d'] = sub_df['net_sentiment'].rolling(window=7, min_periods=2).std().fillna(0)
        sub_df['sentiment_volatility_14d'] = sub_df['net_sentiment'].rolling(window=14, min_periods=2).std().fillna(0)
        return sub_df
    
    ts_features = grouped.groupby('Ticker', include_groups=False).apply(compute_time_series_features).reset_index(drop=True)
    
    if 'Reddit_Topic' in df.columns and 'Source' in df.columns:
        df_topic = df.copy()
        df_topic['Reddit_Topic'] = df_topic['Reddit_Topic'].fillna("").str.upper()
        df_topic['Source'] = df_topic['Source'].fillna("").str.upper()
        for src in ["WSB", "INVESTING"]:
            src_mask = df_topic['Source'].str.contains(src, case=False, na=False)
            df_src = df_topic[src_mask]
            topic_counts = df_src.groupby(['Ticker','Date'])['Reddit_Topic'].value_counts().unstack(fill_value=0)
            all_topics = [
                "BIOTECH", "CHART", "COMMENTARY", "DAILY DISCUSSION", "DAILY THREAD", "DD", "DISCUSSION", "DISTRESSED",
                "EARNINGS THREAD", "EDUCATION", "ENERGY", "FUNDAMENTALS", "FUTURES", "GAIN", "HELP", "INDUSTRY REPORT",
                "INTERVIEW/PROFILE", "INVESTOR LETTER", "LONG THESIS", "LOSS", "MACRO", "MEME", "MODS", "NEWS", "NONE",
                "OPTIONS", "PROFIT", "QUESTION", "RETAIL", "SATIRE", "SHITPOST", "SHORT THESIS", "SPECIAL SITUATION",
                "STOCKS", "STORYTIME", "STRATEGY", "TAG ME PLS", "TECHNICALS", "THESIS", "WALL ST. \"LEAKS\"",
                "WEEKEND DISCUSSION", "WSBBOOKS", "YOLO"
            ]
            for topic in all_topics:
                col_name = f"{src}_count_{topic}"
                if topic in topic_counts.columns:
                    topic_counts = topic_counts.rename(columns={topic: col_name})
                else:
                    topic_counts[col_name] = 0
            topic_counts = topic_counts.reset_index()[['Ticker','Date'] + [f"{src}_count_{t}" for t in all_topics]]
            ts_features = pd.merge(ts_features, topic_counts, on=['Ticker','Date'], how='left')
            for topic in all_topics:
                col_name = f"{src}_count_{topic}"
                if col_name in ts_features.columns:
                    ts_features[col_name] = ts_features[col_name].fillna(0)
    
    final_df = ts_features.fillna(0)
    return final_df

####################################
# Neural Network Model
####################################

class FeedforwardNet(nn.Module):
    def __init__(self, input_dim):
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
    def __init__(self):
        super(WeightedMSELoss, self).__init__()
        
    def forward(self, pred, target, weights=None):
        if weights is None:
            return nn.MSELoss()(pred, target)
        weights = weights / weights.sum()
        squared_error = (pred - target)**2
        return (weights * squared_error).mean()

####################################
# Multi-Objective Loss Function
####################################
# Note: We'll remove this if we are not optimizing for R^2 exclusively.
class MultiObjectiveLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super(MultiObjectiveLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse_loss = WeightedMSELoss()
    
    def forward(self, pred, target, weights=None):
        mse = self.mse_loss(pred, target, weights)
        target_mean = torch.mean(target)
        sst = torch.sum((target - target_mean)**2) + 1e-6
        sse = torch.sum((pred - target)**2)
        corr_loss = 1 - (1 - sse/sst)  # 1 - R^2
        return self.alpha * mse + self.beta * corr_loss

####################################
# Main Functions: Training and Prediction
####################################

def train_model(sentiment_data, return_data):
    print("Preprocessing sentiment data...")
    sentiment_data = preprocess_sentiment_data(sentiment_data)
    print("Creating features from sentiment data...")
    features_df = create_features(sentiment_data)
    
    return_data = return_data.copy()
    return_data['Date'] = pd.to_datetime(return_data['Date']).dt.normalize()
    return_data['Ticker'] = return_data['Ticker'].str.upper()
    return_data['Return'] = return_data['Return'].apply(convert_return)
    
    print("Merging sentiment features with stock returns...")
    model_data = pd.merge(features_df, return_data[['Date', 'Ticker', 'Return']], on=['Date', 'Ticker'], how='inner')
    model_data = model_data.dropna(subset=['Return'])
    model_data['Return'] = model_data['Return'].apply(convert_return).astype(float)
    
    base_features = ['sentiment_mean', 'sentiment_std', 'post_count', 'avg_confidence',
                     'avg_prob_pos', 'avg_prob_ntr', 'avg_prob_neg', 'weighted_sentiment',
                     'avg_source_weight', 'avg_topic_weight', 'net_sentiment']
    
    new_features = ['net_decayed_sentiment','avg_decayed_sentiment','cumulative_decayed_sentiment',
                    'sentiment_volatility_7d','sentiment_volatility_14d']
    
    time_series_features = ['cumulative_sentiment', 'daily_sentiment_change', 'ma_5', 'ma_10',
                            'past_3_sentiment', 'log_volume']
    
    feature_columns = base_features + new_features + time_series_features
    
    topic_cols = [col for col in model_data.columns if col.startswith('topic_sentiment_') or 
                  col.startswith('topic_decayed_sentiment_')]
    feature_columns.extend(topic_cols)
    
    for src in ["WSB", "INVESTING"]:
        all_topics = [t.upper() for t in [
            "Biotech", "Chart", "Commentary", "Daily Discussion", "Daily Thread", "DD", "Discussion", "Distressed",
            "Earnings Thread", "Education", "Energy", "Fundamentals", "Futures", "Gain", "Help", "Industry Report",
            "Interview/Profile", "Investor Letter", "Long Thesis", "Loss", "Macro", "Meme", "Mods", "News", "None",
            "Options", "Profit", "Question", "Retail", "Satire", "Shitpost", "Short Thesis", "Special Situation",
            "Stocks", "Storytime", "Strategy", "tag me pls", "Technicals", "Thesis", "WALL ST. \"LEAKS\"",
            "Weekend Discussion", "WSBbooks", "YOLO"
        ]]
        for topic in all_topics:
            col_name = f"{src}_count_{topic}"
            if col_name in model_data.columns:
                feature_columns.append(col_name)
    
    for col in feature_columns:
        if col not in model_data.columns:
            model_data[col] = 0
    
    model_data[feature_columns] = model_data[feature_columns].fillna(0)
    model_data = model_data.sort_values('Date')
    model_data = handle_outliers(model_data, ['Return'], method='winsorize')
    
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
    
    optimizer = optim.AdamW(model_net.parameters(), lr=0.0005, weight_decay=1e-4)
    criterion = WeightedMSELoss()
    
    print("Training the model with SGD and weighted loss...")
    epochs = 250
    best_val_loss = float('inf')
    best_model_state = None
    patience = 50
    no_improve_count = 0
    
    for epoch in range(epochs):
        model_net.train()
        optimizer.zero_grad()
        outputs = model_net(X_train_tensor)
        loss = criterion(outputs, y_train_tensor, weights_train_tensor)
        loss.backward()
        optimizer.step()
        
        model_net.eval()
        with torch.no_grad():
            val_outputs = model_net(X_val_tensor)
            val_loss_value = criterion(val_outputs, y_val_tensor, weights_val_tensor).item()
        
        if val_loss_value < best_val_loss:
            best_val_loss = val_loss_value
            best_model_state = model_net.state_dict().copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        if epoch % 25 == 0:
            print(f"Epoch {epoch:3d} | Train Loss: {loss.item():.6f} | Val Loss: {val_loss_value:.6f}")
        
        if no_improve_count >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    if best_model_state is not None:
        model_net.load_state_dict(best_model_state)
    
    # Final retraining on full data (optional)
    X_full_scaled = scaler.transform(X_train)
    Xt_all = torch.tensor(X_full_scaled, dtype=torch.float32).to(device)
    yt_all = torch.tensor(y_train, dtype=torch.float32).to(device)
    final_model = FeedforwardNet(X_full_scaled.shape[1]).to(device)
    final_model.load_state_dict(model_net.state_dict())
    opt_final = optim.SGD(final_model.parameters(), lr=0.0005, momentum=0.9, weight_decay=1e-4)
    for ep in range(50):
        final_model.train()
        opt_final.zero_grad()
        loss_final = criterion(final_model(Xt_all), yt_all, weights_train_tensor)
        loss_final.backward()
        opt_final.step()
        if ep % 10 == 0:
            print(f"Final model - Epoch {ep:3d} | Loss: {loss_final.item():.6f}")
    
    return {
        'model': final_model,
        'scaler': scaler,
        'feature_columns': feature_columns,
        'device': device
    }

def predict_returns(model, sentiment_data_today, stock_universe_today):
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
            default_data.append({'Ticker': t, 'Date': current_date, **{col: 0 for col in model['feature_columns']}})
        features_today = pd.DataFrame(default_data)
    else:
        features_today['Ticker'] = features_today['Ticker'].str.upper()
        features_today = features_today[features_today['Ticker'].isin(universe_upper)]
        existing = set(features_today['Ticker'])
        missing = set(universe_upper) - existing
        if missing:
            default_data = []
            for t in missing:
                default_data.append({'Ticker': t, 'Date': current_date, **{col: 0 for col in model['feature_columns']}})
            if default_data:
                features_today = pd.concat([features_today, pd.DataFrame(default_data)], ignore_index=True)
    
    for col in model['feature_columns']:
        if col not in features_today.columns:
            features_today[col] = 0
    
    features_today = features_today.sort_values('Ticker').reset_index(drop=True)
    
    X = model['scaler'].transform(features_today[model['feature_columns']].fillna(0).values)
    Xt = torch.tensor(X, dtype=torch.float32).to(model['device'])
    model_obj = model['model']
    model_obj.eval()
    with torch.no_grad():
        predictions_tensor = model_obj(Xt)
    
    predictions_array = predictions_tensor.cpu().numpy().flatten()
    
    predictions_array += np.random.normal(0, 1e-6, size=predictions_array.shape)
    features_today['Predicted_Return'] = predictions_array
    features_today['Signal_Rank'] = features_today['Predicted_Return'].rank(pct=True)
    
    predictions = features_today[['Ticker', 'Predicted_Return', 'Signal_Rank']].copy()
    return predictions

if __name__ == "__main__":
    sentiment_data = pd.read_csv('./data/sentiment_train_2017_2021.csv')
    return_data = pd.read_csv('./data/return_train_2017_2021.csv')
    print("Data loaded successfully.")
    return_data['Date'] = pd.to_datetime(return_data['Date']).dt.normalize()
    
    model_info = train_model(sentiment_data, return_data)
    
    sample_day = pd.to_datetime('2021-06-01').normalize()
    sd = preprocess_sentiment_data(sentiment_data)
    sentiment_data_today = sd[sd['Date'] == sample_day].copy()
    
    stock_universe_today = return_data[return_data['Date'] == sample_day]['Ticker'].unique().tolist()
    
    predictions = predict_returns(model=model_info, sentiment_data_today=sentiment_data_today, stock_universe_today=stock_universe_today)
    print("Sample predictions:")
    print(predictions.head())
