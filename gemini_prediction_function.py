# 001_prediction_function.py
# (Replace 001 with your group number)

"""
This file provides two main functions for predicting stock returns based on Reddit sentiment:
    1. train_model(sentiment_data, return_data): Trains a PyTorch neural network.
    2. predict_returns(model, sentiment_data_today, stock_universe_today): Generates predictions.

Feature Engineering: Includes sentiment statistics, volume indicators, temporal patterns,
probability measures (incl. difference), source/topic/relevance weights, and interaction features.

Model: A feedforward neural network with Batch Normalization, implemented in PyTorch with GPU support.
Target Clipping: Clips extreme return values during training.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from sklearn.preprocessing import StandardScaler
from scipy.stats import skew
from datetime import time, timedelta
import warnings

# Suppress potential warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)


# Set device for GPU usage if available
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"Device name: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"Error setting device: {e}. Defaulting to CPU.")
    device = torch.device("cpu")


####################################
# Helper Functions
####################################

def convert_return(x):
    """Converts a return value to a float."""
    if isinstance(x, str):
        x = x.strip()
        if x.endswith('%'):
            try:
                return float(x[:-1].strip()) / 100.0
            except ValueError: return np.nan
        else:
            try: return float(x)
            except ValueError: return np.nan
    elif isinstance(x, (int, float)):
        return float(x)
    else:
        return np.nan

def preprocess_sentiment_data(sentiment_data):
    """Preprocesses raw sentiment data."""
    if not isinstance(sentiment_data, pd.DataFrame):
        raise TypeError("sentiment_data must be a pandas DataFrame.")
    df = sentiment_data.copy()
    if 'Received_Time' not in df.columns: raise ValueError("'Received_Time' not found.")
    try:
        df['Received_Time'] = pd.to_datetime(df['Received_Time'], errors='coerce', utc=True)
        df = df.dropna(subset=['Received_Time'])
    except Exception as e: raise ValueError(f"Error converting 'Received_Time': {e}")
    if df.empty:
        print("Warning: No valid 'Received_Time' entries.")
        df['Received_Time_EST'] = pd.Series(dtype='datetime64[ns, America/New_York]')
        df['Date'] = pd.Series(dtype='datetime64[ns]')
        df['Ticker'] = pd.Series(dtype='object')
        return df
    df['Received_Time_EST'] = df['Received_Time'].dt.tz_convert('America/New_York')
    df['local_date'] = df['Received_Time_EST'].dt.date
    cutoff_time = time(16, 0)
    df['Date'] = df['Received_Time_EST'].apply(
        lambda dt: pd.to_datetime(dt.date() + timedelta(days=1)) if dt.time() > cutoff_time else pd.to_datetime(dt.date())
    ).dt.normalize()
    if 'Ticker' in df.columns:
        df['Ticker'] = df['Ticker'].astype(str).str.upper().str.strip()
    else: raise ValueError("'Ticker' not found.")
    required_cols = {
        'Sentiment': 0, 'Confidence': 0.0, 'Prob_POS': 0.0, 'Prob_NTR': 0.0, 'Prob_NEG': 0.0,
        'Relevance': 0.0, 'SourceWeight': 0.0, 'TopicWeight': 0.0, 'Author': 'missing_author',
        'Novelty': 1, 'StoryID': 'missing_storyid'
    }
    for col, default in required_cols.items():
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found. Adding default.")
            df[col] = default
    numeric_cols = ['Sentiment', 'Confidence', 'Prob_POS', 'Prob_NTR', 'Prob_NEG',
                    'Relevance', 'SourceWeight', 'TopicWeight', 'Novelty']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if col != 'Sentiment': df[col].fillna(0, inplace=True)
    if 'Author' in df.columns: df['Author'] = df['Author'].astype(str)
    return df

def create_features(df_processed):
    """
    Creates aggregated daily features for each Ticker. Includes prob_diff feature.
    """
    if df_processed.empty:
        print("Warning: Input DataFrame to create_features is empty.")
        feature_names = [ # Define expected columns for empty output
            'sentiment_mean', 'sentiment_std', 'sentiment_skew', 'pos_neg_ratio',
            'log_post_count', 'author_count', 'sentiment_am_pm_diff', 'volume_am_pm_ratio',
            'avg_confidence', 'avg_prob_pos', 'avg_prob_ntr', 'avg_prob_neg', 'prob_diff', # Added prob_diff
            'prob_uncertainty', 'avg_source_weight', 'avg_topic_weight',
            'weighted_sentiment_relevance', 'avg_relevance', 'avg_novelty',
            'sentiment_volume_interaction'
        ]
        return pd.DataFrame(columns=['Ticker', 'Date'] + feature_names)

    # Aggregation
    agg_funcs = {
        'Sentiment': ['mean', 'std', lambda x: skew(x.dropna())], 'Confidence': 'mean',
        'Prob_POS': 'mean', 'Prob_NTR': 'mean', 'Prob_NEG': 'mean', 'Relevance': 'mean',
        'SourceWeight': 'mean', 'TopicWeight': 'mean', 'Novelty': 'mean',
        'Author': pd.Series.nunique, 'StoryID': 'count'
    }
    grouped = df_processed.groupby(['Ticker', 'Date']).agg(agg_funcs)
    grouped.columns = [
        'sentiment_mean', 'sentiment_std', 'sentiment_skew', 'avg_confidence',
        'avg_prob_pos', 'avg_prob_ntr', 'avg_prob_neg', 'avg_relevance',
        'avg_source_weight', 'avg_topic_weight', 'avg_novelty',
        'author_count', 'post_count'
    ]
    grouped = grouped.reset_index() # Reset index early for easier merging

    # Log Post Count
    grouped['log_post_count'] = np.log1p(grouped['post_count'])

    # Positive/Negative Ratio
    def calculate_pos_neg_ratio(sub_df):
        pos = (sub_df['Sentiment'] == 1).sum()
        neg = (sub_df['Sentiment'] == -1).sum()
        return float(pos) / neg if neg > 0 else (float(pos) if pos > 0 else 0.0)
    pos_neg = df_processed.groupby(['Ticker', 'Date']).apply(calculate_pos_neg_ratio).reset_index(name='pos_neg_ratio')
    grouped = pd.merge(grouped, pos_neg, on=['Ticker', 'Date'], how='left')

    # Weighted Sentiment (Relevance)
    def calculate_weighted_sentiment(sub_df):
        rel = pd.to_numeric(sub_df['Relevance'], errors='coerce').fillna(0)
        sent = pd.to_numeric(sub_df['Sentiment'], errors='coerce').fillna(0)
        tot_rel = rel.sum()
        return (sent * rel).sum() / tot_rel if tot_rel > 0 else 0.0
    weighted_sent = df_processed.groupby(['Ticker', 'Date']).apply(calculate_weighted_sentiment).reset_index(name='weighted_sentiment_relevance')
    grouped = pd.merge(grouped, weighted_sent, on=['Ticker', 'Date'], how='left')

    # Probability Uncertainty
    def calculate_uncertainty(sub_df):
        prob_cols = ['Prob_POS', 'Prob_NEG', 'Prob_NTR']
        for col in prob_cols: sub_df[col] = pd.to_numeric(sub_df[col], errors='coerce').fillna(0)
        probs = sub_df[prob_cols].values
        if probs.shape[0] == 0: return 0.0
        uncert = 1.0 - np.max(probs, axis=1)
        return uncert.mean() if len(uncert) > 0 else 0.0
    prob_uncert = df_processed.groupby(['Ticker', 'Date']).apply(calculate_uncertainty).reset_index(name='prob_uncertainty')
    grouped = pd.merge(grouped, prob_uncert, on=['Ticker', 'Date'], how='left')

    # Temporal Features (AM/PM)
    noon_est = time(12, 0)
    df_processed['is_am'] = df_processed['Received_Time_EST'].dt.time < noon_est
    def calculate_temporal_diffs(sub_df):
        sub_df['Sentiment'] = pd.to_numeric(sub_df['Sentiment'], errors='coerce')
        sent_am = sub_df.loc[sub_df['is_am'], 'Sentiment'].mean()
        sent_pm = sub_df.loc[~sub_df['is_am'], 'Sentiment'].mean()
        vol_am = sub_df['is_am'].sum(); vol_pm = (~sub_df['is_am']).sum()
        sent_diff = sent_am - sent_pm if pd.notna(sent_am) and pd.notna(sent_pm) else 0.0
        vol_ratio = float(vol_am) / vol_pm if vol_pm > 0 else (float(vol_am) if vol_am > 0 else 0.0)
        return pd.Series({'sentiment_am_pm_diff': sent_diff, 'volume_am_pm_ratio': vol_ratio})
    temporal = df_processed.groupby(['Ticker', 'Date']).apply(calculate_temporal_diffs).reset_index()
    grouped = pd.merge(grouped, temporal, on=['Ticker', 'Date'], how='left')

    # Interaction Feature
    grouped['sentiment_volume_interaction'] = grouped['sentiment_mean'] * grouped['log_post_count']

    # *** NEW: Probability Difference Feature ***
    grouped['prob_diff'] = grouped['avg_prob_pos'] - grouped['avg_prob_neg']

    # Final Cleanup
    grouped = grouped.fillna(0)
    grouped.replace([np.inf, -np.inf], 0, inplace=True)

    # Define final feature set
    feature_order = [
        'sentiment_mean', 'sentiment_std', 'sentiment_skew', 'pos_neg_ratio',
        'log_post_count', 'author_count', 'sentiment_am_pm_diff', 'volume_am_pm_ratio',
        'avg_confidence', 'avg_prob_pos', 'avg_prob_ntr', 'avg_prob_neg', 'prob_diff', # Added prob_diff
        'prob_uncertainty', 'avg_source_weight', 'avg_topic_weight',
        'weighted_sentiment_relevance', 'avg_relevance', 'avg_novelty',
        'sentiment_volume_interaction'
    ]
    final_cols = ['Ticker', 'Date']
    for col in feature_order:
        if col not in grouped.columns: grouped[col] = 0.0
        final_cols.append(col)
    return grouped[final_cols]


####################################
# PyTorch Neural Network Model
####################################

class FeedforwardNet(nn.Module):
    """
    Feedforward neural network with Batch Normalization layers.
    """
    # *** MODIFIED: Simpler architecture (64, 32) + Batch Norm ***
    def __init__(self, input_dim, hidden_dim1=64, hidden_dim2=32, dropout_rate=0.25):
        """
        Initializes the network layers including BatchNorm.

        Parameters:
        -----------
        input_dim : int
            Number of input features.
        hidden_dim1 : int, optional (default is 64).
        hidden_dim2 : int, optional (default is 32).
        dropout_rate : float, optional (default is 0.25).
        """
        super(FeedforwardNet, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim1)
        self.bn1 = nn.BatchNorm1d(hidden_dim1) # Batch Norm after first linear layer
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)

        self.layer2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2) # Batch Norm after second linear layer
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)

        self.output_layer = nn.Linear(hidden_dim2, 1)

    def forward(self, x):
        """
        Defines the forward pass with BatchNorm.
        """
        x = self.layer1(x)
        # Apply BatchNorm only if batch size > 1 during training
        if self.training and x.size(0) > 1:
             x = self.bn1(x)
        elif not self.training and x.size(0) > 0: # Apply running stats during eval if possible
             # Check if bn1 has running_mean to avoid errors on first eval pass if batch size was 1
             if hasattr(self.bn1, 'running_mean') and self.bn1.running_mean is not None:
                  x = self.bn1(x)
             # else: pass through without BN if running stats aren't available yet

        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.layer2(x)
        if self.training and x.size(0) > 1:
             x = self.bn2(x)
        elif not self.training and x.size(0) > 0:
             if hasattr(self.bn2, 'running_mean') and self.bn2.running_mean is not None:
                  x = self.bn2(x)

        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.output_layer(x)
        return x

####################################
# Main Functions: train_model
####################################

def train_model(sentiment_data, return_data):
    """
    Trains a PyTorch neural network model using sentiment features to predict next-day stock returns.
    Includes target variable clipping and uses BatchNorm in the model.
    """
    print("--- Starting Model Training (Improved v2 - BatchNorm & Clipping) ---")

    # --- 1. Preprocessing ---
    print("Preprocessing sentiment data...")
    try: sentiment_processed = preprocess_sentiment_data(sentiment_data)
    except Exception as e: print(f"Error: {e}"); raise
    print("Preprocessing return data...")
    try:
        return_data_processed = return_data.copy()
        return_data_processed['Date'] = pd.to_datetime(return_data_processed['Date'], errors='coerce').dt.normalize()
        return_data_processed['Ticker'] = return_data_processed['Ticker'].astype(str).str.upper().str.strip()
        return_data_processed['Return'] = return_data_processed['Return'].apply(convert_return)
        return_data_processed = return_data_processed.dropna(subset=['Date', 'Ticker', 'Return'])
        if return_data_processed.empty: raise ValueError("Empty return DataFrame after preprocessing.")
    except Exception as e: print(f"Error: {e}"); raise

    # --- 2. Feature Engineering ---
    print("Creating features...")
    try: features_df = create_features(sentiment_processed)
    except Exception as e: print(f"Error: {e}"); raise
    feature_columns = [col for col in features_df.columns if col not in ['Ticker', 'Date', 'post_count']]
    print(f"Using {len(feature_columns)} features: {feature_columns}")

    # --- 3. Merging Data ---
    print("Merging data...")
    model_data = pd.merge(features_df, return_data_processed[['Date', 'Ticker', 'Return']],
                          on=['Date', 'Ticker'], how='inner')
    model_data = model_data.dropna(subset=['Return'])
    model_data[feature_columns] = model_data[feature_columns].fillna(0)
    model_data.replace([np.inf, -np.inf], 0, inplace=True)
    if model_data.empty: raise ValueError("Empty DataFrame after merging.")
    print(f"Merged data shape: {model_data.shape}")

    # --- 4. Train/Validation Split ---
    print("Splitting data...")
    model_data = model_data.sort_values('Date')
    unique_dates = np.sort(model_data['Date'].unique())
    if len(unique_dates) < 5: raise ValueError("Not enough unique dates for split.")
    split_index = int(0.8 * len(unique_dates))
    train_cutoff_date = unique_dates[split_index]
    train_data = model_data[model_data['Date'] < train_cutoff_date]
    val_data = model_data[model_data['Date'] >= train_cutoff_date]
    if train_data.empty or val_data.empty: raise ValueError("Empty train or validation set.")
    print(f"Train shape: {train_data.shape}, Val shape: {val_data.shape}")

    X_train = train_data[feature_columns].values
    y_train = train_data['Return'].values.reshape(-1, 1)
    X_val = val_data[feature_columns].values
    y_val = val_data['Return'].values.reshape(-1, 1)

    # --- *** NEW: Clip Target Variable (Returns) *** ---
    return_clip_threshold = 0.10 # Clip returns at +/- 10%
    y_train_clipped = np.clip(y_train, -return_clip_threshold, return_clip_threshold)
    y_val_clipped = np.clip(y_val, -return_clip_threshold, return_clip_threshold)
    print(f"Clipping target returns to [{-return_clip_threshold:.2f}, {return_clip_threshold:.2f}] for training.")

    # --- 5. Feature Scaling ---
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # --- 6. PyTorch Model Training ---
    print("Converting data to PyTorch tensors...")
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
    # Use clipped target variables for training
    y_train_tensor = torch.tensor(y_train_clipped, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val_clipped, dtype=torch.float32).to(device)

    input_dim = X_train_tensor.shape[1]
    # *** MODIFIED: Instantiate the updated network with BatchNorm ***
    model_net = FeedforwardNet(input_dim, hidden_dim1=64, hidden_dim2=32, dropout_rate=0.25).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model_net.parameters(), lr=0.0005, weight_decay=1e-5)
    # *** MODIFIED: Adjusted scheduler gamma ***
    scheduler = StepLR(optimizer, step_size=30, gamma=0.75) # Slower decay

    epochs = 150
    batch_size = 256
    patience = 20 # Increased patience slightly for BatchNorm stabilization
    best_val_loss = float('inf')
    epochs_no_improve = 0

    print("Starting PyTorch model training...")
    for epoch in range(epochs):
        model_net.train() # Set model to training mode (enables BatchNorm updates, dropout)
        permutation = torch.randperm(X_train_tensor.size(0))
        train_loss_epoch = 0.0
        for i in range(0, X_train_tensor.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X_train_tensor[indices], y_train_tensor[indices]

            # Skip batch if size is 1 and BatchNorm is used, as BN requires >1 sample
            if batch_x.size(0) <= 1 and isinstance(model_net, FeedforwardNet):
                 continue

            optimizer.zero_grad()
            outputs = model_net(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss_epoch += loss.item() * batch_x.size(0) # Use actual batch size

        # Recalculate total size used in epoch if batches were skipped
        total_train_samples = (permutation.size(0) // batch_size) * batch_size + (permutation.size(0) % batch_size if permutation.size(0) % batch_size > 1 else 0)
        if total_train_samples > 0:
             train_loss_epoch /= total_train_samples
        else:
             train_loss_epoch = 0 # Handle case where all batches were skipped


        # Validation Phase
        model_net.eval() # Set model to evaluation mode (uses running stats for BN, disables dropout)
        val_loss_epoch = 0.0
        with torch.no_grad():
             # No need to shuffle validation data usually
             for i in range(0, X_val_tensor.size(0), batch_size):
                  batch_x_val, batch_y_val = X_val_tensor[i:i+batch_size], y_val_tensor[i:i+batch_size]

                  # Handle potential last batch size of 1 during evaluation if BN is used
                  if batch_x_val.size(0) <= 1 and isinstance(model_net, FeedforwardNet):
                      # Optionally skip or handle differently (e.g., predict but don't update loss average)
                      # For simplicity, we might skip calculating loss for this batch,
                      # or ensure BN handles eval mode correctly (which it should with running stats)
                      pass # BatchNorm should use running stats in eval mode

                  if batch_x_val.size(0) > 0: # Ensure batch is not empty
                     val_outputs = model_net(batch_x_val)
                     val_loss = criterion(val_outputs, batch_y_val)
                     val_loss_epoch += val_loss.item() * batch_x_val.size(0)

        if X_val_tensor.size(0) > 0:
             val_loss_epoch /= X_val_tensor.size(0)
        else:
             val_loss_epoch = 0

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
             print(f"Epoch {epoch+1:3d}/{epochs} | Train Loss: {train_loss_epoch:.6f} | Val Loss: {val_loss_epoch:.6f} | LR: {current_lr:.6f}")

        # Early Stopping Check
        if val_loss_epoch < best_val_loss:
            best_val_loss = val_loss_epoch
            epochs_no_improve = 0
            # torch.save(model_net.state_dict(), 'best_model_bn_clip.pth')
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    # --- 7. Prepare Model Info ---
    model_info = {
        'model': model_net, 'scaler': scaler,
        'feature_columns': feature_columns, 'device': device
    }
    print("--- Training complete. Model Info created. ---")
    return model_info


####################################
# Main Functions: predict_returns
####################################

def predict_returns(model, sentiment_data_today, stock_universe_today):
    """
    Generates predictions of next-day returns using the trained model (v2).
    """
    print("--- Starting Prediction ---")
    if not isinstance(model, dict) or not all(k in model for k in ['model', 'scaler', 'feature_columns', 'device']):
        raise ValueError("Invalid 'model' dictionary provided.")
    if not isinstance(sentiment_data_today, pd.DataFrame):
        if not sentiment_data_today is None: raise TypeError("'sentiment_data_today' must be DataFrame or None.")
    if not isinstance(stock_universe_today, list): raise TypeError("'stock_universe_today' must be a list.")
    if not stock_universe_today:
        print("Warning: Empty stock universe. Returning empty predictions.")
        return pd.DataFrame(columns=['Ticker', 'Predicted_Return', 'Signal_Rank'])

    print("Preprocessing today's sentiment data...")
    if sentiment_data_today is None or sentiment_data_today.empty:
        print("No sentiment data provided.")
        sentiment_today_processed = pd.DataFrame()
    else:
        try: sentiment_today_processed = preprocess_sentiment_data(sentiment_data_today)
        except Exception as e: print(f"Error preprocessing: {e}"); sentiment_today_processed = pd.DataFrame()

    print("Creating features for today...")
    if sentiment_today_processed.empty:
        features_today = pd.DataFrame()
    else:
        try:
            if 'Date' in sentiment_today_processed.columns:
                 target_date = sentiment_today_processed['Date'].max()
                 sentiment_today_processed = sentiment_today_processed[sentiment_today_processed['Date'] == target_date]
            features_today = create_features(sentiment_today_processed) # Includes prob_diff
        except Exception as e: print(f"Error creating features: {e}"); features_today = pd.DataFrame()

    print(f"Aligning features with universe ({len(stock_universe_today)} tickers)...")
    universe_df = pd.DataFrame({'Ticker': [t.upper().strip() for t in stock_universe_today]})
    feature_columns = model['feature_columns']
    scaler = model['scaler']; model_device = model['device']; model_obj = model['model']

    if not features_today.empty and 'Ticker' in features_today.columns:
        features_today['Ticker'] = features_today['Ticker'].astype(str).str.upper().str.strip()
        final_features_df = pd.merge(universe_df, features_today, on='Ticker', how='left')
    else:
        final_features_df = universe_df.copy()
        for col in feature_columns: final_features_df[col] = 0.0

    # Ensure all feature columns exist, including the new 'prob_diff'
    for col in feature_columns:
        if col not in final_features_df.columns:
            print(f"Warning: Feature column '{col}' missing. Adding default 0.")
            final_features_df[col] = 0.0
    final_features_df[feature_columns] = final_features_df[feature_columns].fillna(0) # Fill NaNs for missing tickers/features
    final_features_df = final_features_df[['Ticker'] + feature_columns] # Ensure order

    print("Scaling features...")
    X_today = final_features_df[feature_columns].values
    try: X_today_scaled = scaler.transform(X_today)
    except Exception as e: print(f"Error scaling: {e}"); raise

    print("Making predictions...")
    X_today_tensor = torch.tensor(X_today_scaled, dtype=torch.float32).to(model_device)
    model_obj.eval() # Set to evaluation mode
    predictions_array = np.zeros(len(final_features_df))
    try:
        with torch.no_grad():
            # Handle potential batch size of 1 during prediction if BN is used
            # In eval mode, BN should use running stats, so it's generally safe.
            # But loop just in case large prediction sets cause memory issues (unlikely here)
            predictions_tensor = model_obj(X_today_tensor)
        predictions_array = predictions_tensor.cpu().numpy().flatten()
    except Exception as e: print(f"Error during prediction: {e}")

    noise = np.random.normal(0, 1e-7, size=predictions_array.shape)
    final_features_df['Predicted_Return'] = predictions_array + noise

    print("Calculating signal ranks...")
    final_features_df['Signal_Rank'] = final_features_df['Predicted_Return'].rank(pct=True)

    predictions_output = final_features_df[['Ticker', 'Predicted_Return', 'Signal_Rank']].copy()
    predictions_output.fillna({'Predicted_Return': 0.0, 'Signal_Rank': 0.0}, inplace=True)

    print(f"--- Prediction complete. Returning {predictions_output.shape[0]} predictions. ---")
    return predictions_output


####################################
# Test Section (Optional)
####################################
if __name__ == "__main__":
    print("--- Running Test Section (Improved Model v2) ---")
    sentiment_file = './data/sentiment_train_2017_2021.csv'
    return_file = './data/return_train_2017_2021.csv'
    try:
        sentiment_data_train = pd.read_csv(sentiment_file)
        return_data_train = pd.read_csv(return_file)
        print("Data loaded.")
        print("\n--- Testing train_model ---")
        model_info_dict = train_model(sentiment_data_train, return_data_train)
        print("train_model executed.")
        print(f"Device: {model_info_dict['device']}, Features: {len(model_info_dict['feature_columns'])}")

        print("\n--- Testing predict_returns ---")
        sample_date_str = '2021-10-01'
        sample_target_date = pd.to_datetime(sample_date_str).normalize()
        test_received_date = (sample_target_date - pd.Timedelta(days=1)).date()

        sentiment_data_train['Received_Time'] = pd.to_datetime(sentiment_data_train['Received_Time'], errors='coerce', utc=True)
        sentiment_data_train.dropna(subset=['Received_Time'], inplace=True)
        sentiment_data_train['Received_Time_EST'] = sentiment_data_train['Received_Time'].dt.tz_convert('America/New_York')
        sentiment_data_train['local_date'] = sentiment_data_train['Received_Time_EST'].dt.date
        sentiment_data_today_raw_test = sentiment_data_train[sentiment_data_train['local_date'] == test_received_date].copy()

        return_data_train['Date'] = pd.to_datetime(return_data_train['Date'], errors='coerce').dt.normalize()
        stock_universe_today_test = return_data_train[return_data_train['Date'] == sample_target_date]['Ticker'].unique().tolist()

        if not stock_universe_today_test: print(f"Warning: No stocks for {sample_target_date}.")
        elif sentiment_data_today_raw_test.empty: print(f"Warning: No sentiment for {test_received_date}.")
        else: print(f"Using {sentiment_data_today_raw_test.shape[0]} records from {test_received_date} for {len(stock_universe_today_test)} stocks.")

        predictions = predict_returns(model=model_info_dict,
                                      sentiment_data_today=sentiment_data_today_raw_test if not sentiment_data_today_raw_test.empty else pd.DataFrame(),
                                      stock_universe_today=stock_universe_today_test)

        print("\nSample predictions:")
        print(predictions.head())
        print(f"\nPredictions shape: {predictions.shape}")
        if not stock_universe_today_test or predictions.empty: print("Skipping universe check.")
        elif set(predictions['Ticker'].unique()) == set([t.upper().strip() for t in stock_universe_today_test]): print("All universe tickers present.")
        else: print("Warning: Mismatch between predicted tickers and universe.")

    except FileNotFoundError: print(f"Error: Data files not found.")
    except Exception as e: print(f"Error in test section: {e}"); import traceback; traceback.print_exc()
    print("--- Test Section Finished ---")

