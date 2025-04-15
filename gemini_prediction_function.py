# 001_prediction_function.py
# (Replace 001 with your group number)

"""
This file provides two main functions for predicting stock returns based on Reddit sentiment:
    1. train_model(sentiment_data, return_data): Trains a PyTorch neural network.
    2. predict_returns(model, sentiment_data_today, stock_universe_today): Generates predictions.

Feature Engineering: Includes sentiment statistics, volume indicators, temporal patterns,
probability measures, and source/topic/relevance weights.

Model: A feedforward neural network implemented in PyTorch with GPU support.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from scipy.stats import skew
from datetime import time, timedelta
import warnings

# Suppress potential warnings from pandas operations like fillna
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)


# Set device for GPU usage if available, otherwise use CPU.
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
    """
    Converts a return value (string percentage or number) to a float.

    Parameters:
    -----------
    x : str or float or int
        The return value to convert.

    Returns:
    --------
    float
        The return value as a float. Returns NaN if conversion fails.
    """
    if isinstance(x, str):
        x = x.strip()
        if x.endswith('%'):
            try:
                return float(x[:-1].strip()) / 100.0
            except ValueError:
                return np.nan
        else:
            try:
                return float(x)
            except ValueError:
                return np.nan
    elif isinstance(x, (int, float)):
        return float(x)
    else:
        return np.nan # Handle unexpected types

def preprocess_sentiment_data(sentiment_data):
    """
    Preprocesses raw sentiment data.

    Steps:
    1.  Converts 'Received_Time' to timezone-aware datetime (UTC), then to US/Eastern.
    2.  Creates a 'Date' column for merging with returns: posts after 4:00 PM EST
        are assigned to the *next* trading day's date.
    3.  Ensures 'Ticker' is uppercase.
    4.  Handles potential missing columns needed for feature engineering by adding them with default values.

    Parameters:
    -----------
    sentiment_data : pd.DataFrame
        Raw sentiment data. Must contain 'Received_Time' and 'Ticker'.

    Returns:
    --------
    pd.DataFrame
        Processed sentiment data with 'Received_Time_EST' and 'Date' columns added,
        and necessary columns ensured.
    """
    if not isinstance(sentiment_data, pd.DataFrame):
        raise TypeError("sentiment_data must be a pandas DataFrame.")

    df = sentiment_data.copy()

    # --- Time Conversion ---
    if 'Received_Time' not in df.columns:
        raise ValueError("Column 'Received_Time' not found in the sentiment data.")
    # Ensure Received_Time is parsed correctly, handling potential errors
    try:
        df['Received_Time'] = pd.to_datetime(df['Received_Time'], errors='coerce', utc=True)
        df = df.dropna(subset=['Received_Time']) # Drop rows where conversion failed
    except Exception as e:
        raise ValueError(f"Error converting 'Received_Time' to datetime: {e}")

    if df.empty:
        print("Warning: No valid 'Received_Time' entries found after conversion.")
        # Return an empty DataFrame with expected columns for downstream processing
        df['Received_Time_EST'] = pd.Series(dtype='datetime64[ns, America/New_York]')
        df['Date'] = pd.Series(dtype='datetime64[ns]')
        df['Ticker'] = pd.Series(dtype='object')
        return df

    df['Received_Time_EST'] = df['Received_Time'].dt.tz_convert('America/New_York')
    df['local_date'] = df['Received_Time_EST'].dt.date # Keep local date for reference if needed

    # --- Date Assignment for Trading ---
    # Posts after 4 PM EST market close are considered for the next day's return prediction
    cutoff_time = time(16, 0)
    df['Date'] = df['Received_Time_EST'].apply(
        lambda dt: pd.to_datetime(dt.date() + timedelta(days=1)) if dt.time() > cutoff_time else pd.to_datetime(dt.date())
    )
    df['Date'] = df['Date'].dt.normalize() # Ensure date is midnight

    # --- Ticker Normalization ---
    if 'Ticker' in df.columns:
        df['Ticker'] = df['Ticker'].astype(str).str.upper().str.strip()
    else:
        raise ValueError("Column 'Ticker' not found in the sentiment data.")

    # --- Ensure Necessary Columns Exist ---
    # Add columns needed for feature engineering if they are missing, filling with appropriate defaults
    required_cols = {
        'Sentiment': 0, 'Confidence': 0.0, 'Prob_POS': 0.0, 'Prob_NTR': 0.0, 'Prob_NEG': 0.0,
        'Relevance': 0.0, 'SourceWeight': 0.0, 'TopicWeight': 0.0, 'Author': 'missing_author',
        'Novelty': 1, 'StoryID': 'missing_storyid' # Use StoryID for post count if Sentiment is missing
    }
    for col, default_value in required_cols.items():
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found. Adding with default value: {default_value}.")
            df[col] = default_value

    # Convert relevant columns to numeric, coercing errors
    numeric_cols = ['Sentiment', 'Confidence', 'Prob_POS', 'Prob_NTR', 'Prob_NEG',
                    'Relevance', 'SourceWeight', 'TopicWeight', 'Novelty']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Fill NaNs created by coercion in numeric cols (except Sentiment which might be intentionally missing)
    for col in numeric_cols:
         if col != 'Sentiment':
              df[col].fillna(0, inplace=True)

    # Ensure Author is string
    if 'Author' in df.columns:
        df['Author'] = df['Author'].astype(str)

    return df


def create_features(df_processed):
    """
    Creates aggregated daily features for each Ticker based on preprocessed sentiment data.

    Features include:
    - Sentiment Stats: mean, std, skew, positive/negative ratio.
    - Volume Indicators: post count (log-transformed), unique author count.
    - Temporal Patterns: Sentiment difference AM vs PM, Volume ratio AM vs PM.
    - Probability Measures: Averages of Prob_POS/NTR/NEG, avg confidence, avg uncertainty.
    - Weights/Relevance: Averages of SourceWeight, TopicWeight, Relevance. Weighted sentiment.
    - Other: Average Novelty.

    Parameters:
    -----------
    df_processed : pd.DataFrame
        Preprocessed sentiment data from `preprocess_sentiment_data`.
        Must include 'Ticker', 'Date', 'Received_Time_EST', and various sentiment/meta columns.

    Returns:
    --------
    pd.DataFrame
        DataFrame with aggregated features, indexed by ['Ticker', 'Date'].
        Returns an empty DataFrame if input is empty.
    """
    if df_processed.empty:
        print("Warning: Input DataFrame to create_features is empty.")
        # Define expected columns for an empty output DataFrame
        feature_names = [
            'sentiment_mean', 'sentiment_std', 'sentiment_skew', 'pos_neg_ratio',
            'log_post_count', 'author_count', 'sentiment_am_pm_diff', 'volume_am_pm_ratio',
            'avg_confidence', 'avg_prob_pos', 'avg_prob_ntr', 'avg_prob_neg',
            'prob_uncertainty', 'avg_source_weight', 'avg_topic_weight',
            'weighted_sentiment_relevance', 'avg_relevance', 'avg_novelty'
        ]
        return pd.DataFrame(columns=['Ticker', 'Date'] + feature_names)

    # --- Feature Engineering Calculations ---

    # Define aggregation functions
    agg_funcs = {
        'Sentiment': ['mean', 'std', lambda x: skew(x.dropna())], # Use lambda for skew
        'Confidence': 'mean',
        'Prob_POS': 'mean',
        'Prob_NTR': 'mean',
        'Prob_NEG': 'mean',
        'Relevance': 'mean',
        'SourceWeight': 'mean',
        'TopicWeight': 'mean',
        'Novelty': 'mean',
        'Author': pd.Series.nunique, # Count unique authors
        'StoryID': 'count' # Use StoryID count as a robust post count measure
    }

    # Perform initial aggregation
    grouped = df_processed.groupby(['Ticker', 'Date']).agg(agg_funcs)

    # Rename columns for clarity
    grouped.columns = [
        'sentiment_mean', 'sentiment_std', 'sentiment_skew', 'avg_confidence',
        'avg_prob_pos', 'avg_prob_ntr', 'avg_prob_neg', 'avg_relevance',
        'avg_source_weight', 'avg_topic_weight', 'avg_novelty',
        'author_count', 'post_count'
    ]

    # --- Calculate Additional Features Requiring Custom Logic ---

    # 1. Log Post Count (handle count = 0)
    grouped['log_post_count'] = np.log1p(grouped['post_count']) # log1p handles 0 gracefully (log(1+0)=0)

    # 2. Positive/Negative Ratio (handle division by zero)
    def calculate_pos_neg_ratio(sub_df):
        pos_count = (sub_df['Sentiment'] == 1).sum()
        neg_count = (sub_df['Sentiment'] == -1).sum()
        if neg_count == 0:
            # If only positive or neutral posts, ratio is effectively infinite or large.
            # Return pos_count as a representation, or choose a large number like 999.
            return float(pos_count) if pos_count > 0 else 0.0
        return float(pos_count) / neg_count

    pos_neg = df_processed.groupby(['Ticker', 'Date']).apply(calculate_pos_neg_ratio).reset_index(name='pos_neg_ratio')
    grouped = pd.merge(grouped.reset_index(), pos_neg, on=['Ticker', 'Date'], how='left')

    # 3. Weighted Sentiment (by Relevance)
    def calculate_weighted_sentiment(sub_df):
        # Ensure Relevance is numeric and handle potential NaNs
        relevance = pd.to_numeric(sub_df['Relevance'], errors='coerce').fillna(0)
        sentiment = pd.to_numeric(sub_df['Sentiment'], errors='coerce').fillna(0)
        # Avoid division by zero if count is zero or relevance sum is zero
        total_relevance = relevance.sum()
        if total_relevance == 0:
             return 0.0 # Or return np.nan if preferred
        return (sentiment * relevance).sum() / total_relevance


    weighted_sent = df_processed.groupby(['Ticker', 'Date']).apply(calculate_weighted_sentiment).reset_index(name='weighted_sentiment_relevance')
    grouped = pd.merge(grouped, weighted_sent, on=['Ticker', 'Date'], how='left')

    # 4. Probability Uncertainty
    def calculate_uncertainty(sub_df):
        # Ensure probability columns are numeric and handle NaNs
        prob_cols = ['Prob_POS', 'Prob_NEG', 'Prob_NTR']
        for col in prob_cols:
             sub_df[col] = pd.to_numeric(sub_df[col], errors='coerce').fillna(0)

        probs = sub_df[prob_cols].values
        if probs.shape[0] == 0: # Handle empty sub-dataframe
             return 0.0
        max_prob = np.max(probs, axis=1)
        # Uncertainty: 1 - max_prob (higher means less confidence in any single class)
        # Average uncertainty over posts for the day
        uncertainty = 1.0 - max_prob
        return uncertainty.mean() if len(uncertainty) > 0 else 0.0

    prob_uncert = df_processed.groupby(['Ticker', 'Date']).apply(calculate_uncertainty).reset_index(name='prob_uncertainty')
    grouped = pd.merge(grouped, prob_uncert, on=['Ticker', 'Date'], how='left')


    # 5. Temporal Features (AM/PM difference) - Requires Received_Time_EST
    noon_est = time(12, 0)
    df_processed['is_am'] = df_processed['Received_Time_EST'].dt.time < noon_est

    def calculate_temporal_diffs(sub_df):
        # Ensure Sentiment is numeric
        sub_df['Sentiment'] = pd.to_numeric(sub_df['Sentiment'], errors='coerce')

        sentiment_am = sub_df.loc[sub_df['is_am'], 'Sentiment'].mean() # NaNs automatically ignored by mean
        sentiment_pm = sub_df.loc[~sub_df['is_am'], 'Sentiment'].mean() # NaNs automatically ignored by mean
        volume_am = sub_df['is_am'].sum()
        volume_pm = (~sub_df['is_am']).sum()

        # Handle cases where there's no AM or PM data (mean returns NaN)
        sentiment_diff = sentiment_am - sentiment_pm if pd.notna(sentiment_am) and pd.notna(sentiment_pm) else 0.0

        if volume_pm == 0:
             # If only AM volume, ratio is large. Return AM volume or a large number.
             volume_ratio = float(volume_am) if volume_am > 0 else 0.0
        else:
             volume_ratio = float(volume_am) / volume_pm

        return pd.Series({
            'sentiment_am_pm_diff': sentiment_diff,
            'volume_am_pm_ratio': volume_ratio
        })

    temporal_features = df_processed.groupby(['Ticker', 'Date']).apply(calculate_temporal_diffs).reset_index()
    grouped = pd.merge(grouped, temporal_features, on=['Ticker', 'Date'], how='left')


    # --- Final Cleanup ---
    # Reset index if it hasn't been already
    if isinstance(grouped.index, pd.MultiIndex):
         grouped = grouped.reset_index()


    # Fill any remaining NaNs (e.g., std/skew for single posts, ratios with zero denominators) with 0
    # Important: Do this *after* all merges and calculations
    grouped = grouped.fillna(0)

    # Handle potential infinite values resulted from division by zero (e.g., in ratios)
    grouped.replace([np.inf, -np.inf], 0, inplace=True) # Replace inf with 0, consider if a large number is better

    # Ensure correct column order (optional but good practice)
    feature_order = [
        'sentiment_mean', 'sentiment_std', 'sentiment_skew', 'pos_neg_ratio',
        'log_post_count', 'author_count', 'sentiment_am_pm_diff', 'volume_am_pm_ratio',
        'avg_confidence', 'avg_prob_pos', 'avg_prob_ntr', 'avg_prob_neg',
        'prob_uncertainty', 'avg_source_weight', 'avg_topic_weight',
        'weighted_sentiment_relevance', 'avg_relevance', 'avg_novelty', 'post_count' # Keep post_count if needed elsewhere
    ]
    # Add any missing columns from the expected order (e.g., if a feature calculation failed)
    final_cols = ['Ticker', 'Date']
    for col in feature_order:
        if col not in grouped.columns:
            grouped[col] = 0.0
        if col != 'post_count': # Exclude post_count from final feature set for model if log_post_count is used
             final_cols.append(col)


    return grouped[final_cols] # Return with Ticker, Date as columns and selected features


####################################
# PyTorch Neural Network Model
####################################

class FeedforwardNet(nn.Module):
    """
    A simple feedforward neural network with two hidden layers and ReLU activations.
    """
    def __init__(self, input_dim, hidden_dim1=64, hidden_dim2=32):
        """
        Initializes the network layers.

        Parameters:
        -----------
        input_dim : int
            Number of input features.
        hidden_dim1 : int, optional
            Number of neurons in the first hidden layer (default is 64).
        hidden_dim2 : int, optional
            Number of neurons in the second hidden layer (default is 32).
        """
        super(FeedforwardNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Dropout(0.2), # Add dropout for regularization
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Dropout(0.2), # Add dropout for regularization
            nn.Linear(hidden_dim2, 1)  # Output layer predicts a single value (return)
        )

    def forward(self, x):
        """
        Defines the forward pass of the network.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor containing features.

        Returns:
        --------
        torch.Tensor
            Output tensor representing the predicted return.
        """
        return self.net(x)

####################################
# Main Functions: train_model
####################################

def train_model(sentiment_data, return_data):
    """
    Trains a PyTorch neural network model using sentiment features to predict next-day stock returns.

    Steps:
    1. Preprocesses sentiment and return data.
    2. Creates features from sentiment data.
    3. Merges features with next-day returns.
    4. Splits data into training and validation sets based on time.
    5. Scales features using StandardScaler.
    6. Defines and trains the PyTorch FeedforwardNet model.
    7. Saves the trained model, scaler, and feature list.

    Parameters:
    -----------
    sentiment_data : pd.DataFrame
        The Reddit sentiment data for training (e.g., sentiment_train_2017_2021.csv).
        Requires columns like 'Received_Time', 'Ticker', 'Sentiment', etc.
    return_data : pd.DataFrame
        The stock return data for training (e.g., return_train_2017_2021.csv).
        Requires columns 'Date', 'Ticker', 'Return'.

    Returns:
    --------
    dict
        A dictionary `model_info` containing:
        - 'model': The trained PyTorch model (FeedforwardNet instance).
        - 'scaler': The fitted StandardScaler object.
        - 'feature_columns': A list of feature names used for training.
        - 'device': The torch device used for training ('cuda' or 'cpu').
    """
    print("--- Starting Model Training ---")

    # --- 1. Preprocessing ---
    print("Preprocessing sentiment data...")
    try:
        sentiment_processed = preprocess_sentiment_data(sentiment_data)
        if sentiment_processed.empty:
             raise ValueError("Preprocessing resulted in empty sentiment DataFrame.")
    except Exception as e:
        print(f"Error during sentiment preprocessing: {e}")
        raise # Re-raise the exception to stop execution

    print("Preprocessing return data...")
    try:
        return_data_processed = return_data.copy()
        # Ensure Date is datetime and normalized to midnight
        return_data_processed['Date'] = pd.to_datetime(return_data_processed['Date'], errors='coerce').dt.normalize()
        # Ensure Ticker is uppercase string
        return_data_processed['Ticker'] = return_data_processed['Ticker'].astype(str).str.upper().str.strip()
        # Convert Return column, handling potential errors
        return_data_processed['Return'] = return_data_processed['Return'].apply(convert_return)
        # Drop rows where Date or Return conversion failed
        return_data_processed = return_data_processed.dropna(subset=['Date', 'Ticker', 'Return'])
        if return_data_processed.empty:
            raise ValueError("Preprocessing resulted in empty return DataFrame.")
    except Exception as e:
        print(f"Error during return data preprocessing: {e}")
        raise

    # --- 2. Feature Engineering ---
    print("Creating features from sentiment data...")
    try:
        features_df = create_features(sentiment_processed)
        if features_df.empty:
             raise ValueError("Feature creation resulted in an empty DataFrame.")
    except Exception as e:
        print(f"Error during feature creation: {e}")
        raise

    # Define the exact list of features used for model input
    # Exclude 'Ticker', 'Date', and raw 'post_count'
    feature_columns = [col for col in features_df.columns if col not in ['Ticker', 'Date', 'post_count']]
    print(f"Using features: {feature_columns}")


    # --- 3. Merging Data ---
    print("Merging sentiment features with stock returns...")
    # Merge features (for day D) with returns (for day D+1, which corresponds to the 'Date' in return_data)
    model_data = pd.merge(features_df, return_data_processed[['Date', 'Ticker', 'Return']],
                          on=['Date', 'Ticker'], how='inner')

    # Drop rows with NaN returns that might have slipped through
    model_data = model_data.dropna(subset=['Return'])
    # Fill NaNs in feature columns with 0 (safer than dropping rows)
    model_data[feature_columns] = model_data[feature_columns].fillna(0)
    # Handle potential infinities just in case
    model_data.replace([np.inf, -np.inf], 0, inplace=True)


    if model_data.empty:
        raise ValueError("Merging features and returns resulted in an empty DataFrame. Check date alignment and ticker matching.")

    print(f"Merged data shape: {model_data.shape}")

    # --- 4. Train/Validation Split (Time-Based) ---
    print("Splitting data into training and validation sets...")
    model_data = model_data.sort_values('Date')
    unique_dates = np.sort(model_data['Date'].unique())

    if len(unique_dates) < 5: # Need enough dates for a meaningful split
        raise ValueError("Not enough unique dates in the data for a train/validation split.")

    # Use an 80/20 split based on unique dates
    split_index = int(0.8 * len(unique_dates))
    train_cutoff_date = unique_dates[split_index]

    train_data = model_data[model_data['Date'] < train_cutoff_date]
    val_data = model_data[model_data['Date'] >= train_cutoff_date]

    if train_data.empty or val_data.empty:
        raise ValueError("Train or validation set is empty after time-based split. Check date range.")

    print(f"Training data shape: {train_data.shape}, Dates: {train_data['Date'].min()} to {train_data['Date'].max()}")
    print(f"Validation data shape: {val_data.shape}, Dates: {val_data['Date'].min()} to {val_data['Date'].max()}")

    X_train = train_data[feature_columns].values
    y_train = train_data['Return'].values.reshape(-1, 1)
    X_val = val_data[feature_columns].values
    y_val = val_data['Return'].values.reshape(-1, 1)


    # --- 5. Feature Scaling ---
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train) # Fit only on training data
    X_val_scaled = scaler.transform(X_val)         # Transform validation data


    # --- 6. PyTorch Model Training ---
    print("Converting data to PyTorch tensors...")
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)

    input_dim = X_train_tensor.shape[1]
    model_net = FeedforwardNet(input_dim).to(device)

    # Define Loss and Optimizer
    criterion = nn.MSELoss() # Mean Squared Error for regression
    optimizer = optim.Adam(model_net.parameters(), lr=0.001, weight_decay=1e-5) # Adam optimizer with L2 regularization

    # Training Loop Parameters
    epochs = 100 # Reduced epochs for faster example, increase for better convergence
    batch_size = 256 # Process data in batches
    patience = 10 # Early stopping patience
    best_val_loss = float('inf')
    epochs_no_improve = 0


    print("Starting PyTorch model training...")
    for epoch in range(epochs):
        model_net.train() # Set model to training mode
        permutation = torch.randperm(X_train_tensor.size(0))

        train_loss_epoch = 0.0
        for i in range(0, X_train_tensor.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X_train_tensor[indices], y_train_tensor[indices]

            optimizer.zero_grad()       # Clear gradients
            outputs = model_net(batch_x) # Forward pass
            loss = criterion(outputs, batch_y) # Calculate loss
            loss.backward()             # Backward pass
            optimizer.step()            # Update weights

            train_loss_epoch += loss.item() * batch_x.size(0)

        train_loss_epoch /= X_train_tensor.size(0)

        # Validation Phase
        model_net.eval() # Set model to evaluation mode
        val_loss_epoch = 0.0
        with torch.no_grad(): # Disable gradient calculation for validation
             permutation_val = torch.randperm(X_val_tensor.size(0))
             for i in range(0, X_val_tensor.size(0), batch_size):
                  indices_val = permutation_val[i:i+batch_size]
                  batch_x_val, batch_y_val = X_val_tensor[indices_val], y_val_tensor[indices_val]

                  val_outputs = model_net(batch_x_val)
                  val_loss = criterion(val_outputs, batch_y_val)
                  val_loss_epoch += val_loss.item() * batch_x_val.size(0)

        val_loss_epoch /= X_val_tensor.size(0)


        if (epoch + 1) % 10 == 0 or epoch == 0: # Print every 10 epochs and the first epoch
             print(f"Epoch {epoch+1:3d}/{epochs} | Train Loss: {train_loss_epoch:.6f} | Val Loss: {val_loss_epoch:.6f}")

        # Early Stopping Check
        if val_loss_epoch < best_val_loss:
            best_val_loss = val_loss_epoch
            epochs_no_improve = 0
            # Optionally save the best model state here
            # torch.save(model_net.state_dict(), 'best_model.pth')
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    # --- 7. Prepare Model Info ---
    # This dictionary is the object passed as 'model' to predict_returns
    model_info = {
        'model': model_net,
        'scaler': scaler,
        'feature_columns': feature_columns,
        'device': device
    }
    print("--- Training complete. Model Info created. ---")
    return model_info


####################################
# Main Functions: predict_returns
####################################

def predict_returns(model, sentiment_data_today, stock_universe_today):
    """
    Generates predictions of next-day returns for a given list of stocks using a trained model.

    Steps:
    1. Preprocesses the sentiment data for the current day.
    2. Creates features for the current day's sentiment data.
    3. Creates a DataFrame encompassing all stocks in today's universe.
    4. Merges calculated features, filling missing values for stocks with no sentiment data.
    5. Ensures all required feature columns are present and ordered correctly.
    6. Scales the features using the scaler from the `model` dictionary.
    7. Uses the trained model object from the `model` dictionary to predict returns.
    8. Calculates the percentile rank ('Signal_Rank') for each prediction.
    9. Formats and returns the predictions.

    Parameters:
    -----------
    model : dict
        Dictionary returned by `train_model`, containing the trained model object ('model'),
        the fitted scaler ('scaler'), feature column list ('feature_columns'), and device ('device').
    sentiment_data_today : pd.DataFrame or None
        Raw sentiment data for the *current* day (posts received up to 4 PM EST).
        Can be None or an empty DataFrame if no sentiment data is available.
    stock_universe_today : list
        List of stock tickers (strings) available for trading *tomorrow*.
        Predictions should be generated for all tickers in this list.

    Returns:
    --------
    pd.DataFrame
        A DataFrame with columns ['Ticker', 'Predicted_Return', 'Signal_Rank'].
        Contains one row for each ticker in `stock_universe_today`.
        'Signal_Rank' is the percentile rank (0-1) of 'Predicted_Return'.
    """
    print("--- Starting Prediction ---")

    # --- Input Validation ---
    # Check if the 'model' dictionary has the required keys
    if not isinstance(model, dict) or not all(k in model for k in ['model', 'scaler', 'feature_columns', 'device']):
        raise ValueError("Invalid 'model' dictionary provided. Must contain 'model', 'scaler', 'feature_columns', 'device'.")
    if not isinstance(sentiment_data_today, pd.DataFrame):
        # Allow empty DataFrame or None as valid input, handle downstream
        if not sentiment_data_today is None:
             raise TypeError("'sentiment_data_today' must be a pandas DataFrame or None.")
    if not isinstance(stock_universe_today, list):
        raise TypeError("'stock_universe_today' must be a list of tickers.")
    if not stock_universe_today:
        print("Warning: 'stock_universe_today' is empty. Returning empty predictions.")
        return pd.DataFrame(columns=['Ticker', 'Predicted_Return', 'Signal_Rank'])


    # --- 1. Preprocess Today's Sentiment Data ---
    print("Preprocessing today's sentiment data...")
    if sentiment_data_today is None or sentiment_data_today.empty:
        print("No sentiment data provided for today.")
        sentiment_today_processed = pd.DataFrame() # Create empty df with expected structure later
    else:
        try:
            # Important: The date assigned by preprocess_sentiment_data aligns
            # with the *target return date*.
            sentiment_today_processed = preprocess_sentiment_data(sentiment_data_today)
            if not sentiment_today_processed.empty:
                 current_prediction_target_date = sentiment_today_processed['Date'].max()
                 print(f"Sentiment data processed for target prediction date: {current_prediction_target_date}")
            else:
                 print("Preprocessing resulted in empty sentiment DataFrame for today.")

        except Exception as e:
            print(f"Error preprocessing today's sentiment data: {e}. Proceeding with default features.")
            sentiment_today_processed = pd.DataFrame() # Ensure it's an empty DF


    # --- 2. Create Features for Today ---
    print("Creating features for today...")
    if sentiment_today_processed.empty:
        features_today = pd.DataFrame() # Will be handled in step 3
    else:
        try:
            # Filter for the relevant date if multiple dates ended up in the input
            if 'Date' in sentiment_today_processed.columns:
                 target_date = sentiment_today_processed['Date'].max()
                 sentiment_today_processed = sentiment_today_processed[sentiment_today_processed['Date'] == target_date]

            features_today = create_features(sentiment_today_processed)
            if features_today.empty and not sentiment_today_processed.empty :
                 print("Warning: Feature creation resulted in an empty DataFrame despite non-empty input.")
            elif not features_today.empty:
                 print(f"Features created for {features_today.shape[0]} tickers.")

        except Exception as e:
            print(f"Error creating features for today: {e}. Proceeding with default features.")
            features_today = pd.DataFrame()


    # --- 3. Align with Stock Universe ---
    print(f"Aligning features with today's stock universe ({len(stock_universe_today)} tickers)...")
    # Create a DataFrame for the full universe
    universe_df = pd.DataFrame({'Ticker': [t.upper().strip() for t in stock_universe_today]})

    # Retrieve necessary components from the 'model' dictionary
    feature_columns = model['feature_columns']
    scaler = model['scaler']
    model_device = model['device']
    model_obj = model['model'] # The actual PyTorch model object


    # Merge features_today with the universe_df
    if not features_today.empty and 'Ticker' in features_today.columns:
        # Ensure Ticker case matches
        features_today['Ticker'] = features_today['Ticker'].astype(str).str.upper().str.strip()
        # Merge, keeping all tickers from the universe
        final_features_df = pd.merge(universe_df, features_today, on='Ticker', how='left')
    else:
        # If no features were calculated, start with the universe and add empty feature columns
        final_features_df = universe_df.copy()
        for col in feature_columns:
            final_features_df[col] = 0.0 # Initialize with default value


    # --- 4. Fill Missing Features & Ensure Column Order ---
    # Fill NaNs for tickers that had no sentiment data (or where feature calculation failed)
    # Use 0 as the default fill value for all features
    final_features_df[feature_columns] = final_features_df[feature_columns].fillna(0)

    # Ensure all required feature columns exist and are in the correct order
    for col in feature_columns:
        if col not in final_features_df.columns:
            print(f"Warning: Feature column '{col}' was missing. Adding with default value 0.")
            final_features_df[col] = 0.0
    # Reorder columns to match training order
    final_features_df = final_features_df[['Ticker'] + feature_columns]


    # --- 5. Scale Features ---
    print("Scaling features for prediction...")
    X_today = final_features_df[feature_columns].values
    try:
        X_today_scaled = scaler.transform(X_today) # Use the *fitted* scaler from the 'model' dictionary
    except Exception as e:
        print(f"Error scaling features: {e}. Check scaler compatibility.")
        # Fallback: predict using unscaled features (likely poor results)
        # Or handle more gracefully (e.g., return default predictions)
        raise # Re-raise for now


    # --- 6. Predict Returns ---
    print("Making predictions with the PyTorch model...")
    X_today_tensor = torch.tensor(X_today_scaled, dtype=torch.float32).to(model_device)

    model_obj.eval() # Set model to evaluation mode using the object from the 'model' dictionary
    predictions_array = np.zeros(len(final_features_df)) # Default prediction is 0

    try:
        with torch.no_grad():
            predictions_tensor = model_obj(X_today_tensor)
        predictions_array = predictions_tensor.cpu().numpy().flatten() # Move to CPU and flatten
        print(f"Predictions generated for {len(predictions_array)} tickers.")
    except Exception as e:
        print(f"Error during model prediction: {e}. Returning default predictions (0).")
        # predictions_array remains zeros


    # Add a tiny amount of noise to break ties for ranking - crucial for stable ranks
    noise = np.random.normal(0, 1e-7, size=predictions_array.shape)
    final_features_df['Predicted_Return'] = predictions_array + noise


    # --- 7. Calculate Signal Rank ---
    print("Calculating signal ranks...")
    # rank(pct=True) gives percentile rank from 0 to 1
    final_features_df['Signal_Rank'] = final_features_df['Predicted_Return'].rank(pct=True)


    # --- 8. Format Output ---
    predictions_output = final_features_df[['Ticker', 'Predicted_Return', 'Signal_Rank']].copy()
    # Ensure no NaNs in final output (shouldn't happen with fillna, but double-check)
    predictions_output.fillna({'Predicted_Return': 0.0, 'Signal_Rank': 0.0}, inplace=True)


    print(f"--- Prediction complete. Returning {predictions_output.shape[0]} predictions. ---")
    return predictions_output
