# 001_prediction_function.py
# (Assuming a group number like 001 for the filename)

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from datetime import datetime, time, timedelta
import warnings
import traceback # For detailed error printing

# Suppress specific warnings if necessary, e.g., PerformanceWarning from pandas
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Degrees of freedom <= 0 for slice")


# Set device for GPU usage if available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

####################################
# Helper Functions
####################################

def convert_return(x):
    """
    Converts a return value to a float.
    If x is a string ending in '%', removes the '%' and divides by 100.
    Handles potential non-numeric values gracefully by returning NaN.
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
        return np.nan # Return NaN for other types or unparseable input


def preprocess_sentiment_data(sentiment_data):
    """
    Preprocess sentiment data:
     - Convert 'Received_Time' to a timezone-aware datetime (UTC) then to US/Eastern.
     - Create a 'Date' column by shifting posts received after 4:00 PM (EST) to the next day's date (normalized).
     - Ensure 'Ticker' is uppercase.
     - Convert relevant numeric columns to numeric type, coercing errors.

    Parameters
    ----------
    sentiment_data : DataFrame
        Raw sentiment data containing at least 'Received_Time' and 'Ticker'.

    Returns
    -------
    df : DataFrame
        Processed sentiment data with added 'Received_Time_EST' and 'Date' columns.
    """
    if not isinstance(sentiment_data, pd.DataFrame):
        raise TypeError("sentiment_data must be a pandas DataFrame")

    df = sentiment_data.copy()

    if 'Received_Time' not in df.columns:
        raise ValueError("Column 'Received_Time' not found in the sentiment data.")
    if 'Ticker' not in df.columns:
        # Allow processing even if Ticker is missing initially, but log it.
        print("Warning: 'Ticker' column not found in sentiment data during preprocessing.")
        df['Ticker'] = 'MISSING'

    # Convert Received_Time, handle potential errors
    df['Received_Time'] = pd.to_datetime(df['Received_Time'], errors='coerce', utc=True)
    df = df.dropna(subset=['Received_Time']) # Drop rows where conversion failed

    if df.empty:
        print("Warning: No valid 'Received_Time' entries found after conversion.")
        df['Date'] = pd.NaT # Assign NaT if DataFrame becomes empty
        df['Received_Time_EST'] = pd.NaT
        # Return early with correct columns if possible, or just the empty df
        if 'Ticker' not in df.columns: df['Ticker'] = pd.NA
        return df

    # Convert to EST timezone
    try:
        df['Received_Time_EST'] = df['Received_Time'].dt.tz_convert('America/New_York')
    except Exception as e:
        print(f"Error converting timezone: {e}. Check if pytz is installed or if source timezone is consistent.")
        # Fallback or raise error depending on requirements
        df['Received_Time_EST'] = pd.NaT # Assign NaT on error
        df = df.dropna(subset=['Received_Time_EST'])
        if df.empty:
             df['Date'] = pd.NaT
             return df


    # Define the market close cutoff time
    cutoff_time = time(16, 0) # 4:00 PM EST

    # Calculate the 'Date' for which the sentiment applies (next trading day if after close)
    # Assign the date part directly, add timedelta only if needed. Normalize to midnight.
    # The resulting 'Date' column will be timezone-aware (America/New_York)
    df['Date'] = df['Received_Time_EST'].apply(
        lambda dt: (dt + timedelta(days=1)).normalize() if dt.time() > cutoff_time else dt.normalize()
    )

    # Convert Ticker to uppercase
    if 'Ticker' in df.columns:
         df['Ticker'] = df['Ticker'].astype(str).str.upper() # Ensure it's string first

    # Convert potential numeric columns, coercing errors to NaN
    numeric_cols = ['Sentiment', 'Confidence', 'Prob_POS', 'Prob_NTR', 'Prob_NEG',
                    'Relevance', 'SourceWeight', 'TopicWeight', 'Novelty', 'Comment_Count']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
             # Add missing expected numeric columns with default NaN or 0 if needed later
             df[col] = 0.0 # Or np.nan if preferred, but 0 works better for later fillna

    # Ensure Author is string
    if 'Author' in df.columns:
        df['Author'] = df['Author'].astype(str)
    else:
        df['Author'] = 'UNKNOWN' # Assign default if missing

    return df


def create_features(df):
    """
    Create advanced daily features by aggregating sentiment data for each Ticker and Date.

    Features include statistical moments, volume, probability-based metrics,
    weighted scores, author counts, and other indicators.

    Parameters
    ----------
    df : DataFrame
        Preprocessed sentiment data with columns including ['Ticker', 'Date', 'Sentiment',
        'Confidence', 'Prob_POS', 'Prob_NTR', 'Prob_NEG', 'Relevance', 'SourceWeight',
        'TopicWeight', 'Author', 'Novelty', 'Comment_Count']. Assumes `preprocess_sentiment_data` ran.

    Returns
    -------
    grouped : DataFrame
        Aggregated features for each Ticker and Date. Returns empty DataFrame if input is empty.
        The 'Date' column retains the timezone from the input df.
    """
    if df.empty or 'Ticker' not in df.columns or 'Date' not in df.columns:
        # Return empty DF with expected columns if possible (though columns determined later)
        return pd.DataFrame()

    # Define columns expected for aggregation
    required_cols = ['Sentiment', 'Confidence', 'Prob_POS', 'Prob_NTR', 'Prob_NEG',
                     'Relevance', 'SourceWeight', 'TopicWeight', 'Author', 'Novelty', 'Comment_Count']

    # Ensure required columns exist, fill with 0 if missing (or handle appropriately)
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0.0 # Default value for missing numeric/feature columns
            if col == 'Author':
                 df[col] = 'UNKNOWN' # Default for Author

    # Define aggregation functions
    agg_funcs = {
        'Sentiment': ['mean', 'std', 'skew', 'count'],
        'Confidence': ['mean', 'std'],
        'Prob_POS': ['mean', 'std'],
        'Prob_NTR': ['mean', 'std'],
        'Prob_NEG': ['mean', 'std'],
        'Relevance': ['mean', 'std'],
        'SourceWeight': ['mean', 'std'],
        'TopicWeight': ['mean', 'std'],
        'Novelty': 'mean',
        'Comment_Count': 'mean',
        'Author': pd.Series.nunique # Count unique authors
    }

    # Perform aggregation
    # Wrap in try-except for robustness against empty groups etc.
    try:
        # Grouping by Ticker and Date (Date is timezone-aware here)
        grouped = df.groupby(['Ticker', 'Date']).agg(agg_funcs)
    except Exception as e:
        print(f"Error during aggregation: {e}")
        return pd.DataFrame() # Return empty if aggregation fails


    # Flatten MultiIndex columns and rename appropriately
    grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
    grouped = grouped.reset_index() # 'Date' column is now timezone-aware

    # Rename columns for clarity and consistency
    grouped = grouped.rename(columns={
        'Sentiment_mean': 'sentiment_mean',
        'Sentiment_std': 'sentiment_std',
        'Sentiment_skew': 'sentiment_skew',
        'Sentiment_count': 'post_count',
        'Confidence_mean': 'avg_confidence', 'Confidence_std': 'std_confidence',
        'Prob_POS_mean': 'avg_prob_pos', 'Prob_POS_std': 'std_prob_pos',
        'Prob_NTR_mean': 'avg_prob_ntr', 'Prob_NTR_std': 'std_prob_ntr',
        'Prob_NEG_mean': 'avg_prob_neg', 'Prob_NEG_std': 'std_prob_neg',
        'Relevance_mean': 'avg_relevance', 'Relevance_std': 'std_relevance',
        'SourceWeight_mean': 'avg_source_weight', 'SourceWeight_std': 'std_source_weight',
        'TopicWeight_mean': 'avg_topic_weight', 'TopicWeight_std': 'std_topic_weight',
        'Novelty_mean': 'avg_novelty',
        'Comment_Count_mean': 'avg_comment_count',
        'Author_nunique': 'unique_authors'
    })

    # --- Calculate additional features requiring per-group operations ---

    # Weighted sentiment scores (handle potential division by zero if count is 0)
    # These require access to original columns within each group
    def calculate_weighted_metrics(sub_df):
        metrics = {}
        count = sub_df['Sentiment'].count()
        if count == 0: # Should not happen if grouped correctly, but safeguard
            metrics['relevance_weighted_sentiment'] = 0.0
            metrics['confidence_weighted_sentiment'] = 0.0
            metrics['prob_pos_weighted_sentiment'] = 0.0
            metrics['prob_neg_weighted_sentiment'] = 0.0
            metrics['source_weighted_sentiment'] = 0.0
            metrics['topic_weighted_sentiment'] = 0.0
            metrics['pos_neg_prob_ratio'] = 0.0

        else:
             # Calculate weighted means safely
             metrics['relevance_weighted_sentiment'] = (sub_df['Sentiment'] * sub_df['Relevance']).sum() / count if count > 0 else 0.0
             metrics['confidence_weighted_sentiment'] = (sub_df['Sentiment'] * sub_df['Confidence']).sum() / count if count > 0 else 0.0
             metrics['prob_pos_weighted_sentiment'] = (sub_df['Sentiment'] * sub_df['Prob_POS']).sum() / count if count > 0 else 0.0
             metrics['prob_neg_weighted_sentiment'] = (sub_df['Sentiment'] * sub_df['Prob_NEG']).sum() / count if count > 0 else 0.0 # Usually negative sentiment * neg prob
             metrics['source_weighted_sentiment'] = (sub_df['Sentiment'] * sub_df['SourceWeight']).sum() / count if count > 0 else 0.0
             metrics['topic_weighted_sentiment'] = (sub_df['Sentiment'] * sub_df['TopicWeight']).sum() / count if count > 0 else 0.0

             # Ratio of mean positive probability to mean negative probability (add epsilon for stability)
             avg_prob_pos = sub_df['Prob_POS'].mean()
             avg_prob_neg = sub_df['Prob_NEG'].mean()
             epsilon = 1e-6
             # Handle case where avg_prob_neg is zero or very close to it
             if abs(avg_prob_neg) < epsilon:
                  # Assign a large number if pos > 0, or 0 if pos is also 0 or neg
                  metrics['pos_neg_prob_ratio'] = (avg_prob_pos / epsilon) if avg_prob_pos > 0 else 0.0
             else:
                  metrics['pos_neg_prob_ratio'] = avg_prob_pos / avg_prob_neg

        return pd.Series(metrics)

    # Apply the function to each group
    # Use the original df grouped by Ticker and Date
    try:
        # Ensure Date is present for grouping
        if 'Date' not in df.columns:
             print("Warning: 'Date' column missing for weighted metric calculation.")
             # Create dummy columns for weighted metrics if calculation fails
             weighted_metrics_df = grouped[['Ticker', 'Date']].copy()
             wm_cols = ['relevance_weighted_sentiment', 'confidence_weighted_sentiment',
                       'prob_pos_weighted_sentiment', 'prob_neg_weighted_sentiment',
                       'source_weighted_sentiment', 'topic_weighted_sentiment', 'pos_neg_prob_ratio']
             for col in wm_cols:
                 weighted_metrics_df[col] = 0.0
        else:
              # Grouping here also uses the timezone-aware 'Date'
              weighted_metrics_df = df.groupby(['Ticker', 'Date']).apply(calculate_weighted_metrics).reset_index()
    except Exception as e:
         print(f"Error calculating weighted metrics: {e}")
         # Create dummy columns if calculation fails
         weighted_metrics_df = grouped[['Ticker', 'Date']].copy()
         wm_cols = ['relevance_weighted_sentiment', 'confidence_weighted_sentiment',
                    'prob_pos_weighted_sentiment', 'prob_neg_weighted_sentiment',
                    'source_weighted_sentiment', 'topic_weighted_sentiment', 'pos_neg_prob_ratio']
         for col in wm_cols:
             weighted_metrics_df[col] = 0.0


    # Merge these new features back into the main grouped DataFrame
    # Both 'Date' columns should be timezone-aware at this point and match
    try:
         # No need to convert to datetime again, they should already be
         # Verify dtypes before merge for debugging if needed
         # print(f"Grouped Date dtype: {grouped['Date'].dtype}")
         # print(f"Weighted Metrics Date dtype: {weighted_metrics_df['Date'].dtype}")
         grouped = pd.merge(grouped, weighted_metrics_df, on=['Ticker', 'Date'], how='left')
    except Exception as e:
         print(f"Error merging weighted metrics: {e}. Weighted metrics might be missing.")
         # Add columns with 0 if merge fails but columns expected
         wm_cols = ['relevance_weighted_sentiment', 'confidence_weighted_sentiment',
                    'prob_pos_weighted_sentiment', 'prob_neg_weighted_sentiment',
                    'source_weighted_sentiment', 'topic_weighted_sentiment', 'pos_neg_prob_ratio']
         for col in wm_cols:
             if col not in grouped.columns:
                 grouped[col] = 0.0


    # --- Post-aggregation features ---
    # Log transform post count (add 1 to avoid log(0))
    if 'post_count' in grouped.columns:
        grouped['log_post_count'] = np.log1p(grouped['post_count'])
    else:
        grouped['log_post_count'] = 0.0 # Add column if missing


    # Fill NaNs that might arise from std/skew calculations with 0
    # Important: Do this *after* all calculations involving potential NaNs
    grouped = grouped.fillna(0)

    # The 'Date' column in the returned 'grouped' DataFrame is still timezone-aware
    return grouped


####################################
# PyTorch Neural Network Model
####################################

class FeedforwardNet(nn.Module):
    def __init__(self, input_dim, hidden_dim1=128, hidden_dim2=64, dropout_rate=0.2):
        """
        Feedforward neural network with two hidden layers, ReLU activation, and Dropout.
        """
        super(FeedforwardNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout_rate), # Added Dropout
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout_rate), # Added Dropout
            nn.Linear(hidden_dim2, 1) # Output layer for regression
        )

    def forward(self, x):
        return self.net(x)

####################################
# Main Functions
####################################

def train_model(sentiment_data, return_data):
    """
    Train a model using engineered sentiment features to predict next-day returns.

    Parameters:
    -----------
    sentiment_data : DataFrame
        The Reddit sentiment data for training (e.g., sentiment_train_2017_2021.csv)
    return_data : DataFrame
        The stock return data for training (e.g., return_train_2017_2021.csv)

    Returns:
    --------
    model_info : dict
        Dictionary containing the trained model object, scaler, feature columns, and device.
        Returns None if training fails.
    """
    print("Starting model training...")
    try:
        # --- 1. Preprocessing ---
        print("Preprocessing sentiment data...")
        sentiment_processed = preprocess_sentiment_data(sentiment_data)
        if sentiment_processed.empty:
             print("Error: Preprocessing resulted in empty sentiment DataFrame.")
             return None
        # 'sentiment_processed' now has a timezone-aware 'Date' column (America/New_York)

        print("Creating features from sentiment data...")
        features_df = create_features(sentiment_processed)
        if features_df.empty:
             print("Error: Feature creation resulted in empty DataFrame.")
             return None
        # 'features_df' also has a timezone-aware 'Date' column inherited from sentiment_processed


        # Preprocess return_data
        print("Preprocessing return data...")
        return_data = return_data.copy()
        # Convert to datetime and normalize (results in timezone-naive datetime)
        return_data['Date'] = pd.to_datetime(return_data['Date']).dt.normalize()
        return_data['Ticker'] = return_data['Ticker'].astype(str).str.upper()
        return_data['Return'] = return_data['Return'].apply(convert_return)
        # Crucial: Drop rows where return is NaN AFTER conversion
        return_data = return_data.dropna(subset=['Return'])
        if return_data.empty:
            print("Error: No valid return data found after preprocessing.")
            return None
        # 'return_data' now has a timezone-naive 'Date' column (datetime64[ns])


        # --- 2. Merging ---
        print("Merging sentiment features with stock returns...")
        # Ensure 'Date' is datetime in features_df (should be, but safe check)
        features_df['Date'] = pd.to_datetime(features_df['Date'])

        # ***** FIX: Make 'Date' columns consistent before merge *****
        # features_df['Date'] is timezone-aware (e.g., datetime64[ns, America/New_York])
        # return_data['Date'] is timezone-naive (datetime64[ns])
        # We need to make features_df['Date'] naive to match return_data['Date'] for the merge.

        if pd.api.types.is_datetime64_any_dtype(features_df['Date']) and features_df['Date'].dt.tz is not None:
            print(f"Detected timezone {features_df['Date'].dt.tz} in features_df['Date']. Converting to naive for merge.")
            features_df['Date'] = features_df['Date'].dt.tz_localize(None)
        # ************************************************************

        # Now both 'Date' columns should be datetime64[ns] (naive)
        print(f"features_df['Date'] dtype after tz removal: {features_df['Date'].dtype}")
        print(f"return_data['Date'] dtype: {return_data['Date'].dtype}")


        # Perform the merge with consistent naive datetime columns
        model_data = pd.merge(features_df, return_data[['Date', 'Ticker', 'Return']],
                              on=['Date', 'Ticker'], how='inner') # Inner join ensures we have both features and returns

        if model_data.empty:
             print("Error: Merged data is empty. Check date alignment and ticker overlap.")
             # Add check: Print date ranges/types if merge fails
             print("Debug Info:")
             if not features_df.empty:
                 print(f"  features_df Date range (naive): {features_df['Date'].min()} to {features_df['Date'].max()}, dtype: {features_df['Date'].dtype}")
             else:
                 print("  features_df is empty.")
             if not return_data.empty:
                  print(f"  return_data Date range (naive): {return_data['Date'].min()} to {return_data['Date'].max()}, dtype: {return_data['Date'].dtype}")
             else:
                   print("  return_data is empty.")
             return None

        # Define feature columns (excluding Ticker, Date, Return)
        # Dynamically get feature names from the features_df columns, excluding identifiers
        potential_feature_cols = list(features_df.columns)
        exclude_cols = ['Ticker', 'Date']
        feature_columns = sorted([col for col in potential_feature_cols if col not in exclude_cols]) # Sort for consistency

        # Check if feature columns list is empty
        if not feature_columns:
             print("Error: No feature columns were generated.")
             return None

        # Ensure all feature columns are numeric and handle any residual NaNs (should be done in create_features, but safeguard)
        model_data[feature_columns] = model_data[feature_columns].apply(pd.to_numeric, errors='coerce').fillna(0)


        # --- 3. Train/Validation Split ---
        print("Splitting data into train/validation sets...")
        # Sort by the now naive 'Date' column
        model_data = model_data.sort_values('Date')
        unique_dates = np.sort(model_data['Date'].unique())

        if len(unique_dates) < 5: # Need enough dates for a meaningful split
            print("Warning: Very few unique dates in training data. Consider using more data.")
            # Proceed with a simple split if possible, otherwise might need to adjust
            split_ratio = 0.8
        else:
            split_ratio = 0.8

        split_index = int(split_ratio * len(unique_dates))
        if split_index == 0 and len(unique_dates) > 0:
            split_index = 1 # Ensure validation set has at least one date if possible
        if split_index == len(unique_dates) and len(unique_dates) > 0: # Ensure training set has at least one date
             split_index = max(0, len(unique_dates) - 1)


        train_dates = unique_dates[:split_index]
        val_dates = unique_dates[split_index:]

        # Handle case where split results in empty sets
        if len(train_dates) == 0 or len(val_dates) == 0:
             print("Warning: Train or validation set has zero dates after split. Adjusting split or using all data for training.")
             # Fallback: Use all data for training, no validation during this run
             train_data = model_data
             val_data = model_data.iloc[0:0] # Empty dataframe for validation
             print("Using all data for training due to split issue.")
        else:
             train_data = model_data[model_data['Date'].isin(train_dates)]
             val_data = model_data[model_data['Date'].isin(val_dates)]


        if train_data.empty:
             print("Error: Training data is empty after split. Cannot train model.")
             return None

        X_train = train_data[feature_columns].values
        y_train = train_data['Return'].values.reshape(-1, 1)

        perform_validation = not val_data.empty
        if perform_validation:
            X_val = val_data[feature_columns].values
            y_val = val_data['Return'].values.reshape(-1, 1)


        # --- 4. Scaling ---
        print("Scaling features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train) # Fit only on training data
        if perform_validation:
            X_val_scaled = scaler.transform(X_val)         # Transform validation data

        # Convert to PyTorch tensors
        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
        if perform_validation:
            X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
            y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)

        # --- 5. Model Initialization and Training ---
        input_dim = X_train_tensor.shape[1]
        model_net = FeedforwardNet(input_dim).to(device)
        criterion = nn.MSELoss() # Mean Squared Error for regression
        optimizer = optim.Adam(model_net.parameters(), lr=0.001, weight_decay=1e-5) # Added weight decay (L2 regularization)

        print("Training the model...")
        epochs = 100 # Reduced epochs for example, tune as needed
        patience = 10 # Early stopping patience
        best_val_loss = float('inf')
        epochs_no_improve = 0


        for epoch in range(epochs):
            model_net.train() # Set model to training mode
            optimizer.zero_grad()
            outputs = model_net(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()

            val_loss_str = "N/A"
            if perform_validation:
                # Validation phase
                model_net.eval() # Set model to evaluation mode
                with torch.no_grad():
                    val_outputs = model_net(X_val_tensor)
                    val_loss = criterion(val_outputs, y_val_tensor)
                    val_loss_item = val_loss.item()
                    val_loss_str = f"{val_loss_item:.6f}"


                # Early Stopping Check (only if validation is performed)
                if val_loss_item < best_val_loss:
                    best_val_loss = val_loss_item
                    epochs_no_improve = 0
                    # Optionally save the best model state here
                    # torch.save(model_net.state_dict(), 'best_model_state.pth')
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1} due to no improvement in validation loss.")
                    break
            else:
                 # If no validation, just train for fixed epochs
                 if epoch == epochs - 1:
                      print("Training finished (no validation set for early stopping).")


            if epoch % 10 == 0 or epoch == epochs - 1 or not perform_validation:
                print(f"Epoch {(epoch+1):3d}/{epochs} | Train Loss: {loss.item():.6f} | Val Loss: {val_loss_str}")

        # Load the best model state if saved, otherwise use the last state
        # if perform_validation and 'best_model_state.pth' exists:
        #     model_net.load_state_dict(torch.load('best_model_state.pth'))


        # --- 6. Prepare Output ---
        model_info = {
            'model': model_net,
            'scaler': scaler,
            'feature_columns': feature_columns, # Store the exact feature columns used
            'device': device
        }
        print("Training complete. Model info dictionary created.")
        return model_info

    except Exception as e:
        print(f"An error occurred during training: {e}")
        traceback.print_exc() # Print detailed traceback
        return None # Return None indication failure


# UPDATED FUNCTION SIGNATURE AND INTERNAL REFERENCES
def predict_returns(model, sentiment_data_today, stock_universe_today):
    """
    Generate predictions of next-day returns for all stocks in the universe for a given day.

    Parameters:
    -----------
    model : dict
        Dictionary containing the trained model object ('model'), scaler ('scaler'),
        feature columns ('feature_columns'), and device ('device').
    sentiment_data_today : DataFrame
        Raw sentiment data for the *current* day (posts up to 4 PM EST).
    stock_universe_today : list
        List of stock tickers (strings) available for trading *today* (prediction is for *next* day's return).

    Returns:
    --------
    predictions : DataFrame
        DataFrame with columns ['Ticker', 'Predicted_Return', 'Signal_Rank'].
        Contains entries for all tickers in stock_universe_today.
        Returns empty DataFrame with correct columns if prediction fails.
    """
    # --- Output DataFrame Structure ---
    output_columns = ['Ticker', 'Predicted_Return', 'Signal_Rank']
    empty_predictions_df = pd.DataFrame(columns=output_columns)


    # --- Input Validation ---
    # Use 'model' instead of 'model_info'
    if model is None or not isinstance(model, dict) or not all(k in model for k in ['model', 'scaler', 'feature_columns', 'device']):
        print("Error: Invalid or incomplete model dictionary provided.")
        return empty_predictions_df
    if not isinstance(sentiment_data_today, pd.DataFrame):
         print("Warning: sentiment_data_today is not a DataFrame. Assuming no sentiment.")
         sentiment_data_today = pd.DataFrame() # Treat as empty if not DF
    if not isinstance(stock_universe_today, list) or not stock_universe_today:
        print("Error: stock_universe_today must be a non-empty list.")
        return empty_predictions_df
    # Validate feature_columns is a list
    if 'feature_columns' not in model or not isinstance(model['feature_columns'], list):
        print("Error: 'feature_columns' missing or not a list in the model dictionary.")
        return empty_predictions_df


    # --- 1. Preprocess Today's Sentiment Data ---
    try:
        # Important: Filter sentiment data *before* preprocessing if it contains multiple dates
        # Determine the 'prediction date' which is the date the sentiment *applies to*
        # If today is Mon, posts before 4pm apply to Tue. Posts after 4pm apply to Wed.
        # The 'Date' column created by preprocessing handles this. We need the max 'Date'
        # present in the input data.

        if not sentiment_data_today.empty:
            # Preprocessing creates timezone-aware 'Date' (America/New_York)
            sentiment_processed_today = preprocess_sentiment_data(sentiment_data_today)

            if sentiment_processed_today.empty or 'Date' not in sentiment_processed_today.columns or sentiment_processed_today['Date'].isnull().all():
                 print("Warning: No valid sentiment data found for today after preprocessing.")
                 current_pred_date = None # We'll handle this later
                 sentiment_features_input = pd.DataFrame() # No features available
            else:
                # Find the most recent date the sentiment applies to (this is timezone-aware)
                current_pred_date = sentiment_processed_today['Date'].max()
                # Filter for records relevant to this specific prediction date
                sentiment_features_input = sentiment_processed_today[sentiment_processed_today['Date'] == current_pred_date].copy()
        else:
             print("Warning: sentiment_data_today is empty.")
             current_pred_date = None
             sentiment_features_input = pd.DataFrame()


        # --- 2. Create Features for Today ---
        if not sentiment_features_input.empty:
            # Feature creation uses the timezone-aware 'Date' for grouping
            features_today_raw = create_features(sentiment_features_input)
            # The 'Date' column in features_today_raw is still timezone-aware here
        else:
            features_today_raw = pd.DataFrame() # Start with empty if no relevant sentiment


        # --- 3. Align with Stock Universe ---
        # Ensure universe tickers are uppercase and unique
        # Use the passed 'model' dictionary
        universe_upper = sorted(list(set([str(t).upper() for t in stock_universe_today])))


        # Create a base DataFrame for the universe for merging
        universe_df = pd.DataFrame({'Ticker': universe_upper})


        # Merge features onto the universe. Use left merge to keep all universe tickers.
        # Use 'model' dictionary for feature columns
        feature_cols_from_model = model['feature_columns']

        if not features_today_raw.empty and 'Ticker' in features_today_raw.columns:
             # Ensure Ticker column is string type for merging
             features_today_raw['Ticker'] = features_today_raw['Ticker'].astype(str)
             # Select only the necessary feature columns + Ticker for merging
             merge_cols = ['Ticker'] + feature_cols_from_model
             # Ensure all feature columns exist in features_today_raw before merge attempt
             for col in feature_cols_from_model:
                  if col not in features_today_raw.columns:
                       features_today_raw[col] = 0.0 # Add missing feature cols with default 0

             # Perform the merge, handle potential duplicate columns if Ticker is somehow in feature_cols_from_model
             # We are merging on 'Ticker' only, so timezone of 'Date' in features_today_raw doesn't matter here
             cols_to_use = [c for c in merge_cols if c in features_today_raw.columns]
             # Keep only Ticker and feature columns from features_today_raw before merge
             predict_df = pd.merge(universe_df,
                                   features_today_raw[['Ticker'] + feature_cols_from_model], # Use unique cols
                                   on='Ticker',
                                   how='left')
        else:
            # If no features were generated, start with universe and add zeroed feature columns
            predict_df = universe_df.copy()
            for col in feature_cols_from_model:
                predict_df[col] = 0.0


        # Fill NaNs *after* the merge - these represent tickers in the universe with no sentiment features
        # Use the feature columns list from the trained model
        predict_df[feature_cols_from_model] = predict_df[feature_cols_from_model].fillna(0)

        # --- 4. Prepare for Prediction ---
        # Ensure correct column order and extract feature values
        # Reorder columns to match the order used during training/scaling
        # Make sure all expected columns are present, add if missing (shouldn't happen ideally)
        for col in feature_cols_from_model:
             if col not in predict_df.columns:
                  predict_df[col] = 0.0

        predict_df = predict_df[['Ticker'] + feature_cols_from_model] # Ensure order and presence
        X_today = predict_df[feature_cols_from_model].values


        if X_today.shape[0] == 0:
            print("Warning: No data available for prediction after processing.")
            # Return empty dataframe with correct columns
            return empty_predictions_df


        # --- 5. Scale Features ---
        # Use 'model' dictionary
        scaler = model['scaler']
        X_today_scaled = scaler.transform(X_today)


        # --- 6. Predict ---
        # Use 'model' dictionary
        X_today_tensor = torch.tensor(X_today_scaled, dtype=torch.float32).to(model['device'])
        model_obj = model['model'] # Get the actual nn.Module object
        model_obj.eval() # Set model to evaluation mode

        with torch.no_grad():
            predictions_tensor = model_obj(X_today_tensor)

        # Convert predictions back to a numpy array
        predictions_array = predictions_tensor.cpu().numpy().flatten()


        # --- 7. Format Output ---
        predict_df['Predicted_Return'] = predictions_array

        # Add small random noise to break ties before ranking (optional but good practice)
        # noise = np.random.normal(0, 1e-7, size=predictions_array.shape)
        # predict_df['Predicted_Return_Noisy'] = predict_df['Predicted_Return'] + noise

        # Calculate percentile rank
        # Handle case where all predictions are identical (rank would give 0.5 or similar)
        if predict_df['Predicted_Return'].nunique() > 1:
             predict_df['Signal_Rank'] = predict_df['Predicted_Return'].rank(pct=True)
        else:
             # If all predictions are the same, assign rank 0.5 to all
             predict_df['Signal_Rank'] = 0.5


        # Select and return the required columns
        predictions_final = predict_df[output_columns]

        return predictions_final.sort_values('Ticker').reset_index(drop=True) # Sort for consistency

    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        traceback.print_exc() # Print detailed traceback
        # Return empty dataframe with correct columns in case of error
        return empty_predictions_df


####################################
# Example Usage (Optional - for testing)
# Guarded by if __name__ == "__main__":
####################################
if __name__ == "__main__":
    print("Running example usage...")

    # --- Create Dummy Data for Testing ---
    # More realistic dummy sentiment data
    sentiment_list = []
    # Use a timezone-aware base time.
    try:
        import pytz
        est = pytz.timezone('America/New_York')
        utc = pytz.utc
        # Base time for "training" data (May 31st EST)
        base_time_train_aware = datetime(2021, 5, 31, 10, 0, 0, tzinfo=est)
        # Base time for "prediction" data (June 1st EST)
        base_time_predict_aware = datetime(2021, 6, 1, 9, 0, 0, tzinfo=est)

    except ImportError:
        print("Warning: pytz not installed. Timezone handling might be less accurate.")
        # Fallback if pytz not installed (less accurate timezone handling)
        est = None # Indicate pytz is not available
        utc = None
        base_time_train_aware = datetime(2021, 5, 31, 10, 0, 0) # Naive datetime
        base_time_predict_aware = datetime(2021, 6, 1, 9, 0, 0) # Naive datetime


    # Generate "training" sentiment data (posts on May 31st EST)
    print("Generating dummy training sentiment data...")
    for i in range(500):
         ticker = np.random.choice(['AAPL', 'GOOG', 'MSFT', 'TSLA', 'AMZN'])
         sentiment = np.random.choice([-1, 0, 1], p=[0.2, 0.3, 0.5])
         conf = np.random.rand()
         prob_pos = np.random.uniform(0.3, 0.8) if sentiment == 1 else np.random.uniform(0.0, 0.4)
         prob_neg = np.random.uniform(0.3, 0.8) if sentiment == -1 else np.random.uniform(0.0, 0.4)
         prob_ntr = 1.0 - prob_pos - prob_neg
         # Generate time offset (covering times before and after 4 PM EST on May 31st)
         time_offset = timedelta(minutes=np.random.randint(-3*60, 10*60)) # Offsets around 10 AM
         recv_time_aware = base_time_train_aware + time_offset
         # Convert to UTC string for the 'Received_Time' column format
         if utc:
            recv_time_utc = recv_time_aware.astimezone(utc)
            recv_time_str = recv_time_utc.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] # Simulate UTC format
         else: # Fallback for naive datetime
            recv_time_utc = recv_time_aware # Cannot convert timezone
            recv_time_str = recv_time_utc.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] # Store as naive time string

         sentiment_list.append({
             'Received_Time': recv_time_str, # Store as string
             'Ticker': ticker,
             'Sentiment': sentiment,
             'Confidence': conf,
             'Prob_POS': prob_pos,
             'Prob_NTR': max(0, prob_ntr), # Ensure non-negative
             'Prob_NEG': prob_neg,
             'Relevance': np.random.rand(),
             'SourceWeight': np.random.rand(),
             'TopicWeight': np.random.rand(),
             'Author': f'author_{np.random.randint(1, 50)}',
             'Novelty': np.random.randint(1, 5),
             'Comment_Count': np.random.randint(0, 10)
         })

    # Generate "prediction day" sentiment data (posts on June 1st EST, before 4 PM)
    print("Generating dummy prediction sentiment data...")
    for i in range(50):
         ticker = np.random.choice(['AAPL', 'GOOG', 'MSFT', 'TSLA', 'AMZN', 'NVDA']) # Include NVDA
         sentiment = np.random.choice([-1, 0, 1], p=[0.3, 0.4, 0.3])
         conf = np.random.rand()
         # Generate time offset (before 4 PM EST on June 1st)
         # Max offset from 9 AM to stay before 4 PM (16:00) is 7 hours * 60 = 420 minutes
         time_offset = timedelta(minutes=np.random.randint(0, 420))
         recv_time_aware = base_time_predict_aware + time_offset
         # Convert to UTC string
         if utc:
            recv_time_utc = recv_time_aware.astimezone(utc)
            recv_time_str = recv_time_utc.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
         else:
            recv_time_utc = recv_time_aware
            recv_time_str = recv_time_utc.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

         sentiment_list.append({
             'Received_Time': recv_time_str, # Store as string
             'Ticker': ticker, 'Sentiment': sentiment, 'Confidence': conf,
             'Prob_POS': np.random.rand()*0.8, 'Prob_NTR': np.random.rand()*0.4, 'Prob_NEG': np.random.rand()*0.8, # Simplified probabilities
             'Relevance': np.random.rand(), 'SourceWeight': np.random.rand(), 'TopicWeight': np.random.rand(),
             'Author': f'author_{np.random.randint(51, 100)}', 'Novelty': 1, 'Comment_Count': np.random.randint(0,5)
         })

    dummy_sentiment_data = pd.DataFrame(sentiment_list)
    # Ensure Received_Time is object/string type as expected from read_csv initially
    dummy_sentiment_data['Received_Time'] = dummy_sentiment_data['Received_Time'].astype(str)


    # Dummy return data
    # Needs returns for the dates *assigned* by preprocess_sentiment_data
    # May 31st posts before 4pm -> Date = May 31st -> Need Return for May 31st
    # May 31st posts after 4pm -> Date = June 1st -> Need Return for June 1st
    print("Generating dummy return data...")
    dates_return = pd.to_datetime(['2021-05-31', '2021-06-01']) # Naive dates
    tickers_return = ['AAPL', 'GOOG', 'MSFT', 'TSLA', 'AMZN'] # Tickers present in "training" sentiment
    return_list_dummy = []
    for date in dates_return:
        for ticker in tickers_return:
            return_list_dummy.append({'Date': date, 'Ticker': ticker, 'Return': np.random.randn() * 0.02})

    dummy_return_data = pd.DataFrame(return_list_dummy)
    # Ensure Date is datetime object (will be normalized in train_model)
    dummy_return_data['Date'] = pd.to_datetime(dummy_return_data['Date'])

    print(f"\nDummy Sentiment Data Shape: {dummy_sentiment_data.shape}")
    print(f"Dummy Return Data Shape: {dummy_return_data.shape}")


    print("\n--- Training Model ---")
    # Pass the full dummy sentiment data; train_model preprocesses and splits
    model_info_output = train_model(dummy_sentiment_data, dummy_return_data)

    if model_info_output:
        print("\n--- Predicting Returns ---")
        # Prepare data for prediction day (June 1st sentiment -> predict for June 2nd returns)
        # Filter the *raw* dummy sentiment data for the relevant day's posts
        # This simulates receiving only the data posted on June 1st
        if utc: # If pytz was available, filter based on UTC strings corresponding to June 1st EST
             start_utc = base_time_predict_aware.astimezone(utc).strftime('%Y-%m-%d')
             end_utc_dt = (base_time_predict_aware + timedelta(days=1)).astimezone(utc)
             sentiment_today_predict = dummy_sentiment_data[
                 (dummy_sentiment_data['Received_Time'] >= start_utc) &
                 (dummy_sentiment_data['Received_Time'] < end_utc_dt.strftime('%Y-%m-%d %H:%M:%S'))
             ].copy()
        else: # Fallback: filter based on naive date string
             predict_date_str = base_time_predict_aware.strftime('%Y-%m-%d')
             sentiment_today_predict = dummy_sentiment_data[
                 dummy_sentiment_data['Received_Time'].str.startswith(predict_date_str)
             ].copy()


        print(f"Shape of sentiment data passed for prediction: {sentiment_today_predict.shape}")

        stock_universe_june_1 = ['AAPL', 'GOOG', 'MSFT', 'TSLA', 'AMZN', 'NVDA', 'IBM'] # Universe for prediction day


        # Call predict_returns using the 'model' dictionary from training output
        predictions_output = predict_returns(model=model_info_output, # Pass the dict here
                                           sentiment_data_today=sentiment_today_predict,
                                           stock_universe_today=stock_universe_june_1)


        print("\nSample Predictions:")
        if not predictions_output.empty:
             print(predictions_output.head())
             print(f"\nTotal predictions made: {len(predictions_output)}")
             predicted_tickers_set = set(predictions_output['Ticker'])
             expected_tickers_set = set([t.upper() for t in stock_universe_june_1])
             print(f"Tickers predicted: {sorted(list(predicted_tickers_set))}")
             print(f"Expected universe: {sorted(list(expected_tickers_set))}")
             # Check if all universe tickers are present
             if predicted_tickers_set == expected_tickers_set:
                  print("All tickers from the universe are present in the predictions.")
             else:
                   print("Mismatch between predicted tickers and stock universe!")
                   print(f"Missing from predictions: {sorted(list(expected_tickers_set - predicted_tickers_set))}")
                   print(f"Extra in predictions: {sorted(list(predicted_tickers_set - expected_tickers_set))}")
        else:
             print("Prediction resulted in an empty DataFrame.")

    else:
        print("\nModel training failed, skipping prediction.")

    print("\nExample usage finished.")