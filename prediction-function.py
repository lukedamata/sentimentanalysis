# Sentiment-Based Return Prediction Function
# Team Group X

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import warnings
warnings.filterwarnings('ignore')

##############################################
# Helper: Ensure sentiment data has a 'Date' column
##############################################
def ensure_date_column(df):
    """
    Ensure that the DataFrame has a 'Date' column.
    If 'Date' is missing, attempt to create it from 'Received_Time' or 'Post_Time'.
    The date is taken as the day portion (floored to 'D').
    """
    if 'Date' not in df.columns:
        if 'Received_Time' in df.columns:
            df['Date'] = pd.to_datetime(df['Received_Time']).dt.floor('D')
        elif 'Post_Time' in df.columns:
            df['Date'] = pd.to_datetime(df['Post_Time']).dt.floor('D')
        else:
            raise KeyError("No 'Date', 'Received_Time', or 'Post_Time' column found in sentiment data.")
    else:
        df['Date'] = pd.to_datetime(df['Date'])
    return df

##############################################
# Feature Engineering Function
##############################################
def engineer_features(sentiment_data, return_data=None, training=True):
    """
    Engineer features from sentiment data.
    
    Parameters:
    -----------
    sentiment_data : DataFrame
        The Reddit sentiment data.
    return_data : DataFrame, optional
        The stock return data (used during training).
    training : bool, default True
        Whether this is for training (merging with returns) or prediction.
        
    Returns:
    --------
    features_df : DataFrame
        DataFrame with engineered features.
    """
    # Ensure the 'Date' column is present
    sentiment_data = ensure_date_column(sentiment_data)
    
    # Make a copy to avoid modifying the original data
    sentiment = sentiment_data.copy()
    
    # Group by Date and Ticker
    grouped = sentiment.groupby(['Date', 'Ticker'])
    
    # Create a list to store daily features for each ticker
    daily_features = []
    
    # Process each day and ticker combination
    for (date, ticker), group in grouped:
        # Skip if too few posts (need at least 3 posts for meaningful stats)
        if len(group) < 3:
            continue
            
        # 1. Basic sentiment statistics
        mean_sentiment = group['Sentiment'].mean()
        median_sentiment = group['Sentiment'].median()
        std_sentiment = group['Sentiment'].std()
        min_sentiment = group['Sentiment'].min()
        max_sentiment = group['Sentiment'].max()
        
        # Handle NaN in std calculation for single posts
        if pd.isna(std_sentiment):
            std_sentiment = 0
            
        # 2. Volume indicators
        post_count = len(group)
        log_post_count = np.log1p(post_count)
        
        # 3. Probability-based features
        mean_prob_pos = group['Prob_POS'].mean()
        mean_prob_neg = group['Prob_NEG'].mean()
        mean_prob_ntr = group['Prob_NTR'].mean()
        
        # Sentiment certainty (max probability regardless of class)
        max_probs = group[['Prob_POS', 'Prob_NEG', 'Prob_NTR']].max(axis=1)
        mean_certainty = max_probs.mean()
        
        # Entropy of probability distribution (measure of ambiguity)
        def entropy(row):
            probs = [row['Prob_POS'], row['Prob_NEG'], row['Prob_NTR']]
            # Filter out zero probabilities to avoid log(0)
            probs = [p for p in probs if p > 0]
            return -sum(p * np.log(p) for p in probs)
        mean_entropy = group.apply(entropy, axis=1).mean()
        
        # 4. Author-based features
        unique_authors = group['Author'].nunique()
        author_ratio = unique_authors / post_count  # Ratio of unique authors to total posts
        
        # 5. Sentiment polarity and distribution
        pos_ratio = (group['Sentiment'] > 0).mean()  # Ratio of positive sentiment
        neg_ratio = (group['Sentiment'] < 0).mean()  # Ratio of negative sentiment
        ntr_ratio = (group['Sentiment'] == 0).mean()  # Ratio of neutral sentiment
        
        # Check if there are enough samples for skewness calculation
        if len(group) >= 3:
            sentiment_skew = group['Sentiment'].skew()
            if pd.isna(sentiment_skew):
                sentiment_skew = 0
        else:
            sentiment_skew = 0
            
        # 6. Weighted sentiment
        weighted_sentiment = (group['Sentiment'] * group['Prob_POS']).sum() / group['Prob_POS'].sum() \
            if group['Prob_POS'].sum() > 0 else mean_sentiment
            
        # Collect all features in a dictionary
        features = {
            'Date': date,
            'Ticker': ticker,
            'mean_sentiment': mean_sentiment,
            'median_sentiment': median_sentiment,
            'std_sentiment': std_sentiment,
            'min_sentiment': min_sentiment,
            'max_sentiment': max_sentiment,
            'sentiment_range': max_sentiment - min_sentiment,
            'post_count': post_count,
            'log_post_count': log_post_count,
            'mean_prob_pos': mean_prob_pos,
            'mean_prob_neg': mean_prob_neg,
            'mean_prob_ntr': mean_prob_ntr,
            'mean_certainty': mean_certainty,
            'mean_entropy': mean_entropy,
            'unique_authors': unique_authors,
            'author_ratio': author_ratio,
            'pos_ratio': pos_ratio,
            'neg_ratio': neg_ratio,
            'ntr_ratio': ntr_ratio,
            'sentiment_skew': sentiment_skew,
            'weighted_sentiment': weighted_sentiment
        }
        
        daily_features.append(features)
    
    # Convert list to DataFrame
    features_df = pd.DataFrame(daily_features)
    
    if len(features_df) == 0:
        return pd.DataFrame()  # Return empty DataFrame if no features were created
        
    # If training, merge with return data to obtain the target variable
    if training and return_data is not None:
        # Ensure the return data 'Date' column is datetime
        return_data['Date'] = pd.to_datetime(return_data['Date'])
        
        # Merge features with next-day returns on Date and Ticker
        features_df = pd.merge(
            features_df,
            return_data[['Date', 'Ticker', 'Return']],
            on=['Date', 'Ticker'],
            how='left'
        )
        
        # Print a warning if returns are missing and drop those rows
        missing_returns = features_df['Return'].isnull().sum()
        if missing_returns > 0:
            print(f"Warning: {missing_returns} rows have missing returns and will be dropped.")
            features_df = features_df.dropna(subset=['Return'])
    
    return features_df

##############################################
# Training Function
##############################################
def train_model(sentiment_data, return_data):
    """
    Train a model using sentiment features to predict next-day returns.
    
    Parameters:
    -----------
    sentiment_data : DataFrame
        The Reddit sentiment data for training (e.g., sentiment_train_2017_2021.csv).
    return_data : DataFrame
        The stock return data for training (e.g., return_train_2017_2021.csv).
        
    Returns:
    --------
    model_info : dict
        Dictionary containing the trained model object and necessary metadata.
    """
    import xgboost as xgb
    
    print("Engineering features...")
    features_df = engineer_features(sentiment_data, return_data, training=True)
    
    if features_df.empty:
        raise ValueError("No features could be engineered from the provided data.")
    
    # Prepare training data
    X = features_df.drop(['Date', 'Ticker', 'Return'], axis=1)
    y = features_df['Return']
    
    # Fill in any remaining missing values with feature means
    X = X.fillna(X.mean())
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("Training XGBoost model...")
    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_scaled, y)
    
    # Create a dictionary with model and metadata
    model_info = {
        'model': model,
        'scaler': scaler,
        'feature_names': X.columns.tolist(),
        'model_type': 'XGBoost'
    }
    
    return model_info

##############################################
# Prediction Function
##############################################
def predict_returns(model_info, sentiment_data_today, stock_universe_today, historical_data=None):
    """
    Generate predictions of next-day returns for all stocks in the universe.
    
    Parameters:
    -----------
    model_info : dict
        Dictionary containing your trained model and metadata.
    sentiment_data_today : DataFrame
        Sentiment data for the current day.
    stock_universe_today : list
        List of stock tickers available for trading today.
    historical_data : dict, optional
        Dictionary containing historical sentiment and return data.
        
    Returns:
    --------
    predictions : DataFrame
        DataFrame with columns ['Ticker', 'Predicted_Return', 'Signal_Rank'].
    """
    # Extract model components from model_info
    model = model_info['model']
    scaler = model_info['scaler']
    feature_names = model_info['feature_names']
    
    # Ensure today's sentiment data has a 'Date' column
    sentiment_data_today = ensure_date_column(sentiment_data_today)
    
    # Engineer features for today's data (do not merge with returns during prediction)
    features_today_df = engineer_features(sentiment_data_today, training=False)
    
    # Create a DataFrame for the full stock universe
    predictions_df = pd.DataFrame({'Ticker': stock_universe_today})
    predictions_df['Predicted_Return'] = 0.0  # Initialize with neutral prediction
    
    # Update predictions for tickers where sentiment features exist
    if not features_today_df.empty:
        # Keep only tickers that are in the stock universe
        valid_features = features_today_df[features_today_df['Ticker'].isin(stock_universe_today)]
        
        if not valid_features.empty:
            # Ensure that all required features are present
            X_pred = pd.DataFrame(index=valid_features.index)
            for feature in feature_names:
                if feature in valid_features.columns:
                    X_pred[feature] = valid_features[feature]
                else:
                    X_pred[feature] = 0.0  # Default value if missing
            
            X_pred = X_pred.fillna(0)
            
            # Scale the features
            X_pred_scaled = scaler.transform(X_pred)
            
            # Make predictions
            y_pred = model.predict(X_pred_scaled)
            
            # Create a DataFrame of predictions
            pred_df = pd.DataFrame({
                'Ticker': valid_features['Ticker'].values,
                'Predicted_Return': y_pred
            })
            
            # Merge these predictions with the full universe
            predictions_df = pd.merge(predictions_df, pred_df, on='Ticker', how='left')
            # If duplicate columns occur, prioritize the new predictions
            if 'Predicted_Return_x' in predictions_df.columns:
                predictions_df['Predicted_Return'] = predictions_df['Predicted_Return_y'].fillna(predictions_df['Predicted_Return_x'])
                predictions_df = predictions_df[['Ticker', 'Predicted_Return']]
    
    # Calculate signal rank as the percentile ranking of predicted returns
    if predictions_df['Predicted_Return'].nunique() > 1:
        predictions_df['Signal_Rank'] = predictions_df['Predicted_Return'].rank(pct=True)
    else:
        predictions_df['Signal_Rank'] = 0.5  # Assign neutral rank if all predictions are identical
    
    return predictions_df[['Ticker', 'Predicted_Return', 'Signal_Rank']]

##############################################
# Optional Helper Functions for Saving/Loading Models
##############################################
def load_model(model_path):
    """Helper function to load a saved model."""
    with open(model_path, 'rb') as f:
        model_info = pickle.load(f)
    return model_info

def save_model(model_info, model_path):
    """Helper function to save a model."""
    with open(model_path, 'wb') as f:
        pickle.dump(model_info, f)
    return True


if __name__ == "__main__":
    print("Sentiment-Based Return Prediction Function Module")
    print("To use this module, import the functions: train_model and predict_returns.")
