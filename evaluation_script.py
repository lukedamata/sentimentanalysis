# Evaluation Script for Deliverable #2
# This script evaluates student submissions on out-of-sample data (2022-2024)
# File structure (relative to this script):
#
# evaluation/
# ├── data/
# │   ├── sentiment_train_2017_2021.csv
# │   ├── return_train_2017_2021.csv
# │   ├── sentiment_test_2022_2024.csv
# │   └── return_test_2022_2024.csv
# │
# ├── student_submissions/
# │   ├── group1_prediction_function.py
# │   ├── group2_prediction_function.py
# │   ├── ...
# │   └── example_prediction_function.py
# │
# ├── evaluation_script.py        <-- This script
# │
# ├── results/
# │   ├── group1/
# │   │   ├── evaluation_metrics_group1.csv
# │   │   ├── predictions_group1.csv
# │   │   └── ...
# │   ├── group2/
# │   │   └── ...
# │   └── ...



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta, time
import seaborn as sns
from importlib.machinery import SourceFileLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ----------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------
DATA_DIR = "data/"
OUT_SAMPLE_SENTIMENT_FILE = "sentiment_test_2022_2024.csv"
OUT_SAMPLE_RETURNS_FILE = "return_test_2022_2024.csv"
TRAIN_SENTIMENT_FILE = "sentiment_train_2017_2021.csv"
TRAIN_RETURNS_FILE = "return_train_2017_2021.csv"
PREDICTION_FUNCTION_DIR = "student_submissions/"
RESULTS_DIR = "results/"

# Create results directory if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)

# ----------------------------------------------------------------
# Load Test Data
# ----------------------------------------------------------------
def load_test_data():
    """Load the out-of-sample test data (2022-2024)"""
    print("Loading test data...")
    
    try:
        sentiment_data = pd.read_csv(os.path.join(DATA_DIR, OUT_SAMPLE_SENTIMENT_FILE))
        return_data = pd.read_csv(os.path.join(DATA_DIR, OUT_SAMPLE_RETURNS_FILE))
        print(f"Successfully loaded data from {DATA_DIR}")
    except FileNotFoundError:
        try:
            sentiment_data = pd.read_csv(OUT_SAMPLE_SENTIMENT_FILE)
            return_data = pd.read_csv(OUT_SAMPLE_RETURNS_FILE)
            print("Successfully loaded data from current directory")
        except FileNotFoundError:
            print("Could not find data files in either location. Please check file paths.")
            raise
    
    # Print basic info
    print(f"Sentiment data: {sentiment_data.shape[0]} rows, {sentiment_data.shape[1]} columns")
    print(f"Return data: {return_data.shape[0]} rows, {return_data.shape[1]} columns")
    
    return sentiment_data, return_data

# ----------------------------------------------------------------
# Load Training Data
# ----------------------------------------------------------------
def load_training_data():
    """Load the training data (2017-2021)"""
    print("Loading training data...")
    
    try:
        sentiment_data = pd.read_csv(os.path.join(DATA_DIR, TRAIN_SENTIMENT_FILE))
        return_data = pd.read_csv(os.path.join(DATA_DIR, TRAIN_RETURNS_FILE))
        print(f"Successfully loaded training data from {DATA_DIR}")
    except FileNotFoundError:
        try:
            sentiment_data = pd.read_csv(TRAIN_SENTIMENT_FILE)
            return_data = pd.read_csv(TRAIN_RETURNS_FILE)
            print("Successfully loaded training data from current directory")
        except FileNotFoundError:
            print("Could not find training data files. Please check file paths.")
            return None, None
    
    # Print basic info
    print(f"Training sentiment data: {sentiment_data.shape[0]} rows, {sentiment_data.shape[1]} columns")
    print(f"Training return data: {return_data.shape[0]} rows, {return_data.shape[1]} columns")
    
    return sentiment_data, return_data

# ----------------------------------------------------------------
# Import Student's Functions
# ----------------------------------------------------------------
def import_student_functions(group_name):
    """Import the student's prediction and training functions"""
    try:
        module_path = os.path.join(PREDICTION_FUNCTION_DIR, f"{group_name}_prediction_function.py")
        if not os.path.exists(module_path):
            print(f"Error: File not found - {module_path}")
            return None, None
            
        module = SourceFileLoader(f"{group_name}_prediction", module_path).load_module()
        
        # Check for required functions
        predict_func = getattr(module, 'predict_returns', None)
        train_func = getattr(module, 'train_model', None)
        
        if predict_func is None:
            print(f"Error: {group_name}'s submission is missing the 'predict_returns' function")
        if train_func is None:
            print(f"Warning: {group_name}'s submission is missing the 'train_model' function")
            
        return predict_func, train_func
    except Exception as e:
        print(f"Error importing {group_name}'s module: {str(e)}")
        return None, None

# ----------------------------------------------------------------
# Evaluation Functions
# ----------------------------------------------------------------
def evaluate_predictions(predictions_df, actual_returns_df, date):
    """
    Evaluate predictions for a single day with enhanced metrics
    
    Parameters:
    -----------
    predictions_df : DataFrame
        DataFrame with columns ['Ticker', 'Predicted_Return', 'Signal_Rank']
    
    actual_returns_df : DataFrame
        DataFrame with actual returns
        
    date : datetime
        The date to evaluate
        
    Returns:
    --------
    results : dict
        Dictionary of evaluation metrics
    """
    # Merge predictions with actual returns
    merged = pd.merge(
        predictions_df, 
        actual_returns_df[actual_returns_df['Date'] == date][['Ticker', 'Return']],
        on='Ticker',
        how='inner'
    )
    
    if len(merged) == 0:
        print(f"Warning: No matching tickers found for date {date}")
        return None
    
    if len(merged) < 5:
        print(f"Warning: Only {len(merged)} tickers found for date {date}. This may not be enough for reliable quintile analysis.")
    
    # Calculate regression metrics
    mse = mean_squared_error(merged['Return'], merged['Predicted_Return'])
    mae = mean_absolute_error(merged['Return'], merged['Predicted_Return'])
    
    # R-squared (can be negative if predictions are worse than mean)
    r2 = r2_score(merged['Return'], merged['Predicted_Return'])
        
    # Assign quintiles based on Signal_Rank
    try:
        # Try using qcut with ranked data (handles duplicate values better)
        signal_rank_quantiles = np.linspace(0, 1, 6)  # [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        
        # Create quintile labels
        quintile_labels = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
        
        # Assign quintiles (fallback method for handling duplicate values)
        conditions = [
            (merged['Signal_Rank'] <= signal_rank_quantiles[1]),
            (merged['Signal_Rank'] > signal_rank_quantiles[1]) & (merged['Signal_Rank'] <= signal_rank_quantiles[2]),
            (merged['Signal_Rank'] > signal_rank_quantiles[2]) & (merged['Signal_Rank'] <= signal_rank_quantiles[3]),
            (merged['Signal_Rank'] > signal_rank_quantiles[3]) & (merged['Signal_Rank'] <= signal_rank_quantiles[4]),
            (merged['Signal_Rank'] > signal_rank_quantiles[4])
        ]
        merged['Quintile'] = np.select(conditions, quintile_labels, default='Q3')
        
    except Exception as e:
        print(f"Error creating quintiles for {date}: {str(e)}")
        # Fallback to equal frequency binning ignoring duplicate values
        merged['Quintile'] = pd.qcut(
            merged['Signal_Rank'].rank(method='first'), 
            5, 
            labels=quintile_labels
        )
    
    # Calculate quintile returns
    quintile_returns = merged.groupby('Quintile')['Return'].mean()
    
    # Ensure all quintiles are represented (even if empty)
    for quintile in quintile_labels:
        if quintile not in quintile_returns.index:
            quintile_returns[quintile] = np.nan
    
    # Calculate long-short return (Q5 - Q1)
    if 'Q5' in quintile_returns.index and 'Q1' in quintile_returns.index:
        if not pd.isna(quintile_returns['Q5']) and not pd.isna(quintile_returns['Q1']):
            long_short_return = quintile_returns['Q5'] - quintile_returns['Q1']
        else:
            long_short_return = np.nan
    else:
        long_short_return = np.nan
    
    return {
        'Date': date,
        'MSE': mse,
        'MAE': mae,
        'R2': r2,
        'Long_Short_Return': long_short_return,
        'Quintile_Returns': quintile_returns.to_dict()
    }

def run_evaluation(group_name):
    """
    Run a complete evaluation for a student group:
    - Combine train and test data at the start
    - For each test day, filter data up to that day for prediction
    """
    print(f"\n{'='*60}")
    print(f"Evaluating submission for group: {group_name}")
    print(f"{'='*60}")
    
    # Create group results directory
    group_dir = os.path.join(RESULTS_DIR, group_name)
    os.makedirs(group_dir, exist_ok=True)
    
    # Load train and test data
    try:
        train_sentiment, train_returns = load_training_data()
        test_sentiment, test_returns = load_test_data()
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None
    
    if train_sentiment is None or train_returns is None:
        print("Warning: Training data not available. Will use only test data.")
    
    # Combine train and test data
    all_sentiment = pd.concat([train_sentiment, test_sentiment]) if train_sentiment is not None else test_sentiment.copy()
    
    # Import student functions
    predict_func, train_model_func = import_student_functions(group_name)
    
    if predict_func is None:
        print("Cannot continue evaluation without predict_returns function.")
        return None
    
    # Train the model if train_model_func exists
    model = None
    if train_model_func is not None and train_sentiment is not None and train_returns is not None:
        try:
            print("Training model...")
            model = train_model_func(train_sentiment, train_returns)
            print("Model training completed")
        except Exception as e:
            print(f"Error training model: {str(e)}")
            print("Will attempt to continue evaluation using just the predict_returns function.")
    
    # Get all unique dates in the test set
    test_dates = sorted(pd.to_datetime(test_returns['Date']).unique())
    
    # Convert the 'Received_Time' column to datetime and localize to EST
    received_time = pd.to_datetime(all_sentiment['Received_Time'])
    received_time = received_time.dt.tz_localize('UTC')
    received_time_EST = received_time.dt.tz_convert('America/New_York')


    # Convert 'Return' column to float if needed
    def convert_return(return_val):
        if isinstance(return_val, str):
            if '%' in return_val:
                return float(return_val.replace('%', '')) / 100
            else:
                return float(return_val)
        return return_val
    
    # Convert 'Return' column to float and 'Date' to datetime
    test_returns['Return'] = test_returns['Return'].apply(convert_return)
    test_returns['Date'] = pd.to_datetime(test_returns['Date'])


    # Initialize results
    all_results = []
    all_predictions = []


    # Evaluate day by day
    for i, date in enumerate(test_dates):
        print(f"Processing {date.strftime('%Y-%m-%d')} ({i+1}/{len(test_dates)})", end='\r')
        
        # Create a timestamp for 4pm EST on the cutoff date
        cutoff_timestamp = pd.Timestamp(date.year, date.month, date.day, 16, 0, 0, tz='America/New_York')
        
        # Filter sentiment data up to 4pm EST on the current date
        historical_sentiment = all_sentiment[received_time_EST <= cutoff_timestamp].copy()
                            
        stock_universe = test_returns[test_returns['Date'] == date]['Ticker'].unique().tolist()
        
        # Skip days with empty universe
        if len(stock_universe) == 0:
            print(f"\nWarning: Empty stock universe for {date}. Skipping.")
            continue
        
        # Get predictions from the student's function
        try:
            # This is the simplified part - we pass today's sentiment and the historical data
            # through the sentiment_data_today parameter
            predictions = predict_func(
                model=model,
                sentiment_data_today=historical_sentiment,  # All historical data up to today
                stock_universe_today=stock_universe
            )
            
            # Check if predictions are in the correct format
            required_cols = ['Ticker', 'Predicted_Return', 'Signal_Rank']
            missing_cols = [col for col in required_cols if col not in predictions.columns]
            
            if missing_cols:
                print(f"\nError: Missing columns in prediction output: {missing_cols}")
                continue
                
            # Fill any NaN values in predictions with 0 (for robustness)
            if predictions['Predicted_Return'].isna().any():
                print(f"\nWarning: NaN values found in predictions for {date}. Filling with 0.")
                predictions['Predicted_Return'] = predictions['Predicted_Return'].fillna(0)
                
            if predictions['Signal_Rank'].isna().any():
                print(f"\nWarning: NaN values found in signal ranks for {date}. Filling with 0.5.")
                predictions['Signal_Rank'] = predictions['Signal_Rank'].fillna(0.5)
            
            # Add date to predictions
            predictions['Date'] = date
            all_predictions.append(predictions)
            
            # Evaluate this day's predictions
            day_results = evaluate_predictions(predictions, test_returns, date)
            if day_results:
                all_results.append(day_results)
                
        except Exception as e:
            print(f"\nError on {date.strftime('%Y-%m-%d')}: {str(e)}")
            continue
    
    print("\nEvaluation complete!")
    
    if len(all_results) == 0:
        print("No valid results generated.")
        return None
        
    # Combine all results
    results_df = pd.DataFrame(all_results)
    
    # Combine all predictions
    predictions_df = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    
    # Save results to files
    results_df.to_csv(os.path.join(group_dir, f"evaluation_metrics_{group_name}.csv"), index=False)
    if len(predictions_df) > 0:
        predictions_df.to_csv(os.path.join(group_dir, f"predictions_{group_name}.csv"), index=False)
    
    return {
        'summary': results_df,
        'predictions': predictions_df
    }

# ----------------------------------------------------------------
# Performance Metrics Calculation
# ----------------------------------------------------------------
def calculate_performance_metrics(results_summary):
    """
    Calculate comprehensive performance metrics from evaluation results
    
    Parameters:
    -----------
    results_summary : DataFrame
        DataFrame with daily evaluation results
        
    Returns:
    --------
    metrics : dict
        Dictionary of performance metrics
    """
    # Extract the long-short returns series
    long_short_returns = results_summary['Long_Short_Return'].dropna()
    
    if len(long_short_returns) == 0:
        return {
            'error': 'No valid long-short returns found'
        }
    
    # Basic return metrics
    avg_daily_return = long_short_returns.mean()
    median_daily_return = long_short_returns.median()
    
    # Risk metrics
    volatility = long_short_returns.std()
    annualized_volatility = volatility * np.sqrt(252)  # Assuming 252 trading days
    
    # Sharpe Ratio (assuming 0% risk-free rate for simplicity)
    sharpe_ratio = avg_daily_return / volatility if volatility > 0 else 0
    annualized_sharpe = sharpe_ratio * np.sqrt(252)
    
    # Drawdown analysis
    cumulative_returns = (1 + long_short_returns).cumprod()
    rolling_max = cumulative_returns.cummax()
    drawdowns = (cumulative_returns / rolling_max) - 1
    max_drawdown = drawdowns.min()

    # Cumulative return
    total_return = cumulative_returns.iloc[-1] - 1 if len(cumulative_returns) > 0 else 0
    
    # Average metrics
    avg_mse = results_summary['MSE'].mean()
    avg_mae = results_summary['MAE'].mean()
    avg_r2 = results_summary['R2'].mean()
    
    # Calculate quintile average returns
    all_quintile_returns = {}
    
    # Using the first row to get quintile keys
    if len(results_summary) > 0 and 'Quintile_Returns' in results_summary.columns:
        first_row = results_summary.iloc[0]
        if isinstance(first_row['Quintile_Returns'], dict):
            quintile_keys = sorted(list(first_row['Quintile_Returns'].keys()))
            
            for q in quintile_keys:
                # Extract returns for this quintile across all days
                q_returns = []
                for idx, row in results_summary.iterrows():
                    if isinstance(row['Quintile_Returns'], dict) and q in row['Quintile_Returns']:
                        q_returns.append(row['Quintile_Returns'][q])
                
                all_quintile_returns[q] = np.nanmean(q_returns)
    
    return {
        # Return metrics
        'avg_daily_return': avg_daily_return,
        'median_daily_return': median_daily_return,
        'annualized_return': (1 + avg_daily_return) ** 252 - 1,
        'total_return': total_return,
        
        # Risk metrics
        'volatility': volatility,
        'annualized_volatility': annualized_volatility,
        'sharpe_ratio': sharpe_ratio,
        'annualized_sharpe': annualized_sharpe,
        'max_drawdown': max_drawdown,
        
        # Prediction accuracy metrics
        'avg_mse': avg_mse,
        'avg_mae': avg_mae,
        'avg_r2': avg_r2,
        
        # Quintile returns
        'quintile_returns': all_quintile_returns,
        
        # Sample size
        'days_evaluated': len(results_summary),
        'trading_days': len(long_short_returns)
    }

# ----------------------------------------------------------------
# Visualization Functions
# ----------------------------------------------------------------
def plot_evaluation_results(results, group_name):
    """
    Generate comprehensive visualizations from evaluation results
    
    Parameters:
    -----------
    results : dict
        Dictionary with evaluation results
    
    group_name : str
        Name of the student group
    """
    if results is None or 'summary' not in results:
        print("No results to visualize.")
        return
    
    summary = results['summary']
    
    # Create group results directory
    group_dir = os.path.join(RESULTS_DIR, group_name)
    os.makedirs(group_dir, exist_ok=True)
    
    # Calculate additional metrics
    metrics = calculate_performance_metrics(summary)
    
    # Check for errors
    if 'error' in metrics:
        print(f"Error calculating metrics: {metrics['error']}")
        return
    
    # 1. Regression Metrics Plot
    plt.figure(figsize=(14, 8))
    plt.subplot(2, 1, 1)
    plt.plot(summary['Date'], summary['MAE'], label='MAE', color='blue')
    plt.plot(summary['Date'], summary['MSE'], label='MSE', color='red')
    plt.title('Prediction Error Metrics Over Time')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(summary['Date'], summary['R2'], label='R²', color='green')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.title('R² Over Time')
    plt.ylabel('R²')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(group_dir, f"{group_name}_regression_metrics.png"))
    plt.close()
        
    # 2. Long-Short Returns
    plt.figure(figsize=(14, 6))
    plt.plot(summary['Date'], summary['Long_Short_Return'], label='Daily Return', color='green')
    plt.axhline(y=0, color='red', linestyle='--', label='Breakeven')
    plt.title(f'Long-Short Portfolio Daily Returns (Avg: {metrics["avg_daily_return"]:.4f})')
    plt.ylabel('Return')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(group_dir, f"{group_name}_longshort_returns.png"))
    plt.close()
    
    # 3. Cumulative Returns by Quintile
    # Create a dataframe with daily returns for each quintile
    quintile_df = pd.DataFrame(index=summary['Date'])
    
    if len(summary) > 0 and 'Quintile_Returns' in summary.columns:
        # Get quintile keys from first row (assuming all rows have same quintiles)
        first_row = summary.iloc[0]
        if isinstance(first_row['Quintile_Returns'], dict):
            quintile_keys = sorted(list(first_row['Quintile_Returns'].keys()))
            
            # Extract returns for each quintile
            for q in quintile_keys:
                quintile_df[q] = [
                    row['Quintile_Returns'].get(q, np.nan) if isinstance(row['Quintile_Returns'], dict) else np.nan
                    for _, row in summary.iterrows()
                ]
            
            # Add long-short
            quintile_df['Long_Short'] = summary['Long_Short_Return']
            
            # Calculate cumulative returns
            cumul_returns = (1 + quintile_df.fillna(0)).cumprod()
            
            # Plot
            plt.figure(figsize=(14, 8))
            
            # Use a consistent color scheme
            colors = {
                'Q1': 'blue',
                'Q2': 'cyan',
                'Q3': 'gray',
                'Q4': 'orange',
                'Q5': 'red',
                'Long_Short': 'green'
            }
            
            for col in cumul_returns.columns:
                if col in colors:
                    plt.plot(cumul_returns.index, cumul_returns[col], label=col, color=colors[col])
                else:
                    plt.plot(cumul_returns.index, cumul_returns[col], label=col)
                    
            plt.title('Cumulative Returns by Quintile')
            plt.ylabel('Cumulative Return (Start = 1.0)')
            plt.xlabel('Date')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(group_dir, f"{group_name}_cumulative_returns.png"))
            plt.close()
    
    # 4. Drawdown Analysis
    if 'Long_Short' in quintile_df.columns:
        ls_returns = quintile_df['Long_Short'].dropna()
        
        if len(ls_returns) > 0:
            cumulative = (1 + ls_returns).cumprod()
            running_max = cumulative.cummax()
            drawdown = (cumulative / running_max) - 1
            
            plt.figure(figsize=(14, 6))
            plt.plot(drawdown.index, drawdown * 100, color='red')
            plt.title(f'Long-Short Strategy Drawdown (Max: {metrics["max_drawdown"]*100:.2f}%)')
            plt.ylabel('Drawdown (%)')
            plt.xlabel('Date')
            plt.grid(True)
            plt.fill_between(drawdown.index, drawdown * 100, 0, color='red', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(group_dir, f"{group_name}_drawdown.png"))
            plt.close()
    
# 6. Performance summary
    fig, ax = plt.subplots(figsize=(10, 6))
    performance_text = (
        f"Performance Summary for {group_name}\n"
        f"---------------------------------------\n"
        f"Annualized Return: {metrics['annualized_return']*100:.2f}%\n"
        f"Annualized Volatility: {metrics['annualized_volatility']*100:.2f}%\n"
        f"Sharpe Ratio: {metrics['annualized_sharpe']:.2f}\n"
        f"Max Drawdown: {metrics['max_drawdown']*100:.2f}%\n"
        f"Average MAE: {metrics['avg_mae']:.6f}\n"
        f"Average R²: {metrics['avg_r2']:.4f}\n"
        f"Days Evaluated: {metrics['days_evaluated']}"
    )
    
    # For this plot, we're just using the Axes to display text
    ax.text(0.5, 0.5, performance_text, fontsize=12, ha='center', va='center', transform=ax.transAxes)
    ax.axis('off')  # Hide the axes
    
    plt.tight_layout()
    plt.savefig(os.path.join(group_dir, f"{group_name}_performance_summary.png"))
    plt.close()
    
    return metrics

# ----------------------------------------------------------------
# Generate Summary Report
# ----------------------------------------------------------------
def generate_summary_report(results, group_name):
    """
    Generate a comprehensive summary report for the group's performance
    
    Parameters:
    -----------
    results : dict
        Dictionary with evaluation results
    
    group_name : str
        Name of the student group
        
    Returns:
    --------
    report : str
        Markdown formatted performance report
    """
    if results is None or 'summary' not in results:
        return "No valid results to report."
    
    summary = results['summary']
    
    # Create group results directory
    group_dir = os.path.join(RESULTS_DIR, group_name)
    os.makedirs(group_dir, exist_ok=True)
    
    # Calculate metrics
    metrics = calculate_performance_metrics(summary)
    
    # Check for errors
    if 'error' in metrics:
        return f"Error generating report: {metrics['error']}"
    
    # Format the report
    report = f"""
# Performance Summary for {group_name}

## Trading Strategy Performance

### Returns
- **Average Daily Return**: {metrics['avg_daily_return']*100:.4f}%
- **Annualized Return**: {metrics['annualized_return']*100:.2f}%
- **Total Return**: {metrics['total_return']*100:.2f}%
- **Median Daily Return**: {metrics['median_daily_return']*100:.4f}%

### Risk Metrics
- **Daily Volatility**: {metrics['volatility']*100:.4f}%
- **Annualized Volatility**: {metrics['annualized_volatility']*100:.2f}%
- **Sharpe Ratio**: {metrics['annualized_sharpe']:.2f}
- **Maximum Drawdown**: {metrics['max_drawdown']*100:.2f}%

## Prediction Performance

### Error Metrics
- **Mean Squared Error (MSE)**: {metrics['avg_mse']:.6f}
- **Mean Absolute Error (MAE)**: {metrics['avg_mae']:.6f}
- **R² Score**: {metrics['avg_r2']:.4f}

## Quintile Returns

"""
    
    # Add quintile returns if available
    if 'quintile_returns' in metrics and metrics['quintile_returns']:
        for q, ret in sorted(metrics['quintile_returns'].items()):
            report += f"- **{q}**: {ret*100:.4f}%\n"

    report += f"""
## Sample Size
- **Days Evaluated**: {metrics['days_evaluated']}
- **Trading Days with Valid Long-Short Returns**: {metrics['trading_days']}

## Model Characteristics
- The model uses Reddit sentiment data to predict next-day stock returns
- Stocks are ranked daily based on predicted returns and divided into quintiles
- The trading strategy goes long the highest quintile (Q5) and shorts the lowest quintile (Q1)

## Visualizations
See the accompanying PNG files for detailed visualizations of model performance.
"""
    
    # Save report to file
    with open(os.path.join(group_dir, f"evaluation_report_{group_name}.md"), "w") as f:
        f.write(report)
        
    return report

# ----------------------------------------------------------------
# Comparative Analysis Functions
# ----------------------------------------------------------------
def generate_comparative_report(all_results):
    """
    Generate a comparative report across all evaluated groups
    
    Parameters:
    -----------
    all_results : dict
        Dictionary with results for each group
    """
    if not all_results:
        print("No results to compare.")
        return
    
    # Create a DataFrame for comparison
    comparison_data = []
    
    for group_name, metrics in all_results.items():
        if 'error' in metrics:
            continue
            
        comparison_data.append({
            'Group': group_name,
            'Annualized Return (%)': metrics['annualized_return'] * 100,
            'Annualized Volatility (%)': metrics['annualized_volatility'] * 100,
            'Sharpe Ratio': metrics['annualized_sharpe'],
            'Max Drawdown (%)': metrics['max_drawdown'] * 100,
            'MSE': metrics['avg_mse'],
            'MAE': metrics['avg_mae'],
            'R²': metrics['avg_r2'],
            'Days Evaluated': metrics['days_evaluated']
        })
    
    if not comparison_data:
        print("No valid metrics for comparison.")
        return
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Sort by Sharpe Ratio (descending)
    comparison_df = comparison_df.sort_values('Sharpe Ratio', ascending=False)
    
    # Save to CSV
    comparison_df.to_csv(os.path.join(RESULTS_DIR, "group_comparison.csv"), index=False)
    
    # Create comparison charts
    create_comparison_charts(comparison_df)
    
    print(f"Comparative analysis saved to {RESULTS_DIR}/group_comparison.csv")
    
    return comparison_df

def create_comparison_charts(comparison_df):
    """
    Create comparison charts for all groups
    
    Parameters:
    -----------
    comparison_df : DataFrame
        DataFrame with performance metrics for all groups
    """
    if len(comparison_df) == 0:
        return
    
    # 1. Performance Metrics Comparison
    plt.figure(figsize=(12, 8))
    
    # Sharpe Ratio
    plt.subplot(1, 2, 1)
    ax = sns.barplot(x='Group', y='Sharpe Ratio', data=comparison_df, palette='viridis')
    plt.title('Sharpe Ratio by Group')
    plt.xticks(rotation=45, ha='right')
    ax.bar_label(ax.containers[0], fmt='%.2f')
    
    # Annualized Return
    plt.subplot(1, 2, 2)
    ax = sns.barplot(x='Group', y='Annualized Return (%)', data=comparison_df, palette='viridis')
    plt.title('Annualized Return (%)')
    plt.xticks(rotation=45, ha='right')
    ax.bar_label(ax.containers[0], fmt='%.1f%%')
        
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "performance_comparison.png"))
    plt.close()
    
    # 2. Error Metrics Comparison
    plt.figure(figsize=(12, 6))
        
    # MAE
    plt.subplot(1, 2, 1)
    ax = sns.barplot(x='Group', y='MAE', data=comparison_df, palette='viridis')
    plt.title('Mean Absolute Error')
    plt.xticks(rotation=45, ha='right')
    ax.bar_label(ax.containers[0], fmt='%.5f')
    
    # R²
    plt.subplot(1, 2, 2)
    ax = sns.barplot(x='Group', y='R²', data=comparison_df, palette='viridis')
    plt.title('R² Score')
    plt.xticks(rotation=45, ha='right')
    ax.bar_label(ax.containers[0], fmt='%.3f')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "error_metrics_comparison.png"))
    plt.close()

# ----------------------------------------------------------------
# Main Function
# ----------------------------------------------------------------
def main():
    """Main function to run the evaluation"""
    print("Starting evaluation of student submissions...")
    
    # Get list of student submissions
    submissions = [
        f.split('_prediction_function.py')[0] 
        for f in os.listdir(PREDICTION_FUNCTION_DIR) 
        if f.endswith('_prediction_function.py')
    ]
    
    if len(submissions) == 0:
        print("No student submissions found in the directory.")
        return
    
    print(f"Found {len(submissions)} submissions: {', '.join(submissions)}")
    
    # Store metrics for all groups
    all_metrics = {}
    
    # Evaluate each submission
    for group_name in submissions:
        print(f"\n{'='*60}")
        print(f"Evaluating {group_name}")
        print(f"{'='*60}")
        
        # Run evaluation
        results = run_evaluation(group_name)
        
        # Generate visualizations and report
        if results:
            metrics = plot_evaluation_results(results, group_name)
            
            # Generate and print summary report
            report = generate_summary_report(results, group_name)
            print("\nSummary Report generated and saved.")
            
            # Store metrics for comparison
            all_metrics[group_name] = metrics
    
    # Generate comparative analysis if we have multiple groups
    if len(all_metrics) > 1:
        print("\nGenerating comparative analysis...")
        comparison_df = generate_comparative_report(all_metrics)
        
        # Display the top performers
        if comparison_df is not None and len(comparison_df) > 0:
            print("\nTop Performers (by Sharpe Ratio):")
            print(comparison_df[['Group', 'Sharpe Ratio', 'Annualized Return (%)']].head().to_string(index=False))
    
    print("\nEvaluation complete for all submissions!")

if __name__ == "__main__":
    main()