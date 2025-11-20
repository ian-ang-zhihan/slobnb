import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def calculate_mape(y_true, y_pred):
    """
    Calculate Mean Absolute Percentage Error (MAPE).
    
    Formula:
    --------
    MAPE = (1/n) × Σ |actual - predicted| / |actual| × 100%
    
    Parameters:
    -----------
    y_true : array-like
        Actual prices
    y_pred : array-like
        Predicted prices
        
    Returns:
    --------
    mape : float
        Mean Absolute Percentage Error (as percentage, e.g., 35.5%)
    
    Note: We skip any prices that are $0 to avoid division by zero
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    # Avoid division by zero - only calculate for non-zero prices
    non_zero = y_true != 0
    
    # Calculate percentage error for each prediction
    # |actual - predicted| / |actual|
    percentage_errors = np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])
    
    # Return mean as percentage
    return np.mean(percentage_errors) * 100


def calculate_metrics(y_true, y_pred):
    """
    Calculate all regression metrics for a set of predictions.
    
    This is the core function that computes all 4 metrics we care about:
    1. RMSE - Root Mean Squared Error (in dollars)
    2. MAE - Mean Absolute Error (in dollars)
    3. R² - R-Squared (proportion of variance explained)
    4. MAPE - Mean Absolute Percentage Error (as percentage)
    
    Parameters:
    -----------
    y_true : array-like
        Actual prices
    y_pred : array-like
        Predicted prices
        
    Returns:
    --------
    metrics : dict
        Dictionary with keys: 'rmse', 'mae', 'r2', 'mape'
    
    """
    return {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'mape': calculate_mape(y_true, y_pred)
    }


def evaluate_model(model, X_train, y_train, X_test, y_test, model_name="Model"):
    """
    Evaluate a trained model on train and test sets.
    
    Parameters:
    -----------
    model : trained model
        Any sklearn model with .predict() method
    X_train, y_train : Training data
        Features and target for training set
    X_test, y_test : Test data
        Features and target for test set
    model_name : str
        Name of the model (for identification in results)
        
    Returns:
    --------
    results : dict
        {
            'model_name': str,
            'train': {'rmse': float, 'mae': float, 'r2': float, 'mape': float},
            'test': {'rmse': float, 'mae': float, 'r2': float, 'mape': float},
            'predictions': {
                'y_train_pred': array,
                'y_test_pred': array
            }
        }
        
    """
    # Make predictions on both train and test sets
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics for training set
    train_metrics = calculate_metrics(y_train, y_train_pred)
    
    # Calculate metrics for test set
    test_metrics = calculate_metrics(y_test, y_test_pred)
    
    # Return everything in a structured dictionary
    return {
        'model_name': model_name,
        'train': train_metrics,
        'test': test_metrics,
        'predictions': {
            'y_train_pred': y_train_pred,
            'y_test_pred': y_test_pred
        }
    }


def display_results(results, y_test=None, show_examples=True):
    """
    Display evaluation results in a formatted, readable way.
    
    What it prints:
    ---------------
    1. Training set performance (all 4 metrics)
    2. Test set performance (all 4 metrics)
    3. Overfitting check (train vs test R²)
    4. Example predictions (optional, if y_test provided)
    
    Parameters:
    -----------
    results : dict
        Results dictionary from evaluate_model()
    y_test : pandas.Series or array-like, optional
        Actual test values (needed for example predictions)
    show_examples : bool
        Whether to show example predictions (default: True)
    
    """
    model_name = results['model_name']
    train = results['train']
    test = results['test']
    
    # Print training set performance
    print(f"\n{'='*70}")
    print(f"TRAINING SET PERFORMANCE - {model_name.upper()}")
    print(f"{'='*70}")
    print(f"RMSE: ${train['rmse']:>10.2f}  →  On average, off by ~${train['rmse']:.2f}")
    print(f"MAE:  ${train['mae']:>10.2f}  →  Average absolute error")
    print(f"R²:   {train['r2']:>11.4f}  →  Explains {train['r2']*100:.2f}% of variance")
    print(f"MAPE: {train['mape']:>10.2f}%  →  Average % error")
    
    # Print test set performance
    print(f"\n{'='*70}")
    print(f"TEST SET PERFORMANCE - {model_name.upper()}")
    print(f"{'='*70}")
    print(f"RMSE: ${test['rmse']:>10.2f}  →  On average, off by ~${test['rmse']:.2f}")
    print(f"MAE:  ${test['mae']:>10.2f}  →  Average absolute error")
    print(f"R²:   {test['r2']:>11.4f}  →  Explains {test['r2']*100:.2f}% of variance")
    print(f"MAPE: {test['mape']:>10.2f}%  →  Average % error")
    
    # Overfitting check
    print(f"\n{'='*70}")
    print(f"OVERFITTING CHECK")
    print(f"{'='*70}")
    r2_diff = train['r2'] - test['r2']
    print(f"Train R²: {train['r2']:>8.4f}")
    print(f"Test R²:  {test['r2']:>8.4f}")
    print(f"Diff:     {r2_diff:>8.4f}")
    
    if r2_diff < 0.05:
        print(f"  → Minimal overfitting (diff < 0.05)")
    elif r2_diff < 0.15:
        print(f"  → Acceptable generalization (diff < 0.15)")
    else:
        print(f"  → Warning! Significant overfitting (diff >= 0.15)")
        print(f"     Model may be memorizing training data")
    
    # Example predictions (if requested and y_test provided)
    if show_examples and y_test is not None:
        print(f"\n{'='*70}")
        print(f"EXAMPLE PREDICTIONS (First 5 test samples)")
        print(f"{'='*70}")
        print(f"{'Actual':>12} | {'Predicted':>12} | {'Error':>12} | {'% Error':>10}")
        print(f"{'-'*70}")
        
        y_test_pred = results['predictions']['y_test_pred']
        for i in range(min(5, len(y_test))):
            actual = y_test.iloc[i] if hasattr(y_test, 'iloc') else y_test[i]
            predicted = y_test_pred[i]
            error = actual - predicted
            pct_error = (abs(error) / actual) * 100
            print(f"${actual:>11.2f} | ${predicted:>11.2f} | ${error:>11.2f} | {pct_error:>9.1f}%")


def compare_models(results_list, sort_by='test_r2', ascending=False):
    """
    Compare multiple models and rank them by a chosen metric.
    
    What it does:
    -------------
    1. Takes results from multiple models
    2. Extracts all metrics into a table
    3. Sorts by your chosen metric
    4. Prints a nice comparison table
    
    Parameters:
    -----------
    results_list : list of dict
        List of results dictionaries from evaluate_model()
        Example: [baseline_results, linear_results, rf_results]
    sort_by : str
        Metric to sort by (default: 'test_r2')
        Options: 'test_r2', 'test_rmse', 'test_mae', 'test_mape',
                 'train_r2', 'train_rmse', 'train_mae', 'train_mape'
    ascending : bool
        Sort order (default: False for R², True for error metrics)
        
    Returns:
    --------
    comparison_df : pandas.DataFrame
        DataFrame with all models and their metrics, sorted
    
    """
    comparison_data = []
    
    for results in results_list:
        comparison_data.append({
            'Model': results['model_name'],
            'Train RMSE': results['train']['rmse'],
            'Test RMSE': results['test']['rmse'],
            'Train MAE': results['train']['mae'],
            'Test MAE': results['test']['mae'],
            'Train R²': results['train']['r2'],
            'Test R²': results['test']['r2'],
            'Train MAPE': results['train']['mape'],
            'Test MAPE': results['test']['mape']
        })
    
    df = pd.DataFrame(comparison_data)
    
    # Map sort_by to column name
    column_map = {
        'test_r2': 'Test R²',
        'test_rmse': 'Test RMSE',
        'test_mae': 'Test MAE',
        'test_mape': 'Test MAPE',
        'train_r2': 'Train R²',
        'train_rmse': 'Train RMSE',
        'train_mae': 'Train MAE',
        'train_mape': 'Train MAPE'
    }
    
    sort_column = column_map.get(sort_by, 'Test R²')
    
    # Auto-determine sort order if not specified
    if sort_by in ['test_r2', 'train_r2']:
        ascending = False  # Higher R² is better
    else:
        ascending = True   # Lower error is better
    
    df = df.sort_values(sort_column, ascending=ascending)
    
    # Print comparison table
    print(f"\n{'='*80}")
    print(f"MODEL COMPARISON (sorted by {sort_column}, {'ascending' if ascending else 'descending'})")
    print(f"{'='*80}")
    print(df.to_string(index=False))
    print(f"{'='*80}")
    
    return df


def compare_to_baseline(model_results, baseline_results):
    """
    Compare a model's results to the baseline (dummy) model.
    
    What it shows:
    --------------
    1. Model results vs baseline results
    2. Absolute improvements (e.g., RMSE decreased by $50)
    3. Percentage improvements (e.g., RMSE improved by 20%)
    4. Success/warning messages based on performance
    
    Parameters:
    -----------
    model_results : dict
        Results from evaluate_model() for your model
    baseline_results : dict
        Results from evaluate_model() for baseline model
        
    """
    print(f"\n{'='*70}")
    print(f"COMPARISON TO BASELINE")
    print(f"{'='*70}")
    
    model_name = model_results['model_name']
    model_test = model_results['test']
    baseline_test = baseline_results['test']
    
    # Print both results
    print(f"\n{model_name} Results:")
    print(f"   RMSE: ${model_test['rmse']:.2f}")
    print(f"   MAE:  ${model_test['mae']:.2f}")
    print(f"   R²:   {model_test['r2']:.4f}")
    print(f"   MAPE: {model_test['mape']:.2f}%")
    
    print(f"\nBaseline Results:")
    print(f"   RMSE: ${baseline_test['rmse']:.2f}")
    print(f"   MAE:  ${baseline_test['mae']:.2f}")
    print(f"   R²:   {baseline_test['r2']:.4f}")
    print(f"   MAPE: {baseline_test['mape']:.2f}%")
    
    # Calculate improvements
    rmse_improvement = ((baseline_test['rmse'] - model_test['rmse']) / baseline_test['rmse']) * 100
    mae_improvement = ((baseline_test['mae'] - model_test['mae']) / baseline_test['mae']) * 100
    r2_improvement = model_test['r2'] - baseline_test['r2']
    mape_improvement = ((baseline_test['mape'] - model_test['mape']) / baseline_test['mape']) * 100
    
    print(f"\nIMPROVEMENTS:")
    print(f"   RMSE: {rmse_improvement:+.1f}% (${baseline_test['rmse'] - model_test['rmse']:+.2f})")
    print(f"   MAE:  {mae_improvement:+.1f}% (${baseline_test['mae'] - model_test['mae']:+.2f})")
    print(f"   R²:   {r2_improvement:+.4f} ({r2_improvement*100:+.2f} percentage points)")
    print(f"   MAPE: {mape_improvement:+.1f}% ({baseline_test['mape'] - model_test['mape']:+.2f} percentage points)")
    
    # Verdict
    print(f"\nVERDICT:")
    if rmse_improvement > 20 and model_test['r2'] > 0.3:
        print(f"   SUCCESS! {model_name} significantly beats baseline!")
    elif rmse_improvement > 10 and model_test['r2'] > 0.1:
        print(f"   GOOD! {model_name} improves over baseline")
    elif rmse_improvement > 0:
        print(f"   MODEST improvement over baseline")
    else:
        print(f"   WARNING! {model_name} performs WORSE than baseline!")
        print(f"      → Something is wrong - check for bugs or data issues")
        print(f"      → Baseline should be the absolute minimum performance")
    
    print(f"{'='*70}")
