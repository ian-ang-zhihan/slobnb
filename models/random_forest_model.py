import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from preprocessing import run_preprocessing_pipeline
from evaluation import evaluate_model, display_results, compare_to_baseline
from visualization import create_all_visualizations


def train_random_forest(X_train, y_train, n_estimators=100, max_depth=None, 
                       min_samples_split=2, random_state=42):
    """
    Train a Random Forest Regressor.
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training features (one-hot encoded room_type + numerical features)
    y_train : pandas.Series
        Training prices (target variable)
    n_estimators : int, default=100
        Number of trees in the forest
    max_depth : int or None, default=None
        Maximum depth of trees (None = unlimited)
    min_samples_split : int, default=2
        Minimum samples required to split a node
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    model : RandomForestRegressor
        Trained Random Forest model
    """
    print("\nTraining Random Forest model...")
    
    print(f"\nTraining data:")
    print(f"  - Samples: {len(X_train)}")
    print(f"  - Features: {len(X_train.columns)}")
    print(f"  - Feature names: {list(X_train.columns)}")
    print(f"\nTarget (price) range:")
    print(f"  - Min: ${y_train.min():.2f}")
    print(f"  - Max: ${y_train.max():.2f}")
    print(f"  - Mean: ${y_train.mean():.2f}")
    print(f"  - Median: ${y_train.median():.2f}")
    
    print(f"\nRandom Forest hyperparameters:")
    print(f"  - Number of trees (n_estimators): {n_estimators}")
    print(f"  - Max depth (max_depth): {max_depth if max_depth else 'Unlimited'}")
    print(f"  - Min samples to split (min_samples_split): {min_samples_split}")
    print(f"  - Random state: {random_state}")
    
    # Create and train the model
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state,
        n_jobs=-1  # Use all CPU cores for faster training
    )
    model.fit(X_train, y_train)
    
    print(f"\nModel trained!")
    print(f"   Built {len(model.estimators_)} trees in the forest")
    
    return model


def analyze_feature_importance(model, feature_names, top_n=None):
    """
    Analyze and display feature importances from the Random Forest.
    
    Parameters:
    -----------
    model : RandomForestRegressor
        Trained Random Forest model
    feature_names : list
        Names of the features
    top_n : int or None
        Show only top N features (None = show all)
        
    Returns:
    --------
    importance_df : pandas.DataFrame
        DataFrame with features and their importances, sorted by importance
    """
    print("\n" + "="*70)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*70)
    
    print(f"\nFeature importances (sum to 100%):")
    
    # Get feature importances from the model
    importances = model.feature_importances_
    
    # Create dataframe for easier analysis
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })
    
    # Sort by importance (descending)
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    # Convert to percentage
    importance_df['importance_pct'] = importance_df['importance'] * 100
    
    # Limit to top N if specified
    display_df = importance_df.head(top_n) if top_n else importance_df
    
    print(f"\n{'Feature':<30} {'Importance':>12} {'Percentage':>12}")
    print(f"{'-'*70}")
    
    for _, row in display_df.iterrows():
        feature = row['feature']
        importance = row['importance']
        pct = row['importance_pct']
        
        print(f"{feature:<30} {importance:>12.4f} {pct:>11.2f}%")
    
    if top_n and len(importance_df) > top_n:
        remaining = len(importance_df) - top_n
        print(f"\n   ... and {remaining} more features with lower importance")
    
    return importance_df


def main():
    """
    Main function to run the Random Forest pipeline.
    
    """
    
    # Step 1: Get preprocessed data
    X_train, X_test, y_train, y_test = run_preprocessing_pipeline()
    
    # Step 2: Train Random Forest
    rf_model = train_random_forest(
        X_train, y_train,
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        random_state=42
    )
    
    # Step 3: Analyze feature importances
    importance_df = analyze_feature_importance(rf_model, X_train.columns.tolist())
    
    # Step 4: Evaluate model using shared evaluation function
    results = evaluate_model(rf_model, X_train, y_train, X_test, y_test,
                           model_name="Random Forest")
    
    # Step 5: Display results using shared display function
    display_results(results, y_test=y_test, show_examples=True)
    
    # Step 6: Compare to baseline
    # (using hardcoded values from baseline run)
    baseline_results = {
        'model_name': 'Baseline (Mean Predictor)',
        'train': {'rmse': 318.03, 'mae': 221.50, 'r2': 0.0000, 'mape': 97.31},
        'test': {'rmse': 334.83, 'mae': 236.63, 'r2': -0.0012, 'mape': 99.71},
        'predictions': {}
    }
    compare_to_baseline(results, baseline_results)
    
    # Step 7: Compare to Linear Regression (manual comparison for now)
    print("\n" + "="*70)
    print("COMPARISON: RANDOM FOREST vs LINEAR REGRESSION")
    print("="*70)
    print(f"\nRandom Forest Results (this run):")
    print(f"   Test RMSE: ${results['test']['rmse']:.2f}")
    print(f"   Test R²:   {results['test']['r2']:.4f}")
    print(f"   Test MAPE: {results['test']['mape']:.2f}%")
    
    lr_rmse = 198.06
    lr_r2 = 0.6497
    lr_mape = 44.94
    print(f"\nLinear Regression Results (from previous run):")
    print(f"   Test RMSE: ${lr_rmse}")
    print(f"   Test R²:   {lr_r2}")
    print(f"   Test MAPE: {lr_mape}%")
    
    # Calculate improvement
    rmse_improvement = ((lr_rmse - results['test']['rmse']) / lr_rmse) * 100
    r2_improvement = results['test']['r2'] - lr_r2
    mape_improvement = ((lr_mape - results['test']['mape']) / lr_mape) * 100
    
    print(f"\nIMPROVEMENTS (Random Forest vs Linear Regression):")
    print(f"   RMSE: {rmse_improvement:+.1f}% ({'better' if rmse_improvement > 0 else 'worse'})")
    print(f"   R²:   {r2_improvement:+.4f} ({abs(r2_improvement)*100:.1f} percentage points)")
    print(f"   MAPE: {mape_improvement:+.1f}% ({'better' if mape_improvement > 0 else 'worse'})")
    
    print(f"\nVERDICT:")
    if results['test']['r2'] > lr_r2:
        print(f"   Random Forest BEATS Linear Regression!")
    elif results['test']['r2'] > lr_r2 - 0.02:
        print(f"   Random Forest and Linear Regression perform similarly")
    else:
        print(f"   Linear Regression slightly outperforms Random Forest")
    
    print("\nRandom forest pipeline complete!")
    
    # Step 8: Create visualizations
    figures = create_all_visualizations(
        model=rf_model,
        importance_df=importance_df,
        y_true=y_test,
        y_pred=results['predictions']['y_test_pred'],
        model_name="Random Forest",
        save_dir="../figures",
        show=False  # Don't block execution (set to True to display)
    )
    
    print("Check the 'figures/' directory for saved visualizations")
    
    return rf_model, results, importance_df, figures


if __name__ == "__main__":
    # Run the Random Forest pipeline
    model, results, importances, figures = main()
