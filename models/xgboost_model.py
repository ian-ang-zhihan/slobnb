import numpy as np
import pandas as pd
import xgboost as xgb
from preprocessing import run_preprocessing_pipeline
from evaluation import evaluate_model, display_results, compare_to_baseline
from visualization import create_all_visualizations


def train_xgboost(X_train, y_train, n_estimators=200, max_depth=6, learning_rate=0.1,
                  subsample=0.8, colsample_bytree=0.8, min_child_weight=1, 
                  gamma=0, reg_alpha=0, reg_lambda=1, random_state=42):
    """
    Train an XGBoost Regressor.
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training features
    y_train : pandas.Series
        Training prices (target variable)
    n_estimators : int, default=200
        Number of boosting rounds (trees to build)
    max_depth : int, default=6
        Maximum depth of each tree
    learning_rate : float, default=0.1
        Step size shrinkage to prevent overfitting (0.01-0.3)
    subsample : float, default=0.8
        Fraction of samples used per tree (prevents overfitting)
    colsample_bytree : float, default=0.8
        Fraction of features used per tree (prevents overfitting)
    min_child_weight : int, default=1
        Minimum sum of instance weight needed in a child (higher = more conservative)
    gamma : float, default=0
        Minimum loss reduction required to make a split (higher = more conservative)
    reg_alpha : float, default=0
        L1 regularization on weights (higher = more regularization)
    reg_lambda : float, default=1
        L2 regularization on weights (higher = more regularization)
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    model : xgb.XGBRegressor
        Trained XGBoost model
    """
    print("\nTraining XGBoost model...")
    
    print(f"\nTraining data:")
    print(f"  - Samples: {len(X_train)}")
    print(f"  - Features: {len(X_train.columns)}")
    print(f"  - Feature names: {list(X_train.columns)}")
    print(f"\nTarget (price) range:")
    print(f"  - Min: ${y_train.min():.2f}")
    print(f"  - Max: ${y_train.max():.2f}")
    print(f"  - Mean: ${y_train.mean():.2f}")
    print(f"  - Median: ${y_train.median():.2f}")
    
    print(f"\nXGBoost hyperparameters:")
    print(f"  - Number of boosting rounds (n_estimators): {n_estimators}")
    print(f"  - Max depth per tree (max_depth): {max_depth}")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Subsample ratio: {subsample}")
    print(f"  - Feature sampling ratio (colsample_bytree): {colsample_bytree}")
    print(f"  - Min child weight (min_child_weight): {min_child_weight}")
    print(f"  - Gamma (min split loss): {gamma}")
    print(f"  - L1 regularization (reg_alpha): {reg_alpha}")
    print(f"  - L2 regularization (reg_lambda): {reg_lambda}")
    print(f"  - Random state: {random_state}")
    
    # Create and train the model
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_weight=min_child_weight,
        gamma=gamma,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        random_state=random_state,
        n_jobs=-1,  # Use all CPU cores
        tree_method='hist'  # Faster histogram-based algorithm
    )
    
    model.fit(X_train, y_train)
    
    print(f"\nModel trained!")
    print(f"   Built {n_estimators} trees sequentially (gradient boosting)")
    
    return model


def analyze_feature_importance(model, feature_names, top_n=None):
    """
    Analyze and display feature importances from XGBoost.
    
    XGBoost uses 'gain' by default - the average improvement in loss
    when a feature is used for splitting.
    
    Parameters:
    -----------
    model : xgb.XGBRegressor
        Trained XGBoost model
    feature_names : list
        List of feature names
    top_n : int or None, default=None
        Number of top features to display (None = all)
        
    Returns:
    --------
    importance_df : pandas.DataFrame
        DataFrame with features and their importance scores
    """
    print("\n" + "="*70)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*70)
    
    # Get feature importances (gain-based by default)
    importances = model.feature_importances_
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    # Normalize to percentages
    importance_df['percentage'] = importance_df['importance'] / importance_df['importance'].sum()
    
    print("\nFeature importances (sum to 100%):\n")
    print(f"{'Feature':<30} {'Importance':>12} {'Percentage':>12}")
    print("-"*70)
    
    # Display top N or all features
    display_df = importance_df.head(top_n) if top_n else importance_df
    
    for _, row in display_df.iterrows():
        print(f"{row['feature']:<30} {row['importance']:>12.4f} {row['percentage']:>11.2%}")
    
    return importance_df


def main():
    """
    Main function to run the XGBoost pipeline.
    """

    # Step 1: Load and preprocess data
    X_train, X_test, y_train, y_test = run_preprocessing_pipeline()
    
    # Step 2: Train XGBoost with strong regularization to prevent overfitting
    # XGBoost overfitted badly with default params (train R²=0.987, test R²=0.696, gap=0.291)
    # Applying aggressive regularization:
    # - Shallower trees: max_depth 6 → 4 (less complex per tree)
    # - Slower learning: learning_rate 0.1 → 0.05 (more conservative updates)
    # - Higher min_child_weight: 1 → 10 (require more samples per leaf)
    # - L2 regularization: reg_lambda 1 → 5 (penalize large weights)
    xgb_model = train_xgboost(
        X_train, y_train,
        n_estimators=200,       # Keep 200 trees
        max_depth=4,            # REDUCED from 6 to 4 (shallower trees)
        learning_rate=0.05,     # REDUCED from 0.1 to 0.05 (slower learning)
        subsample=0.8,          # Keep 80% sample rate
        colsample_bytree=0.8,   # Keep 80% feature rate
        min_child_weight=10,    # INCREASED from 1 to 10 (more conservative splits)
        gamma=0.1,              # ADDED: minimum loss reduction to split
        reg_alpha=0,            # No L1 regularization
        reg_lambda=5,           # INCREASED from 1 to 5 (stronger L2 penalty)
        random_state=42
    )
    
    # Step 3: Analyze feature importances
    importance_df = analyze_feature_importance(xgb_model, X_train.columns.tolist())
    
    # Step 4: Evaluate model using shared evaluation function
    results = evaluate_model(xgb_model, X_train, y_train, X_test, y_test,
                           model_name="XGBoost")
    
    # Step 5: Display results using shared display function
    display_results(results, y_test=y_test, show_examples=True)
    
    # Step 6: Compare to baseline
    baseline_results = {
        'model_name': 'Baseline (Mean Predictor)',
        'train': {'rmse': 318.03, 'mae': 221.50, 'r2': 0.0000, 'mape': 97.31},
        'test': {'rmse': 334.83, 'mae': 236.63, 'r2': -0.0012, 'mape': 99.71},
        'predictions': {}
    }
    compare_to_baseline(results, baseline_results)
    
    # Step 7: Compare to other models
    print("\n" + "="*70)
    print("COMPARISON: XGBOOST vs OTHER MODELS")
    print("="*70)
    
    print(f"\nXGBoost Results (this run):")
    print(f"   Test RMSE: ${results['test']['rmse']:.2f}")
    print(f"   Test MAE:  ${results['test']['mae']:.2f}")
    print(f"   Test R²:   {results['test']['r2']:.4f}")
    print(f"   Test MAPE: {results['test']['mape']:.2f}%")
    
    # Linear Regression baseline (with 26 features)
    lr_rmse = 181.61
    lr_mae = 128.98
    lr_r2 = 0.7055
    lr_mape = 43.10
    
    print(f"\nLinear Regression Results (26 features):")
    print(f"   Test RMSE: ${lr_rmse:.2f}")
    print(f"   Test MAE:  ${lr_mae:.2f}")
    print(f"   Test R²:   {lr_r2:.4f}")
    print(f"   Test MAPE: {lr_mape:.2f}%")
    
    # Random Forest baseline (200 trees, 26 features)
    rf_rmse = 175.04
    rf_mae = 116.15
    rf_r2 = 0.7283
    rf_mape = 37.92
    
    print(f"\nRandom Forest Results (200 trees, 26 features):")
    print(f"   Test RMSE: ${rf_rmse:.2f}")
    print(f"   Test MAE:  ${rf_mae:.2f}")
    print(f"   Test R²:   {rf_r2:.4f}")
    print(f"   Test MAPE: {rf_mape:.2f}%")
    
    # Compare to Random Forest (current best)
    rmse_vs_rf = ((rf_rmse - results['test']['rmse']) / rf_rmse) * 100
    mae_vs_rf = ((rf_mae - results['test']['mae']) / rf_mae) * 100
    r2_vs_rf = results['test']['r2'] - rf_r2
    mape_vs_rf = ((rf_mape - results['test']['mape']) / rf_mape) * 100
    
    print(f"\nXGBOOST vs RANDOM FOREST:")
    print(f"   RMSE: {rmse_vs_rf:+.1f}% ({'better' if rmse_vs_rf > 0 else 'worse'})")
    print(f"   MAE:  {mae_vs_rf:+.1f}% ({'better' if mae_vs_rf > 0 else 'worse'})")
    print(f"   R²:   {r2_vs_rf:+.4f} ({abs(r2_vs_rf)*100:.1f} percentage points {'better' if r2_vs_rf > 0 else 'worse'})")
    print(f"   MAPE: {mape_vs_rf:+.1f}% ({'better' if mape_vs_rf > 0 else 'worse'})")
    
    # Compare to Linear Regression
    rmse_vs_lr = ((lr_rmse - results['test']['rmse']) / lr_rmse) * 100
    mae_vs_lr = ((lr_mae - results['test']['mae']) / lr_mae) * 100
    r2_vs_lr = results['test']['r2'] - lr_r2
    
    print(f"\nXGBOOST vs LINEAR REGRESSION:")
    print(f"   RMSE: {rmse_vs_lr:+.1f}% ({'better' if rmse_vs_lr > 0 else 'worse'})")
    print(f"   MAE:  {mae_vs_lr:+.1f}% ({'better' if mae_vs_lr > 0 else 'worse'})")
    print(f"   R²:   {r2_vs_lr:+.4f} ({abs(r2_vs_lr)*100:.1f} percentage points {'better' if r2_vs_lr > 0 else 'worse'})")
    
    print(f"\nVERDICT:")
    if results['test']['r2'] > rf_r2 + 0.01:
        print(f"   XGBoost BEATS Random Forest!")
    elif results['test']['r2'] > rf_r2 - 0.01:
        print(f"   XGBoost and Random Forest perform similarly")
        print(f"      → Both are strong models for this dataset")
    else:
        print(f"   Random Forest still outperforms XGBoost")
        print(f"      → Random Forest remains the best model")
    
    # Save model for transfer learning
    model_path = 'xgboost_model.json'
    xgb_model.save_model(model_path)
    print(f"\nModel saved to {model_path}")
    
    print("\nXGBoost pipeline complete!")
    
    # Step 8: Create visualizations
    figures = create_all_visualizations(
        model=xgb_model,
        importance_df=importance_df,
        y_true=y_test,
        y_pred=results['predictions']['y_test_pred'],
        model_name="XGBoost",
        save_dir="../figures",
        show=False  # Don't block execution
    )
    
    print("Check the 'figures/' directory for saved visualizations")
    
    return xgb_model, results, importance_df, figures


if __name__ == "__main__":
    # Run the XGBoost pipeline
    model, results, importances, figures = main()
