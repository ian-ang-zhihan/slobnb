import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from preprocessing import run_preprocessing_pipeline
from evaluation import evaluate_model, display_results, compare_to_baseline


def train_linear_model(X_train, y_train):
    """
    Train a Linear Regression model.
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training features (one-hot encoded room_type + numerical features)
        Columns: [rt_Entire home/apt, rt_Private room, rt_Shared room, 
                  rt_Hotel room, accommodates, bedrooms]
    y_train : pandas.Series
        Training prices
        
    Returns:
    --------
    model : LinearRegression
        Trained model with learned coefficients
        
    """
    print("\nTraining linear regression model...")
    
    print(f"\nTraining data:")
    print(f"  - Samples: {len(X_train)}")
    print(f"  - Features: {len(X_train.columns)}")
    print(f"  - Feature names: {list(X_train.columns)}")
    print(f"\nTarget (price) range:")
    print(f"  - Min: ${y_train.min():.2f}")
    print(f"  - Max: ${y_train.max():.2f}")
    print(f"  - Mean: ${y_train.mean():.2f}")
    print(f"  - Median: ${y_train.median():.2f}")
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    print(f"\nModel trained!")
    print(f"   The model has learned {len(model.coef_)} coefficients")
    
    return model


def analyze_coefficients(model, feature_names):
    """
    Analyze and explain what the model learned.
    
    Each coefficient tells us: "If this feature increases by 1 unit,
    how much does the predicted price change?"
    
    Parameters:
    -----------
    model : LinearRegression
        Trained model with learned coefficients
    feature_names : list
        Names of the features (column names from X_train)
        
    """
    print("\n" + "="*70)
    print("MODEL COEFFICIENTS ANALYSIS")
    print("="*70)
    
    print(f"\nWhat the model learned:")
    print(f"\nIntercept (β₀): ${model.intercept_:.2f}")
    print(f"   → This is the base price when all features = 0")
    
    print(f"\nFeature Coefficients:")
    print(f"{'Feature':<30} {'Coefficient':>15} {'Interpretation'}")
    print(f"{'-'*70}")
    
    # Create a dataframe for easier analysis
    coef_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': model.coef_
    })
    
    # Sort by absolute value to see most important features
    coef_df['abs_coef'] = np.abs(coef_df['coefficient'])
    coef_df = coef_df.sort_values('abs_coef', ascending=False)
    
    # Print each feature's coefficient with interpretation
    for _, row in coef_df.iterrows():
        feature = row['feature']
        coef = row['coefficient']
        
        # Interpret what the coefficient means
        if feature.startswith('rt_'):
            # Room type features (categorical)
            room_type = feature.replace('rt_', '')
            if coef > 0:
                interpretation = f"→ {room_type}s add ${coef:.2f} to price"
            else:
                interpretation = f"→ {room_type}s reduce price by ${abs(coef):.2f}"
        elif feature == 'accommodates':
            if coef > 0:
                interpretation = f"→ Each extra person adds ${coef:.2f}"
            else:
                interpretation = f"→ Each extra person reduces ${abs(coef):.2f}"
        elif feature == 'bedrooms':
            if coef > 0:
                interpretation = f"→ Each bedroom adds ${coef:.2f}"
            else:
                interpretation = f"→ Each bedroom reduces ${abs(coef):.2f}"
        else:
            interpretation = ""
        
        print(f"{feature:<30} ${coef:>14.2f}  {interpretation}")
    
    return coef_df


def main():
    """
    Main function to run the Linear Regression pipeline.
    
    """
    
    # Step 1: Get preprocessed data
    X_train, X_test, y_train, y_test = run_preprocessing_pipeline()
    
    # Step 2: Train Linear Regression
    linear_model = train_linear_model(X_train, y_train)
    
    # Step 3: Analyze coefficients
    coef_df = analyze_coefficients(linear_model, X_train.columns.tolist())
    
    # Step 4: Evaluate model using shared evaluation function
    results = evaluate_model(linear_model, X_train, y_train, X_test, y_test,
                           model_name="Linear Regression")
    
    # Step 5: Display results
    display_results(results, y_test=y_test, show_examples=True)
    
    # Step 6: Create baseline results dict for comparison
    # (using hardcoded values from baseline run)
    baseline_results = {
        'model_name': 'Baseline (Mean Predictor)',
        'train': {'rmse': 318.03, 'mae': 221.50, 'r2': 0.0000, 'mape': 97.31},
        'test': {'rmse': 334.83, 'mae': 236.63, 'r2': -0.0012, 'mape': 99.71},
        'predictions': {}
    }
    compare_to_baseline(results, baseline_results)

    print("\nLinear regression pipeline complete!")
    
    return linear_model, results, coef_df


if __name__ == "__main__":
    # Run the Linear Regression pipeline
    model, results, coefficients = main()
