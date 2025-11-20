import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from preprocessing import run_preprocessing_pipeline
from evaluation import evaluate_model, display_results


def train_baseline_model(X_train, y_train):
    """
    Train a dummy baseline model that always predicts the mean.
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training features (we won't actually use these)
    y_train : pandas.Series
        Training prices (we'll just compute the mean)
        
    Returns:
    --------
    model : DummyRegressor
        Trained baseline model (just stores the mean)
        
    """
    print("\nTraining baseline model...")
    
    # Create the dummy model
    # strategy='mean' means it will always predict the average
    model = DummyRegressor(strategy='mean')
    
    print(f"\nStrategy: Always predict the mean price")
    print(f"Training data: {len(X_train)} samples")
    print(f"Price range in training: ${y_train.min():.2f} to ${y_train.max():.2f}")
    print(f"Mean price (what model will always predict): ${y_train.mean():.2f}")
    print(f"Median price (for comparison): ${y_train.median():.2f}")
    
    # "Train" the model (just calculates and stores the mean)
    model.fit(X_train, y_train)
    
    print(f"\nModel trained!")
    print(f"  Model will always predict: ${model.constant_.item():.2f}")
    
    return model


def main():
    """
    Main function to run the baseline model pipeline.
    
    """
    
    # Step 1: Get preprocessed data
    X_train, X_test, y_train, y_test = run_preprocessing_pipeline()
    
    # Step 2: Train baseline model
    baseline_model = train_baseline_model(X_train, y_train)
    
    # Step 3: Evaluate model using shared evaluation function
    results = evaluate_model(baseline_model, X_train, y_train, X_test, y_test, 
                           model_name="Baseline (Mean Predictor)")
    
    # Step 4: Display results using shared display function
    display_results(results, y_test=y_test, show_examples=True)
    
    print("\nBaseline model pipeline complete!")
    
    return baseline_model, results


if __name__ == "__main__":
    # Run the baseline model pipeline
    model, results = main()
