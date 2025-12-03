"""
Transfer Learning Test: Santa Cruz Model → San Luis Obispo Predictions

This script tests whether our Santa Cruz-trained model can generalize to 
San Luis Obispo Airbnb listings (similar college beach town market).

Manually collected 3 SLO listings from airbnb.com and filled in the features below.
"""

import pandas as pd
import numpy as np
import sys
import os
import re

# Add models directory to path
sys.path.append(os.path.dirname(__file__))


def parse_bathrooms(bathrooms_text):
    """
    Parse bathroom text into count and type.
    
    Examples:
        '1 bath' -> {'bathrooms_count': 1.0, 'bathroom_type': 'standard'}
        '2.5 shared baths' -> {'bathrooms_count': 2.5, 'bathroom_type': 'shared'}
        '1 private bath' -> {'bathrooms_count': 1.0, 'bathroom_type': 'private'}
    """
    # Extract count
    match = re.search(r'(\d+\.?\d*)', str(bathrooms_text))
    count = float(match.group(1)) if match else 1.0
    
    # Extract type
    text_lower = str(bathrooms_text).lower()
    if 'private' in text_lower:
        bath_type = 'private'
    elif 'shared' in text_lower:
        bath_type = 'shared'
    else:
        bath_type = 'standard'
    
    return {'bathrooms_count': count, 'bathroom_type': bath_type}


slo_listings = [
    {
        # Listing 1
        'listing_url': 'https://www.airbnb.com/rooms/23305004?viralityEntryPoint=1&s=76',
        'actual_price': 383,
        
        # Basic Info
        'room_type': 'Private room',  # Options: 'Entire home/apt', 'Private room', 'Shared room'
        'accommodates': 2,  # Max number of guests
        'bedrooms': 1,  # Number of bedrooms
        'beds': 1,  # Number of beds
        'bathrooms_text': '1 bath',  # e.g., '2 baths', '1.5 shared baths'
        
        # Quality Signals
        'review_scores_rating': 4.96,  # Overall rating (e.g., 4.89)
        'minimum_nights': 2,  # Minimum stay requirement
        'instant_bookable': False,  # True if instant book available
        'host_is_superhost': True,  # True if host has superhost badge
        
        # Amenity Category Scores (count matching keywords in amenities list)
        'parking_score': 1,  # Keywords: parking, garage
        'kitchen_score': 7,  # Keywords: kitchen, oven, stove, refrigerator, fridge, dishwasher, microwave, coffee, toaster, dishes, blender, freezer
        'entertainment_score': 0,  # Keywords: tv, television, netflix, hulu, cable, apple tv, sound system, hdtv, amazon prime, chromecast, disney+, hbo
        'climate_control_score': 1,  # Keywords: heating, air conditioning, ac, fireplace, fan
        'laundry_score': 2,  # Keywords: washer, dryer, laundry, essentials, iron
        'safety_score': 4,  # Keywords: smoke, carbon monoxide, fire extinguisher, first aid, lock, security cameras
        'convenience_score': 2,  # Keywords: self check-in, keypad, lockbox, smart lock, private entrance, wifi
        'toiletries_score': 2,  # Keywords: hair dryer, shampoo, conditioner, body soap, shower gel, bed linens, extra pillows and blankets, clothing storage, hangers
        
        # Binary Flags (0 or 1)
        'has_beach_access': 0,  # Keywords: beach access, beachfront, waterfront
        'pets_allowed': 0,  # Keywords: pets allowed
    },
    
    {
        # Listing 2
        'listing_url': 'https://www.airbnb.com/rooms/45164131?viralityEntryPoint=1&s=76',
        'actual_price': 277,
        
        'room_type': 'Entire home/apt',
        'accommodates': 4,
        'bedrooms': 1,
        'beds': 2,
        'bathrooms_text': '1 bath',
        
        'review_scores_rating': 4.87,
        'minimum_nights': 2,
        'instant_bookable': False,
        'host_is_superhost': True,
        
        'parking_score': 2,  # Keywords: parking, garage
        'kitchen_score': 7,  # Keywords: kitchen, oven, stove, refrigerator, fridge, dishwasher, microwave, coffee, toaster, dishes, blender, freezer
        'entertainment_score': 6,  # Keywords: tv, television, netflix, hulu, cable, apple tv, sound system, hdtv, amazon prime, chromecast, disney+, hbo
        'climate_control_score': 2,  # Keywords: heating, air conditioning, ac, fireplace, fan
        'laundry_score': 1,  # Keywords: washer, dryer, laundry, essentials, iron
        'safety_score': 4,  # Keywords: smoke, carbon monoxide, fire extinguisher, first aid, lock, security cameras
        'convenience_score': 3,  # Keywords: self check-in, keypad, lockbox, smart lock, private entrance
        'toiletries_score': 5,  # Keywords: hair dryer, shampoo, conditioner, body soap, shower gel, bed linens, extra pillows and blankets, clothing storage, hangers
        
        'has_beach_access': 0,  # Keywords: beach access, beachfront, waterfront
        'pets_allowed': 0,  # Keywords: pets allowed
    },
    
    {
        # Listing 3
        'listing_url': 'https://www.airbnb.com/rooms/47216363?viralityEntryPoint=1&s=76',
        'actual_price': 344,
        
        'room_type': 'Entire home/apt',
        'accommodates': 3,
        'bedrooms': 1,
        'beds': 2,
        'bathrooms_text': '1 bath',
        
        'review_scores_rating': 4.94,
        'minimum_nights': 2,
        'instant_bookable': False,
        'host_is_superhost': True,
        
        'parking_score': 1,  # Keywords: parking, garage
        'kitchen_score': 3,  # Keywords: kitchen, oven, stove, refrigerator, fridge, dishwasher, microwave, coffee, toaster, dishes, blender, freezer
        'entertainment_score': 0,  # Keywords: tv, television, netflix, hulu, cable, apple tv, sound system, hdtv, amazon prime, chromecast, disney+, hbo
        'climate_control_score': 3,  # Keywords: heating, air conditioning, ac, fireplace, fan
        'laundry_score': 1,  # Keywords: washer, dryer, laundry, essentials, iron
        'safety_score': 4,  # Keywords: smoke, carbon monoxide, fire extinguisher, first aid, lock, security cameras
        'convenience_score': 4,  # Keywords: self check-in, keypad, lockbox, smart lock, private entrance
        'toiletries_score': 8,  # Keywords: hair dryer, shampoo, conditioner, body soap, shower gel, bed linens, extra pillows and blankets, clothing storage, hangers
        
        'has_beach_access': 0,  # Keywords: beach access, beachfront, waterfront
        'pets_allowed': 0,  # Keywords: pets allowed
    },
]


def preprocess_slo_listings(listings_data):
    """
    Preprocess SLO listings to match Santa Cruz training data format.
    
    Returns:
        X_slo: DataFrame with all 26 features in correct format
        actual_prices: List of actual prices for comparison
    """
    df = pd.DataFrame(listings_data)
    
    # Extract actual prices for later comparison
    actual_prices = df['actual_price'].values
    listing_urls = df['listing_url'].values
    
    # Drop non-feature columns
    df = df.drop(['actual_price', 'listing_url'], axis=1)
    
    # Parse bathrooms (creates bathrooms_count and bathroom_type)
    bathroom_features = df['bathrooms_text'].apply(parse_bathrooms)
    df['bathrooms_count'] = [bf['bathrooms_count'] for bf in bathroom_features]
    df['bathroom_type'] = [bf['bathroom_type'] for bf in bathroom_features]
    df = df.drop('bathrooms_text', axis=1)
    
    # Convert boolean columns to int
    df['instant_bookable'] = df['instant_bookable'].astype(int)
    df['host_is_superhost'] = df['host_is_superhost'].astype(int)
    
    # One-hot encode room_type with rt_ prefix (to match training data)
    df = pd.get_dummies(df, columns=['room_type'], prefix='rt')
    
    # One-hot encode bathroom_type with bt_ prefix (to match training data)
    df = pd.get_dummies(df, columns=['bathroom_type'], prefix='bt')
    
    # Ensure all expected columns exist (in case some room types missing)
    expected_room_types = ['rt_Entire home/apt', 'rt_Hotel room', 'rt_Private room', 'rt_Shared room']
    for col in expected_room_types:
        if col not in df.columns:
            df[col] = 0
    
    expected_bathroom_types = ['bt_private', 'bt_shared', 'bt_standard']
    for col in expected_bathroom_types:
        if col not in df.columns:
            df[col] = 0
    
    # Add amenities_count (sum of all amenity scores)
    df['amenities_count'] = (df['parking_score'] + df['kitchen_score'] + 
                             df['entertainment_score'] + df['climate_control_score'] + 
                             df['laundry_score'] + df['safety_score'] + 
                             df['convenience_score'] + df['toiletries_score'])
    
    # Reorder columns to match training data EXACTLY
    feature_order = [
        'rt_Entire home/apt', 'rt_Hotel room', 'rt_Private room', 'rt_Shared room',
        'bt_private', 'bt_shared', 'bt_standard',
        'accommodates', 'bedrooms', 'beds', 'bathrooms_count',
        'review_scores_rating', 'host_is_superhost', 'instant_bookable', 'minimum_nights',
        'amenities_count', 'parking_score', 'kitchen_score', 'entertainment_score', 
        'climate_control_score', 'laundry_score', 'safety_score', 'convenience_score', 
        'toiletries_score', 'has_beach_access', 'pets_allowed'
    ]
    
    df = df[feature_order]
    
    return df, actual_prices, listing_urls


def load_all_models():
    """Load all three trained models: Linear Regression, Random Forest, XGBoost."""
    import pickle
    import xgboost as xgb
    
    models = {}
    
    # Load Linear Regression
    lr_path = os.path.join(os.path.dirname(__file__), 'linear_model.pkl')
    if os.path.exists(lr_path):
        with open(lr_path, 'rb') as f:
            models['Linear Regression'] = pickle.load(f)
        print(f"Loaded Linear Regression from {lr_path}")
    else:
        print(f"Linear Regression model not found at {lr_path}")
        print("   Run: python linear_model.py")
    
    # Load Random Forest
    rf_path = os.path.join(os.path.dirname(__file__), 'random_forest_model.pkl')
    if os.path.exists(rf_path):
        with open(rf_path, 'rb') as f:
            models['Random Forest'] = pickle.load(f)
        print(f"Loaded Random Forest from {rf_path}")
    else:
        print(f"Random Forest model not found at {rf_path}")
        print("   Run: python random_forest_model.py")
    
    # Load XGBoost
    xgb_path = os.path.join(os.path.dirname(__file__), 'xgboost_model.json')
    if os.path.exists(xgb_path):
        xgb_model = xgb.XGBRegressor()
        xgb_model.load_model(xgb_path)
        models['XGBoost'] = xgb_model
        print(f"Loaded XGBoost from {xgb_path}")
    else:
        print(f"XGBoost model not found at {xgb_path}")
        print("   Run: python xgboost_model.py")
    
    if not models:
        print("\nERROR: No models found! Please train at least one model first.")
        return None
    
    return models


def main():
    """
    Main function to test transfer learning from Santa Cruz to SLO.
    """
    print("TRANSFER LEARNING TEST: Santa Cruz Model → San Luis Obispo")
    
    # Check if listings have been filled in
    if slo_listings[0]['actual_price'] == 0:
        print("\nWARNING: Please fill in the SLO listing data above before running!")
        print("   Go to airbnb.com, find 3 SLO listings, and fill in all the features.")
        return
    
    # Load all trained models
    models = load_all_models()
    if models is None:
        return
    
    # Preprocess SLO listings
    print("PREPROCESSING SLO LISTINGS")
    
    X_slo, actual_prices, listing_urls = preprocess_slo_listings(slo_listings)
    
    print(f"\nProcessed {len(X_slo)} SLO listings")
    print(f"Features: {list(X_slo.columns)}")
    print(f"\nFeature values:")
    print(X_slo)
    
    # Make predictions with all models and collect results
    print("PREDICTIONS FROM ALL MODELS")
    
    all_predictions = {}
    for model_name, model in models.items():
        predictions = model.predict(X_slo)
        all_predictions[model_name] = predictions
        print(f"\n{model_name} predictions complete")
    
    # Display results for each listing
    print("DETAILED RESULTS BY LISTING")
    
    for i, (actual, url) in enumerate(zip(actual_prices, listing_urls)):
        print(f"\nListing {i+1}: ${actual:.2f} (actual)")
        print(f"  URL: {url}")
        print(f"  {'Model':<20} {'Predicted':<15} {'Error':<12} {'Error %':<10}")
        print(f"  {'-'*60}")
        
        for model_name in models.keys():
            predicted = all_predictions[model_name][i]
            error = predicted - actual
            error_pct = (abs(error) / actual) * 100
            
            print(f"  {model_name:<20} ${predicted:<13.2f} ${error:<10.2f} {error_pct:<8.1f}%")
    
    # Summary statistics for each model
    print("MODEL COMPARISON SUMMARY")
    
    print(f"\n{'Model':<20} {'MAE':<12} {'MAPE':<12} {'Santa Cruz R²':<15}")
    print("-"*80)
    
    model_performance = {
        'Linear Regression': {'r2': 0.7058, 'rmse': 181.49, 'mae': 129.06, 'mape': 43.36},
        'Random Forest': {'r2': 0.7283, 'rmse': 174.43, 'mae': 115.67, 'mape': 37.91},
        'XGBoost': {'r2': 0.7326, 'rmse': 173.03, 'mae': 114.25, 'mape': 38.73}
    }
    
    for model_name in models.keys():
        predictions = all_predictions[model_name]
        mae = np.mean([abs(pred - actual) for pred, actual in zip(predictions, actual_prices)])
        mape = np.mean([abs(pred - actual) / actual * 100 for pred, actual in zip(predictions, actual_prices)])
        
        sc_r2 = model_performance[model_name]['r2']
        
        print(f"{model_name:<20} ${mae:<10.2f} {mape:<10.2f}% {sc_r2:<15.4f}")
    
    print("INTERPRETATION")
    print("\nSanta Cruz Test Set Performance (for reference):")
    for model_name, perf in model_performance.items():
        print(f"\n{model_name}:")
        print(f"  - R² = {perf['r2']:.4f}")
        print(f"  - RMSE = ${perf['rmse']:.2f}")
        print(f"  - MAE = ${perf['mae']:.2f}")
        print(f"  - MAPE = {perf['mape']:.2f}%")
    
    print("\n" + "-"*80)
    print("Transfer Learning Insights:")
    print("-"*80)
    
    # Find best performing model on SLO data
    slo_maes = {name: np.mean([abs(pred - actual) for pred, actual in zip(all_predictions[name], actual_prices)]) 
                for name in models.keys()}
    best_model = min(slo_maes, key=slo_maes.get)
    
    print(f"\nBest performer on SLO: {best_model} (MAE: ${slo_maes[best_model]:.2f})")
    print(f"\nNote: With only {len(slo_listings)} samples, these metrics are anecdotal.")
    print("      They demonstrate generalization capability but aren't statistically robust.")
    print("\nTransfer learning test complete!")


if __name__ == "__main__":
    main()