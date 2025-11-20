import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_data(filepath='../data/28june2025listings.csv'):
    # Read CSV file into memory
    df = pd.read_csv(filepath)
    
    # Print basic info so we know what we're working with
    print(f"Loaded {len(df)} listings")
    print(f"Dataset has {len(df.columns)} columns")
    
    return df


def clean_price(df):
    df = df.copy()
    
    df['price_clean'] = df['price'].str.replace('$', '', regex=False).str.replace(',', '', regex=False).astype(float)
    
    # Print stats so we can verify it worked
    print(f"\nPrice cleaned:")
    print(f"  - Min: ${df['price_clean'].min():.2f}")
    print(f"  - Max: ${df['price_clean'].max():.2f}")
    print(f"  - Median: ${df['price_clean'].median():.2f}")
    print(f"  - Mean: ${df['price_clean'].mean():.2f}")
    
    return df


def remove_outliers(df, max_price=49999):
    # Count before removal
    # before_count = len(df)
    
    # Filter: keep only rows where price_clean <= max_price
    df = df[(df['price_clean'] >= 0) & (df['price_clean'] <= max_price)]
    
    # Count after removal
    # after_count = len(df)
    # removed = before_count - after_count
    
    # print(f"\nOutliers removed:")
    # print(f"  - Before: {before_count} listings")
    # print(f"  - After: {after_count} listings")
    # print(f"  - Removed: {removed} listings ({removed/before_count*100:.1f}%)")
    
    return df


def select_simple_features(df):
    """
    Select a small set of features to start with.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Full dataset with all 79 columns
        
    Returns:
    --------
    df_subset : pandas.DataFrame
        Dataset with only selected features + target (price_clean)
    """
    features_to_keep = [
        'room_type',
        'accommodates',
        'bedrooms',
        'price_clean'      # Target variable
    ]
    
    df_subset = df[features_to_keep].copy()
    
    # print(f"\nFeature selection:")
    # print(f"  - Selected {len(features_to_keep)-1} features (+ 1 target)")
    # print(f"  - Features: {features_to_keep[:-1]}")
    # print(f"  - Target: {features_to_keep[-1]}")
    
    return df_subset


def handle_missing_values(df):
    """
    Handle missing values in our selected features.
    
    What this does:
    ---------------
    1. Check how many missing values we have
    2. For 'bedrooms': fill with median (middle value)
    3. For 'room_type': fill with mode (most common value)
    4. Drop any remaining rows with missing target (price_clean)
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset that might have missing values
        
    Returns:
    --------
    df : pandas.DataFrame
        Dataset with missing values handled
        
    """
    # Check for missing values BEFORE handling them
    print("\nMissing values check:")
    missing = df.isnull().sum()
    print("missing:\n", missing)

    for col, count in missing.items():
        if count > 0:
            print(f"  - {col}: {count} missing ({count/len(df)*100:.1f}%)")
    
    # Handle numerical feature (bedrooms)
    if 'bedrooms' in df.columns:
        median_bedrooms = df['bedrooms'].median()
        df['bedrooms'] = df['bedrooms'].fillna(median_bedrooms)
        print(f"  → Filled 'bedrooms' missing with median: {median_bedrooms}")
    
    # Handle categorical feature (room_type)
    if 'room_type' in df.columns:
        mode_room_type = df['room_type'].mode()[0]  # [0] gets first mode
        df['room_type'] = df['room_type'].fillna(mode_room_type)
        print(f"  → Filled 'room_type' missing with mode: '{mode_room_type}'")
    
    # Drop rows with missing target (can't predict if we don't know the answer)
    before_drop = len(df)
    df = df.dropna(subset=['price_clean'])
    after_drop = len(df)
    dropped = before_drop - after_drop
    
    if dropped > 0:
        print(f"  → Dropped {dropped} rows with missing price_clean")
    
    return df


def encode_categorical_features(df):
    """
    Convert categorical features to numbers using One-Hot Encoding
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset with categorical 'room_type' column
        
    Returns:
    --------
    df : pandas.DataFrame
        Dataset with one-hot encoded columns for room_type

    """
    encoded_df = pd.get_dummies(df, columns=['room_type'], prefix='rt', drop_first=False)
    
    return encoded_df


def create_train_test_split(df, test_size=0.2, random_state=42):
    # After one-hot encoding, we have multiple room_type columns (rt_*)
    # Get all columns that start with 'rt_' plus numerical features
    feature_columns = [col for col in df.columns if col.startswith('rt_')] + ['accommodates', 'bedrooms']
    X = df[feature_columns]
    y = df['price_clean']
    
    # print(f"\nCreating train/test split:")
    # print(f"  - Total samples: {len(df)}")
    # print(f"  - Features (X): {len(feature_columns)} total")
    # print(f"    - Categorical (one-hot): {[col for col in feature_columns if col.startswith('rt_')]}")
    # print(f"    - Numerical: ['accommodates', 'bedrooms']")
    # print(f"  - Target (y): price_clean")
    
    # This randomly assigns rows to train or test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )
    
    # Print results
    # print(f"  - Train set: {len(X_train)} samples ({len(X_train)/len(df)*100:.1f}%)")
    # print(f"  - Test set: {len(X_test)} samples ({len(X_test)/len(df)*100:.1f}%)")
    
    # Show example of what the data looks like
    # print(f"\n  Example training sample:")
    # print(X_train.head(1))
    # print(f"  → Predicting price: ${y_train.iloc[0]:.2f}")
    
    return X_train, X_test, y_train, y_test


def run_preprocessing_pipeline():
    """
    Run the complete preprocessing pipeline.
    
    Returns:
    --------
    X_train, X_test, y_train, y_test : Training and testing data ready for modeling
    
    """
    print("Beginning preprocessing...")
    
    # Step 1: Load data
    df = load_data()
    
    # Step 2: Clean price (target variable)
    df = clean_price(df)
    
    # Step 3: Remove extreme outliers
    df = remove_outliers(df, 49999)
    
    # Step 4: Select simple feature set
    df = select_simple_features(df)
    
    # Step 5: Handle missing values
    df = handle_missing_values(df)
    
    # Step 6: Encode categorical features
    df = encode_categorical_features(df)
    
    # Step 7: Create train/test split
    X_train, X_test, y_train, y_test = create_train_test_split(df)
    

    print("\nPreprocessing complete!")
    print(f"  - Training samples: {len(X_train)}")
    print(f"  - Testing samples: {len(X_test)}")
    print(f"  - Features: {list(X_train.columns)}")
    print(f"  - Target range: ${y_train.min():.2f} - ${y_train.max():.2f}")
    
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = run_preprocessing_pipeline()
