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


def parse_bathroom_features(df):
    """
    Parse bathroom information from both 'bathrooms' and 'bathrooms_text' columns.
    
    Creates two new features:
    1. bathrooms_count (numeric): Number of bathrooms (1.0, 1.5, 2.0, etc.)
    2. bathroom_type (categorical): 'private', 'shared', or 'standard'
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset with 'bathrooms' and 'bathrooms_text' columns
        
    Returns:
    --------
    df : pandas.DataFrame
        Dataset with new 'bathrooms_count' and 'bathroom_type' columns
    """
    import re
    
    def extract_bathroom_count(row):
        """Extract numeric bathroom count from either column."""
        # First, try the numeric 'bathrooms' column
        if pd.notna(row['bathrooms']):
            return row['bathrooms']
        
        # Fall back to parsing 'bathrooms_text'
        if pd.notna(row['bathrooms_text']):
            # Extract first number (handles "2.5 baths", "1 bath", "1 private bath")
            match = re.search(r'(\d+\.?\d*)', str(row['bathrooms_text']))
            if match:
                return float(match.group(1))
        
        # If both are missing, return NaN
        return np.nan
    
    def extract_bathroom_type(bathroom_text):
        """Extract bathroom type (private, shared, or standard) from text."""
        if pd.isna(bathroom_text):
            return 'standard'  # Unknown, assume standard
        
        text_lower = str(bathroom_text).lower()
        
        if 'private' in text_lower:
            return 'private'
        elif 'shared' in text_lower:
            return 'shared'
        else:
            return 'standard'  # No modifier = standard bathroom
    
    # Create bathrooms_count (best of both columns)
    df['bathrooms_count'] = df.apply(extract_bathroom_count, axis=1)
    
    # Create bathroom_type (from bathrooms_text only)
    df['bathroom_type'] = df['bathrooms_text'].apply(extract_bathroom_type)
    
    print("\nBathroom feature engineering:")
    print(f"  - Created 'bathrooms_count' (numeric)")
    print(f"    → Min: {df['bathrooms_count'].min()}, Max: {df['bathrooms_count'].max()}, Median: {df['bathrooms_count'].median()}")
    print(f"  - Created 'bathroom_type' (categorical)")
    print(f"    → Value counts: {df['bathroom_type'].value_counts().to_dict()}")
    
    return df


def parse_amenity_features(df):
    """
    Parse amenity information from 'amenities' column into multiple features.

    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset with 'amenities' column (JSON-like string list)
        
    Returns:
    --------
    df : pandas.DataFrame
        Dataset with new amenity features
    """
    import json
    
    def parse_amenities_list(amenities_str):
        """Safely parse amenities JSON string into Python list."""
        if pd.isna(amenities_str):
            return []
        try:
            # Use eval() since data is formatted like Python list
            amenities_list = eval(amenities_str)
            # Normalize: lowercase for case-insensitive matching
            return [a.lower() for a in amenities_list]
        except:
            return []
    
    def count_category_amenities(amenities_list, keywords):
        """
        Count how many amenities in the list match any keyword.
        Uses substring matching to handle variations:
        - "Free parking" matches "parking"
        - "Washer in unit" matches "washer"
        """
        count = 0
        for amenity in amenities_list:
            if any(keyword in amenity for keyword in keywords):
                count += 1
        return count
    
    def has_keyword(amenities_list, keywords):
        """Binary check: does list contain any keyword?"""
        return int(any(keyword in amenity for amenity in amenities_list for keyword in keywords))
    
    # Parse amenities into lists (do once, reuse for all features)
    amenities_lists = df['amenities'].apply(parse_amenities_list)
    
    # Feature 1: Total count
    df['amenities_count'] = amenities_lists.apply(len)
    
    # Feature 2: Parking score
    parking_keywords = ['parking', 'garage']
    df['parking_score'] = amenities_lists.apply(lambda x: count_category_amenities(x, parking_keywords))
    
    # Feature 3: Kitchen score
    kitchen_keywords = ['kitchen', 'oven', 'stove', 'refrigerator', 'fridge', 'dishwasher', 'microwave', 'coffee', 'toaster', 'dishes', 'blender', 'freezer']
    df['kitchen_score'] = amenities_lists.apply(lambda x: count_category_amenities(x, kitchen_keywords))
    
    # Feature 4: Entertainment score
    entertainment_keywords = ['tv', 'television', 'netflix', 'hulu', 'cable', 'apple tv', 'sound system', 'hdtv', 'amazon prime', 'chromecast', 'disney+', 'hbo']
    df['entertainment_score'] = amenities_lists.apply(lambda x: count_category_amenities(x, entertainment_keywords))
    
    # Feature 5: Climate control score
    climate_keywords = ['heating', 'air conditioning', 'ac', 'fireplace', 'fan']
    df['climate_control_score'] = amenities_lists.apply(lambda x: count_category_amenities(x, climate_keywords))
    
    # Feature 6: Laundry score
    laundry_keywords = ['washer', 'dryer', 'laundry', 'essentials', 'iron']
    df['laundry_score'] = amenities_lists.apply(lambda x: count_category_amenities(x, laundry_keywords))
    
    # Feature 7: Safety score
    safety_keywords = ['smoke', 'carbon monoxide', 'fire extinguisher', 'first aid', 'lock', 'security cameras']
    df['safety_score'] = amenities_lists.apply(lambda x: count_category_amenities(x, safety_keywords))
    
    # Feature 8: Convenience score
    convenience_keywords = ['self check-in', 'keypad', 'lockbox', 'smart lock', 'private entrance', 'wifi']
    df['convenience_score'] = amenities_lists.apply(lambda x: count_category_amenities(x, convenience_keywords))
    
    # Feature 9: Toiletries score
    toiletries_keywords = ['hair dryer', 'shampoo', 'conditioner', 'body soap', 'shower gel', 'bed linens', 'extra pillows and blankets', 'clothing storage', 'hangers']
    df['toiletries_score'] = amenities_lists.apply(lambda x: count_category_amenities(x, toiletries_keywords))
    
    # Feature 10: Beach access
    beach_keywords = ['beach access', 'beachfront', 'waterfront']
    df['has_beach_access'] = amenities_lists.apply(lambda x: has_keyword(x, beach_keywords))
    
    # Feature 11: Pets allowed
    pets_keywords = ['pets allowed']
    df['pets_allowed'] = amenities_lists.apply(lambda x: has_keyword(x, pets_keywords))
    
    print("\nAmenity feature engineering:")
    print(f"  - Created 'amenities_count' (total amenities)")
    print(f"    → Min: {df['amenities_count'].min()}, Max: {df['amenities_count'].max()}, Median: {df['amenities_count'].median()}")
    print(f"  - Created category scores (parking, kitchen, entertainment, climate, laundry, safety, convenience, toiletries)")
    print(f"    → parking_score median: {df['parking_score'].median()}")
    print(f"    → kitchen_score median: {df['kitchen_score'].median()}")
    print(f"    → laundry_score median: {df['laundry_score'].median()}")
    print(f"    → convenience_score median: {df['convenience_score'].median()}")
    print(f"    → toiletries_score median: {df['toiletries_score'].median()}")
    print(f"  - Created 2 binary flags:")
    print(f"    → has_beach_access: {df['has_beach_access'].sum()} listings ({df['has_beach_access'].sum()/len(df)*100:.1f}%)")
    print(f"    → pets_allowed: {df['pets_allowed'].sum()} listings ({df['pets_allowed'].sum()/len(df)*100:.1f}%)")
    
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
        'beds',
        'bathrooms',           # Numeric bathroom count (old column)
        'bathrooms_text',      # Text bathroom description (new column)
        # Batch 1: Trust/Quality features
        'review_scores_rating',
        'host_is_superhost',
        # Batch 2: Booking flexibility features
        'instant_bookable',
        'minimum_nights',
        # Batch 3: Amenity features
        'amenities',           # Raw amenities list (needs parsing)
        'price_clean'          # Target variable
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

    Impute missing values with:
        median for numerical features
        mode for categorical features
    
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
    
    # Handle bedrooms
    if 'bedrooms' in df.columns:
        median_bedrooms = df['bedrooms'].median()
        df['bedrooms'] = df['bedrooms'].fillna(median_bedrooms)
        print(f"  → Filled 'bedrooms' missing with median: {median_bedrooms}")
    
    # Handle beds
    if 'beds' in df.columns:
        median_beds = df['beds'].median()
        df['beds'] = df['beds'].fillna(median_beds)
        print(f"  → Filled 'beds' missing with median: {median_beds}")
    
    # Handle bathrooms_count
    if 'bathrooms_count' in df.columns:
        median_bathrooms = df['bathrooms_count'].median()
        df['bathrooms_count'] = df['bathrooms_count'].fillna(median_bathrooms)
        print(f"  → Filled 'bathrooms_count' missing with median: {median_bathrooms}")
    
    # Handle bathroom_type
    if 'bathroom_type' in df.columns:
        mode_bathroom_type = df['bathroom_type'].mode()[0]
        df['bathroom_type'] = df['bathroom_type'].fillna(mode_bathroom_type)
        print(f"  → Filled 'bathroom_type' missing with mode: '{mode_bathroom_type}'")
    
    # Batch 1: Trust/Quality features
    # Handle review_scores_rating
    if 'review_scores_rating' in df.columns:
        # Option 1: Fill with median (assumes missing = average quality)
        median_rating = df['review_scores_rating'].median()
        df['review_scores_rating'] = df['review_scores_rating'].fillna(median_rating)
        print(f"  → Filled 'review_scores_rating' missing with median: {median_rating}")
    
    # Handle host_is_superhost
    if 'host_is_superhost' in df.columns:
        # Fill with 'f' (not superhost)
        df['host_is_superhost'] = df['host_is_superhost'].fillna('f')
        # Convert to numeric: t=1, f=0
        df['host_is_superhost'] = (df['host_is_superhost'] == 't').astype(int)
        print(f"  → Filled 'host_is_superhost' missing with 'f' (not superhost), converted to binary")
    
    # Batch 2: Handle booking flexibility features
    # Handle instant_bookable
    if 'instant_bookable' in df.columns:
        # Fill with 'f' (not instant bookable - most conservative default)
        df['instant_bookable'] = df['instant_bookable'].fillna('f')
        # Convert to numeric: t=1, f=0
        df['instant_bookable'] = (df['instant_bookable'] == 't').astype(int)
        print(f"  → Filled 'instant_bookable' missing with 'f' (not instant), converted to binary")
    
    # Handle minimum_nights
    if 'minimum_nights' in df.columns:
        median_min_nights = df['minimum_nights'].median()
        df['minimum_nights'] = df['minimum_nights'].fillna(median_min_nights)
        print(f"  → Filled 'minimum_nights' missing with median: {median_min_nights}")
    
    # Batch 3: Handle amenity features (11 total)
    # Amenity count and category scores (already filled with 0 by parse function)
    amenity_features = ['amenities_count', 'parking_score', 'kitchen_score', 
                        'entertainment_score', 'climate_control_score', 
                        'laundry_score', 'safety_score', 'convenience_score', 'toiletries_score',
                        'has_beach_access', 'pets_allowed']
    
    for feature in amenity_features:
        if feature in df.columns:
            df[feature] = df[feature].fillna(0)  # Default to 0 (no amenities/not present)
    
    print(f"  → Filled amenity features missing values with 0 (default: not present)")
    
    # Handle categorical feature (room_type)
    if 'room_type' in df.columns:
        mode_room_type = df['room_type'].mode()[0]
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
        Dataset with categorical 'room_type' and 'bathroom_type' columns
        
    Returns:
    --------
    df : pandas.DataFrame
        Dataset with one-hot encoded columns for room_type and bathroom_type

    """
    # One-hot encode room_type
    encoded_df = pd.get_dummies(df, columns=['room_type'], prefix='rt', drop_first=False)
    
    # One-hot encode bathroom_type (if it exists)
    if 'bathroom_type' in encoded_df.columns:
        encoded_df = pd.get_dummies(encoded_df, columns=['bathroom_type'], prefix='bt', drop_first=False)
    
    return encoded_df


def create_train_test_split(df, test_size=0.2, random_state=42):
    # After one-hot encoding, we have multiple room_type columns (rt_*) and bathroom_type columns (bt_*)
    # Get all columns that start with 'rt_' or 'bt_' plus numerical features
    feature_columns = (
        [col for col in df.columns if col.startswith('rt_')] +
        [col for col in df.columns if col.startswith('bt_')] +
        ['accommodates', 'bedrooms', 'beds', 'bathrooms_count',
         'review_scores_rating', 'host_is_superhost',
         # Batch 2: Booking flexibility
         'instant_bookable', 'minimum_nights',
         # Batch 3: Amenity features (11 total: 1 count + 8 category scores + 2 binary flags)
         'amenities_count', 'parking_score', 'kitchen_score', 'entertainment_score',
         'climate_control_score', 'laundry_score', 'safety_score', 'convenience_score', 'toiletries_score', 'has_beach_access', 'pets_allowed']
    )
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
    
    # Step 4: Select feature set (includes raw bathroom columns)
    df = select_simple_features(df)
    
    # Engineer bathroom features (bathrooms_count + bathroom_type)
    df = parse_bathroom_features(df)
    
    # Engineer amenity features
    df = parse_amenity_features(df)
    
    # Drop original bathroom and amenities columns (we have engineered features now)
    df = df.drop(columns=['bathrooms', 'bathrooms_text', 'amenities'], errors='ignore')
    
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
