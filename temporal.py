# import pandas as pd
# import numpy as np
# import os
# import gzip

# # --- Configuration ---
# WINDOW_SIZE = 30
# SAMPLE_SIZE = 2000
# PATHSTR = "/Users/tejasmacipad/Desktop/Extras/amazon_hackon/amazon_hackon/Data/Reviews with images"

# FILE_INFO = {
#     "Cell_Phones_and_Accessories_5.json": "Cell_Phones_and_Accessories",
#     "Magazine_Subscriptions_5.json.gz": "Magazine_Subscriptions",
#     "Appliances_5 (1).json.gz": "Appliances",
#     "All_Beauty_5 (1).json.gz": "All_Beauty",
#     "AMAZON_FASHION_5 (1).json.gz": "AMAZON_FASHION"
# }

# # --- Data Loading (No Changes) ---
# def read_and_filter(file_path, category, sample_size=1000):
#     print(f"-> Reading {os.path.basename(file_path)}...")
#     try:
#         compression = 'gzip' if file_path.endswith('.gz') else 'infer'
#         df = pd.read_json(file_path, lines=True, compression=compression)
#     except Exception as e:
#         print(f"[ERROR] Failed to read {file_path}: {e}")
#         return pd.DataFrame()
#     df['category'] = category
#     return df.sample(n=min(sample_size, len(df)), random_state=42) if not df.empty else pd.DataFrame()

# # --- Feature Engineering (Corrected Functions) ---

# def _get_slope(series):
#     """Calculates the slope. np.polyfit is often more direct for this."""
#     if len(series) < 2:
#         return np.nan # Return NaN for consistency, will be imputed later
#     # Using np.polyfit to fit a 1st degree polynomial (a line) is standard and robust.
#     return np.polyfit(np.arange(len(series)), series, 1)[0]

# def _pos_neg_ratio(series):
#     """
#     Calculates the ratio of positive (>=4) to negative (<=2) ratings.
#     LOGIC FIX: Returning np.nan is more appropriate for an undefined ratio.
#     """
#     pos = np.sum(series >= 4)
#     neg = np.sum(series <= 2)
#     if neg == 0:
#         return np.nan # FIX: Returning the count of 'pos' is misleading. NaN is correct.
#     return pos / neg

# def create_temporal_features(df, window_size):
#     """
#     Generates temporal features using rolling windows.
#     FIXED to be consistent in column creation and access, resolving the KeyError.
#     """
#     print(f"\nGenerating temporal features with window size: {window_size}...")
    
#     df = df.sort_values(by=['asin', 'reviewerID', 'category', 'unixReviewTime']).reset_index(drop=True)
#     df['time_in_days'] = df['unixReviewTime'] / (24 * 3600)

#     # --- Group-wise Rolling Objects ---
#     product_rolling = df.groupby('asin').rolling(window=window_size, min_periods=1)
#     category_rolling = df.groupby('category').rolling(window=window_size, min_periods=1)

#     # --- Feature Creation (Product-Level) ---
#     print("Calculating product-level features...")
#     time_since_last_review = df.groupby('asin')['time_in_days'].diff()
#     df['review_arrival_rate'] = 1 / time_since_last_review
    
#     # FIX: Use single brackets to get simple column names ('mean', 'std').
#     prod_stats = product_rolling['overall'].agg(['mean', 'std'])
    
#     # FIX: Access columns by their simple names and remove the unnecessary renaming line.
#     df['product_rolling_mean_rating'] = prod_stats['mean'].reset_index(level=0, drop=True)
#     df['product_rolling_std_rating'] = prod_stats['std'].reset_index(level=0, drop=True)
    
#     df['product_rating_trend'] = product_rolling['overall'].apply(_get_slope, raw=False).reset_index(level=0, drop=True)
#     df['product_pos_neg_ratio'] = product_rolling['overall'].apply(_pos_neg_ratio, raw=True).reset_index(level=0, drop=True)
#     df['product_cumulative_reviews'] = df.groupby('asin').cumcount() + 1
    
#     # --- Feature Creation (Reviewer-Level) ---
#     print("Calculating reviewer-level features...")
#     time_between_reviews = df.groupby('reviewerID')['time_in_days'].diff()
#     df['reviewer_review_frequency'] = 1 / time_between_reviews

#     reviewer_time_diff_rolling = time_between_reviews.groupby(df['reviewerID']).rolling(window=window_size, min_periods=1)
#     reviewer_time_stats = reviewer_time_diff_rolling.agg(['mean', 'std'])
    
#     df['reviewer_avg_time_between_reviews'] = reviewer_time_stats['mean'].reset_index(level=0, drop=True)
#     df['reviewer_time_variance'] = reviewer_time_stats['std'].reset_index(level=0, drop=True)
    
#     # --- Feature Creation (Category-Level) ---
#     print("Calculating category-level features...")
#     # FIX: Use single brackets here as well for consistency.
#     cat_stats = category_rolling['overall'].agg(['mean', 'std'])

#     # FIX: Access the columns by their correct, simple names ('mean', 'std'). This was the line that failed.
#     df['category_rolling_mean_rating'] = cat_stats['mean'].reset_index(level=0, drop=True)
#     df['category_rolling_std_rating'] = cat_stats['std'].reset_index(level=0, drop=True)
#     df['category_rating_trend'] = category_rolling['overall'].apply(_get_slope, raw=False).reset_index(level=0, drop=True)
    
#     # Clean up temporary columns
#     df = df.drop(columns=['time_in_days'])
    
#     return df

# # --- The rest of your script (impute_features, main block) will now work correctly ---
# # ... (insert the rest of your original script here) ...


# # --- Imputation and Main Block (No Changes) ---
# def impute_features(df):
#     print("\nImputing NaN values...")
#     for col in df.columns:
#         if 'rate' in col or 'frequency' in col or 'time_between' in col or 'variance' in col:
#             df[col].fillna(0, inplace=True)
#             df[col].replace([np.inf, -np.inf], 0, inplace=True)
#     for col in df.columns:
#         if 'rolling' in col or 'trend' in col or 'ratio' in col:
#              if pd.api.types.is_numeric_dtype(df[col]):
#                 df[col].fillna(df[col].median(), inplace=True)
#     print("Imputation complete.")
#     return df

# if __name__ == "__main__":
#     all_samples = []
#     for file_name, category_name in FILE_INFO.items():
#         file_path = os.path.join(PATHSTR, file_name)
#         if os.path.exists(file_path):
#             sample = read_and_filter(file_path, category_name, sample_size=SAMPLE_SIZE)
#             if not sample.empty:
#                 all_samples.append(sample)
#         else:
#             print(f"[ERROR] File does not exist: {file_path}")

#     if all_samples:
#         df_combined = pd.concat(all_samples, ignore_index=True)
#         print(f"\nSuccessfully loaded and combined data. Shape: {df_combined.shape}")
#         final_df = create_temporal_features(df_combined, window_size=WINDOW_SIZE)
#         final_df = impute_features(final_df)
#         print(f"\nFinal DataFrame with temporal features is ready. Shape: {final_df.shape}")
#         print("\n--- Final DataFrame Head ---")
#         print(final_df.head())
#         print("\n--- Final DataFrame Columns ---")
#         print(final_df.columns.tolist())
#         print("\n--- NaN Check ---")
#         print(final_df.isnull().sum().to_string())
#     else:
#         print("No data was loaded. Exiting.")


# import pandas as pd
# import numpy as np
# import os
# import gzip

# # --- Configuration (No Changes) ---
# WINDOW_SIZE = 30
# SAMPLE_SIZE = 2000
# PATHSTR = "/teamspace/studios/this_studio/amazon_hackon/Data/Reviews with images/"

# FILE_INFO = {
#     "Cell_Phones_and_Accessories_5.json.gz": "Cell_Phones_and_Accessories",
#     "Magazine_Subscriptions_5.json.gz": "Magazine_Subscriptions",
#     "Appliances_5 (1).json.gz": "Appliances",
#     "All_Beauty_5 (1).json.gz": "All_Beauty",
#     "AMAZON_FASHION_5 (1).json.gz": "AMAZON_FASHION"
# }

# # --- Data Loading (No Changes) ---
# def read_and_filter(file_path, category, sample_size=1000):
#     print(f"-> Reading {os.path.basename(file_path)}...")
#     try:
#         compression = 'gzip' if file_path.endswith('.gz') else 'infer'
#         df = pd.read_json(file_path, lines=True, compression=compression)
#     except Exception as e:
#         print(f"[ERROR] Failed to read {file_path}: {e}")
#         return pd.DataFrame()
#     df['category'] = category
#     return df.sample(n=min(sample_size, len(df)), random_state=42) if not df.empty else pd.DataFrame()

# # --- Feature Engineering Helper Functions (No Changes) ---
# def _get_slope(series):
#     if len(series) < 2: return np.nan
#     return np.polyfit(np.arange(len(series)), series, 1)[0]

# def _pos_neg_ratio(series):
#     pos = np.sum(series >= 4)
#     neg = np.sum(series <= 2)
#     if neg == 0: return np.nan
#     return pos / neg

# # --- MODIFIED: Temporal Feature Creation (Reviewer Features Removed) ---
# def create_temporal_features(df, window_size):
#     """
#     Generates temporal features for products and categories only.
#     All reviewer-level calculations have been removed.
#     """
#     print(f"\nGenerating temporal features with window size: {window_size} (Product & Category only)...")
    
#     # Sort by product and category time, removing reviewerID
#     df = df.sort_values(by=['asin', 'category', 'unixReviewTime']).reset_index(drop=True)
#     df['time_in_days'] = df['unixReviewTime'] / (24 * 3600)

#     # Group-wise Rolling Objects
#     product_rolling = df.groupby('asin').rolling(window=window_size, min_periods=1)
#     category_rolling = df.groupby('category').rolling(window=window_size, min_periods=1)

#     # --- Feature Creation (Product-Level) ---
#     print("Calculating product-level features...")
#     time_since_last_review = df.groupby('asin')['time_in_days'].diff()
#     df['review_arrival_rate'] = 1 / time_since_last_review
    
#     prod_stats = product_rolling['overall'].agg(['mean', 'std'])
#     df['product_rolling_mean_rating'] = prod_stats['mean'].reset_index(level=0, drop=True)
#     df['product_rolling_std_rating'] = prod_stats['std'].reset_index(level=0, drop=True)
    
#     df['product_rating_trend'] = product_rolling['overall'].apply(_get_slope, raw=False).reset_index(level=0, drop=True)
#     df['product_pos_neg_ratio'] = product_rolling['overall'].apply(_pos_neg_ratio, raw=True).reset_index(level=0, drop=True)
#     df['product_cumulative_reviews'] = df.groupby('asin').cumcount() + 1
    
#     # --- Feature Creation (Category-Level) ---
#     print("Calculating category-level features...")
#     cat_stats = category_rolling['overall'].agg(['mean', 'std'])
#     df['category_rolling_mean_rating'] = cat_stats['mean'].reset_index(level=0, drop=True)
#     df['category_rolling_std_rating'] = cat_stats['std'].reset_index(level=0, drop=True)
#     df['category_rating_trend'] = category_rolling['overall'].apply(_get_slope, raw=False).reset_index(level=0, drop=True)
    
#     # Clean up temporary columns
#     df = df.drop(columns=['time_in_days'])
    
#     return df

# # --- Imputation (No Changes) ---
# def impute_features(df):
#     print("\nImputing NaN values...")
#     # Impute rate and frequency related columns
#     for col in df.columns:
#         if 'rate' in col or 'frequency' in col:
#             df[col].fillna(0, inplace=True)
#             df[col].replace([np.inf, -np.inf], 0, inplace=True)
#     # Impute rolling stats with median
#     for col in df.columns:
#         if 'rolling' in col or 'trend' in col or 'ratio' in col:
#              if pd.api.types.is_numeric_dtype(df[col]):
#                 df[col].fillna(df[col].median(), inplace=True)
#     print("Imputation complete.")
#     return df

# # --- NEW: Function to Prepare Data for MLP ---
# def prepare_data_for_mlp(df):
#     """
#     Prepares the final dataframe for MLP input.
#     - Selects only the engineered features and the target variable.
#     - One-hot encodes the 'category' column.
#     - Separates features (X) from the target (y).
#     """
#     print("\nPreparing data for MLP input...")
    
#     # Define target and features
#     target_col = 'overall'
#     feature_cols = [
#         'review_arrival_rate', 'product_rolling_mean_rating', 'product_rolling_std_rating',
#         'product_rating_trend', 'product_pos_neg_ratio', 'product_cumulative_reviews',
#         'category_rolling_mean_rating', 'category_rolling_std_rating', 'category_rating_trend'
#     ]
    
#     # Add 'category' to be one-hot encoded
#     df_for_mlp = df[feature_cols + ['category', target_col]].copy()
    
#     # One-hot encode the 'category' column
#     df_for_mlp = pd.get_dummies(df_for_mlp, columns=['category'], prefix='cat')
    
#     # Separate features (X) and target (y)
#     y_mlp = df_for_mlp[target_col]
#     X_mlp = df_for_mlp.drop(columns=[target_col])
    
#     print("MLP data preparation complete.")
#     return X_mlp, y_mlp

# # --- Main Execution Block ---
# if __name__ == "__main__":
#     all_samples = []
#     for file_name, category_name in FILE_INFO.items():
#         file_path = os.path.join(PATHSTR, file_name)
#         if os.path.exists(file_path):
#             sample = read_and_filter(file_path, category_name, sample_size=SAMPLE_SIZE)
#             if not sample.empty:
#                 all_samples.append(sample)
#         else:
#             print(f"[ERROR] File does not exist: {file_path}")

#     if all_samples:
#         df_combined = pd.concat(all_samples, ignore_index=True)
#         print(f"\nSuccessfully loaded and combined data. Shape: {df_combined.shape}")
        
#         # 1. Create temporal features (product and category only)
#         final_df = create_temporal_features(df_combined, window_size=WINDOW_SIZE)
        
#         # 2. Impute missing values
#         final_df = impute_features(final_df)
#         print(f"\nDataFrame with temporal features is ready. Shape: {final_df.shape}")
        
#         # 3. Prepare the data for the MLP model
#         X_mlp, y_mlp = prepare_data_for_mlp(final_df)
        
#         print("\n--- MLP-Ready Features (X_mlp) ---")
#         print(f"Shape: {X_mlp.shape}")
#         print(X_mlp.head())
        
#         print("\n--- MLP-Ready Target (y_mlp) ---")
#         print(f"Shape: {y_mlp.shape}")
#         print(y_mlp.head())
        
#         print("\n--- NaN Check in MLP Features ---")
#         print(X_mlp.isnull().sum().to_string())
#     else:
#         print("No data was loaded. Exiting.")


# import pandas as pd
# import numpy as np
# import os
# import gzip

# # --- Configuration (No Changes) ---
# WINDOW_SIZE = 30
# SAMPLE_SIZE = 2000
# PATHSTR = "/Users/tejasmacipada/Desktop/Extras/amazon_hackon/amazon_hackon/Data/Reviews with images"

# FILE_INFO = {
#     "Cell_Phones_and_Accessories_5.json": "Cell_Phones_and_Accessories",
#     "Magazine_Subscriptions_5.json.gz": "Magazine_Subscriptions",
#     "Appliances_5 (1).json.gz": "Appliances",
#     "All_Beauty_5 (1).json.gz": "All_Beauty",
#     "AMAZON_FASHION_5 (1).json.gz": "AMAZON_FASHION"
# }

# # --- Data Loading (No Changes) ---
# def read_and_filter(file_path, category, sample_size=1000):
#     print(f"-> Reading {os.path.basename(file_path)}...")
#     try:
#         compression = 'gzip' if file_path.endswith('.gz') else 'infer'
#         df = pd.read_json(file_path, lines=True, compression=compression)
#     except Exception as e:
#         print(f"[ERROR] Failed to read {file_path}: {e}")
#         return pd.DataFrame()
#     df['category'] = category
#     return df.sample(n=min(sample_size, len(df)), random_state=42) if not df.empty else pd.DataFrame()

# # --- Feature Engineering Helper Functions (No Changes) ---
# def _get_slope(series):
#     if len(series) < 2: return np.nan
#     return np.polyfit(np.arange(len(series)), series, 1)[0]

# def _pos_neg_ratio(series):
#     pos = np.sum(series >= 4)
#     neg = np.sum(series <= 2)
#     if neg == 0: return np.nan
#     return pos / neg

# # --- MODIFIED: Temporal Feature Creation (Reviewer Features Removed) ---
# def create_temporal_features(df, window_size):
#     """
#     Generates temporal features for products and categories only.
#     All reviewer-level calculations have been removed.
#     """
#     print(f"\nGenerating temporal features with window size: {window_size} (Product & Category only)...")
    
#     # Sort by product and category time, removing reviewerID
#     df = df.sort_values(by=['asin', 'category', 'unixReviewTime']).reset_index(drop=True)
#     df['time_in_days'] = df['unixReviewTime'] / (24 * 3600)

#     # Group-wise Rolling Objects
#     product_rolling = df.groupby('asin').rolling(window=window_size, min_periods=1)
#     category_rolling = df.groupby('category').rolling(window=window_size, min_periods=1)

#     # --- Feature Creation (Product-Level) ---
#     print("Calculating product-level features...")
#     time_since_last_review = df.groupby('asin')['time_in_days'].diff()
#     df['review_arrival_rate'] = 1 / time_since_last_review
    
#     prod_stats = product_rolling['overall'].agg(['mean', 'std'])
#     df['product_rolling_mean_rating'] = prod_stats['mean'].reset_index(level=0, drop=True)
#     df['product_rolling_std_rating'] = prod_stats['std'].reset_index(level=0, drop=True)
    
#     df['product_rating_trend'] = product_rolling['overall'].apply(_get_slope, raw=False).reset_index(level=0, drop=True)
#     df['product_pos_neg_ratio'] = product_rolling['overall'].apply(_pos_neg_ratio, raw=True).reset_index(level=0, drop=True)
#     df['product_cumulative_reviews'] = df.groupby('asin').cumcount() + 1
    
#     # --- Feature Creation (Category-Level) ---
#     print("Calculating category-level features...")
#     cat_stats = category_rolling['overall'].agg(['mean', 'std'])
#     df['category_rolling_mean_rating'] = cat_stats['mean'].reset_index(level=0, drop=True)
#     df['category_rolling_std_rating'] = cat_stats['std'].reset_index(level=0, drop=True)
#     df['category_rating_trend'] = category_rolling['overall'].apply(_get_slope, raw=False).reset_index(level=0, drop=True)
    
#     # Clean up temporary columns
#     df = df.drop(columns=['time_in_days'])
    
#     return df

# # --- Imputation (No Changes) ---
# def impute_features(df):
#     print("\nImputing NaN values...")
#     # Impute rate and frequency related columns
#     for col in df.columns:
#         if 'rate' in col or 'frequency' in col:
#             df[col].fillna(0, inplace=True)
#             df[col].replace([np.inf, -np.inf], 0, inplace=True)
#     # Impute rolling stats with median
#     for col in df.columns:
#         if 'rolling' in col or 'trend' in col or 'ratio' in col:
#              if pd.api.types.is_numeric_dtype(df[col]):
#                 df[col].fillna(df[col].median(), inplace=True)
#     print("Imputation complete.")
#     return df

# # --- NEW: Function to Prepare Data for MLP ---
# def prepare_data_for_mlp(df):
#     """
#     Prepares the final dataframe for MLP input.
#     - Selects only the engineered features and the target variable.
#     - One-hot encodes the 'category' column.
#     - Separates features (X) from the target (y).
#     """
#     print("\nPreparing data for MLP input...")
    
#     # Define target and features
#     target_col = 'overall'
#     feature_cols = [
#         'review_arrival_rate', 'product_rolling_mean_rating', 'product_rolling_std_rating',
#         'product_rating_trend', 'product_pos_neg_ratio', 'product_cumulative_reviews',
#         'category_rolling_mean_rating', 'category_rolling_std_rating', 'category_rating_trend'
#     ]
    
#     # Add 'category' to be one-hot encoded
#     df_for_mlp = df[feature_cols + ['category', target_col]].copy()
    
#     # One-hot encode the 'category' column
#     df_for_mlp = pd.get_dummies(df_for_mlp, columns=['category'], prefix='cat')
    
#     # Separate features (X) and target (y)
#     y_mlp = df_for_mlp[target_col]
#     X_mlp = df_for_mlp.drop(columns=[target_col])
    
#     print("MLP data preparation complete.")
#     return X_mlp, y_mlp

# # --- Main Execution Block ---
# if _name_ == "_main_":
#     all_samples = []
#     for file_name, category_name in FILE_INFO.items():
#         file_path = os.path.join(PATHSTR, file_name)
#         if os.path.exists(file_path):
#             sample = read_and_filter(file_path, category_name, sample_size=SAMPLE_SIZE)
#             if not sample.empty:
#                 all_samples.append(sample)
#         else:
#             print(f"[ERROR] File does not exist: {file_path}")

#     if all_samples:
#         df_combined = pd.concat(all_samples, ignore_index=True)
#         print(f"\nSuccessfully loaded and combined data. Shape: {df_combined.shape}")
        
#         # 1. Create temporal features (product and category only)
#         final_df = create_temporal_features(df_combined, window_size=WINDOW_SIZE)
        
#         # 2. Impute missing values
#         final_df = impute_features(final_df)
#         print(f"\nDataFrame with temporal features is ready. Shape: {final_df.shape}")
        
#         # 3. Prepare the data for the MLP model
#         X_mlp, y_mlp = prepare_data_for_mlp(final_df)
        
#         print("\n--- MLP-Ready Features (X_mlp) ---")
#         print(f"Shape: {X_mlp.shape}")
#         print(X_mlp.head())
        
#         print("\n--- MLP-Ready Target (y_mlp) ---")
#         print(f"Shape: {y_mlp.shape}")
#         print(y_mlp.head())
        
#         print("\n--- NaN Check in MLP Features ---")
#         print(X_mlp.isnull().sum().to_string())
#     else:
#         print("No data was loaded. Exiting.")
#     # The file will be saved in the same directory as your script.
# # You can change 'processed_reviews.csv' to any filename you prefer.
#     print(final_df.columns)
#     final_df.to_csv('processed_reviews.csv', index=False)


import pandas as pd
import numpy as np
import os
import gzip

# --- Configuration ---
WINDOW_SIZE = 30
SAMPLE_SIZE = 2000
PATHSTR = ""

FILE_INFO = {
    "amazon_hackon/Data/Reviews with images/Cell_Phones_and_Accessories_5.json": "Cell_Phones_and_Accessories",
    "amazon_hackon/Data/Reviews with images/Magazine_Subscriptions_5.json.gz": "Magazine_Subscriptions",
    "amazon_hackon/Data/Reviews with images/Appliances_5 (1).json.gz": "Appliances",
    "amazon_hackon/Data/Reviews with images/All_Beauty_5 (1).json.gzAll_Beauty_5 (1).json.gz": "All_Beauty",
    "amazon_hackon/Data/Reviews with images/AMAZON_FASHION_5 (1).json.gz": "AMAZON_FASHION"
}

# --- Data Loading ---
def read_and_filter(file_path, category, sample_size=1000):
    print(f"-> Reading {os.path.basename(file_path)}...")
    try:
        compression = 'gzip' if file_path.endswith('.gz') else 'infer'
        df = pd.read_json(file_path, lines=True, compression=compression)
    except Exception as e:
        print(f"[ERROR] Failed to read {file_path}: {e}")
        return pd.DataFrame()
    df['category'] = category
    return df.sample(n=min(sample_size, len(df)), random_state=42) if not df.empty else pd.DataFrame()

# --- Feature Engineering Helper Functions ---
def _get_slope(series):
    """Calculates the slope of a series."""
    if len(series) < 2: return np.nan
    # Correctly return only the slope (the first element of the polyfit result)
    return np.polyfit(np.arange(len(series)), series, 1)[0]

def _pos_neg_ratio(series):
    """Calculates the ratio of positive to negative ratings."""
    pos = np.sum(series >= 4)
    neg = np.sum(series <= 2)
    if neg == 0: return np.nan
    return pos / neg

# --- Temporal Feature Creation (Product & Category Only) ---
def create_temporal_features(df, window_size):
    """Generates temporal features for products and categories."""
    print(f"\nGenerating temporal features with window size: {window_size} (Product & Category only)...")
    
    df = df.sort_values(by=['asin', 'category', 'unixReviewTime']).reset_index(drop=True)
    df['time_in_days'] = df['unixReviewTime'] / (24 * 3600)

    product_rolling = df.groupby('asin').rolling(window=window_size, min_periods=1)
    category_rolling = df.groupby('category').rolling(window=window_size, min_periods=1)

    print("Calculating product-level features...")
    df['review_arrival_rate'] = 1 / df.groupby('asin')['time_in_days'].diff()
    
    prod_stats = product_rolling['overall'].agg(['mean', 'std'])
    df['product_rolling_mean_rating'] = prod_stats['mean'].reset_index(level=0, drop=True)
    df['product_rolling_std_rating'] = prod_stats['std'].reset_index(level=0, drop=True)
    
    df['product_rating_trend'] = product_rolling['overall'].apply(_get_slope, raw=False).reset_index(level=0, drop=True)
    df['product_pos_neg_ratio'] = product_rolling['overall'].apply(_pos_neg_ratio, raw=True).reset_index(level=0, drop=True)
    df['product_cumulative_reviews'] = df.groupby('asin').cumcount() + 1
    
    print("Calculating category-level features...")
    cat_stats = category_rolling['overall'].agg(['mean', 'std'])
    df['category_rolling_mean_rating'] = cat_stats['mean'].reset_index(level=0, drop=True)
    df['category_rating_trend'] = category_rolling['overall'].apply(_get_slope, raw=False).reset_index(level=0, drop=True)
    
    df = df.drop(columns=['time_in_days'])
    
    return df

# --- Imputation ---
def impute_features(df):
    """Fills NaN and infinity values in the dataframe."""
    print("\nImputing NaN values...")
    for col in df.columns:
        if 'rate' in col:
            df[col].fillna(0, inplace=True)
            df[col].replace([np.inf, -np.inf], 0, inplace=True)
    for col in df.columns:
        if 'rolling' in col or 'trend' in col or 'ratio' in col:
             if pd.api.types.is_numeric_dtype(df[col]):
                df[col].fillna(df[col].median(), inplace=True)
    print("Imputation complete.")
    return df

# --- Main Execution Block ---
if __name__ == "__main__":
    all_samples = []
    for file_name, category_name in FILE_INFO.items():
        file_path = os.path.join(PATHSTR, file_name)
        if os.path.exists(file_path):
            sample = read_and_filter(file_path, category_name, sample_size=SAMPLE_SIZE)
            if not sample.empty:
                all_samples.append(sample)
        else:
            print(f"[ERROR] File does not exist: {file_path}")

    if all_samples:
        df_combined = pd.concat(all_samples, ignore_index=True)
        print(f"\nSuccessfully loaded and combined data. Shape: {df_combined.shape}")
        
        # 1. Create temporal features
        final_df = create_temporal_features(df_combined, window_size=WINDOW_SIZE)
        
        # 2. Impute missing values
        final_df = impute_features(final_df)

        # 3. Reduce features to the desired smaller subset
        reduced_features = [
            'asin', 
            'category', 
            'overall', 
            'review_arrival_rate',
            'product_rolling_mean_rating', 
            'product_rating_trend',
            'category_rolling_mean_rating', 
            'category_rating_trend'
        ]
        # Ensure only existing columns are selected
        existing_features = [col for col in reduced_features if col in final_df.columns]
        final_df_reduced = final_df[existing_features].copy()

        print(f"\nFinal reduced DataFrame shape: {final_df_reduced.shape}")
        print("\n--- Final Reduced DataFrame Head ---")
        print(final_df_reduced.head())

        # 4. Save the final reduced DataFrame to a CSV file
        output_filename = 'processed_reviews_reduced.csv'
        final_df_reduced.to_csv(output_filename, index=False)
        print(f"\nSaved reduced DataFrame to '{output_filename}'")
    else:
        print("No data was loaded. Exiting.")