import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, mode
import gzip

class AmazonReviewAnalyzer:
    def __init__(self, file_paths, sample_size=1000):
        self.file_paths = file_paths
        self.sample_size = sample_size
        self.df_with_images = None

    def read_and_filter(self, file_path):
        try:
            df = pd.read_json(file_path, lines=True, compression='gzip')
        except (OSError, gzip.BadGzipFile, ValueError):
            try:
                df = pd.read_json(file_path, lines=True)
            except Exception as e:
                print(f"[ERROR] Failed to read {file_path}: {e}")
                return pd.DataFrame()
        if 'image' in df.columns:
            df = df[df['image'].apply(lambda x: isinstance(x, list) and len(x) > 0)]
        else:
            print(f"[WARNING] No 'image' column in {file_path}")
            return pd.DataFrame()
        return df.sample(n=min(self.sample_size, len(df)), random_state=42) if not df.empty else df

    def load_all_samples(self):
        all_samples = []
        for fp in self.file_paths:
            print(f"Reading file: {fp}")
            sample = self.read_and_filter(fp)
            if not sample.empty:
                all_samples.append(sample)
        if all_samples:
            self.df_with_images = pd.concat(all_samples, ignore_index=True)
        else:
            self.df_with_images = pd.DataFrame()
        print(f"Final shape: {self.df_with_images.shape}")
        print("Columns:", self.df_with_images.columns.tolist())
        print(self.df_with_images.head())
        return self.df_with_images

    def compute_rating_features(self, df=None):
        if df is None:
            df = self.df_with_images
        if df is None or 'overall' not in df.columns or df.empty:
            print("[ERROR] DataFrame is empty or missing 'overall' column.")
            return {}
        ratings = df['overall'].dropna()
        if ratings.empty:
            print("[WARNING] No ratings available.")
            return {}
        return {
            'mean': ratings.mean(),
            'variance': ratings.var(),
            'std_dev': ratings.std(),
            'min': ratings.min(),
            'max': ratings.max(),
            'median': ratings.median(),
            'mode': mode(ratings, keepdims=True).mode[0] if len(ratings) > 0 else np.nan,
            'skewness': skew(ratings),
            'kurtosis': kurtosis(ratings),
            'review_count': len(ratings)
        }

    def compute_features_by_category(self, category_col='category'):
        if self.df_with_images is None or self.df_with_images.empty:
            print("[ERROR] No data loaded.")
            return {}
        if category_col not in self.df_with_images.columns:
            print(f"[INFO] No '{category_col}' column found. Assigning by file order.")
            categories = [fp.split('/')[-1].split('_5')[0] for fp in self.file_paths]
            sizes = [len(self.read_and_filter(fp)) for fp in self.file_paths]
            cat_list = []
            for cat, size in zip(categories, sizes):
                cat_list.extend([cat] * size)
            self.df_with_images[category_col] = cat_list
        features_by_cat = {}
        for cat in self.df_with_images[category_col].unique():
            subdf = self.df_with_images[self.df_with_images[category_col] == cat]
            features_by_cat[cat] = self.compute_rating_features(subdf)
        return features_by_cat

if __name__ == "__main__":
    pathstr = "/Users/tejasmacipad/Desktop/Extras/amazon_hackon/dataclone/amazon_hackon/Data/Reviews with images"
    file_names = [
        "/Cell_Phones_and_Accessories_5.json.gz",
        "/Magazine_Subscriptions_5.json.gz",
        "/Appliances_5 (1).json.gz",
        "/All_Beauty_5 (1).json.gz",
        "/AMAZON_FASHION_5 (1).json.gz"
    ]
    file_paths = [pathstr + fp for fp in file_names]

    analyzer = AmazonReviewAnalyzer(file_paths, sample_size=1000)
    df_combined = analyzer.load_all_samples()

    print("\n--- Overall Rating Statistics ---")
    overall_features = analyzer.compute_rating_features()
    for k, v in overall_features.items():
        print(f"{k}: {v}")

    print("\n--- Per-Category Rating Statistics ---")
    features_by_category = analyzer.compute_features_by_category()
    for cat, feats in features_by_category.items():
        print(f"\nCategory: {cat}")
        for k, v in feats.items():
            print(f"  {k}: {v}")

    # df_combined is your combined DataFrame for all categories
