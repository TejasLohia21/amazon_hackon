import os
import numpy as np
import pandas as pd
from PIL import Image
from model import ReviewPipeline 
from filetype import FileTypeChecker
from videosampler import VideoSampler
from transformers import CLIPProcessor, CLIPModel as HuggingFaceCLIPModel

def run_visual_verification_pipeline(df_with_images: pd.DataFrame, api_key: str, temp_dir: str = "temp_images") -> pd.DataFrame:
    print("[INFO] Initializing pipeline...")
    pipeline = ReviewPipeline(api_key=api_key)
    
    os.makedirs(temp_dir, exist_ok=True)
    results_list = []

    print(f"[INFO] Processing {len(df_with_images)} reviews...")
    for idx, row in df_with_images.iterrows():
        review_text = row.get("reviewText", "")
        img_array = row.get("image_array")

        # Skip invalid image data
        if img_array is None or not isinstance(img_array, np.ndarray):
            continue

        # Save image array to temporary JPEG file
        img = Image.fromarray(img_array.astype('uint8'))
        img_path = os.path.join(temp_dir, f"{idx}.jpg")
        img.save(img_path, format="JPEG")

        # Run pipeline
        result = pipeline.run(review_text, img_path)

        # Store result
        results_list.append({
            "index": idx,
            "asin": row.get("asin", ""),
            "reviewerID": row.get("reviewerID", ""),
            "visual_verification": result
        })

    # Merge back results
    df_results = pd.DataFrame(results_list).set_index("index")
    df_final = df_with_images.join(df_results, how="left")

    return df_final

if __name__ == "__main__":
    # === CONFIGURATION ===
    API_KEY = "your_actual_api_key_here"  # Replace or load securely
    PICKLE_PATH = "/teamspace/studios/this_studio/amazon_hackon/Data/Reviews with images/dataset_with_images.pkl"
    TEMP_IMG_DIR = "temp_images"

    print("[INFO] Loading dataset...")
    try:
        df_with_images = pd.read_pickle(PICKLE_PATH)
    except Exception as e:
        raise RuntimeError(f"[ERROR] Could not load dataset: {e}")

    df_final = run_visual_verification_pipeline(df_with_images, API_KEY, TEMP_IMG_DIR)

    print("\n[INFO] Sample verified results:")
    print(df_final[["asin", "reviewerID", "visual_verification"]].head())

    print("\n[INFO] Saving results to: final_verified_dataset.pkl")
    df_final.to_pickle("final_verified_dataset.pkl")
