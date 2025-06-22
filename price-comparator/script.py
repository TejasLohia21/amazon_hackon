import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

df = pd.read_csv('combined_top1500_with_500_extra.csv')
for col in ['main_category', 'title', 'features', 'description']:
    df[col] = df[col].fillna('')
df['price'] = df['price'].fillna(0)


df['text'] = (
    df['main_category'].astype(str) + ' ' +
    df['title'].astype(str) + ' ' +
    df['features'].astype(str) + ' ' +
    df['description'].astype(str)
)

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(df['text'].tolist(), batch_size=64, show_progress_bar=True)
embeddings = np.array(embeddings, dtype=np.float32)


dimension = embeddings.shape[1]
M = 32
index = faiss.IndexHNSWFlat(dimension, M)
index.hnsw.efConstruction = 200
index.add(embeddings)
index.hnsw.efSearch = 50


def find_similar_with_different_price(query_idx, top_k=10):
    query_vector = embeddings[query_idx].reshape(1, -1)
    distances, indices = index.search(query_vector, top_k)
    similar_df = df.iloc[indices[0]]
    query_row = df.iloc[query_idx]

    # Filter: same title, same category, different price (and not the query itself)
    filtered = similar_df[
        (similar_df['title'] == query_row['title']) &
        (similar_df['main_category'] == query_row['main_category']) &
        (similar_df['price'] != query_row['price'])
    ]
    return filtered
print("\nTop 5 similar products to the first product:")
print(find_similar_products := df.iloc[index.search(embeddings[0].reshape(1, -1), 5)[1][0]][['main_category', 'title', 'features', 'description', 'price']])

print("\nSimilar products (same title & category, different price):")
similar_diff_price = find_similar_with_different_price(0, top_k=10)
if not similar_diff_price.empty:
    print(similar_diff_price[['main_category', 'title', 'features', 'description', 'price']])
else:
    print("No similar product with a different price found.")

faiss.write_index(index, "hnsw_index.faiss")
