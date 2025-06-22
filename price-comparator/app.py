from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import faiss

# ---- Request models ----
class ByID(BaseModel):
    product_id: str
    top_k: int = 10

class ByName(BaseModel):
    product_name: str
    top_k: int = 10

# ---- FastAPI setup ----
app = FastAPI()

@app.get("/")
async def health():
    return {"status": "ok"}

# ---- Load index & metadata at startup ----
@app.on_event("startup")
def load_data():
    global index, id_map, meta
    # load FAISS index
    index = faiss.read_index("hnsw_index.faiss")
    # load metadata CSV to get product_id and title
    df = pd.read_csv("combined_top1500_with_500_extra.csv")
    id_map = df["product_id"].tolist()
    meta = df.set_index("product_id")

    # If you have extra JSONL metadata, merge it here:
    # extra = pd.read_json("meta_Amazon_Fashion.jsonl", lines=True)
    # meta = pd.concat([meta, extra.set_index("product_id")], axis=0)

# ---- Endpoint: list all products ----
@app.get("/products")
async def list_products():
    # return list of {product_id, title}
    return [
        {"product_id": pid, "title": meta.loc[pid, "title"]}
        for pid in id_map
    ]

# ---- Endpoint: find similar by ID ----
@app.post("/similar")
async def similar_by_id(q: ByID):
    if q.product_id not in id_map:
        raise HTTPException(404, "Product not found")
    idx = id_map.index(q.product_id)
    D, I = index.search(
        np.array([index.reconstruct_n(idx, 1)[0]]), q.top_k
    )
    results = []
    for dist, i in zip(D[0], I[0]):
        pid = id_map[i]
        info = meta.loc[pid].to_dict()
        info.update(distance=float(dist))
        results.append(info)
    return {"results": results}

# ---- Endpoint: find similar by Name ----
@app.post("/similar-by-name")
async def similar_by_name(q: ByName):
    # simple substring match (case-insensitive)
    matches = meta[meta["title"].str.contains(q.product_name, case=False, na=False)]
    if matches.empty:
        raise HTTPException(404, "No product matches that name")
    # pick the first match
    pid = matches.index[0]
    # delegate to the ID-based search
    return await similar_by_id(ByID(product_id=pid, top_k=q.top_k))
