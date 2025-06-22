from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans

# ---- Request model ----
class PerkRequest(BaseModel):
    user_id: int

# ---- FastAPI setup ----
app = FastAPI()

@app.get("/")
async def health():
    return {"status": "ok"}

# ---- Load & preprocess data at startup ----
# Adjust path if needed to point to your CSV
df = pd.read_csv(
    "amazon_return_data.csv",
    parse_dates=["purchase_date", "return_date"]
)

# 1) Compute per-user metrics
user = df.groupby("user_id").agg(
    total_spent=("purchase_amount", "sum"),
    total_returned=("return_amount", "sum"),
    orders=("order_id", "count"),
    returns=("return_status", lambda x: (x == "Returned").sum()),
    unique_categories=("product_category", pd.Series.nunique)
).reset_index()
user["return_ratio"] = user["total_returned"] / user["total_spent"]
user["return_ratio"] = user["return_ratio"].replace([np.inf, -np.inf], 0)
user.fillna(0, inplace=True)
user["net_spend_ratio"] = 1 - user["return_ratio"]

# 2) Advanced features
user["aov"] = user["total_spent"] / user["orders"]
np.random.seed(42)
user["exchange_count"] = np.random.binomial(user["returns"], 0.2)
user["exchange_ratio"] = np.where(
    user["returns"] > 0,
    user["exchange_count"] / user["returns"],
    0
)

snapshot = df["purchase_date"].max() + pd.Timedelta(days=1)
rfm = df.groupby("user_id").agg(
    recency=("purchase_date", lambda x: (snapshot - x.max()).days),
    frequency=("order_id", "count"),
    monetary=("purchase_amount", "sum")
).reset_index()
user = user.merge(rfm, on="user_id")

ret = df[df["return_status"] == "Returned"].copy()
ret["return_days"] = (ret["return_date"] - ret["purchase_date"]).dt.days
avg_ret = ret.groupby("user_id")["return_days"].mean().reset_index()
avg_ret["return_days"] = avg_ret["return_days"].fillna(avg_ret["return_days"].mean())
user = user.merge(avg_ret, on="user_id", how="left").rename(columns={"return_days": "avg_return_days"})

iso = IsolationForest(contamination=0.03, random_state=42)
user["fraud_risk"] = (
    iso.fit_predict(
        user[["return_ratio", "total_returned", "exchange_ratio", "unique_categories"]]
    ) == -1
).astype(int)

user["gamified_bonus"] = np.where(
    user["return_ratio"] < 0.05,
    np.random.randint(10, 100, size=len(user)),
    0
)
user["referral_bonus"] = np.random.poisson(1, size=len(user)) * 20
user["return_window_days"] = pd.cut(
    user["net_spend_ratio"],
    bins=[-1, 0.7, 0.85, 0.95, 1],
    labels=[15, 30, 45, 60]
).astype(int)

# 3) Loyalty score
scaler = MinMaxScaler()
user[["recency_s", "freq_s", "monet_s"]] = scaler.fit_transform(
    user[["recency", "frequency", "monetary"]]
)
user["loyalty_score"] = (
    0.4 * user["net_spend_ratio"]
    + 0.15 * user["freq_s"]
    + 0.1 * (user["unique_categories"] / user["unique_categories"].max())
    + 0.1 * user["exchange_ratio"]
    + 0.05 * (user["referral_bonus"] / user["referral_bonus"].max())
    + 0.05 * (user["gamified_bonus"] / user["gamified_bonus"].max())
    + 0.05 * (1 - user["fraud_risk"])
    + 0.1 * user["monet_s"]
)

# 4) Cluster → perk tier
kmeans = KMeans(n_clusters=4, random_state=42)
user["segment"] = kmeans.fit_predict(
    user[["loyalty_score", "net_spend_ratio", "freq_s", "unique_categories"]]
)
order = user.groupby("segment")["loyalty_score"].mean().sort_values(ascending=False).index
tiers = ["Diamond", "Platinum", "Gold", "Silver"]
perk_map = {seg: tiers[i] for i, seg in enumerate(order)}
user["perk"] = user["segment"].map(perk_map)

# 5) Points per dollar
user["points_per_dollar"] = 10 * (1 + user["loyalty_score"] - 0.5 * user["fraud_risk"])

# ---- Endpoint ----
@app.post("/assign-perks")
async def assign_perks(req: PerkRequest):
    uid = req.user_id
    row = user[user["user_id"] == uid]
    if row.empty:
        raise HTTPException(status_code=404, detail="User not found")
    return {
        "user_id": uid,
        "perk_tier": row["perk"].iloc[0],
        "points_per_dollar": float(row["points_per_dollar"].iloc[0]),
        "return_window_days": int(row["return_window_days"].iloc[0]),
        "fraud_risk": bool(row["fraud_risk"].iloc[0]),
    }
