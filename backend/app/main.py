# backend/app/main.py
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI(title="SuperHack Backend")

# -------------------------
# Paths (robust)
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "..", "ml", "model.joblib")
MARKET_PATH = os.path.join(BASE_DIR, "..", "..", "data", "marketplace_offers.csv")

# Load model
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model at {MODEL_PATH}: {e}")

# -------------------------
# Pydantic Request Models
# -------------------------
class PredictRequest(BaseModel):
    lag1: float
    lag24: float

# -------------------------
# Endpoints
# -------------------------
@app.get("/")
def root():
    return {"message": "Welcome to SuperHack backend!"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    X = [[req.lag1, req.lag24]]
    try:
        pred = model.predict(X)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")
    return {"prediction": float(pred)}

@app.get("/market")
def market():
    if not os.path.exists(MARKET_PATH):
        raise HTTPException(status_code=404, detail="Marketplace CSV not found")
    df = pd.read_csv(MARKET_PATH)
    return df.to_dict(orient="records")

@app.post("/bid")
def bid(
    offer_id: str = Query(..., description="ID of the offer"),
    bid_price: float = Query(..., description="Your bid price")
):
    if not os.path.exists(MARKET_PATH):
        raise HTTPException(status_code=404, detail="Marketplace CSV not found")
    
    df = pd.read_csv(MARKET_PATH)
    
    if offer_id not in df["offer_id"].values:
        raise HTTPException(status_code=404, detail="Offer not found")
    
    row = df[df["offer_id"] == offer_id].iloc[0]
    accepted = bid_price >= row["price_per_hour"] * 0.9
    
    return {
        "offer_id": offer_id,
        "bid_price": bid_price,
        "accepted": bool(accepted)
    }

# -------------------------
# Hybrid endpoint
# -------------------------
@app.get("/hybridaction/zybTrackerStatisticsAction")
def get_statistics(data: str = "{}"):
    # Your logic here
    return {"status": "ok", "data": data}
