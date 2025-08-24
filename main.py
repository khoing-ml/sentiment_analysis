# main.py
from fastapi import FastAPI, HTTPException
import pandas as pd
from pydantic import BaseModel, field_validator

MAX_REVIEW_LEN = 5000
MAX_BATCH_SIZE = 100

class SentimentRequest(BaseModel):
    review: str

    @field_validator("review")
    @classmethod
    def review_not_empty(cls, v: str):
        if not v or not v.strip():
            raise ValueError("review must not be empty")
        if len(v) > MAX_REVIEW_LEN:
            raise ValueError(f"review too long (>{MAX_REVIEW_LEN} chars)")
        return v

class BatchSentimentRequest(BaseModel):
    reviews: list[str]

    @field_validator("reviews")
    @classmethod
    def reviews_valid(cls, v: list[str]):
        if not v:
            raise ValueError("reviews list must not be empty")
        if len(v) > MAX_BATCH_SIZE:
            raise ValueError(f"batch too large (>{MAX_BATCH_SIZE})")
        cleaned = []
        for idx, r in enumerate(v):
            if not r or not r.strip():
                raise ValueError(f"review at index {idx} empty")
            if len(r) > MAX_REVIEW_LEN:
                raise ValueError(f"review at index {idx} too long (>{MAX_REVIEW_LEN} chars)")
            cleaned.append(r)
        return cleaned

import mlflow
from pathlib import Path
from typing import Optional

# Initialize the app
app = FastAPI(title="Sentiment Analysis API")

MODEL_NAME = "sentiment-model"
_model = None  # lazy loaded

def load_model():
    global _model
    if _model is not None:
        return _model
    # Try registry first (latest version)
    try:
        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions(MODEL_NAME)
        if versions:
            latest_version = versions[0].version
            uri = f"models:/{MODEL_NAME}/{latest_version}"
            _model = mlflow.pyfunc.load_model(uri)
            return _model
    except Exception:
        pass
    # Fallback: use saved run id file
    run_id_path = Path("latest_run_id.txt")
    if run_id_path.exists():
        run_id = run_id_path.read_text().strip()
        # Support both artifact_path and name logging possibilities
        for candidate in [f"runs:/{run_id}/{MODEL_NAME}", f"runs:/{run_id}/sentiment-model"]:
            try:
                _model = mlflow.pyfunc.load_model(candidate)
                return _model
            except Exception:
                continue
    raise RuntimeError("Could not load sentiment model via registry or run id fallback.")

@app.get("/")
def read_root():
    return {"message": "Sentiment Analysis API is running!"}

@app.get("/health")
def health():
    try:
        load_model()
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/info")
def model_info():
    info = {}
    # Attempt registry metadata
    try:
        client = mlflow.tracking.MlflowClient()
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        if versions:
            # pick highest version numerically
            latest = sorted(versions, key=lambda v: int(v.version))[-1]
            info["registry_version"] = latest.version
            info["run_id"] = latest.run_id
    except Exception:
        pass
    # Fallback run id file
    run_id_path = Path("latest_run_id.txt")
    if "run_id" not in info and run_id_path.exists():
        info["run_id"] = run_id_path.read_text().strip()
    info["model_name"] = MODEL_NAME
    if not info.get("run_id"):
        raise HTTPException(status_code=404, detail="Model metadata not found")
    return info

@app.post("/predict")
def predict_sentiment(payload: SentimentRequest):
    """Predict sentiment for supplied review text."""
    try:
        mdl = load_model()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model not available: {e}")
    # Ensure input matches signature (expects DataFrame with 'review' column)
    input_df = pd.DataFrame({"review": [payload.review]})
    prediction = mdl.predict(input_df)
    # Some MLflow-wrapped models may return ndarray or list
    pred_val = prediction[0] if isinstance(prediction, (list, tuple)) else getattr(prediction, 'iloc', lambda idx: prediction[idx])(0)
    sentiment = "Positive" if pred_val == 1 else "Negative"
    return {"review": payload.review, "sentiment": sentiment}

@app.post("/predict/batch")
def predict_sentiment_batch(payload: BatchSentimentRequest):
    try:
        mdl = load_model()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model not available: {e}")
    df = pd.DataFrame({"review": payload.reviews})
    preds = mdl.predict(df)
    # Ensure we have one prediction per input row
    if getattr(preds, 'shape', None) and preds.shape == (1,) and len(df) > 1:
        # Force row-wise predictions as fallback
        row_outputs = []
        for text in df['review'].tolist():
            row_outputs.append(mdl.predict(pd.DataFrame({"review": [text]}))[0])
        preds_list = row_outputs
    else:
        preds_list = preds.tolist() if hasattr(preds, 'tolist') else list(preds)
    sentiments = ["Positive" if p == 1 else "Negative" for p in preds_list]
    return {"count": len(sentiments), "sentiments": sentiments}