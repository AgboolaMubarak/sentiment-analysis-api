import os
import joblib
import torch
from contextlib import asynccontextmanager
from typing import List, Dict

from fastapi import FastAPI, HTTPException
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from fastapi.responses import RedirectResponse
from .predictors import SentimentPredictor  

# Import the custom schemas
from .schemas import (
    SentimentRequest, 
    BatchSentimentRequest, 
    SentimentResponse, 
    BatchSentimentResponse,
    HealthCheckResponse
)

# Configuration 
MODEL_TYPE = os.getenv("MODEL_TYPE", "advanced")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Determine which path to pass to the predictor
if MODEL_TYPE == "baseline":
    MODEL_PATH = os.path.join(BASE_DIR, "models", "baseline_sentiment_model.joblib")
else:
    MODEL_PATH = os.path.join(BASE_DIR, "models", "distilbert_sentiment")



@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown events.
    Loads the predictor into the application state.
    """
    print(f"Loading {MODEL_TYPE} model...")
    try:
        # initialize the Predictor class and store it in app.state
        app.state.predictor = SentimentPredictor(MODEL_TYPE, MODEL_PATH)
        print(f"Model ({MODEL_TYPE}) loaded successfully.")
    except Exception as e:
        print(f"Critical error loading model: {e}")
        
    
    yield
    # Clean up on shutdown
    if hasattr(app.state, "predictor"):
        del app.state.predictor

app = FastAPI(
    title="Sentiment Analysis API",
    description="API for classifying text as positive, negative, or neutral.",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/", include_in_schema=False)
async def root():
    """Redirects the root URL to the API documentation."""
    return RedirectResponse(url="/docs")

@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Endpoint to check if the API and models are ready."""
    # Check if predictor exists in state
    if not hasattr(app.state, "predictor"):
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_version": MODEL_TYPE}

@app.post("/predict", response_model=SentimentResponse)
async def predict(request: SentimentRequest):
    """Classify a single text input."""
    label, confidence = app.state.predictor.predict(request.text)
    return SentimentResponse(
        text=request.text,
        label=label,
        confidence=confidence
    )

@app.post("/predict/batch", response_model=BatchSentimentResponse)
async def predict_batch(request: BatchSentimentRequest):
    """Classify multiple texts in a single request."""
    predictions = []
    for text in request.texts:
        label, confidence = app.state.predictor.predict(text)
        predictions.append(SentimentResponse(text=text, label=label, confidence=confidence))
    
    return BatchSentimentResponse(predictions=predictions)