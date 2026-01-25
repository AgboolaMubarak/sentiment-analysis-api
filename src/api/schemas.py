from pydantic import BaseModel, Field
from typing import List

class SentimentRequest(BaseModel):
    """Schema for a single text prediction request."""
    text: str = Field(
        ..., 
        min_length=1, 
        description="The text string to be classified for sentiment.",
        examples=["I really enjoyed this movie!"]
    )

class BatchSentimentRequest(BaseModel):
    """Schema for multiple text prediction requests."""
    texts: List[str] = Field(
        ..., 
        min_length=1, 
        description="A list of text strings to be classified.",
        examples=["The service was great.", "I am very disappointed."]
    )

class SentimentResponse(BaseModel):
    """Schema for a single prediction output."""
    text: str
    label: str
    confidence: float

class BatchSentimentResponse(BaseModel):
    """Schema for multiple prediction outputs."""
    predictions: List[SentimentResponse]

class HealthCheckResponse(BaseModel):
    """Schema for the API health status."""
    status: str = Field(..., description="Indicates if the API is 'healthy'.")
    model_version: str = Field(..., description="The version or type of the model currently loaded.")