import os
import joblib
import torch
import numpy as np
from typing import Tuple, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification, PreTrainedTokenizerBase
from typing import Union, Optional

class SentimentPredictor:
    """
    Handles model loading and inference logic for both 
    Baseline (Scikit-Learn) and Advanced (DistilBERT) models.
    """

    def __init__(self, model_type: str, model_path: str):
        self.model_type = model_type
        self.model_path = model_path
        self.model: Any = None
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_path)
        
        # Label mapping for tweet_eval (0: negative, 1: neutral, 2: positive)
        self.labels_map = {0: "negative", 1: "neutral", 2: "positive"}
        
        self._load_model()

    def _load_model(self) -> None:
        """Loads the model artifacts from disk based on the specified model_type."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model path not found: {self.model_path}")

        if self.model_type == "baseline":
            self.model = joblib.load(self.model_path)
        else:
            # Load transformer model and tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.model.eval()  # Set to evaluation mode for inference

    def predict(self, text: str) -> Tuple[str, float]:
        """
        Runs inference on a single string.
        Returns: (label_string, confidence_score)
        """
        if self.model_type == "baseline":
            # scikit-learn pipeline expects a list/array of strings
            label_idx = int(self.model.predict([text])[0])
            # Extract probability for the predicted class
            probs = self.model.predict_proba([text])[0]
            confidence = float(probs[label_idx])
        else:
            # DistilBERT inference

            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                padding=True, 
                max_length=128
            )
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Apply Softmax to get probabilities
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                confidence = float(torch.max(probs))
                label_idx = int(torch.argmax(probs))

        label_str = self.labels_map.get(label_idx, "unknown")
        return label_str, confidence