import os
import joblib  #Importing necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# Import the data loader
from data_processing import load_processed_data

def train_baseline():
    """
    Trains a TF-IDF + Logistic Regression baseline model.
    """
    # 1. Load the cleaned data
    print("Loading and preprocessing data...")
    train_df, val_df, test_df = load_processed_data()
    
    X_train, y_train = train_df["text"], train_df["label"]
    X_test, y_test = test_df["text"], test_df["label"]

    # 2. Define the Pipeline
    # I included n-grams (1,2) to capture some context like "not good"
    baseline_model = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 2), 
            max_features=10000,
            sublinear_tf=True
        )),
        ('clf', LogisticRegression(solver='lbfgs', max_iter=1000))
    ])

    # 3. Train the model
    print("Training Baseline Model (Logistic Regression)...")
    baseline_model.fit(X_train, y_train)

    # 4. Evaluation
    y_pred = baseline_model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted') # 'weighted' handles potential imbalance
    cm = confusion_matrix(y_test, y_pred)

    print("\n--- Baseline Model Evaluation ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))

    # 5. Saving the model for inference
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_path = os.path.join(BASE_DIR, "models", "baseline_sentiment_model.joblib")
    os.makedirs("models", exist_ok=True)
    joblib.dump(baseline_model, model_path)
    print(f"\nModel saved to {model_path}")

if __name__ == "__main__":
    train_baseline()
    