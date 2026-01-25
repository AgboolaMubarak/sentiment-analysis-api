import pandas as pd
import re
import os
from typing import Tuple

def clean_tweet(text: str) -> str:
    """
    Cleans tweet text by removing handles, URLs, and special characters.
    Required for production-grade NLP pipelines.
    """
    # Convert to lowercase
    text = text.lower()
    # Remove twitter handles (contain @)
    text = re.sub(r"@[^\s]+", "", text)
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    # Remove special characters and numbers 
    text = re.sub(r"[^a-z\s]", "", text)
    # Remove extra whitespace
    text = " ".join(text.split())
    return text

def load_processed_data(data_dir: str = "data/raw") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads the CSVs created and apply cleaning. 
    Returns: (train_df, val_df, test_df)
    """
    splits = []
    for name in ["train", "validation", "test"]:
        path = os.path.join(data_dir, f"{name}.csv")
        df = pd.read_csv(path)
        
        # Apply cleaning to the text column
        df["text"] = df["text"].apply(clean_tweet)
        
        # Drop any rows that became empty after cleaning
        df = df[df["text"].str.strip() != ""]
        splits.append(df)
        
    return tuple(splits)

if __name__ == "__main__":
    # Verification
    train, val, test = load_processed_data()
    print(f"Cleaned Train Size: {len(train)}")
    print(f"Sample Tweet: {train['text'].iloc[0]}")