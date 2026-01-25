from datasets import load_dataset
import pandas as pd
import os

DATA_DIR = "data/raw"

def main():
    # 1. Load dataset
    dataset = load_dataset("tweet_eval", "sentiment")

    # 2. Create directory if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)

    # 3. Save splits
    for split in ["train", "validation", "test"]:
        df = pd.DataFrame({
            "text": dataset[split]["text"],
            "label": dataset[split]["label"]
        })

        file_path = os.path.join(DATA_DIR, f"{split}.csv")
        df.to_csv(file_path, index=False)

        print(f"Saved {split} to {file_path}")

if __name__ == "__main__":
    main()


    