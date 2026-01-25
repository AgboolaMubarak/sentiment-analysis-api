import os
import torch
import numpy as np
from typing import Dict, cast, Any
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer
)
from sklearn.metrics import accuracy_score, f1_score

# Import your custom data loader
from data_processing import load_processed_data

# Set paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_OUT_DIR = os.path.join(BASE_DIR, "models", "distilbert_sentiment")

def compute_metrics(eval_pred) -> Dict[str, float]:
    """
    Computes evaluation metrics during training with explicit type casting 
    to satisfy static type checkers.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
   
    acc = float(accuracy_score(labels, predictions))
    f1 = float(f1_score(labels, predictions, average="weighted"))
    
    return {
        "accuracy": acc,
        "f1": f1
    }
def train_advanced():
    """
    Fine-tunes DistilBERT using processed data from load_processed_data.
    Fulfills Requirement 1.b.ii and 1.c.
    """
    # 1. Load the cleaned data from your processing script
    print("Loading cleaned data...")
    train_df, val_df, test_df = load_processed_data()

    # Convert Pandas DataFrames to Hugging Face Datasets
    raw_datasets = DatasetDict({
        "train": Dataset.from_pandas(train_df),
        "validation": Dataset.from_pandas(val_df),
        "test": Dataset.from_pandas(test_df)
    })

    # 2. Setup Tokenizer and Model
    model_ckpt = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    
    # tweet_eval sentiment has 3 labels (0: Neg, 1: Neu, 2: Pos)
    model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=3)

    # 3. Tokenization
    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

    print("Tokenizing datasets...")
    tokenized_datasets = raw_datasets.map(tokenize_fn, batched=True)

    # 4. Training Configuration
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        logging_dir="./logs",
        push_to_hub=False,
        report_to="none" # Prevents external logging prompts during recruitment test
    )

    # 5. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        compute_metrics=compute_metrics,
    )

    # 6. Execution
    print("Starting DistilBERT training (this may take a while on CPU)...")
    trainer.train()

  
    
    # 7. Final Evaluation on Test Set
    print("\nEvaluating on test dataset...")
    trainer.eval_dataset = tokenized_datasets["test"]
    test_results = trainer.evaluate()

    print(f"\n--- Advanced Model Test Results ---")
    print(f"Accuracy: {test_results['eval_accuracy']:.4f}")
    print(f"F1-Score: {test_results['eval_f1']:.4f}")

    # 8. Save Artifacts
    os.makedirs(MODEL_OUT_DIR, exist_ok=True)
    model.save_pretrained(MODEL_OUT_DIR)
    tokenizer.save_pretrained(MODEL_OUT_DIR)
    print(f"\nModel and tokenizer saved to {MODEL_OUT_DIR}")

if __name__ == "__main__":
    train_advanced()