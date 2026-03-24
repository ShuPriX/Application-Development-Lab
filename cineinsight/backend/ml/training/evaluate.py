"""
Model Evaluation Script

Generates classification reports, confusion matrices, and metrics
for both BERT and BiLSTM models.
"""

import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset


def evaluate_bert(model_path: str):
    print("=" * 60)
    print("BERT Sentiment Model Evaluation")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    dataset = load_dataset("imdb", split="test")

    all_preds = []
    all_labels = []

    batch_size = 32
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i : i + batch_size]
        inputs = tokenizer(
            batch["text"],
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=-1).tolist()

        all_preds.extend(preds)
        all_labels.extend(batch["label"])

        if i % (batch_size * 10) == 0:
            print(f"  Processed {i}/{len(dataset)}")

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=["Negative", "Positive"]))
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))


if __name__ == "__main__":
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else "../models/bert_sentiment"
    evaluate_bert(model_path)
