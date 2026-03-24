"""
BERT Fine-tuning Script for Sentiment Classification

Run this on Google Colab with T4 GPU:
1. Upload this script or copy into a Colab notebook
2. Install: pip install transformers datasets torch scikit-learn
3. Run training
4. Download saved model weights to backend/ml/models/bert_sentiment/

Dataset: IMDB 50K (25K train + 25K test)
Model: bert-base-uncased → 3-class (positive/negative/neutral)
"""

import os
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, f1_score, classification_report


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted"),
    }


def main():
    MODEL_NAME = "bert-base-uncased"
    OUTPUT_DIR = "../models/bert_sentiment"
    NUM_LABELS = 2  # IMDB is binary; extend to 3 with neutral threshold
    EPOCHS = 3
    BATCH_SIZE = 16
    LR = 2e-5

    print("Loading dataset...")
    dataset = load_dataset("imdb")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

    print("Tokenizing...")
    tokenized = dataset.map(tokenize, batched=True)
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_steps=500,
        weight_decay=0.01,
        learning_rate=LR,
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()

    print("Evaluating...")
    results = trainer.evaluate()
    print(f"Results: {results}")

    print(f"Saving model to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("Done! Download the model directory for local inference.")


if __name__ == "__main__":
    main()
