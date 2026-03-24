"""
BERT fine-tuning on Modal with T4 GPU.
Trains on IMDB 50K dataset, saves weights to Modal Volume, then downloads locally.
"""

import modal

app = modal.App("cineinsight-bert-training")

# Persistent volume to store model weights
volume = modal.Volume.from_name("cineinsight-models", create_if_missing=True)

# GPU image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.47.1",
        "datasets==3.2.0",
        "scikit-learn==1.6.0",
        "accelerate>=0.26.0",
        "numpy<2",
    )
)


@app.function(
    image=image,
    gpu="T4",
    timeout=3600,
    volumes={"/models": volume},
)
def train_bert():
    import os
    import numpy as np
    import torch
    from datasets import load_dataset
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
    )
    from sklearn.metrics import accuracy_score, f1_score

    MODEL_NAME = "bert-base-uncased"
    OUTPUT_DIR = "/models/bert_sentiment"
    NUM_LABELS = 2
    EPOCHS = 3
    BATCH_SIZE = 16
    LR = 2e-5

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading IMDB dataset...")
    dataset = load_dataset("imdb")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(examples):
        return tokenizer(
            examples["text"], truncation=True, padding="max_length", max_length=512
        )

    print("Tokenizing dataset...")
    tokenized = dataset.map(tokenize, batched=True, num_proc=2)
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    print(f"Train: {len(tokenized['train'])}, Test: {len(tokenized['test'])}")

    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, predictions),
            "f1": f1_score(labels, predictions, average="weighted"),
        }

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=32,
        warmup_steps=500,
        weight_decay=0.01,
        learning_rate=LR,
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=True,
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        compute_metrics=compute_metrics,
    )

    print("Starting BERT training...")
    trainer.train()

    print("Evaluating...")
    results = trainer.evaluate()
    print(f"Final results: {results}")

    print(f"Saving model to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Commit volume so files persist
    volume.commit()

    print("BERT training complete!")
    return results


@app.local_entrypoint()
def main():
    print("Starting BERT training on Modal T4 GPU...")
    results = train_bert.remote()
    print(f"\nTraining complete! Results: {results}")
    print("Weights saved to Modal volume 'cineinsight-models'")
    print("Run modal_download.py to download weights locally.")
