"""
BiLSTM aspect-based sentiment training on Modal with T4 GPU.
Uses Claude Haiku to generate high-quality aspect labels from IMDB reviews.
"""

import modal
import os

app = modal.App("cineinsight-bilstm-training")

volume = modal.Volume.from_name("cineinsight-models", create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.5.1",
    "datasets==3.2.0",
    "anthropic>=0.40.0",
    "numpy<2",
)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")


@app.function(
    image=image,
    gpu="T4",
    timeout=7200,
    volumes={"/models": volume},
    secrets=[modal.Secret.from_dict({"ANTHROPIC_API_KEY": ANTHROPIC_API_KEY})],
)
def train_bilstm():
    import json
    import random
    import time
    from collections import Counter
    from pathlib import Path

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import numpy as np
    from datasets import load_dataset

    ASPECTS = ["acting", "plot", "visuals", "music", "direction"]
    NUM_CLASSES = 3  # 0=negative, 1=neutral, 2=positive
    EMBED_DIM = 300
    HIDDEN_DIM = 256
    NUM_LAYERS = 2
    BATCH_SIZE = 64
    EPOCHS = 20
    LR = 1e-3
    MAX_LEN = 200
    OUTPUT_DIR = "/models/bilstm_aspect"
    LABEL_CACHE = "/models/bilstm_aspect/labeled_data.json"
    NUM_REVIEWS_TO_LABEL = 10000

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- Step 1: Label reviews with Claude Haiku ----

    def label_with_claude(reviews: list[dict]) -> list[dict]:
        """Label reviews using Claude Haiku in batches of 5."""
        import anthropic

        client = anthropic.Anthropic()
        labeled = []
        batch_size = 5
        total_batches = (len(reviews) + batch_size - 1) // batch_size

        for batch_idx in range(0, len(reviews), batch_size):
            batch = reviews[batch_idx : batch_idx + batch_size]
            batch_num = batch_idx // batch_size + 1

            if batch_num % 50 == 0:
                print(f"  Labeling batch {batch_num}/{total_batches}...")

            reviews_text = ""
            for i, rev in enumerate(batch):
                text = rev["text"][:800]  # Truncate to save tokens
                reviews_text += f"\n[Review {i + 1}]: {text}\n"

            prompt = f"""Analyze these movie reviews and rate each on 5 aspects.
For each review, output a JSON object with scores for: acting, plot, visuals, music, direction.
Each score should be: 0 (negative/bad), 1 (neutral/not mentioned), or 2 (positive/good).

{reviews_text}

Respond with ONLY a JSON array of objects, one per review. Example:
[{{"acting": 2, "plot": 1, "visuals": 2, "music": 1, "direction": 0}}]"""

            try:
                response = client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=300,
                    messages=[{"role": "user", "content": prompt}],
                )
                text_resp = response.content[0].text.strip()

                # Parse JSON - handle potential markdown wrapping
                if text_resp.startswith("```"):
                    text_resp = text_resp.split("```")[1]
                    if text_resp.startswith("json"):
                        text_resp = text_resp[4:]
                    text_resp = text_resp.strip()

                scores_list = json.loads(text_resp)

                for rev, scores in zip(batch, scores_list):
                    label = [
                        scores.get("acting", 1),
                        scores.get("plot", 1),
                        scores.get("visuals", 1),
                        scores.get("music", 1),
                        scores.get("direction", 1),
                    ]
                    # Clamp to valid range
                    label = [max(0, min(2, int(v))) for v in label]
                    labeled.append({"text": rev["text"], "labels": label})

            except Exception as e:
                # Fallback: use overall sentiment
                for rev in batch:
                    overall = 0 if rev["sentiment"] == 0 else 2
                    labeled.append(
                        {
                            "text": rev["text"],
                            "labels": [overall, overall, 1, 1, 1],
                        }
                    )
                if batch_num % 100 == 0:
                    print(f"  Batch {batch_num} failed ({e}), used fallback")

            # Rate limit: ~50 RPM for Haiku
            if batch_num % 40 == 0:
                time.sleep(2)

        return labeled

    # Check if we already have labeled data cached
    if os.path.exists(LABEL_CACHE):
        print(f"Loading cached labeled data from {LABEL_CACHE}...")
        with open(LABEL_CACHE) as f:
            labeled_data = json.load(f)
        print(f"Loaded {len(labeled_data)} labeled reviews from cache")
    else:
        print("Loading IMDB dataset...")
        dataset = load_dataset("imdb")

        # Sample reviews for labeling
        all_texts = dataset["train"]["text"]
        all_sentiments = dataset["train"]["label"]
        indices = list(range(len(all_texts)))
        random.seed(42)
        random.shuffle(indices)
        indices = indices[:NUM_REVIEWS_TO_LABEL]

        reviews_to_label = [
            {"text": all_texts[i], "sentiment": all_sentiments[i]} for i in indices
        ]

        print(f"Labeling {len(reviews_to_label)} reviews with Claude Haiku...")
        labeled_data = label_with_claude(reviews_to_label)

        # Cache labeled data
        print(f"Caching {len(labeled_data)} labeled reviews...")
        with open(LABEL_CACHE, "w") as f:
            json.dump(labeled_data, f)
        volume.commit()

    # ---- Step 2: Train BiLSTM ----

    print(f"\nTraining BiLSTM on {len(labeled_data)} labeled reviews...")

    # Check label distribution
    for i, aspect in enumerate(ASPECTS):
        counts = Counter(d["labels"][i] for d in labeled_data)
        print(
            f"  {aspect}: neg={counts.get(0, 0)}, neutral={counts.get(1, 0)}, pos={counts.get(2, 0)}"
        )

    class BiLSTMAspectModel(nn.Module):
        def __init__(
            self,
            vocab_size,
            embed_dim=EMBED_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            num_aspects=5,
            num_classes=NUM_CLASSES,
        ):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.lstm = nn.LSTM(
                embed_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=0.3,
            )
            self.attention = nn.Linear(hidden_dim * 2, 1)
            self.aspect_heads = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(hidden_dim * 2, 128),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(128, num_classes),
                    )
                    for _ in range(num_aspects)
                ]
            )

        def forward(self, x):
            embedded = self.embedding(x)
            lstm_out, _ = self.lstm(embedded)
            attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
            context = (lstm_out * attn_weights).sum(dim=1)
            return [head(context) for head in self.aspect_heads]

    class AspectDataset(Dataset):
        def __init__(self, texts, labels, vocab, max_len=MAX_LEN):
            self.texts = texts
            self.labels = labels
            self.vocab = vocab
            self.max_len = max_len

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            tokens = self.texts[idx].lower().split()[: self.max_len]
            indices = [self.vocab.get(t, 1) for t in tokens]
            if len(indices) < self.max_len:
                indices += [0] * (self.max_len - len(indices))
            return torch.tensor(indices), torch.tensor(self.labels[idx])

    # Build vocab
    print("Building vocabulary...")
    texts = [d["text"] for d in labeled_data]
    labels = [d["labels"] for d in labeled_data]

    word_counter = Counter()
    for text in texts:
        word_counter.update(text.lower().split())
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, count in word_counter.items():
        if count >= 3:
            vocab[word] = len(vocab)
    print(f"Vocabulary size: {len(vocab)}")

    # Split 90/10
    random.seed(42)
    indices = list(range(len(texts)))
    random.shuffle(indices)
    split = int(0.9 * len(indices))

    train_texts = [texts[i] for i in indices[:split]]
    train_labels = [labels[i] for i in indices[:split]]
    val_texts = [texts[i] for i in indices[split:]]
    val_labels = [labels[i] for i in indices[split:]]

    train_dataset = AspectDataset(train_texts, train_labels, vocab)
    val_dataset = AspectDataset(val_texts, val_labels, vocab)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=2)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Build model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = BiLSTMAspectModel(vocab_size=len(vocab)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    # Training loop
    best_val_loss = float("inf")
    patience = 5
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()

            outputs = model(batch_x)
            loss = sum(
                criterion(outputs[i], batch_y[:, i]) for i in range(len(ASPECTS))
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        correct = [0] * len(ASPECTS)
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = sum(
                    criterion(outputs[i], batch_y[:, i]) for i in range(len(ASPECTS))
                )
                val_loss += loss.item()
                total += batch_y.size(0)
                for i in range(len(ASPECTS)):
                    preds = outputs[i].argmax(dim=-1)
                    correct[i] += (preds == batch_y[:, i]).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        accuracies = [c / total for c in correct]
        avg_acc = sum(accuracies) / len(accuracies)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} - "
            f"Train: {avg_train_loss:.4f} - Val: {avg_val_loss:.4f} - "
            f"Acc: {avg_acc:.3f} ({', '.join(f'{a}={acc:.3f}' for a, acc in zip(ASPECTS, accuracies))})"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(
                model.state_dict(), os.path.join(OUTPUT_DIR, "bilstm_weights.pt")
            )
            torch.save(vocab, os.path.join(OUTPUT_DIR, "vocab.pt"))
            print("  -> Saved best model")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

    volume.commit()
    print("BiLSTM training complete!")
    return {
        "best_val_loss": best_val_loss,
        "final_accuracies": dict(zip(ASPECTS, accuracies)),
    }


@app.local_entrypoint()
def main():
    print("Starting BiLSTM training on Modal T4 GPU...")
    print("Step 1: Label 10K IMDB reviews with Claude Haiku")
    print("Step 2: Train BiLSTM on labeled data")
    print()
    results = train_bilstm.remote()
    print(f"\nTraining complete! Results: {results}")
    print("Weights saved to Modal volume 'cineinsight-models'")
    print("Run modal_download.py to download weights locally.")
