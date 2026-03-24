"""
BiLSTM with Attention Training Script for Aspect-Based Sentiment Analysis

Run this on Google Colab:
1. Upload this script or copy into a Colab notebook
2. Install: pip install torch numpy scikit-learn
3. Run training
4. Download bilstm_weights.pt and vocab.pt to backend/ml/models/bilstm_aspect/

Architecture: Embedding (GloVe 300d) → BiLSTM (2 layers, 256 hidden) → Attention → 5 aspect heads → Softmax
Aspects: Acting, Plot, Visuals, Music, Direction
"""

import os
import json
import random
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np


ASPECTS = ["acting", "plot", "visuals", "music", "direction"]
NUM_CLASSES = 3  # negative, neutral, positive
EMBED_DIM = 300
HIDDEN_DIM = 256
NUM_LAYERS = 2
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-3
MAX_LEN = 200


class BiLSTMAspectModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM,
                 num_layers=NUM_LAYERS, num_aspects=5, num_classes=NUM_CLASSES):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True, dropout=0.3,
        )
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.aspect_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, num_classes),
            )
            for _ in range(num_aspects)
        ])

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
        tokens = self.texts[idx].lower().split()[:self.max_len]
        indices = [self.vocab.get(t, 1) for t in tokens]
        # Pad
        if len(indices) < self.max_len:
            indices += [0] * (self.max_len - len(indices))
        return torch.tensor(indices), torch.tensor(self.labels[idx])


def build_vocab(texts, min_freq=2):
    counter = Counter()
    for text in texts:
        counter.update(text.lower().split())
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, count in counter.items():
        if count >= min_freq:
            vocab[word] = len(vocab)
    return vocab


def load_glove_embeddings(vocab, glove_path="glove.6B.300d.txt"):
    """Load GloVe embeddings for words in vocab."""
    embeddings = np.random.randn(len(vocab), EMBED_DIM) * 0.01
    embeddings[0] = 0  # PAD
    found = 0

    if not os.path.exists(glove_path):
        print(f"GloVe file not found at {glove_path}, using random embeddings")
        return torch.FloatTensor(embeddings)

    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            if word in vocab:
                embeddings[vocab[word]] = np.array([float(x) for x in parts[1:]])
                found += 1

    print(f"Loaded {found}/{len(vocab)} GloVe embeddings")
    return torch.FloatTensor(embeddings)


def main():
    OUTPUT_DIR = "../models/bilstm_aspect"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # NOTE: Replace with actual training data loading
    # Expected format: list of (text, [label_acting, label_plot, label_visuals, label_music, label_direction])
    # Labels: 0=negative, 1=neutral, 2=positive
    print("NOTE: You need to prepare aspect-labeled training data.")
    print("This script provides the model architecture and training loop.")
    print("Use SemEval ABSA datasets or semi-auto label IMDB reviews.")

    # Demo with synthetic data
    texts = [f"This movie has great acting and story {i}" for i in range(1000)]
    labels = [[random.randint(0, 2) for _ in ASPECTS] for _ in range(1000)]

    # Build vocab
    vocab = build_vocab(texts)
    print(f"Vocabulary size: {len(vocab)}")

    # Split
    split = int(0.8 * len(texts))
    train_dataset = AspectDataset(texts[:split], labels[:split], vocab)
    val_dataset = AspectDataset(texts[split:], labels[split:], vocab)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Model
    model = BiLSTMAspectModel(vocab_size=len(vocab))

    # Load GloVe if available
    glove_embeddings = load_glove_embeddings(vocab)
    model.embedding.weight.data.copy_(glove_embeddings)
    model.embedding.weight.requires_grad = True  # Fine-tune embeddings

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Training on {device}")

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
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = sum(
                    criterion(outputs[i], batch_y[:, i]) for i in range(len(ASPECTS))
                )
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "bilstm_weights.pt"))
            torch.save(vocab, os.path.join(OUTPUT_DIR, "vocab.pt"))
            print("  Saved best model")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    print(f"Training complete! Model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
