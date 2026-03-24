import logging
import os
import re

logger = logging.getLogger(__name__)

ASPECTS = ["acting", "plot", "visuals", "music", "direction"]

# Aspect keyword indicators (used as fallback when model not available)
ASPECT_KEYWORDS = {
    "acting": ["act", "perform", "character", "cast", "portray", "role", "actor", "actress"],
    "plot": ["story", "plot", "script", "narrative", "writing", "twist", "storyline", "pacing"],
    "visuals": ["visual", "cinemat", "scene", "shot", "effect", "cgi", "look", "beautiful", "stun"],
    "music": ["music", "score", "sound", "soundtrack", "song", "audio", "compose"],
    "direction": ["direct", "vision", "helm", "filmmaker", "auteur", "craft"],
}


def _create_bilstm_class():
    """Lazy-create the BiLSTM class to avoid importing torch at module level."""
    import torch
    import torch.nn as nn

    class BiLSTMAspectModel(nn.Module):
        """BiLSTM with attention for aspect-based sentiment analysis."""

        def __init__(self, vocab_size: int, embed_dim: int = 300, hidden_dim: int = 256,
                     num_layers: int = 2, num_aspects: int = 5, num_classes: int = 3):
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

    return BiLSTMAspectModel


# Global model reference
_bilstm_model = None
_vocab = None


def load_bilstm_model(model_path: str):
    """Load trained BiLSTM model. Falls back to keyword-based scoring if not found."""
    global _bilstm_model, _vocab

    try:
        import torch
    except ImportError:
        logger.warning("torch not installed, BiLSTM model will not be loaded")
        return

    weights_path = os.path.join(model_path, "bilstm_weights.pt")
    vocab_path = os.path.join(model_path, "vocab.pt")

    if os.path.exists(weights_path) and os.path.exists(vocab_path):
        logger.info(f"Loading BiLSTM model from {model_path}")
        _vocab = torch.load(vocab_path, map_location="cpu", weights_only=True)
        BiLSTMAspectModel = _create_bilstm_class()
        _bilstm_model = BiLSTMAspectModel(vocab_size=len(_vocab))
        _bilstm_model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
        _bilstm_model.eval()
        logger.info("BiLSTM model loaded successfully")
    else:
        logger.warning(f"BiLSTM model not found at {model_path}, using keyword-based fallback")


def predict_aspects(texts: list[str]) -> list[dict]:
    """
    Predict aspect scores for a batch of texts.
    Returns list of {"acting": float, "plot": float, ...} dicts (0-1 scale).
    """
    if _bilstm_model is not None and _vocab is not None:
        return _model_predict(texts)
    return _keyword_predict(texts)


def _model_predict(texts: list[str]) -> list[dict]:
    """Run the BiLSTM model for predictions."""
    import torch

    results = []
    for text in texts:
        tokens = text.lower().split()[:200]
        indices = [_vocab.get(t, 1) for t in tokens]  # 1 = UNK
        input_tensor = torch.tensor([indices])

        with torch.no_grad():
            outputs = _bilstm_model(input_tensor)

        aspect_scores = {}
        for i, aspect in enumerate(ASPECTS):
            probs = torch.softmax(outputs[i], dim=-1)[0]
            # Weighted score: neg=0, neutral=0.5, pos=1.0
            score = probs[0] * 0.0 + probs[1] * 0.5 + probs[2] * 1.0
            aspect_scores[aspect] = round(score.item(), 3)
        results.append(aspect_scores)
    return results


def _keyword_predict(texts: list[str]) -> list[dict]:
    """Keyword-based fallback for aspect scoring."""
    results = []
    for text in texts:
        text_lower = text.lower()
        scores = {}
        for aspect, keywords in ASPECT_KEYWORDS.items():
            mentions = sum(1 for kw in keywords if re.search(rf"\b{kw}", text_lower))
            # Base score + keyword boost, clamped to [0, 1]
            score = min(0.5 + mentions * 0.1, 1.0)
            scores[aspect] = round(score, 3)
        results.append(scores)
    return results
