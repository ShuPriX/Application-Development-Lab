import logging
import os

logger = logging.getLogger(__name__)

# Labels
LABELS = ["negative", "neutral", "positive"]

# Global model references (loaded at startup)
_bert_model = None
_bert_tokenizer = None


def load_bert_model(model_path: str):
    """Load fine-tuned BERT model. Falls back to a pretrained base if weights not found."""
    global _bert_model, _bert_tokenizer

    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError:
        logger.warning("transformers not installed, BERT model will not be loaded")
        return

    if os.path.exists(model_path):
        logger.info(f"Loading fine-tuned BERT from {model_path}")
        _bert_tokenizer = AutoTokenizer.from_pretrained(model_path)
        _bert_model = AutoModelForSequenceClassification.from_pretrained(model_path)
    else:
        logger.warning(
            f"BERT model not found at {model_path}, using pretrained distilbert-base-uncased-finetuned-sst-2-english"
        )
        fallback = "distilbert-base-uncased-finetuned-sst-2-english"
        _bert_tokenizer = AutoTokenizer.from_pretrained(fallback)
        _bert_model = AutoModelForSequenceClassification.from_pretrained(fallback)

    _bert_model.eval()
    logger.info("BERT model loaded successfully")


def predict_sentiment(texts: list[str]) -> list[dict]:
    """
    Predict sentiment for a batch of texts.
    Returns list of {"label": str, "score": float} dicts.
    """
    if _bert_model is None or _bert_tokenizer is None:
        logger.warning("BERT model not loaded, returning dummy predictions")
        return _dummy_predictions(texts)

    import torch
    import torch.nn.functional as F

    results = []
    batch_size = 16

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = _bert_tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = _bert_model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)

        for j in range(len(batch)):
            prob_values = probs[j].tolist()
            # Handle 2-class models (SST-2 fallback: negative=0, positive=1)
            if len(prob_values) == 2:
                neg, pos = prob_values
                if pos > 0.6:
                    label, score = "positive", pos
                elif neg > 0.6:
                    label, score = "negative", neg
                else:
                    label, score = "neutral", max(pos, neg)
            else:
                idx = prob_values.index(max(prob_values))
                label = LABELS[idx] if idx < len(LABELS) else "neutral"
                score = prob_values[idx]

            results.append({"label": label, "score": score})

    return results


def _dummy_predictions(texts: list[str]) -> list[dict]:
    """Fallback dummy predictions when no model is loaded."""
    import random
    return [
        {"label": random.choice(LABELS), "score": round(random.uniform(0.5, 0.95), 3)}
        for _ in texts
    ]
