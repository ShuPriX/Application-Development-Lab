import re
import logging

logger = logging.getLogger(__name__)


def clean_review(text: str) -> str:
    """Clean a single review text for ML inference."""
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Remove URLs
    text = re.sub(r"https?://\S+", " ", text)
    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Truncate very long reviews (BERT max 512 tokens)
    words = text.split()
    if len(words) > 400:
        text = " ".join(words[:400])
    return text


def batch_clean(texts: list[str]) -> list[str]:
    """Clean a batch of review texts."""
    return [clean_review(t) for t in texts]
