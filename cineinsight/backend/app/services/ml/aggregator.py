import logging
import re
from collections import Counter, defaultdict
from datetime import datetime

logger = logging.getLogger(__name__)


def compute_verdict(sentiments: list[dict]) -> tuple[str, float]:
    """
    Compute verdict and confidence from a list of sentiment predictions.
    Each dict has 'label' and 'score' keys.
    Returns (verdict_string, confidence_float).
    """
    if not sentiments:
        return "Mixed Reviews", 0.0

    total = len(sentiments)
    positive_count = sum(1 for s in sentiments if s["label"] == "positive")
    weighted_score = positive_count / total

    # Confidence = average model confidence
    avg_confidence = sum(s["score"] for s in sentiments) / total

    if weighted_score > 0.8:
        verdict = "Strongly Recommended"
    elif weighted_score > 0.6:
        verdict = "Worth Watching"
    elif weighted_score >= 0.4:
        verdict = "Mixed Reviews"
    elif weighted_score >= 0.2:
        verdict = "Below Average"
    else:
        verdict = "Skip It"

    return verdict, round(avg_confidence, 3)


def compute_overall_sentiment(sentiments: list[dict]) -> str:
    """Determine the dominant sentiment label."""
    if not sentiments:
        return "neutral"
    counts = Counter(s["label"] for s in sentiments)
    return counts.most_common(1)[0][0]


def aggregate_aspect_scores(aspect_results: list[dict]) -> dict:
    """Average aspect scores across all reviews."""
    if not aspect_results:
        return {"acting": 0.5, "plot": 0.5, "visuals": 0.5, "music": 0.5, "direction": 0.5}

    totals = defaultdict(float)
    for result in aspect_results:
        for aspect, score in result.items():
            totals[aspect] += score
    count = len(aspect_results)
    return {aspect: round(totals[aspect] / count, 3) for aspect in totals}


def compute_source_comparison(review_sentiments: list[dict]) -> dict:
    """Compute sentiment distribution per source."""
    result: dict[str, dict[str, int]] = {}
    for rs in review_sentiments:
        source = rs.get("source", "tmdb")
        label = rs.get("label", "neutral")
        if source not in result:
            result[source] = {"positive": 0, "negative": 0, "neutral": 0}
        if label in result[source]:
            result[source][label] += 1
    return result


def compute_sentiment_trend(review_sentiments: list[dict]) -> list[dict]:
    """Group sentiments by month for trend chart."""
    monthly: dict[str, dict[str, int]] = {}
    for rs in review_sentiments:
        date = rs.get("review_date")
        if not date:
            continue
        if isinstance(date, datetime):
            key = date.strftime("%Y-%m")
        elif isinstance(date, str):
            key = date[:7]
        else:
            continue
        if key not in monthly:
            monthly[key] = {"positive": 0, "negative": 0, "neutral": 0}
        label = rs.get("label", "neutral")
        if label in monthly[key]:
            monthly[key][label] += 1

    trend = [
        {"date": month, **counts}
        for month, counts in sorted(monthly.items())
    ]
    return trend


def extractive_summary(texts: list[str], sentiments: list[dict], top_k: int = 3) -> tuple[list[str], list[str]]:
    """
    Extract top positive and negative sentences using TF-IDF-like scoring.
    Simple implementation: score sentences by confidence × length factor.
    """
    positive_sentences = []
    negative_sentences = []

    for text, sent in zip(texts, sentiments):
        # Split into sentences
        sentences = re.split(r"[.!?]+", text)
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 30:
                continue
            score = sent["score"] * min(len(sentence.split()) / 20, 1.0)
            if sent["label"] == "positive":
                positive_sentences.append((sentence, score))
            elif sent["label"] == "negative":
                negative_sentences.append((sentence, score))

    positive_sentences.sort(key=lambda x: x[1], reverse=True)
    negative_sentences.sort(key=lambda x: x[1], reverse=True)

    return (
        [s for s, _ in positive_sentences[:top_k]],
        [s for s, _ in negative_sentences[:top_k]],
    )


def generate_word_cloud_data(texts: list[str], sentiments: list[dict]) -> tuple[list[dict], list[dict]]:
    """Generate word frequency data for positive and negative word clouds."""
    import nltk
    try:
        stopwords = set(nltk.corpus.stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        stopwords = set(nltk.corpus.stopwords.words("english"))

    # Add movie-generic stopwords
    stopwords.update({"movie", "film", "one", "like", "also", "would", "could", "really",
                       "much", "even", "get", "make", "see", "good", "bad", "great"})

    pos_words: Counter = Counter()
    neg_words: Counter = Counter()

    for text, sent in zip(texts, sentiments):
        words = re.findall(r"\b[a-z]{3,}\b", text.lower())
        filtered = [w for w in words if w not in stopwords]
        if sent["label"] == "positive":
            pos_words.update(filtered)
        elif sent["label"] == "negative":
            neg_words.update(filtered)

    return (
        [{"text": w, "value": c} for w, c in pos_words.most_common(50)],
        [{"text": w, "value": c} for w, c in neg_words.most_common(50)],
    )
