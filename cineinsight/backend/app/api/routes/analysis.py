import asyncio
import logging

import httpx
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.session import get_db
from app.models.database import Movie, Review, AnalysisResult, ReviewSentiment
from app.models.schemas import AnalysisResponse, ReviewSentimentResponse
from app.services import tmdb
from app.services.cache import (
    get_or_create_movie,
    get_cached_analysis,
    store_reviews,
    save_analysis,
)
from app.services.scraper.imdb import IMDBScraper
from app.services.scraper.rottentomatoes import RottenTomatoesScraper
from app.services.ml.preprocessor import batch_clean
from app.services.ml.sentiment import predict_sentiment
from app.services.ml.aspects import predict_aspects
from app.services.ml.aggregator import (
    compute_verdict,
    compute_overall_sentiment,
    aggregate_aspect_scores,
    compute_source_comparison,
    compute_sentiment_trend,
    extractive_summary,
    generate_word_cloud_data,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/movies", tags=["analysis"])

# Track active WebSocket connections for progress updates
_ws_connections: dict[int, list[WebSocket]] = {}


async def _send_progress(tmdb_id: int, **data):
    """Send progress update to connected WebSocket clients."""
    for ws in _ws_connections.get(tmdb_id, []):
        try:
            await ws.send_json(data)
        except Exception:
            pass


def _build_analysis_response(analysis: AnalysisResult) -> dict:
    """Build the full analysis response dict with review content."""
    review_sentiments = []
    for rs in analysis.review_sentiments:
        review_sentiments.append(ReviewSentimentResponse(
            id=rs.id,
            review_id=rs.review_id,
            content=rs.review.content if rs.review else "",
            author=rs.review.author if rs.review else "",
            source=rs.review.source if rs.review else "",
            sentiment_label=rs.sentiment_label,
            sentiment_score=rs.sentiment_score,
            aspect_scores=rs.aspect_scores,
        ))

    return AnalysisResponse(
        id=analysis.id,
        movie_id=analysis.movie_id,
        movie=analysis.movie,
        overall_sentiment=analysis.overall_sentiment,
        confidence=analysis.confidence,
        verdict=analysis.verdict,
        aspect_scores=analysis.aspect_scores,
        source_comparison=analysis.source_comparison or {"tmdb": {}, "imdb": {}, "rottentomatoes": {}},
        sentiment_trend=analysis.sentiment_trend or [],
        positive_summary=analysis.positive_summary or [],
        negative_summary=analysis.negative_summary or [],
        word_cloud_positive=analysis.word_cloud_positive or [],
        word_cloud_negative=analysis.word_cloud_negative or [],
        review_sentiments=review_sentiments,
        review_count=analysis.review_count,
        analyzed_at=analysis.analyzed_at,
        updated_at=analysis.updated_at,
    )


@router.get("/{tmdb_id}/analysis")
async def get_analysis(tmdb_id: int, db: AsyncSession = Depends(get_db)):
    """Get cached analysis for a movie. Returns 404 if not analyzed yet."""
    stmt = select(Movie).where(Movie.tmdb_id == tmdb_id)
    result = await db.execute(stmt)
    movie = result.scalar_one_or_none()

    if not movie:
        raise HTTPException(status_code=404, detail="Movie not found")

    analysis = await get_cached_analysis(db, movie.id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    return _build_analysis_response(analysis)


@router.post("/{tmdb_id}/analyze")
async def trigger_analysis(tmdb_id: int, db: AsyncSession = Depends(get_db)):
    """Trigger fresh scraping + analysis for a movie."""

    # ── Step 1: Metadata ─────────────────────────────────────────────
    await _send_progress(
        tmdb_id, type="log", stage="metadata", progress=2,
        message="Connecting to TMDB API...",
    )
    try:
        movie_data = await tmdb.get_movie_details(tmdb_id)
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Could not reach TMDB API, please try again")
    movie = await get_or_create_movie(db, movie_data)

    await _send_progress(
        tmdb_id, type="log", stage="metadata", progress=5,
        message=f"Movie identified: {movie.title} ({movie.year})",
    )

    # ── Step 2: Scrape reviews from all sources in parallel ──────────
    await _send_progress(
        tmdb_id, type="log", stage="scraping", progress=8,
        message="Starting review scraping from TMDB, IMDB, Rotten Tomatoes...",
    )

    imdb_scraper = IMDBScraper()
    rt_scraper = RottenTomatoesScraper()

    async def _safe_tmdb_reviews():
        try:
            reviews = await tmdb.get_movie_reviews(tmdb_id)
            await _send_progress(
                tmdb_id, type="source", stage="scraping", progress=20,
                source="tmdb", count=len(reviews), status="done",
                message=f"TMDB API: fetched {len(reviews)} user reviews",
            )
            return reviews
        except httpx.ConnectError:
            await _send_progress(
                tmdb_id, type="source", stage="scraping", progress=20,
                source="tmdb", count=0, status="error",
                message="TMDB reviews: connection failed",
            )
            return []

    async def _imdb_with_progress():
        reviews = await imdb_scraper.scrape(movie.title, movie.year)
        await _send_progress(
            tmdb_id, type="source", stage="scraping", progress=30,
            source="imdb", count=len(reviews),
            status="done" if reviews else "empty",
            message=f"IMDB: scraped {len(reviews)} reviews" if reviews
            else "IMDB: no reviews found",
        )
        return reviews

    async def _rt_with_progress():
        reviews = await rt_scraper.scrape(movie.title, movie.year)
        await _send_progress(
            tmdb_id, type="source", stage="scraping", progress=35,
            source="rottentomatoes", count=len(reviews),
            status="done" if reviews else "skipped",
            message=f"Rotten Tomatoes: scraped {len(reviews)} reviews" if reviews
            else "Rotten Tomatoes: skipped (JS-rendered)",
        )
        return reviews

    tmdb_reviews, imdb_reviews, rt_reviews = await asyncio.gather(
        _safe_tmdb_reviews(),
        _imdb_with_progress(),
        _rt_with_progress(),
    )

    n_total = len(tmdb_reviews) + len(imdb_reviews) + len(rt_reviews)
    logger.info(f"[DEBUG] Scraped counts — TMDB: {len(tmdb_reviews)}, IMDB: {len(imdb_reviews)}, RT: {len(rt_reviews)}")
    await _send_progress(
        tmdb_id, type="log", stage="scraping", progress=40,
        message=f"Scraping complete — {n_total} total reviews collected",
    )

    # Send individual review previews
    all_scraped = list(tmdb_reviews) + [
        {"source": r.source, "author": r.author, "content": r.content,
         "rating": r.rating, "review_date": r.review_date}
        for r in imdb_reviews + rt_reviews
    ]
    logger.info(f"[DEBUG] all_scraped length: {len(all_scraped)}")

    if not all_scraped:
        raise HTTPException(status_code=422, detail="No reviews found for this movie")

    # Send a batch of review previews (first 8)
    preview_reviews = []
    for r in all_scraped[:8]:
        content = r["content"]
        preview_reviews.append({
            "source": r["source"],
            "author": r.get("author", "Anonymous"),
            "snippet": (content[:120] + "...") if len(content) > 120 else content,
        })
    await _send_progress(
        tmdb_id, type="reviews", stage="scraping", progress=42,
        reviews=preview_reviews,
        message=f"Previewing {len(preview_reviews)} of {len(all_scraped)} reviews",
    )

    # ── Step 3: Store reviews ────────────────────────────────────────
    await _send_progress(
        tmdb_id, type="log", stage="storing", progress=45,
        message=f"Deduplicating and storing {len(all_scraped)} reviews in database...",
    )
    stored = await store_reviews(db, movie.id, all_scraped)
    logger.info(f"[DEBUG] store_reviews returned {len(stored)} newly stored reviews")

    stmt = select(Review).where(Review.movie_id == movie.id)
    result = await db.execute(stmt)
    all_reviews = list(result.scalars().all())
    logger.info(f"[DEBUG] DB query returned {len(all_reviews)} reviews for movie_id={movie.id}")

    await _send_progress(
        tmdb_id, type="log", stage="storing", progress=50,
        message=f"{len(all_reviews)} unique reviews ready for analysis",
    )

    # ── Step 4: ML inference ─────────────────────────────────────────
    texts = [r.content for r in all_reviews]
    cleaned_texts = batch_clean(texts)

    # BERT sentiment analysis
    await _send_progress(
        tmdb_id, type="log", stage="bert", progress=52,
        message="Loading BERT sentiment model...",
    )
    await _send_progress(
        tmdb_id, type="log", stage="bert", progress=55,
        message=f"Running BERT inference on {len(cleaned_texts)} reviews...",
    )

    sentiments = predict_sentiment(cleaned_texts)
    logger.info(f"[DEBUG] predict_sentiment returned {len(sentiments)} results for {len(cleaned_texts)} texts")

    # Send live sentiment results in batches
    pos_count = sum(1 for s in sentiments if s["label"] == "positive")
    neg_count = sum(1 for s in sentiments if s["label"] == "negative")
    neu_count = sum(1 for s in sentiments if s["label"] == "neutral")

    # Send individual sentiment predictions (first 10)
    sentiment_previews = []
    for r, s in list(zip(all_reviews, sentiments))[:10]:
        sentiment_previews.append({
            "snippet": (r.content[:80] + "...") if len(r.content) > 80 else r.content,
            "source": r.source,
            "label": s["label"],
            "score": round(s["score"], 3),
        })

    await _send_progress(
        tmdb_id, type="sentiments", stage="bert", progress=68,
        predictions=sentiment_previews,
        counts={"positive": pos_count, "negative": neg_count, "neutral": neu_count},
        message=f"BERT complete: {pos_count} positive, {neg_count} negative, {neu_count} neutral",
    )

    # BiLSTM aspect analysis
    await _send_progress(
        tmdb_id, type="log", stage="bilstm", progress=70,
        message="Loading BiLSTM aspect model (GloVe embeddings)...",
    )
    await _send_progress(
        tmdb_id, type="log", stage="bilstm", progress=73,
        message=f"Running BiLSTM aspect analysis on {len(cleaned_texts)} reviews...",
    )

    aspect_scores = predict_aspects(cleaned_texts)

    avg_aspects = aggregate_aspect_scores(aspect_scores)
    await _send_progress(
        tmdb_id, type="aspects", stage="bilstm", progress=80,
        aspects=avg_aspects,
        message=f"BiLSTM complete — aspect scores: acting={avg_aspects.get('acting', 0):.2f}, "
                f"plot={avg_aspects.get('plot', 0):.2f}, visuals={avg_aspects.get('visuals', 0):.2f}",
    )

    # ── Step 5: Aggregation ──────────────────────────────────────────
    await _send_progress(
        tmdb_id, type="log", stage="aggregation", progress=82,
        message="Computing verdict and confidence scores...",
    )

    verdict, confidence = compute_verdict(sentiments)
    overall_sentiment = compute_overall_sentiment(sentiments)

    await _send_progress(
        tmdb_id, type="verdict", stage="aggregation", progress=85,
        verdict=verdict, confidence=round(confidence, 3),
        overall_sentiment=overall_sentiment,
        message=f'Verdict: "{verdict}" (confidence: {confidence:.1%})',
    )

    review_data_for_agg = [
        {"source": r.source, "label": s["label"], "review_date": r.review_date}
        for r, s in zip(all_reviews, sentiments)
    ]
    source_comparison = compute_source_comparison(review_data_for_agg)
    sentiment_trend = compute_sentiment_trend(review_data_for_agg)

    await _send_progress(
        tmdb_id, type="log", stage="aggregation", progress=87,
        message="Generating review summaries...",
    )
    pos_summary, neg_summary = extractive_summary(cleaned_texts, sentiments)

    await _send_progress(
        tmdb_id, type="log", stage="aggregation", progress=89,
        message="Building word frequency data...",
    )
    wc_pos, wc_neg = generate_word_cloud_data(cleaned_texts, sentiments)

    # ── Step 6: Save ─────────────────────────────────────────────────
    await _send_progress(
        tmdb_id, type="log", stage="saving", progress=92,
        message="Persisting analysis results to database...",
    )

    analysis_data = {
        "overall_sentiment": overall_sentiment,
        "confidence": confidence,
        "verdict": verdict,
        "aspect_scores": avg_aspects,
        "source_comparison": source_comparison,
        "sentiment_trend": sentiment_trend,
        "positive_summary": pos_summary,
        "negative_summary": neg_summary,
        "word_cloud_positive": wc_pos,
        "word_cloud_negative": wc_neg,
        "review_count": len(all_reviews),
    }

    review_sentiments_data = [
        {
            "review_id": r.id,
            "label": s["label"],
            "score": s["score"],
            "aspect_scores": a,
        }
        for r, s, a in zip(all_reviews, sentiments, aspect_scores)
    ]
    logger.info(f"[DEBUG] review_sentiments_data length: {len(review_sentiments_data)} "
                f"(all_reviews={len(all_reviews)}, sentiments={len(sentiments)}, aspects={len(aspect_scores)})")

    await save_analysis(db, movie.id, analysis_data, review_sentiments_data)

    await _send_progress(
        tmdb_id, type="log", stage="complete", progress=100,
        message="Analysis complete!",
    )

    analysis = await get_cached_analysis(db, movie.id)
    logger.info(f"[DEBUG] Final analysis: review_count={analysis.review_count}, "
                f"review_sentiments loaded={len(analysis.review_sentiments)}")
    return _build_analysis_response(analysis)
