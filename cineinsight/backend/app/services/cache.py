import logging
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.database import Movie, Review, AnalysisResult, ReviewSentiment

logger = logging.getLogger(__name__)


async def get_or_create_movie(db: AsyncSession, movie_data: dict) -> Movie:
    """Get existing movie by tmdb_id or create new one."""
    stmt = select(Movie).where(Movie.tmdb_id == movie_data["tmdb_id"])
    result = await db.execute(stmt)
    movie = result.scalar_one_or_none()

    if movie:
        # Update metadata
        for key in ("title", "year", "poster_url", "backdrop_url", "genre", "overview", "vote_average"):
            if key in movie_data:
                setattr(movie, key, movie_data[key])
        await db.commit()
        return movie

    movie = Movie(**movie_data)
    db.add(movie)
    await db.commit()
    await db.refresh(movie)
    return movie


async def get_cached_analysis(db: AsyncSession, movie_id: int) -> AnalysisResult | None:
    """Get cached analysis for a movie, with all related data."""
    stmt = (
        select(AnalysisResult)
        .where(AnalysisResult.movie_id == movie_id)
        .options(
            selectinload(AnalysisResult.movie),
            selectinload(AnalysisResult.review_sentiments).selectinload(ReviewSentiment.review),
        )
    )
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def store_reviews(db: AsyncSession, movie_id: int, scraped_reviews: list[dict]) -> list[Review]:
    """Store new reviews, skipping duplicates by content hash."""
    stored = []
    for review_data in scraped_reviews:
        content_hash = Review.compute_hash(review_data["content"])

        # Check for duplicate
        stmt = select(Review).where(Review.content_hash == content_hash)
        result = await db.execute(stmt)
        if result.scalar_one_or_none():
            continue

        review = Review(
            movie_id=movie_id,
            source=review_data["source"],
            author=review_data.get("author", "Anonymous"),
            content=review_data["content"],
            rating=review_data.get("rating"),
            review_date=review_data.get("review_date"),
            content_hash=content_hash,
        )
        db.add(review)
        stored.append(review)

    await db.commit()
    for review in stored:
        await db.refresh(review)
    return stored


async def save_analysis(
    db: AsyncSession,
    movie_id: int,
    analysis_data: dict,
    review_sentiments_data: list[dict],
) -> AnalysisResult:
    """Save or overwrite analysis results for a movie."""
    # Delete existing analysis if any
    stmt = select(AnalysisResult).where(AnalysisResult.movie_id == movie_id)
    result = await db.execute(stmt)
    existing = result.scalar_one_or_none()
    if existing:
        await db.delete(existing)
        await db.flush()

    analysis = AnalysisResult(
        movie_id=movie_id,
        overall_sentiment=analysis_data["overall_sentiment"],
        confidence=analysis_data["confidence"],
        verdict=analysis_data["verdict"],
        aspect_scores=analysis_data["aspect_scores"],
        source_comparison=analysis_data["source_comparison"],
        sentiment_trend=analysis_data["sentiment_trend"],
        positive_summary=analysis_data["positive_summary"],
        negative_summary=analysis_data["negative_summary"],
        word_cloud_positive=analysis_data["word_cloud_positive"],
        word_cloud_negative=analysis_data["word_cloud_negative"],
        review_count=analysis_data["review_count"],
        analyzed_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    db.add(analysis)
    await db.flush()

    for rs_data in review_sentiments_data:
        rs = ReviewSentiment(
            review_id=rs_data["review_id"],
            analysis_id=analysis.id,
            sentiment_label=rs_data["label"],
            sentiment_score=rs_data["score"],
            aspect_scores=rs_data.get("aspect_scores"),
        )
        db.add(rs)

    await db.commit()
    await db.refresh(analysis)
    return analysis


async def get_recent_movies(db: AsyncSession, limit: int = 20) -> list[Movie]:
    """Get recently analyzed movies."""
    stmt = (
        select(Movie)
        .join(AnalysisResult, Movie.id == AnalysisResult.movie_id)
        .order_by(AnalysisResult.updated_at.desc())
        .limit(limit)
    )
    result = await db.execute(stmt)
    return list(result.scalars().all())
