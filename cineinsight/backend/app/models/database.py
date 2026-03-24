import hashlib
from datetime import datetime, timezone

from sqlalchemy import Column, Integer, String, Float, Text, DateTime, ForeignKey, JSON
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class Movie(Base):
    __tablename__ = "movies"

    id = Column(Integer, primary_key=True, autoincrement=True)
    tmdb_id = Column(Integer, unique=True, nullable=False, index=True)
    title = Column(String(500), nullable=False)
    year = Column(Integer)
    poster_url = Column(String(500))
    backdrop_url = Column(String(500))
    genre = Column(String(500))
    overview = Column(Text)
    vote_average = Column(Float)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    reviews = relationship("Review", back_populates="movie", cascade="all, delete-orphan")
    analysis = relationship("AnalysisResult", back_populates="movie", uselist=False,
                            cascade="all, delete-orphan")


class Review(Base):
    __tablename__ = "reviews"

    id = Column(Integer, primary_key=True, autoincrement=True)
    movie_id = Column(Integer, ForeignKey("movies.id"), nullable=False, index=True)
    source = Column(String(50), nullable=False)  # "tmdb", "imdb", or "rottentomatoes"
    author = Column(String(200))
    content = Column(Text, nullable=False)
    rating = Column(Float)
    review_date = Column(DateTime)
    content_hash = Column(String(64), unique=True, nullable=False, index=True)
    scraped_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    movie = relationship("Movie", back_populates="reviews")
    sentiment = relationship("ReviewSentiment", back_populates="review", uselist=False,
                             cascade="all, delete-orphan")

    @staticmethod
    def compute_hash(content: str) -> str:
        return hashlib.sha256(content.strip().encode()).hexdigest()


class AnalysisResult(Base):
    __tablename__ = "analysis_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    movie_id = Column(Integer, ForeignKey("movies.id"), unique=True, nullable=False, index=True)
    overall_sentiment = Column(String(20))  # "positive", "negative", "neutral"
    confidence = Column(Float)
    verdict = Column(String(50))
    aspect_scores = Column(JSON)
    source_comparison = Column(JSON)
    sentiment_trend = Column(JSON)
    positive_summary = Column(JSON)
    negative_summary = Column(JSON)
    word_cloud_positive = Column(JSON)
    word_cloud_negative = Column(JSON)
    review_count = Column(Integer)
    analyzed_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc),
                        onupdate=lambda: datetime.now(timezone.utc))

    movie = relationship("Movie", back_populates="analysis")
    review_sentiments = relationship("ReviewSentiment", back_populates="analysis",
                                     cascade="all, delete-orphan")


class ReviewSentiment(Base):
    __tablename__ = "review_sentiments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    review_id = Column(Integer, ForeignKey("reviews.id"), nullable=False, index=True)
    analysis_id = Column(Integer, ForeignKey("analysis_results.id"), nullable=False, index=True)
    sentiment_label = Column(String(20))
    sentiment_score = Column(Float)
    aspect_scores = Column(JSON)

    review = relationship("Review", back_populates="sentiment")
    analysis = relationship("AnalysisResult", back_populates="review_sentiments")
