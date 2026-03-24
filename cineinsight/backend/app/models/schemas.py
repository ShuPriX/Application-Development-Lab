from datetime import datetime
from pydantic import BaseModel


class MovieBase(BaseModel):
    tmdb_id: int
    title: str
    year: int | None = None
    poster_url: str | None = None
    backdrop_url: str | None = None
    genre: str = ""
    overview: str = ""
    vote_average: float | None = None

    model_config = {"from_attributes": True}


class MovieResponse(MovieBase):
    id: int

    model_config = {"from_attributes": True}


class MovieSearchResponse(BaseModel):
    results: list[MovieResponse]
    total_results: int


class AspectScoresSchema(BaseModel):
    acting: float = 0.0
    plot: float = 0.0
    visuals: float = 0.0
    music: float = 0.0
    direction: float = 0.0


class SentimentDistribution(BaseModel):
    positive: int = 0
    negative: int = 0
    neutral: int = 0


class SourceComparisonSchema(BaseModel):
    tmdb: SentimentDistribution = SentimentDistribution()
    imdb: SentimentDistribution = SentimentDistribution()
    rottentomatoes: SentimentDistribution = SentimentDistribution()

    model_config = {"extra": "allow"}


class SentimentTrendPoint(BaseModel):
    date: str
    positive: int = 0
    negative: int = 0
    neutral: int = 0


class WordCloudItem(BaseModel):
    text: str
    value: int


class ReviewSentimentResponse(BaseModel):
    id: int
    review_id: int
    content: str = ""
    author: str = ""
    source: str = ""
    sentiment_label: str
    sentiment_score: float
    aspect_scores: AspectScoresSchema | None = None

    model_config = {"from_attributes": True}


class AnalysisResponse(BaseModel):
    id: int
    movie_id: int
    movie: MovieBase
    overall_sentiment: str
    confidence: float
    verdict: str
    aspect_scores: AspectScoresSchema
    source_comparison: SourceComparisonSchema
    sentiment_trend: list[SentimentTrendPoint]
    positive_summary: list[str]
    negative_summary: list[str]
    word_cloud_positive: list[WordCloudItem]
    word_cloud_negative: list[WordCloudItem]
    review_sentiments: list[ReviewSentimentResponse]
    review_count: int
    analyzed_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class AnalysisProgress(BaseModel):
    stage: str
    progress: float
    message: str
