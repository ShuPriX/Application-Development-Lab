export interface AspectScores {
  acting: number;
  plot: number;
  visuals: number;
  music: number;
  direction: number;
}

export interface ReviewSentiment {
  id: number;
  review_id: number;
  content: string;
  author: string;
  source: string;
  sentiment_label: "positive" | "negative" | "neutral";
  sentiment_score: number;
  aspect_scores: AspectScores | null;
}

export interface SentimentDistribution {
  positive: number;
  negative: number;
  neutral: number;
}

export interface SourceComparison {
  tmdb?: SentimentDistribution;
  imdb?: SentimentDistribution;
  rottentomatoes?: SentimentDistribution;
  [key: string]: SentimentDistribution | undefined;
}

export interface SentimentTrendPoint {
  date: string;
  positive: number;
  negative: number;
  neutral: number;
}

export interface WordCloudItem {
  text: string;
  value: number;
}

export interface AnalysisResult {
  id: number;
  movie_id: number;
  movie: {
    tmdb_id: number;
    title: string;
    year: number;
    poster_url: string | null;
    backdrop_url: string | null;
    genre: string;
    overview: string;
  };
  overall_sentiment: string;
  confidence: number;
  verdict: string;
  aspect_scores: AspectScores;
  source_comparison: SourceComparison;
  sentiment_trend: SentimentTrendPoint[];
  positive_summary: string[];
  negative_summary: string[];
  word_cloud_positive: WordCloudItem[];
  word_cloud_negative: WordCloudItem[];
  review_sentiments: ReviewSentiment[];
  review_count: number;
  analyzed_at: string;
  updated_at: string;
}

export type Verdict =
  | "Strongly Recommended"
  | "Worth Watching"
  | "Mixed Reviews"
  | "Below Average"
  | "Skip It";

export interface AnalysisProgress {
  type: "log" | "source" | "reviews" | "sentiments" | "aspects" | "verdict";
  stage: string;
  progress: number;
  message: string;
  // source updates
  source?: string;
  count?: number;
  status?: string;
  // review previews
  reviews?: { source: string; author: string; snippet: string }[];
  // sentiment predictions
  predictions?: { snippet: string; source: string; label: string; score: number }[];
  counts?: { positive: number; negative: number; neutral: number };
  // aspect scores
  aspects?: Record<string, number>;
  // verdict
  verdict?: string;
  confidence?: number;
  overall_sentiment?: string;
}
