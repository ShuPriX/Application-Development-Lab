import { useState, useEffect, useRef } from "react";
import { useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { Button, Chip, Skeleton } from "@heroui/react";
import { useAnalysis, useAnalysisMutation, useAnalysisProgress } from "../hooks/useAnalysis";
import LiveAnalysis from "../components/analysis/LiveAnalysis";
import { getMovieDetails } from "../services/api";
import AnalysisHeader from "../components/analysis/AnalysisHeader";
import VerdictBadge from "../components/analysis/VerdictBadge";
import SentimentGauge from "../components/analysis/SentimentGauge";
import AspectRadar from "../components/analysis/AspectRadar";
import SourceComparison from "../components/analysis/SourceComparison";
import SentimentTrend from "../components/analysis/SentimentTrend";
import ReviewSummary from "../components/analysis/ReviewSummary";
import WordCloud from "../components/analysis/WordCloud";
import ReviewList from "../components/analysis/ReviewList";
import Loading from "../components/common/Loading";

const TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p";

export default function Analysis() {
  const { tmdbId } = useParams<{ tmdbId: string }>();
  const id = tmdbId ? parseInt(tmdbId, 10) : undefined;

  const { data: movie, isLoading: isLoadingMovie } = useQuery({
    queryKey: ["movieDetails", id],
    queryFn: () => getMovieDetails(id!),
    enabled: !!id,
    staleTime: 30 * 60 * 1000,
  });

  const { data: analysis, isLoading, error } = useAnalysis(id);
  const mutation = useAnalysisMutation(id);
  const liveState = useAnalysisProgress(id, mutation.isPending);

  // Keep showing live dashboard for 5s after mutation completes
  const [showingLive, setShowingLive] = useState(false);
  const wasPending = useRef(false);
  const frozenLiveState = useRef(liveState);

  useEffect(() => {
    if (mutation.isPending) {
      wasPending.current = true;
      setShowingLive(true);
      frozenLiveState.current = liveState;
    }
  }, [mutation.isPending, liveState]);

  useEffect(() => {
    if (wasPending.current && !mutation.isPending && !mutation.isError) {
      // Mutation just finished successfully — hold live view for 5s
      const timer = setTimeout(() => {
        setShowingLive(false);
        wasPending.current = false;
      }, 5000);
      return () => clearTimeout(timer);
    }
    if (!mutation.isPending && !wasPending.current) {
      setShowingLive(false);
    }
  }, [mutation.isPending, mutation.isError]);

  const backdropUrl = movie?.backdrop_url
    ? `${TMDB_IMAGE_BASE}/w1280${movie.backdrop_url}`
    : null;
  const posterUrl = movie?.poster_url
    ? `${TMDB_IMAGE_BASE}/w500${movie.poster_url}`
    : null;
  const genres = movie?.genre
    ? movie.genre.split(",").map((g) => g.trim()).filter(Boolean)
    : [];

  // No cached analysis -- show Netflix-style movie details with analyze prompt
  if (!isLoading && !analysis && !mutation.isPending && !error) {
    if (isLoadingMovie) {
      return (
        <div className="min-h-screen pt-16">
          <div className="relative w-full h-[70vh]">
            <Skeleton className="absolute inset-0 bg-white/5" />
            <div className="absolute bottom-0 left-0 right-0 p-8">
              <div className="max-w-7xl mx-auto flex gap-8">
                <Skeleton className="w-[200px] h-[300px] rounded-lg bg-white/10 flex-shrink-0" />
                <div className="flex-1 space-y-4 pt-8">
                  <Skeleton className="h-10 w-96 rounded-lg bg-white/10" />
                  <Skeleton className="h-6 w-48 rounded-lg bg-white/10" />
                  <Skeleton className="h-24 w-full max-w-2xl rounded-lg bg-white/10" />
                </div>
              </div>
            </div>
          </div>
        </div>
      );
    }

    return (
      <div className="min-h-screen pt-16">
        {/* Backdrop hero */}
        <div className="relative w-full h-[70vh] overflow-hidden">
          {backdropUrl ? (
            <motion.img
              initial={{ opacity: 0, scale: 1.05 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.8 }}
              src={backdropUrl}
              alt={movie?.title ?? ""}
              className="absolute inset-0 w-full h-full object-cover"
            />
          ) : (
            <div className="absolute inset-0 bg-netflix-dark" />
          )}
          {/* Dark overlays for readability */}
          <div className="absolute inset-0 bg-black/40" />
          <div className="absolute inset-0 bg-gradient-to-t from-[#141414] via-[#141414]/80 to-transparent" />
          <div className="absolute inset-0 bg-gradient-to-r from-[#141414]/90 via-[#141414]/40 to-transparent" />

          {/* Movie details overlay */}
          <div className="absolute bottom-0 left-0 right-0 pb-12 pt-20">
            <div className="max-w-7xl mx-auto px-6 flex gap-8 items-end">
              {/* Poster */}
              {posterUrl ? (
                <motion.img
                  initial={{ opacity: 0, y: 30 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: 0.2 }}
                  src={posterUrl}
                  alt={movie?.title ?? ""}
                  className="w-[200px] rounded-lg shadow-2xl flex-shrink-0 hidden md:block"
                />
              ) : (
                <div className="w-[200px] h-[300px] bg-netflix-card rounded-lg flex items-center justify-center flex-shrink-0 hidden md:block">
                  <span className="text-gray-600 text-5xl">?</span>
                </div>
              )}

              {/* Info */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.3 }}
                className="flex-1 min-w-0"
              >
                <h1 className="text-white text-4xl md:text-5xl font-bold mb-2 drop-shadow-lg">
                  {movie?.title}
                </h1>
                <div className="flex items-center gap-3 mb-4 flex-wrap">
                  {movie?.year && (
                    <span className="text-gray-300 text-lg">{movie.year}</span>
                  )}
                  {movie?.vote_average != null && movie.vote_average > 0 && (
                    <span className="text-yellow-400 text-lg font-semibold">
                      ★ {movie.vote_average.toFixed(1)}
                    </span>
                  )}
                </div>
                {genres.length > 0 && (
                  <div className="flex gap-2 mb-4 flex-wrap">
                    {genres.map((genre) => (
                      <Chip
                        key={genre}
                        size="sm"
                        variant="bordered"
                        classNames={{
                          base: "border-white/20",
                          content: "text-gray-300 text-xs",
                        }}
                      >
                        {genre}
                      </Chip>
                    ))}
                  </div>
                )}
                {movie?.overview && (
                  <p className="text-gray-300 text-sm md:text-base leading-relaxed max-w-2xl mb-6 line-clamp-4">
                    {movie.overview}
                  </p>
                )}
                <Button
                  size="lg"
                  className="bg-netflix-red text-white font-semibold hover:bg-netflix-red-hover px-8 text-base"
                  onPress={() => mutation.mutate()}
                >
                  Analyze Movie
                </Button>
              </motion.div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Loading analysis data
  if (isLoading) {
    return <Loading message="Loading analysis..." />;
  }

  // Analysis in progress or holding for 5s after completion
  if (mutation.isPending || showingLive) {
    const displayState = mutation.isPending ? liveState : frozenLiveState.current;
    return (
      <div className="min-h-screen pt-20 pb-12 px-4 sm:px-6">
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
          <LiveAnalysis
            liveState={displayState}
            movieTitle={movie?.title}
            posterUrl={posterUrl}
          />
        </motion.div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[60vh] gap-4">
        <p className="text-red-400 text-lg">Failed to load analysis</p>
        <p className="text-gray-500 text-sm">
          {error instanceof Error ? error.message : "Unknown error"}
        </p>
      </div>
    );
  }

  if (mutation.isError) {
    const mutErr = mutation.error;
    const statusCode =
      mutErr &&
      typeof mutErr === "object" &&
      "response" in mutErr
        ? (mutErr as any).response?.status
        : null;
    const is422 = statusCode === 422;
    const is503 = statusCode === 503;

    return (
      <div className="min-h-screen pt-16">
        <div className="relative w-full h-[70vh] overflow-hidden">
          {backdropUrl ? (
            <img
              src={backdropUrl}
              alt={movie?.title ?? ""}
              className="absolute inset-0 w-full h-full object-cover"
            />
          ) : (
            <div className="absolute inset-0 bg-netflix-dark" />
          )}
          <div className="absolute inset-0 bg-black/70" />

          <div className="absolute inset-0 flex flex-col items-center justify-center px-6">
            <div className="w-full max-w-md text-center">
              {posterUrl && (
                <img
                  src={posterUrl}
                  alt={movie?.title ?? ""}
                  className="w-24 rounded-lg shadow-xl mx-auto mb-6"
                />
              )}
              {is422 ? (
                <>
                  <h2 className="text-white text-xl font-bold mb-2">
                    No Reviews Found
                  </h2>
                  <p className="text-gray-400 text-sm mb-6 max-w-sm mx-auto">
                    We couldn't find enough reviews for{" "}
                    <span className="text-white font-medium">
                      {movie?.title ?? "this movie"}
                    </span>
                    . This can happen with very new, unreleased, or lesser-known
                    titles.
                  </p>
                </>
              ) : is503 ? (
                <>
                  <h2 className="text-white text-xl font-bold mb-2">
                    Connection Error
                  </h2>
                  <p className="text-gray-400 text-sm mb-6 max-w-sm mx-auto">
                    Could not reach the movie database API. Please check your
                    internet connection and try again.
                  </p>
                </>
              ) : (
                <>
                  <h2 className="text-white text-xl font-bold mb-2">
                    Analysis Failed
                  </h2>
                  <p className="text-gray-400 text-sm mb-6">
                    Something went wrong while analyzing this movie.
                  </p>
                </>
              )}
              <Button
                className="bg-netflix-red text-white font-semibold hover:bg-netflix-red-hover px-6"
                onPress={() => mutation.mutate()}
              >
                {is422 ? "Try Again" : "Retry"}
              </Button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (!analysis) return null;

  // Compute overall sentiment distribution
  const sentimentDist = analysis.review_sentiments
    ? {
        positive: analysis.review_sentiments.filter(
          (r) => r.sentiment_label === "positive",
        ).length,
        neutral: analysis.review_sentiments.filter(
          (r) => r.sentiment_label === "neutral",
        ).length,
        negative: analysis.review_sentiments.filter(
          (r) => r.sentiment_label === "negative",
        ).length,
      }
    : { positive: 0, neutral: 0, negative: 0 };

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
    >
      <AnalysisHeader
        analysis={analysis}
        onUpdate={() => mutation.mutate()}
        isUpdating={mutation.isPending}
      />

      <div className="max-w-7xl mx-auto px-6 py-8 space-y-8">
        {/* Top row: Verdict + Sentiment + Aspects */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <VerdictBadge
            verdict={analysis.verdict}
            confidence={analysis.confidence}
            sentiment={analysis.overall_sentiment}
          />
          <SentimentGauge distribution={sentimentDist} />
          <AspectRadar scores={analysis.aspect_scores} />
        </div>

        {/* Source comparison + Trend */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <SourceComparison data={analysis.source_comparison} />
          <SentimentTrend data={analysis.sentiment_trend} />
        </div>

        {/* Summary */}
        <ReviewSummary
          positive={analysis.positive_summary}
          negative={analysis.negative_summary}
        />

        {/* Word clouds */}
        <WordCloud
          positive={analysis.word_cloud_positive}
          negative={analysis.word_cloud_negative}
        />

        {/* Review list */}
        <ReviewList reviews={analysis.review_sentiments} />
      </div>
    </motion.div>
  );
}
