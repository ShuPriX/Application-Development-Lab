import { Button, Chip } from "@heroui/react";
import { Clock, RefreshCw, MessageSquare, Calendar } from "lucide-react";
import type { AnalysisResult } from "../../types/analysis";

const TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p";

interface AnalysisHeaderProps {
  analysis: AnalysisResult;
  onUpdate: () => void;
  isUpdating: boolean;
}

function timeAgo(dateStr: string): string {
  const date = new Date(dateStr);
  const now = new Date();
  const seconds = Math.floor((now.getTime() - date.getTime()) / 1000);
  if (seconds < 60) return "just now";
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

export default function AnalysisHeader({
  analysis,
  onUpdate,
  isUpdating,
}: AnalysisHeaderProps) {
  const movie = analysis.movie;
  const backdropUrl = movie.backdrop_url
    ? `${TMDB_IMAGE_BASE}/w1280${movie.backdrop_url}`
    : null;
  const posterUrl = movie.poster_url
    ? `${TMDB_IMAGE_BASE}/w342${movie.poster_url}`
    : null;

  return (
    <div className="relative pt-16">
      {/* Backdrop */}
      {backdropUrl && (
        <div className="absolute inset-0 h-[450px] sm:h-[400px]">
          <img
            src={backdropUrl}
            alt=""
            className="w-full h-full object-cover"
          />
          <div className="absolute inset-0 bg-black/40" />
          <div className="absolute inset-0 bg-gradient-to-t from-[#141414] via-[#141414]/80 to-transparent" />
          <div className="absolute inset-0 bg-gradient-to-r from-[#141414]/90 via-[#141414]/30 to-transparent" />
        </div>
      )}

      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 pt-8 pb-6 flex flex-col sm:flex-row gap-6 sm:gap-8 items-start sm:items-end min-h-[300px] sm:min-h-[350px]">
        {/* Poster */}
        {posterUrl && (
          <img
            src={posterUrl}
            alt={movie.title}
            className="w-32 sm:w-48 rounded-lg shadow-2xl hidden sm:block"
          />
        )}

        <div className="flex-1 min-w-0">
          <h1 className="text-3xl sm:text-4xl lg:text-5xl font-extrabold text-white mb-2">
            {movie.title}
          </h1>
          <div className="flex items-center gap-3 mb-3 flex-wrap">
            <span className="flex items-center gap-1.5 text-gray-300 text-base">
              <Calendar className="w-4 h-4" />
              {movie.year}
            </span>
            {movie.genre.split(",").map((g) => (
              <Chip
                key={g.trim()}
                size="sm"
                variant="bordered"
                className="text-gray-300 border-gray-600"
              >
                {g.trim()}
              </Chip>
            ))}
          </div>
          <p className="text-gray-400 text-sm sm:text-base max-w-2xl line-clamp-2 sm:line-clamp-3 mb-4">
            {movie.overview}
          </p>
          <div className="flex items-center gap-3 sm:gap-4 flex-wrap">
            <span className="flex items-center gap-1.5 text-gray-500 text-sm">
              <Clock className="w-4 h-4" />
              {timeAgo(analysis.updated_at)}
            </span>
            <Button
              size="sm"
              variant="bordered"
              className="border-netflix-red text-netflix-red hover:bg-netflix-red/10"
              onPress={onUpdate}
              isLoading={isUpdating}
              startContent={!isUpdating && <RefreshCw className="w-4 h-4" />}
            >
              Update
            </Button>
            <span className="flex items-center gap-1.5 text-gray-500 text-sm">
              <MessageSquare className="w-4 h-4" />
              {analysis.review_count} reviews
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
