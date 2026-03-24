import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { motion, AnimatePresence } from "framer-motion";
import { GitCompareArrows, Check, Search, X } from "lucide-react";
import { Input } from "@heroui/react";
import CompareView from "../components/compare/CompareView";
import { useAnalysis } from "../hooks/useAnalysis";
import { useMovieSearch } from "../hooks/useMovieSearch";
import { getRecentMovies } from "../services/api";
import Loading from "../components/common/Loading";
import type { Movie } from "../types/movie";

const TMDB_IMG = "https://image.tmdb.org/t/p";

function SelectedBadge({ slot }: { slot: "1" | "2" }) {
  const bg = slot === "1" ? "bg-netflix-red" : "bg-blue-500";
  return (
    <div
      className={`absolute top-1.5 right-1.5 ${bg} rounded-full w-5 h-5 flex items-center justify-center shadow-lg z-10`}
    >
      <Check className="w-3 h-3 text-white" />
    </div>
  );
}

function MovieGrid({
  movies,
  movieA,
  movieB,
  onSelect,
}: {
  movies: Movie[];
  movieA: Movie | null;
  movieB: Movie | null;
  onSelect: (movie: Movie) => void;
}) {
  return (
    <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-5 lg:grid-cols-6 gap-3 sm:gap-4">
      {movies.map((movie) => {
        const isA = movieA?.tmdb_id === movie.tmdb_id;
        const isB = movieB?.tmdb_id === movie.tmdb_id;
        const isSelected = isA || isB;

        return (
          <motion.button
            key={movie.tmdb_id}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.97 }}
            onClick={() => onSelect(movie)}
            className={`relative group rounded-lg overflow-hidden border-2 transition-colors ${
              isA
                ? "border-netflix-red"
                : isB
                  ? "border-blue-500"
                  : "border-transparent hover:border-white/20"
            }`}
          >
            {isA && <SelectedBadge slot="1" />}
            {isB && <SelectedBadge slot="2" />}
            {movie.poster_url ? (
              <img
                src={`${TMDB_IMG}/w300${movie.poster_url}`}
                alt={movie.title}
                className={`w-full aspect-[2/3] object-cover transition-opacity ${
                  isSelected ? "opacity-80" : "opacity-100 group-hover:opacity-90"
                }`}
              />
            ) : (
              <div className="w-full aspect-[2/3] bg-white/5 flex items-center justify-center">
                <span className="text-gray-600 text-3xl">?</span>
              </div>
            )}
            <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/90 via-black/50 to-transparent p-2 pt-8">
              <p className="text-white text-xs sm:text-sm font-medium leading-tight line-clamp-2">
                {movie.title}
              </p>
              {movie.year && (
                <p className="text-gray-400 text-[10px] sm:text-xs">{movie.year}</p>
              )}
            </div>
          </motion.button>
        );
      })}
    </div>
  );
}

function SearchSelector({
  label,
  movie,
  onSelect,
  onClear,
  color,
}: {
  label: string;
  movie: Movie | null;
  onSelect: (m: Movie) => void;
  onClear: () => void;
  color: string;
}) {
  const [query, setQuery] = useState("");
  const [showResults, setShowResults] = useState(false);
  const { data: results } = useMovieSearch(query);

  if (movie) {
    return (
      <div className={`flex items-center gap-3 bg-white/5 rounded-lg px-3 py-2 border ${color}`}>
        {movie.poster_url && (
          <img
            src={`${TMDB_IMG}/w92${movie.poster_url}`}
            alt=""
            className="w-10 h-14 rounded object-cover"
          />
        )}
        <div className="flex-1 min-w-0">
          <p className="text-white text-sm font-semibold truncate">{movie.title}</p>
          <p className="text-gray-500 text-xs">{movie.year}</p>
        </div>
        <button
          onClick={onClear}
          className="text-gray-500 hover:text-white p-1 rounded hover:bg-white/5"
        >
          <X className="w-4 h-4" />
        </button>
      </div>
    );
  }

  return (
    <div className="relative">
      <Input
        size="sm"
        placeholder={`Search for ${label}...`}
        value={query}
        onValueChange={(v) => {
          setQuery(v);
          setShowResults(true);
        }}
        onFocus={() => setShowResults(true)}
        onBlur={() => setTimeout(() => setShowResults(false), 200)}
        startContent={<Search className="w-4 h-4 text-gray-500" />}
        classNames={{
          inputWrapper:
            "bg-netflix-dark border border-white/10 hover:border-netflix-red/50",
          input: "text-white text-sm",
        }}
      />
      {showResults && results && results.length > 0 && (
        <div className="absolute top-full left-0 right-0 mt-1 bg-netflix-dark/95 backdrop-blur-xl border border-white/10 rounded-lg shadow-xl z-50 max-h-52 overflow-auto">
          {results.map((m) => (
            <button
              key={m.tmdb_id}
              className="w-full flex items-center gap-3 p-2.5 hover:bg-white/5 text-left"
              onMouseDown={() => {
                onSelect(m);
                setShowResults(false);
                setQuery("");
              }}
            >
              {m.poster_url && (
                <img
                  src={`${TMDB_IMG}/w92${m.poster_url}`}
                  alt=""
                  className="w-8 h-11 rounded object-cover"
                />
              )}
              <div>
                <p className="text-white text-sm">{m.title}</p>
                <p className="text-gray-500 text-xs">{m.year}</p>
              </div>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

export default function Compare() {
  const [movieA, setMovieA] = useState<Movie | null>(null);
  const [movieB, setMovieB] = useState<Movie | null>(null);

  const { data: recentMovies, isLoading: loadingRecent } = useQuery({
    queryKey: ["recentMovies"],
    queryFn: getRecentMovies,
    staleTime: 5 * 60 * 1000,
  });

  const { data: analysisA, isLoading: loadingA } = useAnalysis(movieA?.tmdb_id);
  const { data: analysisB, isLoading: loadingB } = useAnalysis(movieB?.tmdb_id);

  const handleGridSelect = (movie: Movie) => {
    const isA = movieA?.tmdb_id === movie.tmdb_id;
    const isB = movieB?.tmdb_id === movie.tmdb_id;

    if (isA) {
      // Deselect A
      setMovieA(null);
    } else if (isB) {
      // Deselect B
      setMovieB(null);
    } else if (!movieA) {
      setMovieA(movie);
    } else if (!movieB) {
      setMovieB(movie);
    } else {
      // Both slots full — replace B
      setMovieB(movie);
    }
  };

  const bothSelected = movieA && movieB;

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 pt-24 pb-8 min-h-screen">
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
        <h1 className="flex items-center gap-3 text-white text-2xl sm:text-3xl lg:text-4xl font-extrabold mb-2">
          <GitCompareArrows className="w-7 h-7 sm:w-8 sm:h-8 text-netflix-red" />
          Compare Movies
        </h1>
        <p className="text-gray-400 text-sm sm:text-base mb-6">
          Pick two analyzed movies to compare — click from the grid or search below
        </p>

        {/* Selection bar */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-8">
          <div>
            <p className="text-gray-500 text-xs uppercase tracking-wider mb-1.5 flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-netflix-red" />
              Movie 1
            </p>
            <SearchSelector
              label="Movie 1"
              movie={movieA}
              onSelect={setMovieA}
              onClear={() => setMovieA(null)}
              color="border-netflix-red/30"
            />
          </div>
          <div>
            <p className="text-gray-500 text-xs uppercase tracking-wider mb-1.5 flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-blue-500" />
              Movie 2
            </p>
            <SearchSelector
              label="Movie 2"
              movie={movieB}
              onSelect={setMovieB}
              onClear={() => setMovieB(null)}
              color="border-blue-500/30"
            />
          </div>
        </div>

        {/* Analyzed movies grid */}
        {!bothSelected && (
          <AnimatePresence>
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              <p className="text-gray-500 text-xs uppercase tracking-wider mb-3">
                Previously Analyzed Movies
              </p>
              {loadingRecent ? (
                <Loading message="Loading movies..." />
              ) : recentMovies && recentMovies.length > 0 ? (
                <MovieGrid
                  movies={recentMovies}
                  movieA={movieA}
                  movieB={movieB}
                  onSelect={handleGridSelect}
                />
              ) : (
                <div className="text-center py-12">
                  <p className="text-gray-500 text-sm">
                    No analyzed movies yet. Go analyze some movies first!
                  </p>
                </div>
              )}
            </motion.div>
          </AnimatePresence>
        )}

        {/* Loading */}
        {(loadingA || loadingB) && (
          <div className="mt-8">
            <Loading message="Loading analysis..." />
          </div>
        )}

        {/* Compare results */}
        <AnimatePresence>
          {analysisA && analysisB && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="mt-8"
            >
              <CompareView analysisA={analysisA} analysisB={analysisB} />
            </motion.div>
          )}
        </AnimatePresence>

        {/* One or both not analyzed */}
        {movieA && movieB && !loadingA && !loadingB && (!analysisA || !analysisB) && (
          <div className="text-center py-12">
            <p className="text-gray-400 text-base">
              {!analysisA && !analysisB
                ? "Both movies need to be analyzed first."
                : !analysisA
                  ? `"${movieA.title}" needs to be analyzed first.`
                  : `"${movieB.title}" needs to be analyzed first.`}
            </p>
            <p className="text-gray-600 text-sm mt-2">
              Go to each movie&apos;s page and run the analysis.
            </p>
          </div>
        )}
      </motion.div>
    </div>
  );
}
