import { useState, useCallback } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { useNavigate } from "react-router-dom";
import { useMovieSearch } from "../../hooks/useMovieSearch";
import PlaceholderInput from "../ui/PlaceholderInput";

const PLACEHOLDERS = [
  "Search for Inception...",
  "Try The Dark Knight...",
  "Look up Interstellar...",
  "Find Oppenheimer...",
  "Discover Dune Part Two...",
];

export default function SearchBar() {
  const [query, setQuery] = useState("");
  const [showResults, setShowResults] = useState(false);
  const navigate = useNavigate();
  const { data: movies, isLoading } = useMovieSearch(query);

  const handleSelect = useCallback(
    (tmdbId: number) => {
      setShowResults(false);
      setQuery("");
      navigate(`/analysis/${tmdbId}`);
    },
    [navigate],
  );

  return (
    <div className="relative w-full max-w-xl mx-auto">
      <PlaceholderInput
        placeholders={PLACEHOLDERS}
        value={query}
        onChange={(val) => {
          setQuery(val);
          setShowResults(true);
        }}
        onFocus={() => setShowResults(true)}
        onBlur={() => setTimeout(() => setShowResults(false), 200)}
      />

      {/* Dropdown results */}
      <AnimatePresence>
        {showResults && query.length >= 2 && (
          <motion.div
            initial={{ opacity: 0, y: -8, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -8, scale: 0.98 }}
            transition={{ duration: 0.2 }}
            className="absolute top-full left-0 right-0 mt-2 bg-netflix-dark/95 backdrop-blur-xl border border-white/10 rounded-xl shadow-2xl shadow-black/50 z-50 max-h-80 overflow-auto"
          >
            {isLoading ? (
              <div className="p-4 space-y-3">
                {[1, 2, 3].map((i) => (
                  <div key={i} className="flex items-center gap-3 animate-pulse">
                    <div className="w-10 h-14 bg-white/5 rounded" />
                    <div className="flex-1 space-y-2">
                      <div className="h-4 w-32 bg-white/5 rounded" />
                      <div className="h-3 w-20 bg-white/5 rounded" />
                    </div>
                  </div>
                ))}
              </div>
            ) : movies && movies.length > 0 ? (
              <div className="py-1">
                {movies.slice(0, 8).map((movie, index) => (
                  <motion.button
                    key={movie.tmdb_id}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.03 }}
                    onClick={() => handleSelect(movie.tmdb_id)}
                    className="w-full flex items-center gap-3 px-4 py-2.5 hover:bg-white/5 transition-colors text-left"
                  >
                    {movie.poster_url ? (
                      <img
                        src={`https://image.tmdb.org/t/p/w92${movie.poster_url}`}
                        alt=""
                        className="w-10 h-14 object-cover rounded flex-shrink-0"
                      />
                    ) : (
                      <div className="w-10 h-14 bg-white/5 rounded flex items-center justify-center flex-shrink-0">
                        <span className="text-gray-600 text-xs">?</span>
                      </div>
                    )}
                    <div className="min-w-0 flex-1">
                      <p className="text-white text-sm font-medium truncate">
                        {movie.title}
                      </p>
                      <p className="text-gray-500 text-xs truncate">
                        {movie.year}{movie.genre ? ` · ${movie.genre}` : ""}
                      </p>
                    </div>
                  </motion.button>
                ))}
              </div>
            ) : (
              <p className="text-gray-500 text-sm p-4 text-center">
                No movies found
              </p>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
