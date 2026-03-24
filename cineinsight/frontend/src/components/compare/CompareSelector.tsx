import { useState } from "react";
import { Input, Card } from "@heroui/react";
import { Search, X } from "lucide-react";
import { useMovieSearch } from "../../hooks/useMovieSearch";
import type { Movie } from "../../types/movie";

interface CompareSelectorProps {
  label: string;
  selectedMovie: Movie | null;
  onSelect: (movie: Movie) => void;
}

export default function CompareSelector({ label, selectedMovie, onSelect }: CompareSelectorProps) {
  const [query, setQuery] = useState("");
  const [showResults, setShowResults] = useState(false);
  const { data: movies } = useMovieSearch(query);

  return (
    <div className="relative">
      <p className="text-gray-400 text-xs sm:text-sm uppercase tracking-wider mb-2">{label}</p>
      {selectedMovie ? (
        <Card className="bg-netflix-card border-none p-4">
          <div className="flex items-center gap-3">
            {selectedMovie.poster_url && (
              <img src={`https://image.tmdb.org/t/p/w92${selectedMovie.poster_url}`} alt="" className="w-12 h-16 rounded object-cover" />
            )}
            <div className="flex-1 min-w-0">
              <p className="text-white font-semibold text-sm sm:text-base truncate">{selectedMovie.title}</p>
              <p className="text-gray-500 text-xs sm:text-sm">{selectedMovie.year}</p>
            </div>
            <button
              onClick={() => { onSelect(null!); setQuery(""); }}
              className="text-gray-500 hover:text-white p-1.5 rounded-lg hover:bg-white/5 transition-colors"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </Card>
      ) : (
        <>
          <Input
            size="sm"
            placeholder="Search movie..."
            value={query}
            onValueChange={(val) => { setQuery(val); setShowResults(true); }}
            onFocus={() => setShowResults(true)}
            onBlur={() => setTimeout(() => setShowResults(false), 200)}
            startContent={<Search className="w-4 h-4 text-gray-500" />}
            classNames={{
              inputWrapper: "bg-netflix-dark border border-white/10 hover:border-netflix-red/50",
              input: "text-white text-sm sm:text-base",
            }}
          />
          {showResults && movies && movies.length > 0 && (
            <div className="absolute top-full left-0 right-0 mt-1 bg-netflix-dark/95 backdrop-blur-xl border border-white/10 rounded-lg shadow-xl z-50 max-h-60 overflow-auto">
              {movies.map((movie) => (
                <button
                  key={movie.tmdb_id}
                  className="w-full flex items-center gap-3 p-3 hover:bg-white/5 text-left transition-colors"
                  onMouseDown={() => { onSelect(movie); setShowResults(false); setQuery(""); }}
                >
                  {movie.poster_url && (
                    <img src={`https://image.tmdb.org/t/p/w92${movie.poster_url}`} alt="" className="w-8 h-11 rounded object-cover" />
                  )}
                  <div>
                    <p className="text-white text-sm sm:text-base">{movie.title}</p>
                    <p className="text-gray-500 text-xs">{movie.year}</p>
                  </div>
                </button>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}
