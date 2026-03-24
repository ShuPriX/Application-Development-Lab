import { motion } from "framer-motion";
import MovieCard from "../home/MovieCard";
import type { Movie } from "../../types/movie";

interface SearchResultsProps {
  movies: Movie[];
  query: string;
}

export default function SearchResults({ movies, query }: SearchResultsProps) {
  return (
    <div className="px-6 py-8">
      <h2 className="text-white text-xl font-semibold mb-6">
        Results for &ldquo;{query}&rdquo;
      </h2>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-4"
      >
        {movies.map((movie) => (
          <MovieCard key={movie.tmdb_id} movie={movie} />
        ))}
      </motion.div>
    </div>
  );
}
