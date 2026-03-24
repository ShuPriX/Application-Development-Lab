import { Chip } from "@heroui/react";
import { motion } from "framer-motion";
import { useNavigate } from "react-router-dom";
import { Star } from "lucide-react";
import type { Movie } from "../../types/movie";

const TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p";

interface MovieCardProps {
  movie: Movie;
  showVerdict?: string;
}

export default function MovieCard({ movie, showVerdict }: MovieCardProps) {
  const navigate = useNavigate();
  const posterUrl = movie.poster_url ? `${TMDB_IMAGE_BASE}/w342${movie.poster_url}` : null;

  return (
    <motion.div
      whileHover={{ scale: 1.05, zIndex: 10 }}
      transition={{ duration: 0.2 }}
      className="cursor-pointer flex-shrink-0 w-[140px] sm:w-[160px] md:w-[180px] group"
      onClick={() => navigate(`/analysis/${movie.tmdb_id}`)}
    >
      <div className="bg-netflix-card overflow-hidden rounded-lg shadow-lg group-hover:shadow-netflix-red/20 group-hover:ring-1 group-hover:ring-netflix-red/30 transition-all">
        <div className="relative aspect-[2/3]">
          {posterUrl ? (
            <img src={posterUrl} alt={movie.title} className="w-full h-full object-cover" loading="lazy" />
          ) : (
            <div className="w-full h-full bg-netflix-dark flex items-center justify-center">
              <span className="text-gray-600 text-4xl">?</span>
            </div>
          )}
          {/* Hover overlay */}
          <div className="absolute inset-0 bg-gradient-to-t from-black/90 via-black/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity flex flex-col justify-end p-3">
            <p className="text-white text-sm font-semibold line-clamp-2">{movie.title}</p>
            <div className="flex items-center gap-2 mt-1">
              <span className="text-gray-400 text-xs">{movie.year}</span>
              {movie.vote_average != null && movie.vote_average > 0 && (
                <span className="flex items-center gap-0.5 text-yellow-400 text-xs">
                  <Star className="w-3 h-3 fill-yellow-400" />
                  {movie.vote_average.toFixed(1)}
                </span>
              )}
            </div>
            {showVerdict && (
              <Chip size="sm" className="mt-2 text-xs" color={showVerdict.includes("Recommend") || showVerdict.includes("Worth") ? "success" : showVerdict.includes("Mixed") ? "warning" : "danger"}>
                {showVerdict}
              </Chip>
            )}
          </div>
        </div>
      </div>
      <p className="text-gray-300 text-xs sm:text-sm mt-2 font-medium line-clamp-1 px-1">{movie.title}</p>
      <p className="text-gray-600 text-xs px-1">{movie.year}</p>
    </motion.div>
  );
}
