import { useRef } from "react";
import { motion } from "framer-motion";
import { ChevronLeft, ChevronRight } from "lucide-react";
import MovieCard from "./MovieCard";
import type { Movie } from "../../types/movie";

interface MovieCarouselProps {
  title: string;
  movies: Movie[];
}

export default function MovieCarousel({ title, movies }: MovieCarouselProps) {
  const scrollRef = useRef<HTMLDivElement>(null);

  const scroll = (direction: "left" | "right") => {
    if (scrollRef.current) {
      const amount = direction === "left" ? -400 : 400;
      scrollRef.current.scrollBy({ left: amount, behavior: "smooth" });
    }
  };

  if (movies.length === 0) return null;

  return (
    <motion.section initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="mb-10">
      <h2 className="text-white text-lg sm:text-xl md:text-2xl font-semibold mb-4 px-4 sm:px-6">{title}</h2>
      <div className="relative group/carousel">
        <button
          onClick={() => scroll("left")}
          className="absolute left-0 top-0 bottom-8 z-20 w-10 sm:w-12 bg-gradient-to-r from-[#141414] to-transparent flex items-center justify-center opacity-0 group-hover/carousel:opacity-100 transition-opacity"
        >
          <ChevronLeft className="w-6 h-6 sm:w-8 sm:h-8 text-white" />
        </button>
        <div ref={scrollRef} className="flex gap-2 sm:gap-3 overflow-x-auto hide-scrollbar px-4 sm:px-6 pb-4">
          {movies.map((movie) => (
            <MovieCard key={movie.tmdb_id} movie={movie} />
          ))}
        </div>
        <button
          onClick={() => scroll("right")}
          className="absolute right-0 top-0 bottom-8 z-20 w-10 sm:w-12 bg-gradient-to-l from-[#141414] to-transparent flex items-center justify-center opacity-0 group-hover/carousel:opacity-100 transition-opacity"
        >
          <ChevronRight className="w-6 h-6 sm:w-8 sm:h-8 text-white" />
        </button>
      </div>
    </motion.section>
  );
}
