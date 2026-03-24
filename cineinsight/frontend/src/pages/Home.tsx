import { useQuery } from "@tanstack/react-query";
import { Skeleton } from "@heroui/react";
import HeroBanner from "../components/home/HeroBanner";
import MovieCarousel from "../components/home/MovieCarousel";
import { getRecentMovies, getTrendingMovies } from "../services/api";

function CarouselSkeleton() {
  return (
    <div className="px-6 space-y-4">
      <Skeleton className="h-8 w-48 rounded-lg bg-white/5" />
      <div className="flex gap-3">
        {Array.from({ length: 6 }).map((_, i) => (
          <Skeleton
            key={i}
            className="w-[180px] h-[270px] rounded-md bg-white/5 flex-shrink-0"
          />
        ))}
      </div>
    </div>
  );
}

export default function Home() {
  const { data: recentMovies, isLoading: isLoadingRecent } = useQuery({
    queryKey: ["recentMovies"],
    queryFn: getRecentMovies,
    staleTime: 5 * 60 * 1000,
  });

  const { data: trendingMovies, isLoading: isLoadingTrending } = useQuery({
    queryKey: ["trendingMovies"],
    queryFn: getTrendingMovies,
    staleTime: 10 * 60 * 1000,
  });

  const trendingThisWeek = trendingMovies?.slice(0, 10) ?? [];
  const popularMovies = trendingMovies?.slice(10, 20) ?? [];

  return (
    <div className="min-h-screen">
      <HeroBanner />

      <div className="mt-8 space-y-2">
        {/* Trending This Week */}
        {isLoadingTrending ? (
          <CarouselSkeleton />
        ) : (
          trendingThisWeek.length > 0 && (
            <MovieCarousel title="Trending This Week" movies={trendingThisWeek} />
          )
        )}

        {/* Recently Analyzed */}
        {isLoadingRecent ? (
          <CarouselSkeleton />
        ) : (
          recentMovies &&
          recentMovies.length > 0 && (
            <MovieCarousel title="Recently Analyzed" movies={recentMovies} />
          )
        )}

        {/* Popular Movies */}
        {isLoadingTrending ? (
          <CarouselSkeleton />
        ) : (
          popularMovies.length > 0 && (
            <MovieCarousel title="Popular Movies" movies={popularMovies} />
          )
        )}
      </div>
    </div>
  );
}
