import { useState, useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { searchMovies } from "../services/api";

function useDebouncedValue(value: string, delay = 400) {
  const [debounced, setDebounced] = useState(value);

  useEffect(() => {
    const timer = setTimeout(() => setDebounced(value), delay);
    return () => clearTimeout(timer);
  }, [value, delay]);

  return debounced;
}

export function useMovieSearch(query: string) {
  const debouncedQuery = useDebouncedValue(query);

  return useQuery({
    queryKey: ["movieSearch", debouncedQuery],
    queryFn: () => searchMovies(debouncedQuery),
    enabled: debouncedQuery.length >= 2,
    staleTime: 5 * 60 * 1000,
  });
}
