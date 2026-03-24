import axios from "axios";
import type { Movie, MovieSearchResponse } from "../types/movie";
import type { AnalysisResult } from "../types/analysis";

const API_BASE = import.meta.env.VITE_API_URL || "";

const api = axios.create({
  baseURL: `${API_BASE}/api`,
  timeout: 120000,
});

export async function searchMovies(query: string): Promise<Movie[]> {
  const { data } = await api.get<MovieSearchResponse>("/movies/search", {
    params: { query },
  });
  return data.results;
}

export async function getMovieDetails(tmdbId: number): Promise<Movie> {
  const { data } = await api.get<Movie>(`/movies/${tmdbId}`);
  return data;
}

export async function getAnalysis(
  tmdbId: number,
): Promise<AnalysisResult | null> {
  try {
    const { data } = await api.get<AnalysisResult>(
      `/movies/${tmdbId}/analysis`,
    );
    return data;
  } catch (err) {
    if (axios.isAxiosError(err) && err.response?.status === 404) {
      return null;
    }
    throw err;
  }
}

export async function triggerAnalysis(tmdbId: number): Promise<AnalysisResult> {
  const { data } = await api.post<AnalysisResult>(
    `/movies/${tmdbId}/analyze`,
  );
  return data;
}

export async function getRecentMovies(): Promise<Movie[]> {
  const { data } = await api.get<Movie[]>("/movies/recent");
  return data;
}

export async function getTrendingMovies(): Promise<Movie[]> {
  const { data } = await api.get<Movie[]>("/movies/trending");
  return data;
}

export function createAnalysisWebSocket(tmdbId: number): WebSocket {
  const wsUrl = API_BASE
    ? API_BASE.replace(/^http/, "ws")
    : `ws://${window.location.host}`;
  return new WebSocket(`${wsUrl}/ws/analysis/${tmdbId}`);
}

export default api;
