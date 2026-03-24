export interface Movie {
  id?: number;
  tmdb_id: number;
  title: string;
  year: number;
  poster_url: string | null;
  backdrop_url: string | null;
  genre: string;
  overview: string;
  vote_average?: number;
  cast?: string[];
}

export interface TMDBSearchResult {
  id: number;
  title: string;
  release_date: string;
  poster_path: string | null;
  backdrop_path: string | null;
  genre_ids: number[];
  overview: string;
  vote_average: number;
}

export interface MovieSearchResponse {
  results: Movie[];
  total_results: number;
}
