import asyncio

import httpx
from app.config import settings


TMDB_BASE = "https://api.themoviedb.org/3"

# Force IPv4 — TMDB's IPv6 endpoints have flaky TLS on some networks
_transport = httpx.AsyncHTTPTransport(local_address="0.0.0.0")

GENRE_MAP = {
    28: "Action", 12: "Adventure", 16: "Animation", 35: "Comedy",
    80: "Crime", 99: "Documentary", 18: "Drama", 10751: "Family",
    14: "Fantasy", 36: "History", 27: "Horror", 10402: "Music",
    9648: "Mystery", 10749: "Romance", 878: "Sci-Fi", 10770: "TV Movie",
    53: "Thriller", 10752: "War", 37: "Western",
}


async def _request_with_retry(client: httpx.AsyncClient, url: str, params: dict, retries: int = 3, delay: float = 2.0) -> httpx.Response:
    """Make an HTTP GET request with retry logic for ConnectError."""
    for attempt in range(retries + 1):
        try:
            resp = await client.get(url, params=params, timeout=15)
            resp.raise_for_status()
            return resp
        except httpx.ConnectError:
            if attempt < retries:
                await asyncio.sleep(delay * (attempt + 1))
            else:
                raise


async def search_movies(query: str) -> list[dict]:
    async with httpx.AsyncClient(transport=_transport) as client:
        resp = await _request_with_retry(
            client,
            f"{TMDB_BASE}/search/movie",
            params={"api_key": settings.TMDB_API_KEY, "query": query, "language": "en-US"},
        )
        data = resp.json()

    results = []
    for item in data.get("results", [])[:20]:
        year = None
        if item.get("release_date"):
            try:
                year = int(item["release_date"][:4])
            except (ValueError, IndexError):
                pass
        genres = ", ".join(
            GENRE_MAP.get(gid, "") for gid in item.get("genre_ids", []) if gid in GENRE_MAP
        )
        results.append({
            "tmdb_id": item["id"],
            "title": item["title"],
            "year": year,
            "poster_url": item.get("poster_path"),
            "backdrop_url": item.get("backdrop_path"),
            "genre": genres,
            "overview": item.get("overview", ""),
            "vote_average": item.get("vote_average"),
        })
    return results


async def get_movie_details(tmdb_id: int) -> dict:
    async with httpx.AsyncClient(transport=_transport) as client:
        resp = await _request_with_retry(
            client,
            f"{TMDB_BASE}/movie/{tmdb_id}",
            params={"api_key": settings.TMDB_API_KEY, "language": "en-US"},
        )
        item = resp.json()

    year = None
    if item.get("release_date"):
        try:
            year = int(item["release_date"][:4])
        except (ValueError, IndexError):
            pass

    genres = ", ".join(g["name"] for g in item.get("genres", []))

    return {
        "tmdb_id": item["id"],
        "title": item["title"],
        "year": year,
        "poster_url": item.get("poster_path"),
        "backdrop_url": item.get("backdrop_path"),
        "genre": genres,
        "overview": item.get("overview", ""),
        "vote_average": item.get("vote_average"),
    }


async def get_movie_reviews(tmdb_id: int, max_pages: int = 5) -> list[dict]:
    """Fetch user reviews for a movie from TMDB API."""
    from datetime import datetime

    reviews = []
    async with httpx.AsyncClient(transport=_transport) as client:
        for page in range(1, max_pages + 1):
            resp = await _request_with_retry(
                client,
                f"{TMDB_BASE}/movie/{tmdb_id}/reviews",
                params={
                    "api_key": settings.TMDB_API_KEY,
                    "language": "en-US",
                    "page": page,
                },
            )
            data = resp.json()
            for item in data.get("results", []):
                content = item.get("content", "").strip()
                if len(content) < 20:
                    continue
                review_date = None
                if item.get("created_at"):
                    try:
                        review_date = datetime.fromisoformat(
                            item["created_at"].replace("Z", "+00:00")
                        )
                    except (ValueError, TypeError):
                        pass
                rating = None
                author_details = item.get("author_details", {})
                if author_details.get("rating"):
                    rating = float(author_details["rating"])
                reviews.append({
                    "source": "tmdb",
                    "author": item.get("author", "Anonymous"),
                    "content": content,
                    "rating": rating,
                    "review_date": review_date,
                })
            if page >= data.get("total_pages", 1):
                break
    return reviews


async def get_trending_movies() -> list[dict]:
    """Fetch trending movies for the week from TMDB."""
    async with httpx.AsyncClient(transport=_transport) as client:
        resp = await _request_with_retry(
            client,
            f"{TMDB_BASE}/trending/movie/week",
            params={"api_key": settings.TMDB_API_KEY, "language": "en-US"},
        )
        data = resp.json()

    results = []
    for item in data.get("results", [])[:20]:
        year = None
        if item.get("release_date"):
            try:
                year = int(item["release_date"][:4])
            except (ValueError, IndexError):
                pass
        genres = ", ".join(
            GENRE_MAP.get(gid, "") for gid in item.get("genre_ids", []) if gid in GENRE_MAP
        )
        results.append({
            "tmdb_id": item["id"],
            "title": item["title"],
            "year": year,
            "poster_url": item.get("poster_path"),
            "backdrop_url": item.get("backdrop_path"),
            "genre": genres,
            "overview": item.get("overview", ""),
            "vote_average": item.get("vote_average"),
        })
    return results
