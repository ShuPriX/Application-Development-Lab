import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.models.schemas import MovieBase, MovieResponse, MovieSearchResponse
from app.services import tmdb
from app.services.cache import get_or_create_movie, get_recent_movies

router = APIRouter(prefix="/movies", tags=["movies"])


@router.get("/search", response_model=MovieSearchResponse)
async def search_movies(
    query: str = Query(..., min_length=1),
    db: AsyncSession = Depends(get_db),
):
    """Search movies via TMDB API."""
    try:
        results = await tmdb.search_movies(query)
    except httpx.ConnectError:
        return MovieSearchResponse(results=[], total_results=0)

    # Store/update in DB
    movies = []
    for movie_data in results:
        movie = await get_or_create_movie(db, movie_data)
        movies.append(movie)

    return MovieSearchResponse(
        results=[MovieResponse.model_validate(m) for m in movies],
        total_results=len(movies),
    )


@router.get("/recent", response_model=list[MovieResponse])
async def recent_movies(db: AsyncSession = Depends(get_db)):
    """Get recently analyzed movies for home page carousel."""
    movies = await get_recent_movies(db)
    return [MovieResponse.model_validate(m) for m in movies]


@router.get("/trending", response_model=list[MovieBase])
async def trending_movies():
    """Get trending movies for the week from TMDB (pass-through, no DB storage)."""
    try:
        results = await tmdb.get_trending_movies()
    except httpx.ConnectError:
        return []
    return [MovieBase(**m) for m in results]


@router.get("/{tmdb_id}", response_model=MovieResponse)
async def get_movie(tmdb_id: int, db: AsyncSession = Depends(get_db)):
    """Get movie details by TMDB ID."""
    try:
        movie_data = await tmdb.get_movie_details(tmdb_id)
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Could not reach TMDB API, please try again")
    movie = await get_or_create_movie(db, movie_data)
    return MovieResponse.model_validate(movie)
