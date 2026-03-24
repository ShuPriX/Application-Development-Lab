import logging

from .base import BaseScraper, ScrapedReview

logger = logging.getLogger(__name__)


class RottenTomatoesScraper(BaseScraper):
    """Rotten Tomatoes scraper.

    NOTE: RT reviews are fully JS-rendered and there is no public API,
    so this scraper cannot extract reviews with httpx + BS4.
    Kept as a stub so the analysis pipeline doesn't break; may be
    replaced with a headless-browser approach in the future.
    """

    async def scrape(self, movie_title: str, movie_year: int | None = None) -> list[ScrapedReview]:
        logger.info(
            f"RT scraper skipped for {movie_title} — reviews are JS-rendered and not scrapable"
        )
        return []
