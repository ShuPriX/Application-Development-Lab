import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime

import httpx
from app.config import settings

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}


@dataclass
class ScrapedReview:
    source: str
    author: str
    content: str
    rating: float | None = None
    review_date: datetime | None = None


class BaseScraper:
    def __init__(self):
        self.delay = settings.SCRAPE_DELAY
        self.max_reviews = settings.MAX_REVIEWS_PER_SOURCE

    async def _get(self, url: str) -> str:
        async with httpx.AsyncClient(headers=HEADERS, follow_redirects=True) as client:
            await asyncio.sleep(self.delay)
            resp = await client.get(url, timeout=15)
            resp.raise_for_status()
            return resp.text

    async def scrape(self, movie_title: str, movie_year: int | None = None) -> list[ScrapedReview]:
        raise NotImplementedError
