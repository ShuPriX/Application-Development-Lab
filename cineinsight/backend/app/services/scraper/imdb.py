import logging
import re
from datetime import datetime
from urllib.parse import quote_plus

from bs4 import BeautifulSoup

from .base import BaseScraper, ScrapedReview

logger = logging.getLogger(__name__)


class IMDBScraper(BaseScraper):
    async def scrape(self, movie_title: str, movie_year: int | None = None) -> list[ScrapedReview]:
        try:
            imdb_id = await self._find_movie_id(movie_title, movie_year)
            if not imdb_id:
                logger.warning(f"Could not find IMDB ID for {movie_title}")
                return []
            return await self._scrape_reviews(imdb_id)
        except Exception as e:
            logger.error(f"IMDB scraping failed for {movie_title}: {e}")
            return []

    async def _find_movie_id(self, title: str, year: int | None) -> str | None:
        search_query = f"{title} {year}" if year else title
        url = f"https://www.imdb.com/find/?q={quote_plus(search_query)}&s=tt&ttype=ft"
        html = await self._get(url)
        soup = BeautifulSoup(html, "lxml")

        for a_tag in soup.select("a[href*='/title/tt']"):
            href = a_tag.get("href", "")
            match = re.search(r"/title/(tt\d+)", href)
            if match:
                return match.group(1)
        return None

    async def _scrape_reviews(self, imdb_id: str) -> list[ScrapedReview]:
        url = f"https://www.imdb.com/title/{imdb_id}/reviews?sort=submissionDate&dir=desc"
        html = await self._get(url)
        soup = BeautifulSoup(html, "lxml")

        reviews = []

        # New IMDB layout (2024+): article.user-review-item
        articles = soup.select("article.user-review-item")
        for article in articles[: self.max_reviews]:
            # Content: inside [data-testid=review-overflow] .ipc-html-content-inner-div
            content_el = article.select_one(
                "[data-testid=review-overflow] .ipc-html-content-inner-div"
            )
            if not content_el:
                continue
            content = content_el.get_text(strip=True)
            if len(content) < 20:
                continue

            # Rating: .ipc-rating-star--rating
            rating = None
            rating_el = article.select_one(".ipc-rating-star--rating")
            if rating_el:
                try:
                    rating = float(rating_el.get_text(strip=True))
                except ValueError:
                    pass

            # Author: [data-testid=author-link] span (last text node)
            author = "Anonymous"
            author_el = article.select_one("[data-testid=author-link] span")
            if author_el:
                author = author_el.get_text(strip=True) or "Anonymous"

            # Date: li.review-date
            review_date = None
            date_el = article.select_one("li.review-date")
            if date_el:
                date_text = date_el.get_text(strip=True)
                for fmt in ("%b %d, %Y", "%d %B %Y", "%B %d, %Y"):
                    try:
                        review_date = datetime.strptime(date_text, fmt)
                        break
                    except ValueError:
                        continue

            reviews.append(
                ScrapedReview(
                    source="imdb",
                    author=author,
                    content=content,
                    rating=rating,
                    review_date=review_date,
                )
            )

        # Fallback: old IMDB layout (.review-container / .lister-item)
        if not reviews:
            for container in soup.select(".review-container, .lister-item")[
                : self.max_reviews
            ]:
                content_el = container.select_one(".text, .content .text")
                if not content_el:
                    continue
                content = content_el.get_text(strip=True)
                if len(content) < 20:
                    continue

                author_el = container.select_one(
                    ".display-name-link a, .lister-item-header a"
                )
                author = author_el.get_text(strip=True) if author_el else "Anonymous"

                rating = None
                rating_el = container.select_one(
                    ".rating-other-user-rating span, .ipl-ratings-bar span"
                )
                if rating_el:
                    try:
                        rating = float(rating_el.get_text(strip=True))
                    except ValueError:
                        pass

                review_date = None
                date_el = container.select_one(".review-date, .lister-item-date")
                if date_el:
                    try:
                        review_date = datetime.strptime(
                            date_el.get_text(strip=True), "%d %B %Y"
                        )
                    except ValueError:
                        pass

                reviews.append(
                    ScrapedReview(
                        source="imdb",
                        author=author,
                        content=content,
                        rating=rating,
                        review_date=review_date,
                    )
                )

        logger.info(f"Scraped {len(reviews)} IMDB reviews for {imdb_id}")
        return reviews
