from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


@dataclass
class SourceDocument:
    url: str
    title: str
    text: str


class WebScraper:
    """Fetches and cleans HTML content from the public web."""

    def __init__(self, timeout: int = 20) -> None:
        self._session = requests.Session()
        self._timeout = timeout
        self._session.headers.update(
            {
                "User-Agent": "CognitEdgeScraper/0.1 (+https://zerosignal.ai)",
                "Accept-Language": "en-US,en;q=0.9",
            }
        )

    def fetch(self, url: str) -> SourceDocument:
        logger.info("Fetching %s", url)
        response = self._session.get(url, timeout=self._timeout)
        response.raise_for_status()
        text, title = self._clean_html(response.text, response.url)
        return SourceDocument(url=response.url, title=title, text=text)

    @staticmethod
    def _clean_html(html: str, fallback_title: Optional[str]) -> tuple[str, str]:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
            tag.decompose()
        title = fallback_title or ""
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
        text = " ".join(part.strip() for part in soup.stripped_strings)
        return text, title
