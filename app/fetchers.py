from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

try:
    import trafilatura
except Exception:  # pragma: no cover - optional dependency
    trafilatura = None

from app.config import settings


HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; MistralHackathonBot/1.0)"
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def fetch_html(url: str) -> str:
    response = requests.get(url, headers=HEADERS, timeout=settings.request_timeout_s)
    response.raise_for_status()
    return response.text


def extract_text(url: str, html: str) -> tuple[str, str | None]:
    title = None
    text = ""

    if trafilatura is not None:
        downloaded = trafilatura.extract(html, include_comments=False, include_tables=True)
        if downloaded:
            text = downloaded.strip()

    if not text:
        soup = BeautifulSoup(html, "html.parser")
        if soup.title and soup.title.text:
            title = soup.title.text.strip()
        for element in soup(["script", "style", "noscript"]):
            element.extract()
        text = "\n".join(line.strip() for line in soup.get_text("\n").splitlines() if line.strip())

    if title is None:
        soup = BeautifulSoup(html, "html.parser")
        if soup.title and soup.title.text:
            title = soup.title.text.strip()

    return text, title


def split_text(text: str, chunk_size: int = 1200, chunk_overlap: int = 180) -> list[str]:
    clean = " ".join(text.split())
    if not clean:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(clean):
        end = min(start + chunk_size, len(clean))
        chunk = clean[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(clean):
            break
        start = max(0, end - chunk_overlap)
    return chunks


def make_chunks(url: str, html: str, source_type: str, agent: str) -> list[dict[str, Any]]:
    text, title = extract_text(url, html)
    pieces = split_text(text)
    chunks: list[dict[str, Any]] = []
    for index, piece in enumerate(pieces):
        chunks.append(
            {
                "id": f"{url}::{index}",
                "text": piece,
                "source_url": url,
                "title": title,
                "section": None,
                "source_type": source_type,
                "agent": agent,
                "date_ingestion": utc_now_iso(),
            }
        )
    return chunks


def crawl_seed_pages(seed_urls: list[str], source_type: str, agent: str, max_pages: int = 8) -> list[dict[str, Any]]:
    visited: set[str] = set()
    pending = list(seed_urls)
    collected: list[dict[str, Any]] = []

    while pending and len(visited) < max_pages:
        url = pending.pop(0)
        if url in visited:
            continue
        visited.add(url)

        try:
            html = fetch_html(url)
        except Exception:
            continue

        collected.extend(make_chunks(url, html, source_type=source_type, agent=agent))

        soup = BeautifulSoup(html, "html.parser")
        base_domain = urlparse(url).netloc
        for link in soup.find_all("a", href=True):
            href = link["href"].strip()
            if href.startswith(("mailto:", "javascript:", "#")):
                continue
            absolute = urljoin(url, href)
            parsed = urlparse(absolute)
            if parsed.scheme in {"http", "https"} and parsed.netloc == base_domain:
                if absolute not in visited and absolute not in pending:
                    if len(pending) + len(visited) < max_pages:
                        pending.append(absolute)

    return collected


def search_github_repositories(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    url = "https://api.github.com/search/repositories"
    headers = dict(HEADERS)
    if settings.github_token:
        headers["Authorization"] = f"Bearer {settings.github_token}"
    response = requests.get(url, params={"q": query, "per_page": max_results}, headers=headers, timeout=settings.request_timeout_s)
    response.raise_for_status()
    items = response.json().get("items", [])
    results: list[dict[str, Any]] = []
    for item in items:
        results.append(
            {
                "id": item["html_url"],
                "text": f"Repository: {item['full_name']}\nDescription: {item.get('description') or ''}\nStars: {item.get('stargazers_count', 0)}\nLanguage: {item.get('language') or ''}",
                "source_url": item["html_url"],
                "title": item["full_name"],
                "section": "GitHub search",
                "source_type": "github",
                "agent": "github",
                "date_ingestion": utc_now_iso(),
            }
        )
    return results


def search_arxiv(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    feed_url = "http://export.arxiv.org/api/query"
    response = requests.get(
        feed_url,
        params={
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        },
        headers=HEADERS,
        timeout=settings.request_timeout_s,
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "xml")
    results: list[dict[str, Any]] = []
    for entry in soup.find_all("entry"):
        title = entry.title.text.strip().replace("\n", " ") if entry.title else ""
        summary = entry.summary.text.strip().replace("\n", " ") if entry.summary else ""
        link = entry.id.text.strip() if entry.id else ""
        text = f"Title: {title}\nSummary: {summary}"
        results.append(
            {
                "id": link or title,
                "text": text,
                "source_url": link,
                "title": title,
                "section": "arXiv search",
                "source_type": "paper",
                "agent": "web_research",
                "date_ingestion": utc_now_iso(),
            }
        )
    return results
