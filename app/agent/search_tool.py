import asyncio
import html as html_lib
import json
import logging
import re
import time
import urllib.parse
from typing import Optional, Union, List, Dict, Any
import httpx
from bs4 import BeautifulSoup

from app.core.config import settings
from app.documents.clients import tavily_client

logger = logging.getLogger(__name__)


# ── Text cleaning for LLM consumption ────────────────────────────────────────

def _clean_search_text(text: str, max_chars: int = 1800) -> str:
    """Clean search/snippet text for LLM consumption and evidence display.

    Removes HTML, nav chrome, tracking junk, citation markers; keeps numbers/dates.
    """
    if not text:
        return ""

    text = re.sub(r"<[^>]+>", " ", text)
    text = html_lib.unescape(text)

    # Drop common web chrome / cookie / share noise lines
    noise = re.compile(
        r"(?i)^\s*(?:cookie|subscribe|sign up|log in|menu|share|follow us|"
        r"advertisement|related articles?|read more|click here|privacy policy|"
        r"terms of (?:use|service)|all rights reserved).*$",
        re.MULTILINE,
    )
    text = noise.sub(" ", text)

    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(
        r"\[(?:citation needed|edit|clarification needed|when\?|who\?)\]",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\|\s*", " ", text)
    text = re.sub(r"[+][-]+[+]", "", text)
    text = re.sub(r"utm_[a-z]+=[^&\s]+&?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"https?://\S+", "", text)  # URLs live in metadata; strip from body

    lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            lines.append("")
            continue
        if len(stripped) <= 2:
            continue
        # Keep lines that look factual (digits, %, years) even if short
        if len(stripped) < 25 and not re.search(r"\d", stripped):
            continue
        lines.append(stripped)
    text = "\n".join(lines)

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"^\s+$", "", text, flags=re.MULTILINE)
    text = text.strip()

    if len(text) > max_chars:
        # Prefer cutting on sentence boundary near the limit
        cut = text[:max_chars]
        last_stop = max(cut.rfind(". "), cut.rfind("; "), cut.rfind("\n"))
        if last_stop > max_chars // 2:
            cut = cut[: last_stop + 1]
        text = cut.strip() + "…"
    return text


def _normalize_result(
    *,
    content: str,
    title: str,
    url: str,
    source: str,
    published_date: Any = None,
    score: float = 0.5,
) -> Dict[str, Any] | None:
    cleaned = _clean_search_text(content)
    if len(cleaned) < 40:
        return None
    return {
        "content": cleaned,
        "title": _clean_search_text(title, max_chars=200) or "Untitled",
        "url": url or "",
        "source": source or (urllib.parse.urlparse(url or "").netloc or "web"),
        "published_date": published_date,
        "score": float(score or 0.5),
    }


def _dedupe_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: set[str] = set()
    out: List[Dict[str, Any]] = []
    for r in results:
        key = (r.get("url") or "").rstrip("/").lower() or (r.get("content") or "")[:120].lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(r)
    out.sort(key=lambda x: float(x.get("score") or 0), reverse=True)
    return out


def _osint_query_variants(query: str) -> list[str]:
    """Expand a query into a small set of OSINT-oriented search strings."""
    q = query.strip()
    if not q:
        return []
    year = time.strftime("%Y")
    variants = [q]
    # Prefer recent official / statistical phrasing without changing user intent
    if not re.search(r"\b(20\d{2}|latest|current|recent)\b", q, re.I):
        variants.append(f"{q} {year}")
        variants.append(f"{q} latest official statistics")
    # Deduplicate while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for v in variants:
        key = v.lower()
        if key not in seen:
            seen.add(key)
            out.append(v)
    return out[:3]


def _extract_wiki_text(soup: BeautifulSoup) -> Optional[str]:
    """Extract the main article text from a Wikipedia page."""
    if not soup:
        return None

    heading = soup.find(id="firstHeading")
    title = heading.get_text() if heading else ""

    body_content = soup.find(id="bodyContent")
    if not body_content:
        return _clean_search_text(title) if title else None

    paragraphs = body_content.find_all("p")
    cleaned = []
    for p in paragraphs:
        text = p.get_text().strip()
        if text:
            cleaned.append(text)

    if not cleaned:
        return _clean_search_text(title) if title else None

    body = "\n\n".join(cleaned)
    result = f"{title}\n\n{body}" if title else body
    return _clean_search_text(result)


async def search_wiki(query: str, lang: str = "en") -> Optional[str]:
    start = time.perf_counter()
    clean_query = query.strip().replace(" ", "_")
    encoded_query = urllib.parse.quote(clean_query)
    
    base_domain = f"https://{lang}.wikipedia.org"
    direct_url = f"{base_domain}/wiki/{encoded_query}"
    
    headers = {
        "User-Agent": "MySelfCorrectingRAGBot/1.0 (contact@your-domain.com)"
    }

    async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
        response = None
        
        try:
            response = await client.get(direct_url, headers=headers)
            response.raise_for_status()
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.info("Direct Wikipedia page missing (404). Falling back to search for: %s", query)
                search_url = f"{base_domain}/w/index.php"
                search_params = {"search": query}
                
                try:
                    response = await client.get(search_url, params=search_params, headers=headers)
                    response.raise_for_status()
                except httpx.HTTPError as search_err:
                    logger.error("Wikipedia search backup route failed: %s", str(search_err))
                    return None
            else:
                logger.warning("Wikipedia returned status code %s for query: %s", e.response.status_code, query)
                return None
                
        except httpx.RequestError as e:
            logger.error("Primary network request failed for Wikipedia query '%s': %s", query, str(e))
            return None

        if not response:
            return None

        soup = BeautifulSoup(response.text, "html.parser")

        if soup.title and "Search results" in soup.title.text:
            first_res = soup.find("div", class_="mw-search-result-heading") or soup.find("div", id="mw-search-result-heading")
            if not first_res:
                logger.info("No matching Wikipedia search results found for: %s", query)
                return None

            link_tag = first_res.find("a")
            if not link_tag or not link_tag.get("href"):
                return None

            target_url = f"{base_domain}{link_tag['href']}"
            try:
                response = await client.get(target_url, headers=headers)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, "html.parser")
            except httpx.HTTPError as e:
                logger.error("Failed to fetch article from search result link: %s", str(e))
                return None

        result = _extract_wiki_text(soup)
        elapsed = time.perf_counter() - start
        logger.info("[wiki] '%s' — %.1fs", query[:40], elapsed)
        return result


async def search_tavily(query: str) -> Optional[str]:
    """
    Search the web via Tavily AI and return formatted snippets.
    Runs in a thread-pool to prevent blocking FastAPI's async event loop.
    Uses advanced search depth for better quality results.
    """
    start = time.perf_counter()
    try:
        response = await asyncio.to_thread(
            tavily_client.search,
            query=query,
            search_depth="advanced",
            max_results=10,
            include_answer=True,
        )
    except Exception as e:
        logger.error("Tavily search failed for query '%s': %s", query, str(e))
        return None

    results = response.get("results", [])
    if not results:
        logger.info("No Tavily results found for query: %s", query)
        return None

    snippets = []
    for idx, res in enumerate(results):
        title = _clean_search_text(res.get("title", "Untitled"))
        content = _clean_search_text(res.get("content", ""))
        url = res.get("url", "")
        if content:  # only include results with actual content
            snippets.append(f"Source={idx} [{url}]: {title}\nContent: {content}")

    # Include Tavily's synthesized answer if available
    answer = _clean_search_text(response.get("answer", ""))
    if answer:
        snippets.insert(0, f"Tavily Summary: {answer}")

    combined = "\n\n".join(snippets)
    elapsed = time.perf_counter() - start
    logger.info("[tavily] '%s' %d results — %.1fs", query[:40], len(results), elapsed)
    return combined


async def search_searxng(query: str) -> Optional[str]:
    """
    Search via local SearXNG instance (meta-search engine).
    Aggregates results from Google, Bing, DuckDuckGo, Wikipedia, and more.
    """
    start = time.perf_counter()
    searxng_url = settings.SEARXNG_URL.rstrip("/")
    url = f"{searxng_url}/search"
    params = {
        "q": query,
        "format": "json",
        "pageno": 1,
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
    except httpx.HTTPError as e:
        logger.error("SearXNG search failed for query '%s': %s", query, str(e))
        return None

    data = response.json()
    results = data.get("results", [])
    if not results:
        logger.info("No SearXNG results found for query: %s", query)
        return None

    # Take top 15 results (SearXNG returns 30-46, let reranker decide the best)
    snippets = []
    for idx, res in enumerate(results[:15]):
        title = _clean_search_text(res.get("title", "Untitled"))
        content = _clean_search_text(res.get("content", ""))
        result_url = res.get("url", "")
        engine = res.get("engine", "")
        score = res.get("score", 0)
        if content:  # only include results with actual content
            snippets.append(f"Source={idx} [{result_url}] (via {engine}, score={score:.1f}): {title}\nContent: {content}")

    combined = "\n\n".join(snippets)
    elapsed = time.perf_counter() - start
    logger.info("[searxng] '%s' %d results — %.1fs", query[:40], len(results), elapsed)
    return combined


async def search_web_fallback(query: str) -> Optional[str]:
    """Fallback strategy: Try SearXNG first (broadest coverage), then Wikipedia, then Tavily."""
    searxng_result = await search_searxng(query)
    if searxng_result:
        return searxng_result

    wiki_result = await search_wiki(query)
    if wiki_result:
        return wiki_result

    tavily_result = await search_tavily(query)
    if tavily_result:
        return tavily_result

    logger.warning("All search sources failed for query: %s", query)
    return None


async def _execute_single_item(data: Dict[str, str]) -> str:
    """Worker helper for parallel execution."""
    flag = data.get('intent', '').lower()
    query = data.get('query', '')

    if not query or not flag:
        logger.warning("Invalid item payload received in search worker: %s", data)
        return ""

    if flag == 'wiki':
        result = await search_wiki(query=query)
    elif flag == 'current':
        result = await search_tavily(query=query)
    else:
        result = await search_web_fallback(query=query)

    return result or ""


async def smart_search(req: Union[str, List[Dict[str, str]]]) -> str:
    """
    Parses JSON array and fires ALL query fetches concurrently via asyncio.gather.
    """
    if isinstance(req, str):
        data_sheet = json.loads(req)
    else:
        data_sheet = req

    if not isinstance(data_sheet, list):
        raise ValueError("Expected a list of query objects.")

    # FIX: Run all sub-queries in PARALLEL simultaneously
    tasks = [_execute_single_item(item) for item in data_sheet]
    results = await asyncio.gather(*tasks)

    # Filter out empty responses and join with visual separators
    valid_results = [r for r in results if r]
    return "\n\n--- SEARCH RESULT ---\n\n".join(valid_results)


async def _tavily_structured(
    query: str,
    max_results: int,
    *,
    topic: str = "general",
    time_range: str | None = "year",
) -> List[Dict[str, Any]]:
    kwargs: Dict[str, Any] = {
        "query": query,
        "search_depth": "advanced",
        "max_results": max_results,
        "include_answer": False,
        "topic": topic,
    }
    if time_range:
        kwargs["time_range"] = time_range
    if topic == "news":
        kwargs["days"] = 365

    response = await asyncio.to_thread(tavily_client.search, **kwargs)
    out: List[Dict[str, Any]] = []
    for res in response.get("results", []):
        item = _normalize_result(
            content=res.get("content", "") or res.get("raw_content", "") or "",
            title=res.get("title", "Untitled"),
            url=res.get("url", ""),
            source=urllib.parse.urlparse(res.get("url", "")).netloc or "tavily",
            published_date=res.get("published_date"),
            score=float(res.get("score", 0.5)),
        )
        if item:
            out.append(item)
    return out


async def _searxng_structured(query: str, max_results: int) -> List[Dict[str, Any]]:
    searxng_url = settings.SEARXNG_URL.rstrip("/")
    url = f"{searxng_url}/search"
    params = {
        "q": query,
        "format": "json",
        "pageno": 1,
        "time_range": "year",
        "categories": "general,news",
    }
    async with httpx.AsyncClient(timeout=12.0) as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
    data = response.json()
    out: List[Dict[str, Any]] = []
    for res in data.get("results", [])[: max_results * 2]:
        result_url = res.get("url", "")
        item = _normalize_result(
            content=res.get("content", "") or res.get("snippet", "") or "",
            title=res.get("title", "Untitled"),
            url=result_url,
            source=urllib.parse.urlparse(result_url).netloc or res.get("engine", "searxng"),
            published_date=res.get("publishedDate") or res.get("published_date"),
            score=float(res.get("score", 0.5)),
        )
        if item:
            out.append(item)
    return out[:max_results]


async def search_structured(
    query: str,
    max_results: int = 10,
    *,
    user_id=None,
    allow_tavily: bool = True,
) -> List[Dict[str, Any]]:
    """
    Multi-source OSINT-style web search with recency bias and cleaned snippets.

    Runs Tavily (general + news) and SearXNG in parallel, merges/dedupes, then
    falls back to Wikipedia only if nothing else returned.
    """
    start = time.perf_counter()
    variants = _osint_query_variants(query)
    primary = variants[0] if variants else query

    async def _safe(coro, label: str) -> List[Dict[str, Any]]:
        try:
            return await coro
        except Exception as exc:
            logger.warning("Structured %s search failed for '%s': %s", label, primary[:60], exc)
            return []

    tasks = []
    if allow_tavily:
        tasks.append(
            _safe(_tavily_structured(primary, max_results, topic="general", time_range="year"), "tavily-general")
        )
        tasks.append(
            _safe(
                _tavily_structured(primary, max(4, max_results // 2), topic="news", time_range="year"),
                "tavily-news",
            )
        )
        if len(variants) > 1:
            tasks.append(
                _safe(
                    _tavily_structured(variants[1], max(3, max_results // 3), topic="general", time_range="year"),
                    "tavily-variant",
                )
            )
    tasks.append(_safe(_searxng_structured(primary, max_results), "searxng"))

    batches = await asyncio.gather(*tasks)
    if allow_tavily and user_id is not None:
        try:
            from app.core.database import AsyncLocalSession
            from app.core.usage import record_usage

            async with AsyncLocalSession() as session:
                await record_usage(session, user_id, "tavily", amount=1)
        except Exception:
            logger.exception("Failed to record Tavily usage")

    merged: List[Dict[str, Any]] = []
    for batch in batches:
        merged.extend(batch)

    results = _dedupe_results(merged)[:max_results]

    if not results:
        try:
            wiki_text = await search_wiki(query)
            if wiki_text:
                item = _normalize_result(
                    content=wiki_text,
                    title=f"Wikipedia: {query}",
                    url=f"https://en.wikipedia.org/wiki/{urllib.parse.quote(query.replace(' ', '_'))}",
                    source="wikipedia.org",
                    published_date=None,
                    score=0.4,
                )
                if item:
                    results = [item]
        except Exception as e:
            logger.warning("Structured Wikipedia search failed for '%s': %s", query, e)

    logger.info(
        "[search_structured] '%s' → %d cleaned results — %.1fs (tavily=%s)",
        primary[:50],
        len(results),
        time.perf_counter() - start,
        allow_tavily,
    )
    return results
