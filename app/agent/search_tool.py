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

def _clean_search_text(text: str) -> str:
    """Clean search output for LLM consumption.

    Removes: HTML, entities, refs, citation markers, markdown tables,
    tracking params, and normalizes whitespace.
    """
    if not text:
        return ""

    # 1. Strip HTML tags and decode entities
    text = re.sub(r'<[^>]+>', '', text)
    text = html_lib.unescape(text)

    # 2. Remove Wikipedia refs [1], citation markers [citation needed], [edit], etc.
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\[(?:citation needed|edit|clarification needed|when\?|who\?)\]',
                  '', text, flags=re.IGNORECASE)

    # 3. Clean markdown tables: pipes → spaces, remove separator lines
    text = re.sub(r'\|\s*', ' ', text)
    text = re.sub(r'[+][-]+[+]', '', text)

    # 4. Remove tracking params and short garbage lines
    text = re.sub(r'utm_[a-z]+=[^&\s]+&?', '', text)
    lines = [l for l in text.split('\n') if len(l.strip()) > 2 or not l.strip()]
    text = '\n'.join(lines)

    # 5. Normalize whitespace: collapse spaces, limit newlines, trim
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'^\s+$', '', text, flags=re.MULTILINE)

    return text.strip()


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


async def search_structured(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """
    Structured web search returning provenance-preserving result dicts.

    Tries Tavily first (rich metadata), then SearXNG, then Wikipedia.
    Each result contains: content, title, url, source, published_date, score.
    """
    results: List[Dict[str, Any]] = []

    # 1. Tavily (has the best structured metadata)
    try:
        response = await asyncio.to_thread(
            tavily_client.search,
            query=query,
            search_depth="advanced",
            max_results=max_results,
            include_answer=False,
        )
        for res in response.get("results", []):
            results.append({
                "content": _clean_search_text(res.get("content", "")),
                "title": _clean_search_text(res.get("title", "Untitled")),
                "url": res.get("url", ""),
                "source": urllib.parse.urlparse(res.get("url", "")).netloc or "tavily",
                "published_date": res.get("published_date"),
                "score": float(res.get("score", 0.5)),
            })
        if results:
            return results
    except Exception as e:
        logger.warning("Structured Tavily search failed for '%s': %s", query, e)

    # 2. SearXNG
    try:
        searxng_url = settings.SEARXNG_URL.rstrip("/")
        url = f"{searxng_url}/search"
        params = {"q": query, "format": "json", "pageno": 1}
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
        data = response.json()
        for res in data.get("results", [])[:max_results]:
            result_url = res.get("url", "")
            results.append({
                "content": _clean_search_text(res.get("content", "")),
                "title": _clean_search_text(res.get("title", "Untitled")),
                "url": result_url,
                "source": urllib.parse.urlparse(result_url).netloc or res.get("engine", "searxng"),
                "published_date": res.get("publishedDate") or res.get("published_date"),
                "score": float(res.get("score", 0.5)),
            })
        if results:
            return results
    except Exception as e:
        logger.warning("Structured SearXNG search failed for '%s': %s", query, e)

    # 3. Wikipedia fallback (single article)
    try:
        wiki_text = await search_wiki(query)
        if wiki_text:
            results.append({
                "content": wiki_text,
                "title": f"Wikipedia: {query}",
                "url": f"https://en.wikipedia.org/wiki/{urllib.parse.quote(query.replace(' ', '_'))}",
                "source": "wikipedia.org",
                "published_date": None,
                "score": 0.5,
            })
    except Exception as e:
        logger.warning("Structured Wikipedia search failed for '%s': %s", query, e)

    return results