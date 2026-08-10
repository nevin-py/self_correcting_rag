import asyncio
import json
import logging
import urllib.parse
from typing import Optional, Union, List, Dict
import httpx
from bs4 import BeautifulSoup

from app.core.config import settings
from app.documents.clients import tavily_client

logger = logging.getLogger(__name__)


def _extract_wiki_text(soup: BeautifulSoup) -> Optional[str]:
    """Extract the main article text from a Wikipedia page."""
    if not soup:
        return None

    heading = soup.find(id="firstHeading")
    title = heading.get_text() if heading else ""

    body_content = soup.find(id="bodyContent")
    if not body_content:
        return title

    paragraphs = body_content.find_all("p")
    cleaned = []
    for p in paragraphs:
        text = p.get_text().strip()
        if text:
            cleaned.append(text)

    if not cleaned:
        return title if title else None

    body = "\n\n".join(cleaned)
    return f"{title}\n\n{body}" if title else body


async def search_wiki(query: str, lang: str = "en") -> Optional[str]:
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

        return _extract_wiki_text(soup)


async def search_tavily(query: str) -> Optional[str]:
    """
    Search the web via Tavily AI and return formatted snippets.
    Runs in a thread-pool to prevent blocking FastAPI's async event loop.
    """
    try:
        # FIX: Non-blocking thread execution for synchronous Tavily SDK
        response = await asyncio.to_thread(
            tavily_client.search,
            query=query,
            search_depth="basic",
            max_results=3,
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
        title = res.get("title", "Untitled")
        content = res.get("content", "")
        snippets.append(f"Source={idx}: {title}\nContent: {content}")

    combined = "\n\n".join(snippets)
    logger.info("Tavily search returned %d results for query: %s", len(results), query)
    return combined


async def search_web_fallback(query: str) -> str:
    """Fallback strategy: Try Wikipedia first, then Tavily."""
    wiki_result = await search_wiki(query)
    if wiki_result:
        return wiki_result

    tavily_result = await search_tavily(query)
    if tavily_result:
        return tavily_result

    logger.warning("Both Wikipedia and Tavily failed for query: %s", query)
    return ""


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