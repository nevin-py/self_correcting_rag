"""Location Tool for geographic context retrieval.

Browser-based location lookup using geocoding APIs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class LocationInfo:
    """Geographic location information."""
    name: str
    country: str
    region_type: str  # "city", "state", "country", "region"
    state: str | None = None
    coordinates: dict[str, float] | None = None  # {"lat": float, "lon": float}
    population: int | None = None


class LocationTool:
    """Browser-based location lookup and enrichment."""

    def __init__(self, api_key: str | None = None):
        """Initialize location tool.

        Args:
            api_key: Optional geocoding API key (e.g., Google Maps, OpenCage, etc.)
        """
        self.api_key = api_key
        logger.info("Location tool initialized (no API key provided, using fallback)")

    async def search_location(self, query: str) -> LocationInfo | None:
        """Search for location information based on query string.

        Args:
            query: Location query (e.g., "Mumbai", "Maharashtra", "New York City")

        Returns:
            LocationInfo with geographic details, or None if not found
        """
        try:
            # Try API-based lookup first if API key provided
            if self.api_key:
                result = await self._search_via_api(query)
                if result:
                    return result

            # Fallback: keyword-based location detection
            return self._search_via_keywords(query)
        except Exception as exc:
            logger.warning("Location search failed: %s", exc)
            return None

    async def _search_via_api(self, query: str) -> LocationInfo | None:
        """Search using geocoding API (Google Maps, OpenCage, etc.)."""
        try:
            import httpx

            # Example using OpenCage API (no auth for low volume)
            url = "https://api.opencagedata.com/geocode/v1/json"
            params = {"q": query, "key": self.api_key or "demo"}

            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    results = data.get("results", [])
                    if results:
                        first = results[0]
                        components = first.get("components", {})

                        # Extract location info
                        location = LocationInfo(
                            name=query,
                            country=components.get("country", ""),
                            state=components.get("state", components.get("state_code")),
                            coordinates={
                                "lat": first["geometry"]["lat"],
                                "lon": first["geometry"]["lng"],
                            },
                            region_type=self._determine_region_type(components),
                            population=None,
                        )

                        return location
        except Exception as exc:
            logger.debug("API location search failed: %s", exc)

        return None

    def _search_via_keywords(self, query: str) -> LocationInfo | None:
        """Fallback keyword-based location detection (no API)."""
        q = query.strip().lower()

        # Common location patterns
        location_db = {
            "mumbai": ("India", "Maharashtra", "city"),
            "delhi": ("India", "Delhi", "city"),
            "bangalore": ("India", "Karnataka", "city"),
            "chennai": ("India", "Tamil Nadu", "city"),
            "kolkata": ("India", "West Bengal", "city"),
            "hyderabad": ("India", "Telangana", "city"),
            "pune": ("India", "Maharashtra", "city"),
            "ahmedabad": ("India", "Gujarat", "city"),
            "maharashtra": ("India", None, "state"),
            "karnataka": ("India", None, "state"),
            "tamil nadu": ("India", None, "state"),
            "gujarat": ("India", None, "state"),
            "west bengal": ("India", None, "state"),
            "uttar pradesh": ("India", None, "state"),
            "india": ("India", None, "country"),
            "usa": ("United States", None, "country"),
            "america": ("United States", None, "country"),
            "new york": ("United States", "New York", "city"),
            "california": ("United States", "California", "state"),
        }

        # Try exact match
        if q in location_db:
            country, state, region_type = location_db[q]
            return LocationInfo(
                name=q,
                country=country,
                state=state,
                region_type=region_type,
                coordinates=None,
                population=None,
            )

        # Try partial match
        for name, (country, state, region_type) in location_db.items():
            if name in q:
                return LocationInfo(
                    name=q,
                    country=country,
                    state=state,
                    region_type=region_type,
                    coordinates=None,
                    population=None,
                )

        return None

    def _determine_region_type(self, components: dict[str, Any]) -> str:
        """Determine region type from geocoding components."""
        if components.get("city"):
            return "city"
        if components.get("state"):
            return "state"
        if components.get("country"):
            return "country"
        return "region"

    async def enrich_context(self, query: str, evidence_text: str) -> dict[str, Any] | None:
        """Add location context to evidence text.

        Returns dict with location info and enriched evidence context if location found.

        Args:
            query: User's query
            evidence_text: Evidence text to enrich

        Returns:
            dict with "location" and "enriched_context" or None if no location
        """
        location = await self.search_location(query)
        if not location:
            return None

        logger.info("Location enrichment: %s, %s", location.name, location.country)

        return {
            "location": {
                "name": location.name,
                "country": location.country,
                "state": location.state,
                "region_type": location.region_type,
                "coordinates": location.coordinates,
            },
            "enriched_context": f"[LOC: {location.name} is in {location.country}]",
        }

    async def retrieve_location_context(self, location_name: str) -> str:
        """Retrieve textual context for a location.

        Args:
            location_name: Name of the location

        Returns:
            String with location context (population, key facts, etc.)
        """
        location = await self.search_location(location_name)
        if not location:
            return f"[LOC: Could not find information about {location_name}]"

        context_parts = []
        if location.population:
            context_parts.append(f"{location.name} has a population of approximately {location.population}")

        if location.state:
            context_parts.append(f"{location.name} is in {location.state}, {location.country}")

        return " ".join(context_parts)


# Singleton instance
_location_tool_instance: LocationTool | None = None


def get_location_tool(api_key: str | None = None) -> LocationTool:
    """Get or create LocationTool singleton instance."""
    global _location_tool_instance
    if _location_tool_instance is None:
        _location_tool_instance = LocationTool(api_key=api_key)
    return _location_tool_instance