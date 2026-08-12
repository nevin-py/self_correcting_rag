"""
Evidence normalization layer.

Pure, deterministic, *general* functions that extract and canonicalize the
structured fields carried on every :class:`Evidence` object:

    metric_type     (GSDP / GVA / GVA_SHARE / OUTPUT_SHARE / GDP / ...)
    price_basis     (CURRENT / CONSTANT)            -- nominal vs real
    temporal_qualifier (ACTUAL / REVISED / ADVANCE / PROJECTED / ...)
    year_period     (2024-25 / FY2023 / 2023)
    geography       (free text, discovered generically -- no hardcoded place list)
    geographic_scope (GLOBAL / NATIONAL / STATE / ...)

Design constraints honored:
- No hardcoded geography lists (states / countries). Place discovery is a generic
  Title-Case recognizer driven by the LLM-classified query geography.
- No hardcoded "facts". Only vocabulary (metric acronyms, scope words, price words).
- Acronyms are preserved so the planner can search explicitly (e.g. "USA GDP",
  "USA growth rate") instead of relying on fuzzy similarity.
"""

from __future__ import annotations

import re
from pydantic import BaseModel

from app.agent.state import (
    GeographicScope,
    MetricType,
    PriceBasis,
    TemporalQualifier,
)


# ── Metric detection (priority ordered) ──────────────────────────────────────
#
# Specific metric names are checked BEFORE generic modifiers ("growth rate",
# "inflation") so that "GDP growth rate" resolves to GDP, not GROWTH_RATE.

_METRIC_PATTERNS: list[tuple[re.Pattern[str], MetricType]] = [
    (re.compile(r"gross state domestic product|\bgsdp\b", re.IGNORECASE), MetricType.GSDP),
    (re.compile(r"gross domestic product|\bgdp\b", re.IGNORECASE), MetricType.GDP),
    # GVA_SHARE must be checked BEFORE plain GVA (it also contains "GVA").
    (re.compile(r"share of (?:gross )?value added|share of gva|\bgva share|\bgva %|gva's share", re.IGNORECASE), MetricType.GVA_SHARE),
    (re.compile(r"gross value added|\bgva\b", re.IGNORECASE), MetricType.GVA),
    (re.compile(r"share of output|output share|share of total output|sector share of output|sectoral share of output", re.IGNORECASE), MetricType.OUTPUT_SHARE),
    (re.compile(r"employment rate|\bemployment\b|\bjobs\b|workforce|labour force|labor force", re.IGNORECASE), MetricType.EMPLOYMENT),
    (re.compile(r"\brevenue\b|tax revenue|tax collection", re.IGNORECASE), MetricType.REVENUE),
    (re.compile(r"\bpopulation\b|headcount|inhabitants", re.IGNORECASE), MetricType.POPULATION),
    (re.compile(r"consumer price|\bcpi\b|\bwpi\b|\binflation\b|price index|price rise", re.IGNORECASE), MetricType.INFLATION),
]

_GENERIC_METRIC_PATTERNS: list[tuple[re.Pattern[str], MetricType]] = [
    (re.compile(r"growth rate|\bcagr\b|year on year|yoy|\bgrowth %|rate of growth|contraction", re.IGNORECASE), MetricType.GROWTH_RATE),
]


def detect_metric_type(text: str) -> MetricType:
    """Detect the canonical metric from text.

    Specific metric acronyms/phrases win over generic modifiers. Returns
    MetricType.UNKNOWN when nothing matches.
    """
    if not text:
        return MetricType.UNKNOWN
    for pattern, metric in _METRIC_PATTERNS:
        if pattern.search(text):
            return metric
    for pattern, metric in _GENERIC_METRIC_PATTERNS:
        if pattern.search(text):
            return metric
    return MetricType.UNKNOWN


# ── Price basis detection (nominal vs real) ──────────────────────────────────

_PRICE_CURRENT = re.compile(
    r"current price|current prices|nominal|at current|in nominal|nominal term|current price term",
    re.IGNORECASE,
)
_PRICE_CONSTANT = re.compile(
    r"constant price|constant prices|real (?:price|term|gdp|gsdp|gva)|"
    r"2011[- ]?12 price|2011[- ]?12 base|inflation[- ]?adjust|chain base|"
    r"base year|real growth|\breal\b",
    re.IGNORECASE,
)


def detect_price_basis(text: str) -> PriceBasis:
    """Detect nominal (CURRENT) vs real (CONSTANT) price basis."""
    if not text:
        return PriceBasis.UNKNOWN
    # Constant takes priority: "real" is explicit about inflation adjustment.
    if _PRICE_CONSTANT.search(text):
        return PriceBasis.CONSTANT
    if _PRICE_CURRENT.search(text):
        return PriceBasis.CURRENT
    return PriceBasis.UNKNOWN


# ── Temporal qualifier detection ─────────────────────────────────────────────

_TEMPORAL_PATTERNS: list[tuple[re.Pattern[str], TemporalQualifier]] = [
    (re.compile(r"advance estimate|advance estimates", re.IGNORECASE), TemporalQualifier.ADVANCE),
    (re.compile(r"preliminary|provisional", re.IGNORECASE), TemporalQualifier.PRELIMINARY),
    (re.compile(r"\brevise\w*|revision|second advance|final estimate", re.IGNORECASE), TemporalQualifier.REVISED),
    (re.compile(r"projected|projection|forecast|expected to", re.IGNORECASE), TemporalQualifier.PROJECTED),
    (re.compile(r"\bestimat\w*|estimated", re.IGNORECASE), TemporalQualifier.ESTIMATE),
    (re.compile(r"\bactual\b|\bfinal\b|confirmed|realized|as reported", re.IGNORECASE), TemporalQualifier.ACTUAL),
]


def detect_temporal_qualifier(text: str) -> TemporalQualifier:
    """Detect the data-status / temporal qualifier from text."""
    if not text:
        return TemporalQualifier.UNKNOWN
    # Order matters: more specific qualifiers first.
    for pattern, qualifier in _TEMPORAL_PATTERNS:
        if pattern.search(text):
            return qualifier
    return TemporalQualifier.UNKNOWN


# ── Year / period extraction ────────────────────────────────────────────────

_FY_PATTERN = re.compile(r"\bFY\s*(20\d{2})\b", re.IGNORECASE)
_RANGE_PATTERN = re.compile(r"\b(20\d{2})\s*[-/]\s*(\d{2})\b")
_SINGLE_YEAR = re.compile(r"\b(20\d{2})\b")


def extract_year_period(text: str) -> str:
    """Extract a fiscal-year / date range from text."""
    if not text:
        return ""
    m = _FY_PATTERN.search(text)
    if m:
        return f"FY{m.group(1)}"
    m = _RANGE_PATTERN.search(text)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    m = _SINGLE_YEAR.search(text)
    if m:
        return m.group(1)
    return ""


def parse_year(year_period: str) -> int | None:
    """Best-effort numeric year from a period string (e.g. '2024-25' -> 2024)."""
    if not year_period:
        return None
    m = re.search(r"(20\d{2})", year_period)
    return int(m.group(1)) if m else None


# ── Generic geography discovery (NO hardcoded place lists) ───────────────────

# Scope-level words -- generic, not a geography enumeration.
_SCOPE_WORDS: list[tuple[set[str], GeographicScope]] = [
    ({"global", "worldwide", "world", "international"}, GeographicScope.GLOBAL),
    ({"national", "country", "countrywide", "nation"}, GeographicScope.NATIONAL),
    ({"state", "state-level", "statewise", "provincial", "province"}, GeographicScope.STATE),
    ({"district", "district-level"}, GeographicScope.DISTRICT),
    ({"city", "city-level", "municipal", "metro", "metropolitan"}, GeographicScope.CITY),
    ({"region", "regional", "sub-continent", "subcontinent"}, GeographicScope.REGION),
]

# Category words that are NOT place names (institutions, orgs, metrics, months).
# This is a *category* list, deliberately NOT a list of specific geographies.
_NON_GEO_TITLE_TOKENS = {
    # institutions / documents
    "Economic", "Survey", "Report", "Reserve", "Bank", "Ministry", "Census",
    "Bulletin", "Department", "Office", "Statistics", "Statistical", "Central",
    "Annual", "Budget", "Paper", "Research", "University", "Institute",
    "Organization", "Corporation", "Limited", "Company", "Group", "Foundation",
    "Authority", "Commission", "Committee", "Council", "Board", "Secretariat",
    # generic scope words (handled separately)
    "National", "State", "District", "City", "Global", "International", "Regional",
    "World", "Country", "Province", "Federal", "Government", "Municipal",
    # media / orgs
    "Reuters", "Bloomberg", "News", "Times", "Post", "Herald", "Guardian",
    "Wikipedia", "Reddit", "Twitter", "Quora", "Prs", "Pib", "Mospi", "Rbi",
    "Niti", "Imf", "Oecd", "Who", "Bis", "Worldbank", "Ap",
    # common English / economic capitalized words that would otherwise be grouped
    # into a (false) place phrase, e.g. "Advance estimates of Karnataka" -> "Karnataka"
    "Advance", "Advances", "Estimate", "Estimates", "Estimated", "Revised",
    "Revision", "Preliminary", "Provisional", "Actual", "Final", "Growth",
    "Share", "Shares", "Services", "Manufacturing", "Sector", "Sectors",
    "Agriculture", "Industry", "Industries", "Total", "Gross", "Net", "Value",
    "Added", "Output", "Domestic", "Product", "Products", "Primary", "Secondary",
    "Tertiary", "Per", "Capita", "Crore", "Crores", "Lakh", "Lakhs", "Billion",
    "Million", "Millions", "Trillion", "Trillions", "Constant", "Current",
    "Prices", "Price", "Real", "Nominal", "Quarterly", "Quarter", "Month",
    "Months", "Year", "Years", "First", "Second", "Third", "Index", "Percent",
    "Percentage", "Rise", "Rose", "Fall", "Fell", "Increase", "Decrease",
    "During", "Compared", "Among", "Between", "According", "This", "That",
    "These", "Those", "From", "With", "Table", "Figure", "Chart", "Note",
    "Source", "Appendix", "Week", "Day",
    # sentence-start / common English capitals
    "The", "A", "An", "In", "On", "At", "For", "To", "Of", "And", "But",
    "As", "It", "Its", "Based", "Following", "Section",
    # months
    "January", "February", "March", "April", "May", "June", "July", "August",
    "September", "October", "November", "December",
}

_METRIC_ACRONYMS = {"GDP", "GSDP", "GVA", "GVA", "CPI", "WPI", "CAGR", "FY", "NSSO", "RBI"}


def detect_scope_cues(text: str) -> GeographicScope:
    """Detect an explicit geographic *scope level* from generic scope words."""
    if not text:
        return GeographicScope.UNKNOWN
    lower = text.lower()
    for words, scope in _SCOPE_WORDS:
        if any(w in lower for w in words):
            return scope
    return GeographicScope.UNKNOWN


def detect_place_mentions(text: str) -> list[str]:
    """Discover candidate place names from Title-Case proper nouns.

    Generic recognizer -- it does NOT enumerate states/countries. It finds
    capitalized phrases and removes institution/metric/common-word tokens, leaving
    genuine proper nouns (e.g. 'Maharashtra', 'United States', 'Tamil Nadu').
    """
    if not text:
        return []
    # Find capitalized word sequences; strip trailing possessive 's.
    raw = re.findall(r"[A-Z][a-zA-Z]*(?:['’][A-Za-z]+)?", text)
    # Group consecutive capitalized tokens into phrases (max 3 words).
    phrases: list[str] = []
    buf: list[str] = []
    for tok in raw:
        base = re.sub(r"['’][A-Za-z]+$", "", tok)
        if base in _NON_GEO_TITLE_TOKENS or base.upper() in _METRIC_ACRONYMS:
            if buf:
                phrases.append(" ".join(buf))
                buf = []
            continue
        buf.append(base)
        if len(buf) == 3:
            phrases.append(" ".join(buf))
            buf = []
    if buf:
        phrases.append(" ".join(buf))

    # Drop single-letter or purely-numeric or empty.
    seen: set[str] = set()
    out: list[str] = []
    for p in phrases:
        p = p.strip()
        if not p or len(p) < 3:
            continue
        if p.lower() in seen:
            continue
        seen.add(p.lower())
        out.append(p)
    return out


def normalize_geography(geo: str) -> str:
    """Normalize a geography string for comparison (lowercase, trimmed)."""
    if not geo:
        return ""
    return re.sub(r"\s+", " ", geo.strip()).lower()


def geographic_match(a: str, b: str) -> float:
    """Similarity of two geography strings in [0, 1].

    1.0 when equal or one contains the other as a phrase; lower otherwise.
    """
    na, nb = normalize_geography(a), normalize_geography(b)
    if not na or not nb:
        return 0.8  # unknown geography -> soft neutral
    if na == nb:
        return 1.0
    if na in nb or nb in na:
        return 1.0
    return 0.15


# ── Search-term composition (LLM-driven explicit queries) ────────────────────

_METRIC_SEARCH_TERM: dict[MetricType, str] = {
    MetricType.GDP: "GDP",
    MetricType.GSDP: "GSDP",
    MetricType.GVA: "GVA",
    MetricType.GVA_SHARE: "GVA share",
    MetricType.OUTPUT_SHARE: "output share",
    MetricType.EMPLOYMENT: "employment",
    MetricType.REVENUE: "revenue",
    MetricType.POPULATION: "population",
    MetricType.GROWTH_RATE: "growth rate",
    MetricType.INFLATION: "inflation",
    MetricType.OTHER: "",
    MetricType.UNKNOWN: "",
}

_TEMPORAL_SEARCH_TERM: dict[TemporalQualifier, str] = {
    TemporalQualifier.ACTUAL: "actual",
    TemporalQualifier.REVISED: "revised estimate",
    TemporalQualifier.ADVANCE: "advance estimate",
    TemporalQualifier.ESTIMATE: "estimate",
    TemporalQualifier.PRELIMINARY: "preliminary",
    TemporalQualifier.PROJECTED: "projection",
    TemporalQualifier.UNKNOWN: "",
}

_PRICE_SEARCH_TERM: dict[PriceBasis, str] = {
    PriceBasis.CURRENT: "current prices",
    PriceBasis.CONSTANT: "constant prices",
    PriceBasis.UNKNOWN: "",
}


def metric_search_term(metric: MetricType) -> str:
    return _METRIC_SEARCH_TERM.get(metric, "")


def temporal_search_term(qualifier: TemporalQualifier) -> str:
    return _TEMPORAL_SEARCH_TERM.get(qualifier, "")


def price_search_term(basis: PriceBasis) -> str:
    return _PRICE_SEARCH_TERM.get(basis, "")


def compose_search_query(
    geography: str = "",
    metric: MetricType = MetricType.UNKNOWN,
    temporal: TemporalQualifier = TemporalQualifier.UNKNOWN,
    price: PriceBasis = PriceBasis.UNKNOWN,
    base: str = "",
) -> str:
    """Compose an explicit, disambiguated search query.

    e.g. geography='USA', metric=GDP, temporal=ACTUAL -> 'USA GDP actual ...'.
    The LLM/planner uses this so searches carry the metric acronym + geography
    rather than relying on fuzzy similarity.
    """
    parts: list[str] = []
    if geography:
        parts.append(geography)
    if metric != MetricType.UNKNOWN:
        parts.append(metric_search_term(metric))
    if price != PriceBasis.UNKNOWN:
        parts.append(price_search_term(price))
    if temporal != TemporalQualifier.UNKNOWN:
        parts.append(temporal_search_term(temporal))
    if base and base not in " ".join(parts):
        parts.append(base)
    return " ".join(p for p in parts if p).strip()


# Query expansion mappings for better retrieval coverage
_METRIC_EXPANSIONS: dict[MetricType, list[str]] = {
    MetricType.GSDP: ["GSDP", "Gross State Domestic Product", "state GDP", "state economy", "MH GSDP", "Karnataka GSDP"],
    MetricType.GDP: ["GDP", "Gross Domestic Product", "economic output", "national economy", "India GDP"],
    MetricType.GVA: ["GVA", "Gross Value Added", "value added", "GVA growth"],
    MetricType.GVA_SHARE: ["GVA share", "share of GVA", "sectoral contribution", "sector share", "GVA percentage"],
    MetricType.OUTPUT_SHARE: ["output share", "share of output", "production share", "sectoral output"],
    MetricType.EMPLOYMENT: ["employment", "jobs", "workforce", "labor force", "unemployment"],
    MetricType.REVENUE: ["revenue", "tax revenue", "government revenue", "fiscal revenue"],
    MetricType.POPULATION: ["population", "headcount", "demographics", "residents"],
    MetricType.INFLATION: ["inflation", "CPI", "WPI", "price index", "price growth"],
}

_GEOGRAPHY_EXPANSIONS: dict[str, list[str]] = {
    "Maharashtra": ["Maharashtra", "MH", "Maharashtra state", "Maharashtrian"],
    "Karnataka": ["Karnataka", "KA", "Karnataka state", "Karnataka's"],
    "Tamil Nadu": ["Tamil Nadu", "TN", "TamilNadu"],
    "Gujarat": ["Gujarat", "GJ", "Gujarat state"],
    "West Bengal": ["West Bengal", "WB", "Bengal"],
    "Uttar Pradesh": ["Uttar Pradesh", "UP", "UP state"],
    "India": ["India", "Indian", "Bharat", "All India", "National"],
    "USA": ["USA", "United States", "America", "US"],
    "World": ["World", "Global", "Worldwide", "International"],
}



def expand_queries(
    base_query: str,
    metric: MetricType = MetricType.UNKNOWN,
    geography: str = "",
    temporal: TemporalQualifier = TemporalQualifier.UNKNOWN,
    price_basis: PriceBasis = PriceBasis.UNKNOWN,
) -> list[str]:
    """Generate multiple query variants for better retrieval coverage.
    
    Creates variations using metric synonyms, geography abbreviations,
    and temporal qualifiers to maximize recall.
    """
    variants = [base_query]
    
    # Add metric variations
    if metric != MetricType.UNKNOWN and metric in _METRIC_EXPANSIONS:
        for expansion in _METRIC_EXPANSIONS[metric]:
            if geography:
                variants.append(f"{geography} {expansion}")
            else:
                variants.append(expansion)
    
    # Add geography variations  
    if geography and geography in _GEOGRAPHY_EXPANSIONS:
        for geo_exp in _GEOGRAPHY_EXPANSIONS[geography]:
            if metric != MetricType.UNKNOWN:
                variants.append(f"{geo_exp} {metric.value}")
            else:
                variants.append(geo_exp)
    
    # Add temporal variations
    if temporal != TemporalQualifier.UNKNOWN:
        temporal_str = temporal.value
        for variant in list(variants):
            if temporal_str not in variant.lower():
                variants.append(f"{variant} {temporal_str}")
    
    # Add price basis variations
    if price_basis != PriceBasis.UNKNOWN:
        price_str = price_basis.value
        for variant in list(variants):
            if price_str not in variant.lower():
                variants.append(f"{variant} {price_str}")
    
    # Deduplicate while preserving order
    seen = set()
    unique_variants = []
    for v in variants:
        v_lower = v.lower().strip()
        if v_lower not in seen:
            seen.add(v_lower)
            unique_variants.append(v)
    
    return unique_variants[:5]  # Max 5 query variants


# Query decomposition for complex multi-part questions
class SubQuery(BaseModel):
    """A sub-query derived from decomposition of a complex question."""
    query: str
    sub_query_type: str = "factual"  # factual, comparative, temporal, procedural
    depends_on: list[int] = []  # indices of sub-queries this depends on
    focus_area: str = ""  # What aspect this sub-query focuses on


class QueryDecomposition(BaseModel):
    """Result of decomposing a complex query into sub-queries."""
    needs_decomposition: bool = False
    sub_queries: list[SubQuery] = []
    reasoning: str = ""  # Why the query was/wasn't decomposed


_DECOMPOSE_PROMPT = """You are a query analyzer. If the user's question has multiple parts or requires comparison, break it down into simpler sub-questions.

Examples:
- "Compare X and Y" → 2 sub-queries, comparative
- "What is X and when did it happen?" → 2 sub-queries, temporal
- "Causes and effects of X" → 2 sub-queries
- Simple "What is X?" → 1 sub-query (no decomposition needed)

For EACH sub-query provide:
- query: the specific question to ask
- sub_query_type: factual, comparative, temporal, procedural
- depends_on: [] if independent, or [index] if depends on another sub-query
- focus_area: what aspect this addresses

User query: {query}

Respond with JSON matching the QueryDecomposition schema."""


def decompose_query_text(query: str) -> QueryDecomposition:
    """Analyze if a query needs decomposition into sub-queries.
    
    Returns QueryDecomposition with sub_queries if needed.
    """
    # Simple heuristic: check for decomposition indicators
    decomposition_indicators = [
        "compare", "versus", "vs", "and then", "both", "also", "additionally",
        "first", "second", "finally", "causes", "effects", "reasons", "results",
        "what is x and", "how does x compare", "difference between",
    ]
    
    query_lower = query.lower()
    needs_decomp = any(indicator in query_lower for indicator in decomposition_indicators)
    
    if not needs_decomp:
        return QueryDecomposition(
            needs_decomposition=False,
            sub_queries=[SubQuery(query=query, sub_query_type="factual", focus_area="main")],
            reasoning="Simple factual query - no decomposition needed",
        )
    
    # Complex query - return for LLM decomposition (caller should use LLM for proper decomposition)
    return QueryDecomposition(
        needs_decomposition=True,
        sub_queries=[SubQuery(query=query, sub_query_type="factual", focus_area="main")],
        reasoning="Query appears complex - may benefit from multi-step retrieval",
    )


def get_retrieval_queries_for_subqueries(
    base_query: str,
    sub_queries: list[SubQuery],
    classification: QueryClassification | None = None,
) -> list[str]:
    """Generate retrieval queries for decomposed sub-queries.
    
    Combines the original query with specific sub-queries for comprehensive retrieval.
    """
    queries = []
    
    # Always include the main query
    queries.append(base_query)
    
    # Add each sub-query if different enough
    seen = {base_query.lower()}
    for sq in sub_queries[:3]:  # Max 3 sub-queries
        q = sq.query.strip()
        if q.lower() not in seen:
            seen.add(q.lower())
            queries.append(q)
    
    # Add classification-based expansion if available
    if classification:
        expanded = expand_queries(
            base_query=base_query,
            metric=classification.metric_hint,
            geography=classification.geography,
            temporal=classification.temporal_qualifier,
        )
        for q in expanded:
            if q.lower() not in seen:
                seen.add(q.lower())
                queries.append(q)
    
    return queries[:5]  # Max 5 total
