"""
Source authority & quality — DNS/TLD based, not hardcoded per-domain.

Authority is derived from the URL's structure (TLD / domain pattern) plus a small
allowlist of *organization-type* domains that are unambiguously primary statistical
sources (e.g. worldbank.org, imf.org). We deliberately do NOT enumerate specific
countries, agencies, or facts.

Tiers (score in [0, 1], higher = more authoritative for official figures):
    GOVERNMENT   (.gov / .gov.<tld>)                      -> primary, ~0.95
    OFFICIAL_ORG (worldbank.org, imf.org, who.int, ...)   -> primary, ~0.92
    ACADEMIC     (.edu / .ac.<tld> / .int)                -> primary,  ~0.90
    NEWS/SECONDARY (.com / .co.<tld> / .net / generic .org) -> secondary, ~0.65
    TERTIARY     (wikipedia, reddit, quora, twitter/x, blogs) -> tertiary, ~0.30
    UNKNOWN      (no URL / document)                      -> default by source type
"""

from __future__ import annotations

from urllib.parse import urlparse

from app.agent.state import SourceQuality, SourceType


# Government TLDs / patterns (suffix-based, country-agnostic).
_GOV_SUFFIXES = (".gov", ".gov.", ".mil")

# Organization-type domains that are unambiguously primary statistical / official
# sources. These are *organization* domains, not a list of specific facts or places.
_PRIMARY_ORG_DOMAINS = {
    "worldbank.org", "imf.org", "oecd.org", "who.int", "bis.org", "un.org",
    "europa.eu", "data.gov", "data.gov.in", "statista.com", "nasa.gov",
    "fao.org", "unesco.org", "iea.org", "oecd-ilibrary.org", "rbi.org.in",
}

# User-generated / aggregator / low-authority domains (general categories).
_TERTIARY_DOMAINS = {
    "wikipedia.org", "reddit.com", "quora.com", "twitter.com", "x.com",
    "medium.com", "blogspot.com", "wordpress.com", "tumblr.com", "youtube.com",
    "linkedin.com", "facebook.com", "instagram.com", "pinterest.com",
}

_DEFAULT_DOC_AUTHORITY = 0.80
_DEFAULT_WEB_AUTHORITY = 0.60

# Document filenames that look like primary reports (category, not specific facts).
_PRIMARY_DOC_INDICATORS = (
    "report", "census", "survey", "annual", "bulletin", "whitepaper",
    "statistical", "statistics", "budget", "yearbook", "gazette",
)


def _domain(url: str | None) -> str:
    if not url:
        return ""
    try:
        d = urlparse(url).netloc.lower()
    except Exception:
        return ""
    if d.startswith("www."):
        d = d[4:]
    return d


def authority_tier(url: str | None, source_type: SourceType) -> tuple[str, float]:
    """Return (tier_name, authority_score) for a source URL."""
    domain = _domain(url)

    # Documents without URLs use a sensible default (they are user-provided KB).
    if source_type == SourceType.DOCUMENT and not domain:
        return "document", _DEFAULT_DOC_AUTHORITY

    if not domain:
        return "unknown", _DEFAULT_WEB_AUTHORITY

    # Government (any country) -- suffix based, handles sub-TLDs like .gov.in / .gov.uk
    if domain.endswith(".gov") or ".gov." in domain or domain.startswith("gov.") or domain == "gov":
        return "government", 0.95

    # Tertiary / user-generated.
    if any(domain == td or domain.endswith("." + td) for td in _TERTIARY_DOMAINS):
        return "tertiary", 0.30

    # Official organizations (primary statistical sources).
    if any(domain == od or domain.endswith("." + od) or domain.startswith(od + ".")
           for od in _PRIMARY_ORG_DOMAINS):
        return "official_org", 0.92

    # Academic / international (.edu, .ac.<tld>, .int).
    if domain.endswith((".edu", ".int")) or ".ac." in domain:
        return "academic", 0.90

    # Everything else: news / analysis / generic org -> secondary.
    if domain.endswith((".com", ".co", ".net", ".io", ".org", ".info", ".biz", ".news")):
        return "secondary", 0.65

    return "unknown", _DEFAULT_WEB_AUTHORITY


def authority_score(url: str | None, source_type: SourceType) -> float:
    """Numeric authority score in [0, 1]."""
    return authority_tier(url, source_type)[1]


def official_search_variants(query: str) -> list[str]:
    """Extra search strings using TLD / org-type patterns already in this module.

    Not a per-country agency list: ``site:.gov`` plus at most one primary-org
    ``site:`` from ``_PRIMARY_ORG_DOMAINS``.
    """
    q = (query or "").strip()
    if not q:
        return []
    variants = [f"{q} site:.gov"]
    # One org-type domain is enough; pick a stable first item for determinism.
    org = next(iter(sorted(_PRIMARY_ORG_DOMAINS)), None)
    if org:
        variants.append(f"{q} site:{org}")
    return variants


def classify_source_quality(
    source_name: str,
    source_url: str | None,
    source_type: SourceType,
) -> SourceQuality:
    """Classify a source as primary / secondary / tertiary.

    Rules:
    - Documents that look like reports are PRIMARY; otherwise UNKNOWN.
    - Web: tertiary for user-generated domains; primary for government / official
      organizations / academic; otherwise SECONDARY.
    Authority therefore depends on the *claim type*: official stats from .gov are
    highly authoritative, but a blog post is not — and a primary source is not
    automatically correct for causal/historical analysis (that is left to the
    verifier, not hardcoded here).
    """
    if source_type == SourceType.DOCUMENT:
        name_lower = (source_name or "").lower()
        if any(ind in name_lower for ind in _PRIMARY_DOC_INDICATORS):
            return SourceQuality.PRIMARY
        return SourceQuality.UNKNOWN

    if not source_url:
        return SourceQuality.UNKNOWN

    tier, _ = authority_tier(source_url, source_type)
    if tier in ("government", "official_org", "academic"):
        return SourceQuality.PRIMARY
    if tier == "tertiary":
        return SourceQuality.TERTIARY
    return SourceQuality.SECONDARY
