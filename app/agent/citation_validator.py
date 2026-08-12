"""
Hard citation validator for generated answers.

Runs before / alongside the LLM verifier to catch:
  - factual assertions with no [evidence_id] / [E#] citation
  - citations that do not resolve to assembled evidence
  - metric / geography keywords asserted without a supporting citation

Does not call an LLM — deterministic and cheap.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.agent.normalization import detect_metric_type, detect_place_mentions
from app.agent.state import Claim, ClaimStatus, ClaimType, Evidence, MetricType
from app.agent.verification import VerificationError
from app.agent.debug_session import agent_debug_log

# Hex evidence ids (legacy) and short cite keys E1..En (cite-while-writing)
_HEX_RE = re.compile(r"[a-fA-F0-9]{6,12}")
_EKEY_RE = re.compile(r"[Ee]\d{1,3}")
CITATION_TOKEN_RE = re.compile(r"\[([Ee]\d{1,3}|[a-fA-F0-9]{6,12})\]")
_HEADING_RE = re.compile(r"^#{1,6}\s+|^\*\*[^*]+\*\*\s*:?\s*$")
_SECTION_SKIP = re.compile(
    r"(?i)^(direct answer|supporting evidence|analysis|caveats|confidence|"
    r"limitations|inference|note:|your thinking)\b"
)
_SECTION_SKIP_BODY = re.compile(
    r"(?i)^(analysis|caveats|confidence|limitations|inference)\b"
)
# Sentences that look like factual claims (numbers, %, metrics, named places)
_FACTUAL_CUES = re.compile(
    r"\d|"
    r"%|"
    r"\b(?:gsdp|gdp|gva|crore|lakh|percent|percentage|share|growth|estimate|"
    r"revised|advance|actual|contributed|largest|fastest)\b",
    re.IGNORECASE,
)


@dataclass
class CitationValidationResult:
    """Outcome of hard citation checks against an answer + evidence set."""

    errors: list[VerificationError] = field(default_factory=list)
    uncited_sentences: list[str] = field(default_factory=list)
    invalid_citation_ids: list[str] = field(default_factory=list)
    cited_ids: list[str] = field(default_factory=list)
    claims: list[Claim] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def high_severity_count(self) -> int:
        return sum(1 for e in self.errors if e.severity == "high")


def extract_citation_tokens(text: str) -> list[str]:
    """Return raw citation tokens inside brackets (E3 or hex)."""
    return CITATION_TOKEN_RE.findall(text or "")


def resolve_citation_token(
    token: str,
    evidence: list[Evidence],
    cite_map: dict[str, str] | None = None,
) -> str | None:
    """Map a citation token to an evidence_id, or None if unknown."""
    if not token:
        return None
    known = {ev.evidence_id: ev for ev in evidence or []}
    cite_map = cite_map or {}

    # Build reverse maps from evidence metadata cite_key
    for ev in evidence or []:
        key = (ev.metadata or {}).get("cite_key")
        if key:
            cite_map.setdefault(str(key).upper(), ev.evidence_id)
            cite_map.setdefault(str(key), ev.evidence_id)

    if _EKEY_RE.fullmatch(token):
        upper = token.upper()
        eid = cite_map.get(upper) or cite_map.get(token)
        if eid and eid in known:
            return eid
        # Direct match if someone used E# as evidence_id (tests)
        if token in known:
            return token
        if upper in known:
            return upper
        return None

    if token in known:
        return token
    return None


def _prepare_line_for_citation_check(line: str) -> tuple[str, list[str]]:
    """Strip markdown chrome; return (body, line-level cite tokens).

    Fact bullets often put the cite on the label:
      ``- **Fact 1 [E1]**: "quoted evidence…"``
    """
    line_cites = list(dict.fromkeys(extract_citation_tokens(line)))
    cleaned = re.sub(r"^[-*•]\s+", "", line)
    cleaned = re.sub(r"^\*\*(.+?)\*\*:?\s*", "", cleaned)
    if cleaned == re.sub(r"^[-*•]\s+", "", line):
        cleaned = re.sub(r"^\*\*[^*]+\*\*:?\s*", "", cleaned)
    return cleaned, line_cites


def _sentences_from_prepared(prepared: str, line_cites: list[str], *, factual_only: bool) -> list[str]:
    """Split prepared line into sentences; inherit line cites onto parts that lack any."""
    chunks: list[str] = []
    cite_suffix = (" " + " ".join(f"[{c}]" for c in line_cites)) if line_cites else ""
    for part in re.split(r"(?<=[.!?])\s+", prepared):
        s = part.strip()
        if len(s) < 20:
            continue
        if factual_only and not _is_factual_assertion(s):
            continue
        if line_cites and not extract_citation_tokens(s):
            s = s + cite_suffix
        chunks.append(s)
    return chunks


def split_checkable_sentences(answer: str) -> list[str]:
    """Split answer into factual sentence-like units; skip Analysis/Caveats body."""
    if not answer:
        return []
    text = answer.replace("\r\n", "\n")
    chunks: list[str] = []
    skip_body = False
    for block in re.split(r"\n+", text):
        line = block.strip()
        if not line:
            continue
        bare = line.lstrip("#*- ")
        if _HEADING_RE.match(line) or _SECTION_SKIP.match(bare):
            skip_body = bool(_SECTION_SKIP_BODY.match(bare))
            continue
        if skip_body:
            continue
        prepared, line_cites = _prepare_line_for_citation_check(line)
        chunks.extend(_sentences_from_prepared(prepared, line_cites, factual_only=True))
    return chunks


def _split_sentences(answer: str) -> list[str]:
    """Split answer into checkable sentence-like units (legacy helper)."""
    if not answer:
        return []
    text = answer.replace("\r\n", "\n")
    chunks: list[str] = []
    skip_body = False
    for block in re.split(r"\n+", text):
        line = block.strip()
        if not line:
            continue
        bare = line.lstrip("#*- ")
        if _HEADING_RE.match(line) or _SECTION_SKIP.match(bare):
            skip_body = bool(_SECTION_SKIP_BODY.match(bare))
            continue
        if skip_body:
            continue
        # #region agent log
        pre_cites = extract_citation_tokens(line)
        # #endregion
        prepared, line_cites = _prepare_line_for_citation_check(line)
        # #region agent log
        sample = _sentences_from_prepared(prepared, line_cites, factual_only=False)
        post_cites = extract_citation_tokens(" ".join(sample)) if sample else []
        if pre_cites and not post_cites:
            agent_debug_log(
                "B",
                "citation_validator.py:_split_sentences",
                "cite_stripped_from_label",
                {
                    "pre_cites": pre_cites,
                    "line_preview": bare[:140],
                    "after_strip_preview": prepared[:140],
                },
            )
        elif pre_cites and post_cites:
            agent_debug_log(
                "B",
                "citation_validator.py:_split_sentences",
                "cite_preserved_from_label",
                {
                    "pre_cites": pre_cites,
                    "post_cites": post_cites,
                    "line_preview": bare[:100],
                },
            )
        # #endregion
        chunks.extend(_sentences_from_prepared(prepared, line_cites, factual_only=False))
    return chunks


def _is_factual_assertion(sentence: str) -> bool:
    if not _FACTUAL_CUES.search(sentence):
        return False
    if re.search(r"(?i)\b(insufficient|could not|unable to|i don't have|unknown)\b", sentence):
        return False
    if sentence.lower().startswith(("confidence:", "according to the most")):
        pass
    return True


def validate_answer_citations(
    answer: str,
    evidence: list[Evidence],
    *,
    require_citations: bool = True,
    cite_map: dict[str, str] | None = None,
) -> CitationValidationResult:
    """Validate that factual assertions in ``answer`` cite real evidence IDs / E# keys."""
    known = {ev.evidence_id: ev for ev in evidence or []}
    cite_map = dict(cite_map or {})
    for ev in evidence or []:
        key = (ev.metadata or {}).get("cite_key")
        if key:
            cite_map.setdefault(str(key).upper(), ev.evidence_id)

    result = CitationValidationResult()
    raw_tokens = extract_citation_tokens(answer or "")
    resolved_ids: list[str] = []
    for tok in raw_tokens:
        eid = resolve_citation_token(tok, evidence, cite_map)
        if eid:
            resolved_ids.append(eid)
            continue
        result.invalid_citation_ids.append(tok)
        result.errors.append(
            VerificationError(
                claim_id=tok,
                issue="INVALID_CITATION",
                severity="high",
                detail=f"Citation [{tok}] does not match any assembled evidence id / cite key",
                suggested_fix="Remove the citation or retrieve matching evidence",
            )
        )
    result.cited_ids = list(dict.fromkeys(resolved_ids))

    if not require_citations:
        return result

    for sentence in _split_sentences(answer or ""):
        if not _is_factual_assertion(sentence):
            continue
        ids: list[str] = []
        for tok in extract_citation_tokens(sentence):
            eid = resolve_citation_token(tok, evidence, cite_map)
            if eid and eid in known:
                ids.append(eid)
        ids = list(dict.fromkeys(ids))
        if ids:
            result.claims.append(
                Claim(
                    text=sentence[:500],
                    status=ClaimStatus.PARTIALLY_VERIFIED,
                    claim_type=ClaimType.FACT,
                    evidence_ids=ids,
                    reasoning="Citation present; pending semantic verify",
                )
            )
            continue

        result.uncited_sentences.append(sentence)
        claim = Claim(
            text=sentence[:500],
            status=ClaimStatus.UNVERIFIED,
            claim_type=ClaimType.FACT,
            evidence_ids=[],
            reasoning="Factual assertion without a resolvable evidence citation",
            repair_action="rephrase",
        )
        result.claims.append(claim)
        result.errors.append(
            VerificationError(
                claim_id=claim.claim_id,
                issue="UNCITED_ASSERTION",
                severity="high",
                detail=f"Uncited factual assertion: {sentence[:160]}",
                suggested_fix="Add [E#] from assembled context or remove the claim",
            )
        )

        metric = detect_metric_type(sentence)
        places = detect_place_mentions(sentence)
        if metric != MetricType.UNKNOWN and not ids:
            result.errors.append(
                VerificationError(
                    claim_id=claim.claim_id,
                    issue="UNCITED_METRIC",
                    severity="medium",
                    detail=f"Metric `{metric.value}` asserted without citation",
                    suggested_fix="Cite the evidence chunk that states this metric",
                )
            )
        if places and not ids:
            result.errors.append(
                VerificationError(
                    claim_id=claim.claim_id,
                    issue="UNCITED_GEOGRAPHY",
                    severity="medium",
                    detail=f"Geography `{places[0]}` asserted without citation",
                    suggested_fix="Cite evidence scoped to this geography",
                )
            )

    # #region agent log
    agent_debug_log(
        "B",
        "citation_validator.py:validate_answer_citations",
        "citation_validation_result",
        {
            "ok": result.ok,
            "uncited_count": len(result.uncited_sentences),
            "invalid_count": len(result.invalid_citation_ids),
            "uncited_previews": [s[:100] for s in result.uncited_sentences[:4]],
            "cited_ids": result.cited_ids[:8],
        },
    )
    # #endregion
    return result


def flag_uncited_in_answer(answer: str, result: CitationValidationResult) -> str:
    """Append a short caveat when hard citation checks fail (flag, don't wipe)."""
    if result.ok or not answer:
        return answer
    n = len(result.uncited_sentences)
    inv = len(result.invalid_citation_ids)
    parts = []
    if n:
        parts.append(f"{n} factual statement(s) lacked evidence citations")
    if inv:
        parts.append(f"{inv} citation id(s) did not resolve to evidence")
    note = "; ".join(parts)
    if "Citation check:" in answer:
        return answer
    return answer.rstrip() + f"\n\n*Citation check: {note}. Treat uncited figures as unverified.*"
