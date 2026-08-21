"""Hard citation validator for generated answers.

Deterministic and cheap — no LLM, no domain knowledge. Catches:
  - citations that do not resolve to assembled evidence
  - factual-looking assertions with no citation at all

Semantic support checking (does the evidence actually back the claim?) is the
LLM judge's job in nodes.verify_answer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.agent.state import Claim, ClaimStatus, Evidence


# Short cite keys E1..En (cite-while-writing) or legacy hex evidence ids
CITATION_TOKEN_RE = re.compile(r"\[([Ee]\d{1,3}|[a-fA-F0-9]{6,12})\]")
_EKEY_RE = re.compile(r"[Ee]\d{1,3}")
_HEADING_RE = re.compile(r"^#{1,6}\s+|^\*\*[^*]+\*\*\s*:?\s*$")
_SECTION_SKIP = re.compile(
    r"(?i)^(direct answer|supporting evidence|analysis|caveats|confidence|"
    r"limitations|inference|note:|your thinking)\b"
)
_SECTION_SKIP_BODY = re.compile(
    r"(?i)^(analysis|caveats|confidence|limitations|inference)\b"
)


@dataclass
class CitationError:
    claim_id: str
    issue: str            # INVALID_CITATION | UNCITED_ASSERTION
    severity: str         # high | medium | low
    detail: str

    def to_dict(self) -> dict:
        return {
            "claim_id": self.claim_id,
            "issue": self.issue,
            "severity": self.severity,
            "detail": self.detail,
        }


@dataclass
class CitationValidationResult:
    """Outcome of hard citation checks against an answer + evidence set."""

    errors: list[CitationError] = field(default_factory=list)
    uncited_sentences: list[str] = field(default_factory=list)
    invalid_citation_ids: list[str] = field(default_factory=list)
    cited_ids: list[str] = field(default_factory=list)
    claims: list[Claim] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


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
    known = {ev.evidence_id for ev in evidence or []}
    cite_map = dict(cite_map or {})
    for ev in evidence or []:
        key = (ev.metadata or {}).get("cite_key")
        if key:
            cite_map.setdefault(str(key).upper(), ev.evidence_id)
            cite_map.setdefault(str(key), ev.evidence_id)

    if _EKEY_RE.fullmatch(token):
        eid = cite_map.get(token.upper()) or cite_map.get(token)
        if eid and eid in known:
            return eid
        # Direct match if someone used E# as evidence_id (tests)
        for cand in (token, token.upper()):
            if cand in known:
                return cand
        return None
    return token if token in known else None


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


def _is_factual_assertion(sentence: str) -> bool:
    """Heuristic pre-filter so only assertion-like sentences need citations.

    Deliberately minimal and domain-agnostic: numbers/percentages/proper-noun-ish
    content counts; hedged "I could not find…" statements do not.
    """
    if not re.search(r"\d|%|\b[A-Z][a-z]+", sentence):
        return False
    if re.search(r"(?i)\b(insufficient|could not|unable to|i don't have|unknown)\b", sentence):
        return False
    return True


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
    return _split(answer, factual_only=True)


def _split(answer: str, *, factual_only: bool) -> list[str]:
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
        chunks.extend(_sentences_from_prepared(prepared, line_cites, factual_only=factual_only))
    return chunks


def validate_answer_citations(
    answer: str,
    evidence: list[Evidence],
    *,
    require_citations: bool = True,
    cite_map: dict[str, str] | None = None,
) -> CitationValidationResult:
    """Validate that factual assertions in ``answer`` cite real evidence IDs / E# keys."""
    known = {ev.evidence_id for ev in evidence or []}
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
            CitationError(
                claim_id=tok,
                issue="INVALID_CITATION",
                severity="high",
                detail=f"Citation [{tok}] does not match any assembled evidence id / cite key",
            )
        )
    result.cited_ids = list(dict.fromkeys(resolved_ids))

    if not require_citations:
        return result

    for sentence in _split(answer or "", factual_only=False):
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
                    status=ClaimStatus.VERIFIED,
                    evidence_ids=ids,
                    reasoning="Citation present; pending semantic verify",
                )
            )
            continue
        if not _is_factual_assertion(sentence):
            continue
        result.uncited_sentences.append(sentence)
        claim = Claim(
            text=sentence[:500],
            status=ClaimStatus.UNVERIFIED,
            evidence_ids=[],
            reasoning="Factual assertion without a resolvable evidence citation",
        )
        result.claims.append(claim)
        result.errors.append(
            CitationError(
                claim_id=claim.claim_id,
                issue="UNCITED_ASSERTION",
                severity="high",
                detail=f"Uncited factual assertion: {sentence[:160]}",
            )
        )

    return result


def flag_uncited_in_answer(answer: str, result: CitationValidationResult) -> str:
    """Only flag invalid citation IDs, not uncited sentences.

    The generator can't cite every sentence — uncertainty statements, transitions,
    and contextual sentences don't need citations.  Only invalid IDs
    (e.g., [E99] when only E1-E10 exist) are actionable.
    """
    if result.ok or not answer:
        return answer
    inv = len(result.invalid_citation_ids)
    if not inv:
        return answer
    note = f"{inv} citation id(s) did not resolve to evidence"
    if "Citation check:" in answer:
        return answer
    return answer.rstrip() + f"\n\n*Citation check: {note}. Treat uncited figures as unverified.*"
