"""Hard citation validator for generated answers.

Deterministic and cheap — no LLM, no domain knowledge. Catches:
  - citations that do not resolve to assembled evidence
  - factual-looking assertions with no citation at all
  - (optional support gate) citations whose evidence does not semantically
    support the sentence citing it — embedding similarity via app.agent.support

Full semantic checking (contradictions, numeric mismatches) remains the LLM
judge's job in nodes.verify_answer; this gate is a cheap deterministic filter.
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
    # Sentences whose cited evidence does not semantically support them:
    # (sentence_text_with_markers, [cite_tokens]). verify_answer strips these
    # markers from the prose so unsupported claims never carry citations.
    unsupported_citations: list[tuple[str, list[str]]] = field(default_factory=list)

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
    entailment_gate: bool = False,
) -> CitationValidationResult:
    """Validate that factual assertions in ``answer`` cite real evidence IDs / E# keys.

    With ``entailment_gate=True`` each cited sentence is additionally embedded
    against its cited evidence texts; below CITATION_SUPPORT_MIN_SIM the claim
    is demoted to UNVERIFIED and its markers are recorded for stripping.
    """
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

    if entailment_gate:
        _apply_support_gate(result, evidence)

    return result


def _apply_support_gate(result: CitationValidationResult, evidence: list[Evidence]) -> None:
    """Demote cited claims whose evidence does not semantically support them.

    Uses local MiniLM embeddings (no network). A demoted claim keeps its text,
    becomes UNVERIFIED, and is registered in ``unsupported_citations`` so the
    caller can strip its markers from the prose.
    """
    from app.agent import support as support_gate

    if not support_gate.gate_enabled() or not result.claims:
        return
    text_by_id = {ev.evidence_id: ev.text for ev in evidence or []}
    for claim in result.claims:
        if not claim.evidence_ids or claim.status != ClaimStatus.VERIFIED:
            continue
        cited_texts = [text_by_id[eid] for eid in claim.evidence_ids if eid in text_by_id]
        if not cited_texts:
            continue
        score = support_gate.max_support(claim.text, cited_texts)
        if score >= support_gate.min_sim():
            continue
        tokens = extract_citation_tokens(claim.text)
        claim.status = ClaimStatus.UNVERIFIED
        claim.reasoning = (
            f"Cited evidence does not semantically support this sentence "
            f"(best similarity {score:.2f} < {support_gate.min_sim():.2f})"
        )
        result.unsupported_citations.append((claim.text, tokens))
        result.errors.append(
            CitationError(
                claim_id=claim.claim_id,
                issue="WEAK_SUPPORT",
                severity="medium",
                detail=(
                    f"Cited evidence similarity {score:.2f} below "
                    f"{support_gate.min_sim():.2f}: {claim.text[:140]}"
                ),
            )
        )


def strip_weak_markers(answer: str, result: CitationValidationResult) -> str:
    """Remove citation markers from sentences the support gate rejected.

    Unsupported claims must not keep citations in the prose — a marker implies
    the evidence backs the statement, which the gate just determined it does
    not. Sentences that cannot be located verbatim are left untouched (the
    claim still lands in Caveats via its UNVERIFIED status).
    """
    out = answer
    for sentence, _tokens in result.unsupported_citations:
        core = re.sub(r"\[[^\]]+\]", "", sentence)
        probe = re.sub(r"\s+", " ", core).strip()[:60]
        # Removing "[E6]" leaves "4-2 ." — trailing orphan punctuation would
        # break the verbatim lookup, so trim to the last word boundary.
        probe = probe.rstrip(" .,;:!?")
        if len(probe) < 20:
            continue
        idx = out.find(probe)
        if idx < 0:
            continue
        span_end = out.find("\n", idx)
        end = span_end if span_end > idx else min(idx + len(sentence) + 40, len(out))
        segment = out[idx:end]
        cleaned = CITATION_TOKEN_RE.sub("", segment)
        cleaned = re.sub(r"\s+([.,;:])", r"\1", cleaned)  # "4-2 ." -> "4-2."
        out = out[:idx] + cleaned + out[end:]
    return out



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
