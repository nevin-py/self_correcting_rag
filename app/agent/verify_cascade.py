"""
Cheap verification cascade: mechanical → numeric → local NLI → batched LLM residual.

When USE_VERIFY_CASCADE is enabled, expensive whole-answer LLM verify is replaced
by stage-wise resolution. Only unresolved claims hit a short-timeout LLM batch.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Callable

from app.agent.citation_validator import (
    CITATION_TOKEN_RE,
    extract_citation_tokens,
    resolve_citation_token,
    split_checkable_sentences,
)
from app.agent.state import Claim, ClaimStatus, ClaimType, Evidence
from app.core.config import settings

logger = logging.getLogger(__name__)

# ── Numeric / date patterns ───────────────────────────────────────────────────

_NUMBER_RE = re.compile(
    r"(?<![\w.])"
    r"("
    r"\d{1,3}(?:,\d{2,3})+(?:\.\d+)?|"  # 1,23,456 or 1,234.5
    r"\d+\.\d+|"
    r"\d+"
    r")"
    r"\s*(%|percent|percentage|crore|lakh|billion|million|trillion)?"
    r"(?![\w.])",
    re.IGNORECASE,
)
_FY_RE = re.compile(
    r"\b(?:FY\s*)?((?:19|20)\d{2})\s*[-–/]\s*((?:19|20)?\d{2})\b",
    re.IGNORECASE,
)
_YEAR_RE = re.compile(r"\b((?:19|20)\d{2})\b")
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "are",
    "was", "were", "be", "as", "by", "with", "from", "that", "this", "it", "at",
}


@dataclass
class CascadeStats:
    mechanical: int = 0
    numeric: int = 0
    nli: int = 0
    llm: int = 0
    escalated: int = 0
    uncertain: int = 0


@dataclass
class CascadeResult:
    claims: list[Claim] = field(default_factory=list)
    stats: CascadeStats = field(default_factory=CascadeStats)


# ── Coverage grading / query dedupe ──────────────────────────────────────────


def normalize_query(q: str) -> str:
    return re.sub(r"\s+", " ", (q or "").strip().lower())


def dedupe_search_queries(queries: list[str], *, max_keep: int = 1) -> list[str]:
    """Collapse near-duplicate search queries; keep at most ``max_keep`` distinct."""
    kept: list[str] = []
    norms: list[str] = []
    for raw in queries:
        q = (raw or "").strip()
        if not q:
            continue
        n = normalize_query(q)
        dup = False
        for existing in norms:
            if n == existing or n in existing or existing in n:
                dup = True
                break
            if SequenceMatcher(None, n, existing).ratio() >= 0.82:
                dup = True
                break
        if dup:
            continue
        kept.append(q)
        norms.append(n)
        if len(kept) >= max_keep:
            break
    return kept


def grade_coverage(
    query: str,
    evidence: list[Evidence],
    classification: Any | None = None,
    *,
    weak_score: float = 0.35,
) -> list[str]:
    """Return human-readable coverage gaps (empty = evidence looks adequate)."""
    gaps: list[str] = []
    if not evidence:
        gaps.append(f"no evidence for: {query[:120]}")
        return gaps

    scores = []
    for ev in evidence:
        s = ev.rerank_score if ev.rerank_score is not None else ev.combined_score
        scores.append(float(s or 0.0))
    top = max(scores) if scores else 0.0
    if top < weak_score:
        gaps.append(f"weak top evidence score ({top:.2f}) for: {query[:100]}")

    needed: list[str] = []
    if classification is not None:
        geo = getattr(classification, "geography", None)
        if geo:
            needed.append(str(geo))
        metric = getattr(classification, "metric_type", None)
        if metric is not None:
            mv = getattr(metric, "value", None) or str(metric)
            if mv and mv.lower() not in ("unknown", "none", ""):
                needed.append(str(mv))

    high = [ev for ev, s in zip(evidence, scores) if s >= weak_score]
    corpus = " ".join(ev.text for ev in (high or evidence[:5])).lower()
    for term in needed:
        t = str(term).strip().lower()
        if t and t not in corpus and t.replace("_", " ") not in corpus:
            gaps.append(f"missing high-score evidence for: {term}")

    return gaps


# ── Mechanical checker ───────────────────────────────────────────────────────


def _tokens(text: str) -> set[str]:
    return {
        t for t in re.findall(r"[a-z0-9%]+", (text or "").lower())
        if t not in _STOPWORDS and len(t) > 1
    }


def token_overlap_ratio(claim: str, evidence_text: str) -> float:
    a, b = _tokens(claim), _tokens(evidence_text)
    if not a:
        return 0.0
    return len(a & b) / len(a)


def mechanical_check_claim(
    sentence: str,
    evidence: list[Evidence],
    cite_map: dict[str, str] | None = None,
    *,
    min_overlap: float = 0.25,
) -> Claim | None:
    """Return a resolved Claim if mechanical check can decide; else None (escalate)."""
    known = {ev.evidence_id: ev for ev in evidence}
    tokens = extract_citation_tokens(sentence)
    if not tokens:
        return Claim(
            text=sentence[:500],
            status=ClaimStatus.UNVERIFIED,
            claim_type=ClaimType.FACT,
            evidence_ids=[],
            reasoning="Mechanical: factual sentence without citation",
            repair_action="rephrase",
        )

    resolved: list[str] = []
    invalid: list[str] = []
    for tok in tokens:
        eid = resolve_citation_token(tok, evidence, cite_map)
        if eid and eid in known:
            resolved.append(eid)
        else:
            invalid.append(tok)

    if invalid and not resolved:
        return Claim(
            text=sentence[:500],
            status=ClaimStatus.UNVERIFIED,
            claim_type=ClaimType.FACT,
            evidence_ids=[],
            reasoning=f"Mechanical: invalid citation(s) {invalid}",
            repair_action="rephrase",
        )

    claim_body = re.sub(CITATION_TOKEN_RE, " ", sentence)
    best = 0.0
    for eid in resolved:
        ev = known[eid]
        best = max(best, token_overlap_ratio(claim_body, ev.text))
    if best >= min_overlap:
        return Claim(
            text=sentence[:500],
            status=ClaimStatus.VERIFIED,
            claim_type=ClaimType.FACT,
            evidence_ids=list(dict.fromkeys(resolved)),
            reasoning=f"Mechanical: citation valid and overlap={best:.2f}",
            repair_action="none",
        )
    if best < 0.12:
        return Claim(
            text=sentence[:500],
            status=ClaimStatus.UNVERIFIED,
            claim_type=ClaimType.FACT,
            evidence_ids=list(dict.fromkeys(resolved)),
            reasoning=f"Mechanical: weak overlap with cited evidence ({best:.2f})",
            repair_action="rephrase",
        )
    # Mid overlap → escalate
    return None


# ── Numeric checker ──────────────────────────────────────────────────────────


def _parse_number(raw: str) -> float | None:
    try:
        return float(raw.replace(",", ""))
    except ValueError:
        return None


def extract_numbers(text: str) -> list[tuple[float, str]]:
    """Return (value, unit) pairs from text (excludes years / FY spans)."""
    masked = text or ""
    # Blank out fiscal-year spans so 2024-25 does not yield orphan 25
    masked = _FY_RE.sub(" ", masked)
    masked = re.sub(r"\b((?:19|20)\d{2})\b", " ", masked)
    out: list[tuple[float, str]] = []
    for m in _NUMBER_RE.finditer(masked):
        val = _parse_number(m.group(1))
        if val is None:
            continue
        unit = (m.group(2) or "").lower()
        if unit in ("percent", "percentage"):
            unit = "%"
        out.append((val, unit))
    return out


def extract_fiscal_years(text: str) -> set[str]:
    years: set[str] = set()
    for m in _FY_RE.finditer(text or ""):
        y1 = m.group(1)
        y2 = m.group(2)
        if len(y2) == 2:
            y2 = y1[:2] + y2
        years.add(f"{y1}-{y2}")
        years.add(f"{y1[-2:]}-{y2[-2:]}")
    return years


def numeric_compare(
    claim_text: str,
    evidence_text: str,
    *,
    rel_tol: float = 0.02,
    abs_tol: float = 0.05,
) -> str:
    """Return 'pass' | 'contradict' | 'skip' for numeric/date agreement."""
    c_nums = extract_numbers(claim_text)
    e_nums = extract_numbers(evidence_text)
    c_fy = extract_fiscal_years(claim_text)
    e_fy = extract_fiscal_years(evidence_text)

    decided = False

    if c_fy and e_fy:
        decided = True
        if c_fy.isdisjoint(e_fy):
            # Also allow plain year overlap
            c_years = set(_YEAR_RE.findall(claim_text))
            e_years = set(_YEAR_RE.findall(evidence_text))
            if not (c_years & e_years):
                return "contradict"

    if not c_nums:
        return "pass" if decided else "skip"

    if not e_nums:
        # Claim asserts a magnitude the cited snippet does not contain.
        return "contradict"

    # For each claim number, require a nearby evidence number with compatible unit
    for cval, cunit in c_nums:
        matched = False
        for eval_, eunit in e_nums:
            if cunit and eunit and cunit != eunit:
                # Allow bare number vs % when values match
                if not ({cunit, eunit} <= {"%", ""} or cunit == eunit):
                    if {cunit, eunit} != {"%", ""} and cunit and eunit:
                        continue
            tol = max(abs_tol, abs(cval) * rel_tol)
            if abs(cval - eval_) <= tol:
                matched = True
                break
            # Common scale slip: 54.5 vs 0.545 (share as fraction)
            if cunit == "%" or eunit == "%":
                if abs(cval - eval_ * 100) <= tol or abs(cval * 100 - eval_) <= tol:
                    matched = True
                    break
        if not matched:
            # If claim has a distinctive number absent from evidence → contradict
            return "contradict"
        decided = True

    return "pass" if decided else "skip"


def numeric_check_claim(
    sentence: str,
    evidence: list[Evidence],
    cite_map: dict[str, str] | None = None,
) -> Claim | None:
    """Resolve via numeric/date check when possible; None to escalate."""
    known = {ev.evidence_id: ev for ev in evidence}
    resolved = []
    for tok in extract_citation_tokens(sentence):
        eid = resolve_citation_token(tok, evidence, cite_map)
        if eid and eid in known:
            resolved.append(eid)
    if not resolved:
        return None

    outcomes = []
    for eid in resolved:
        outcomes.append(numeric_compare(sentence, known[eid].text))

    if any(o == "contradict" for o in outcomes):
        return Claim(
            text=sentence[:500],
            status=ClaimStatus.CONTRADICTED,
            claim_type=ClaimType.FACT,
            evidence_ids=list(dict.fromkeys(resolved)),
            contradicting_evidence_ids=list(dict.fromkeys(resolved)),
            reasoning="Numeric: claim numbers/dates disagree with cited evidence",
            repair_action="rephrase",
        )
    if any(o == "pass" for o in outcomes) and all(o != "contradict" for o in outcomes):
        # Only treat as verified if at least one pass and no skip-only on all
        if all(o == "skip" for o in outcomes):
            return None
        return Claim(
            text=sentence[:500],
            status=ClaimStatus.VERIFIED,
            claim_type=ClaimType.FACT,
            evidence_ids=list(dict.fromkeys(resolved)),
            reasoning="Numeric: numbers/dates consistent with cited evidence",
            repair_action="none",
        )
    return None


# ── Local NLI (lazy DeBERTa) ─────────────────────────────────────────────────

_nli_pipeline = None
_nli_init_attempted = False


def _get_nli_pipeline():
    """Lazy-load tokenizer+model for premise/hypothesis NLI (FlashRank-style)."""
    global _nli_pipeline, _nli_init_attempted
    if _nli_pipeline is not None:
        return _nli_pipeline
    if _nli_init_attempted:
        return None
    _nli_init_attempted = True
    try:
        import torch  # type: ignore
        from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore

        model_name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.eval()
        id2label = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
        _nli_pipeline = (tokenizer, model, torch, id2label)
        logger.info("Loaded NLI model %s", model_name)
    except Exception as exc:
        logger.warning("NLI model unavailable (%s); cascade will escalate to LLM", exc)
        _nli_pipeline = None
    return _nli_pipeline


def nli_check_claim(
    sentence: str,
    evidence: list[Evidence],
    cite_map: dict[str, str] | None = None,
    *,
    entail_threshold: float | None = None,
    contradict_threshold: float | None = None,
) -> Claim | None:
    """Local NLI: pass / fail / escalate (None)."""
    entail_threshold = entail_threshold if entail_threshold is not None else settings.NLI_ENTAIL_THRESHOLD
    contradict_threshold = (
        contradict_threshold if contradict_threshold is not None else settings.NLI_CONTRADICT_THRESHOLD
    )
    bundle = _get_nli_pipeline()
    if bundle is None:
        return None
    tokenizer, model, torch, id2label = bundle

    known = {ev.evidence_id: ev for ev in evidence}
    resolved = []
    for tok in extract_citation_tokens(sentence):
        eid = resolve_citation_token(tok, evidence, cite_map)
        if eid and eid in known:
            resolved.append(eid)
    if not resolved:
        return None

    premise = " ".join(known[eid].text for eid in resolved)[:1500]
    hypothesis = re.sub(CITATION_TOKEN_RE, "", sentence).strip()[:500]
    try:
        inputs = tokenizer(premise, hypothesis, truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits[0]
            probs = torch.softmax(logits, dim=-1).tolist()
        label_scores = {id2label.get(i, str(i)): float(p) for i, p in enumerate(probs)}
    except Exception as exc:
        logger.debug("NLI inference failed: %s", exc)
        return None

    entail = max(
        (label_scores.get(k, 0.0) for k in ("entailment", "entail", "supported")),
        default=0.0,
    )
    contra = max(
        (label_scores.get(k, 0.0) for k in ("contradiction", "contradict", "refuted")),
        default=0.0,
    )
    if entail >= entail_threshold and entail >= contra:
        return Claim(
            text=sentence[:500],
            status=ClaimStatus.VERIFIED,
            claim_type=ClaimType.FACT,
            evidence_ids=list(dict.fromkeys(resolved)),
            reasoning=f"NLI: entailment={entail:.2f}",
            repair_action="none",
        )
    if contra >= contradict_threshold and contra > entail:
        return Claim(
            text=sentence[:500],
            status=ClaimStatus.CONTRADICTED,
            claim_type=ClaimType.FACT,
            evidence_ids=list(dict.fromkeys(resolved)),
            contradicting_evidence_ids=list(dict.fromkeys(resolved)),
            reasoning=f"NLI: contradiction={contra:.2f}",
            repair_action="rephrase",
        )
    return None


# ── LLM residual batch ───────────────────────────────────────────────────────


def _llm_verify_residuals(
    sentences: list[str],
    evidence: list[Evidence],
    cite_map: dict[str, str] | None,
    llm_invoke: Callable[[list[str], list[Evidence]], list[Claim]],
    timeout_s: float | None = None,
) -> list[Claim]:
    """Run batched LLM residual; on failure → UNCERTAIN (never silent pass)."""
    if not sentences:
        return []

    try:
        result = llm_invoke(sentences, evidence)
        if result:
            return result
        raise ValueError("Cascade LLM residual returned empty claims")
    except Exception as exc:
        logger.warning("Cascade LLM residual failed: %s", exc)

    out: list[Claim] = []
    known = {ev.evidence_id: ev for ev in evidence}
    for s in sentences:
        ids = []
        for tok in extract_citation_tokens(s):
            eid = resolve_citation_token(tok, evidence, cite_map)
            if eid and eid in known:
                ids.append(eid)
        out.append(
            Claim(
                text=s[:500],
                status=ClaimStatus.UNCERTAIN,
                claim_type=ClaimType.FACT,
                evidence_ids=list(dict.fromkeys(ids)),
                reasoning="Cascade LLM residual failed or returned empty",
                repair_action="rephrase",
            )
        )
    return out


def run_verify_cascade(
    answer: str,
    evidence: list[Evidence],
    *,
    cite_map: dict[str, str] | None = None,
    llm_invoke: Callable[[list[str], list[Evidence]], list[Claim]] | None = None,
    timeout_s: float | None = None,
) -> CascadeResult:
    """Run mechanical → numeric → NLI → LLM residual on factual sentences."""
    result = CascadeResult()
    stats = result.stats
    sentences = split_checkable_sentences(answer)

    pending: list[str] = []
    for sent in sentences:
        claim = mechanical_check_claim(sent, evidence, cite_map)
        if claim is not None:
            # Numbers in a mechanically-passed sentence must still appear in evidence
            if claim.status == ClaimStatus.VERIFIED and extract_numbers(sent):
                numeric = numeric_check_claim(sent, evidence, cite_map)
                if numeric is not None and numeric.status == ClaimStatus.CONTRADICTED:
                    result.claims.append(numeric)
                    stats.numeric += 1
                    continue
            result.claims.append(claim)
            stats.mechanical += 1
            continue
        pending.append(sent)

    still: list[str] = []
    for sent in pending:
        claim = numeric_check_claim(sent, evidence, cite_map)
        if claim is not None:
            result.claims.append(claim)
            stats.numeric += 1
            continue
        still.append(sent)

    after_nli: list[str] = []
    for sent in still:
        claim = nli_check_claim(sent, evidence, cite_map)
        if claim is not None:
            result.claims.append(claim)
            stats.nli += 1
            continue
        after_nli.append(sent)

    stats.escalated = len(after_nli)
    if after_nli:
        if llm_invoke is None:
            for s in after_nli:
                result.claims.append(
                    Claim(
                        text=s[:500],
                        status=ClaimStatus.UNCERTAIN,
                        claim_type=ClaimType.FACT,
                        evidence_ids=[],
                        reasoning="Escalated but no LLM residual available",
                        repair_action="rephrase",
                    )
                )
                stats.uncertain += 1
        else:
            llm_claims = _llm_verify_residuals(
                after_nli, evidence, cite_map, llm_invoke, timeout_s
            )
            for c in llm_claims:
                result.claims.append(c)
                if c.status == ClaimStatus.UNCERTAIN:
                    stats.uncertain += 1
                else:
                    stats.llm += 1

    logger.info(
        "verify_cascade resolved mechanical=%d numeric=%d nli=%d llm=%d escalated=%d uncertain=%d",
        stats.mechanical,
        stats.numeric,
        stats.nli,
        stats.llm,
        stats.escalated,
        stats.uncertain,
    )
    return result


# ── Surgical sentence patch ──────────────────────────────────────────────────


_PATCH_PROMPT = """You are repairing ONE flagged sentence in a research answer.

Rules:
- Rewrite ONLY the target sentence so it is faithful to the evidence snippets.
- End the sentence with a valid citation from the allowed keys: {allowed_keys}
- Do not invent numbers, years, or sources.
- If evidence cannot support the claim, rewrite as an honest uncertainty statement with no fake stats.
- Return ONLY the replacement sentence (no markdown fences, no commentary).

Neighbor before: {before}
TARGET: {target}
Neighbor after: {after}

Evidence snippets:
{snippets}
"""


def patch_flagged_sentences(
    answer: str,
    failed_claims: list[Claim],
    evidence: list[Evidence],
    cite_map: dict[str, str] | None,
    llm_rewrite: Callable[[str], str],
) -> str:
    """Surgically replace failed claim sentences using a short LLM rewrite."""
    if not answer or not failed_claims:
        return answer

    known = {ev.evidence_id: ev for ev in evidence}
    cite_map = cite_map or {}
    reverse = {v: k for k, v in cite_map.items()}
    allowed = sorted(cite_map.keys()) or [f"[{ev.evidence_id}]" for ev in evidence[:8]]

    text = answer
    for claim in failed_claims:
        target = (claim.text or "").strip()
        if not target or target not in text:
            # Fuzzy: find sentence containing first 40 chars
            needle = target[:40]
            if needle and needle in text:
                # locate surrounding sentence roughly
                idx = text.find(needle)
                start = text.rfind("\n", 0, idx)
                start = 0 if start < 0 else start + 1
                end = text.find("\n", idx)
                if end < 0:
                    end = len(text)
                target = text[start:end].strip()
            else:
                continue

        # Neighbors by newline/sentence split
        parts = re.split(r"(?<=[.!?])\s+|\n+", text)
        before = after = ""
        for i, p in enumerate(parts):
            if target in p or p.strip() == target:
                before = parts[i - 1] if i > 0 else ""
                after = parts[i + 1] if i + 1 < len(parts) else ""
                break

        ids = list(claim.evidence_ids)
        if not ids:
            for tok in extract_citation_tokens(target):
                eid = resolve_citation_token(tok, evidence, cite_map)
                if eid:
                    ids.append(eid)
        snippets = []
        for eid in ids[:3]:
            ev = known.get(eid)
            if ev:
                key = reverse.get(eid, eid)
                snippets.append(f"[{key}] {ev.text[:400]}")
        if not snippets:
            for ev in evidence[:3]:
                key = reverse.get(ev.evidence_id, ev.evidence_id)
                snippets.append(f"[{key}] {ev.text[:400]}")

        keys_str = ", ".join(
            k if str(k).startswith("[") else f"[{k}]" for k in allowed
        )
        prompt = _PATCH_PROMPT.format(
            allowed_keys=keys_str,
            before=before[:200],
            target=target[:500],
            after=after[:200],
            snippets="\n".join(snippets) or "(no snippets)",
        )
        try:
            replacement = (llm_rewrite(prompt) or "").strip()
        except Exception as exc:
            logger.warning("Surgical patch LLM failed: %s", exc)
            continue
        if not replacement or len(replacement) < 8:
            continue
        # Single line / sentence
        replacement = replacement.split("\n")[0].strip().strip('"')
        text = text.replace(target, replacement, 1)

    return text
