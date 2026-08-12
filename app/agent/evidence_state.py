"""
Persistent, structured cross-turn evidence state.

Replaces the previous 'prepend a prior-evidence summary to the query' hack. Evidence
is carried forward as typed records so a later turn can build on *established* facts
while still re-verifying, *superseding*, or flagging *conflicts*. It deliberately does
NOT store raw conversation text -- only structured, provenance-preserving evidence.

Categories:
    established   verified facts from prior turns (reusable, but still re-checked)
    inferences   inference-level evidence carried forward
    superseded   older evidence replaced by a newer figure (kept for audit, penalized)
    conflicts    structured conflict records
    unresolved   claims that could not be resolved

The state is serialized compactly (JSON) and stored alongside each interaction; the
router reads it back at the start of the next turn.
"""

from __future__ import annotations

import json
import re
from typing import Any

from app.agent.normalization import parse_year
from app.agent.ranking import status_priority
from app.agent.state import (
    Claim,
    ClaimStatus,
    ClaimType,
    Evidence,
    EvidenceState,
    MetricType,
    PriceBasis,
    TemporalQualifier,
)

MAX_CARRIED = 8  # cap per category to keep context bounded


def _evidence_signature(ev: Evidence) -> str:
    """Dedup signature ignoring volatile scores."""
    return "|".join([
        ev.metric_type.value,
        ev.geographic_scope.value,
        ev.geography.lower(),
        ev.year_period,
        ev.temporal_qualifier.value,
        (ev.metric_value or ev.text[:60]).lower(),
    ])


def _supersede_key(ev: Evidence) -> tuple[str, str, str]:
    """Key used to detect when a newer figure replaces an older one for the same
    measured quantity (metric + geography + period), independent of status/value."""
    return (ev.metric_type.value, ev.geography.lower(), ev.year_period)


def _evidence_sort_key(ev: Evidence) -> tuple[float, float, int]:
    year = parse_year(ev.year_period) or 0
    return (ev.authority_score, status_priority(ev.temporal_qualifier), year)


def build_evidence_state(
    evidence: list[Evidence],
    claims: list[Claim] | None = None,
    conflicts: list[dict] | None = None,
    turn: int = 0,
) -> EvidenceState:
    """Construct a turn's evidence state from its evidence + verification claims.

    Evidence supporting VERIFIED claims becomes 'established'; inference evidence
    becomes 'inferences'; unresolved/contradicted claims become 'unresolved'.
    """
    claims = claims or []
    conflicts = conflicts or []

    established: list[Evidence] = []
    inferences: list[Evidence] = []
    unresolved: list[str] = []

    ev_by_id = {ev.evidence_id: ev for ev in evidence}

    for claim in claims:
        if claim.claim_type == ClaimType.INFERENCE:
            for eid in claim.evidence_ids:
                ev = ev_by_id.get(eid)
                if ev and ev not in inferences:
                    inferences.append(ev)
        elif claim.status == ClaimStatus.VERIFIED:
            for eid in claim.evidence_ids:
                ev = ev_by_id.get(eid)
                if ev and ev not in established:
                    established.append(ev)
        elif claim.status in (
            ClaimStatus.UNVERIFIED,
            ClaimStatus.UNCERTAIN,
            ClaimStatus.CONTRADICTED,
            ClaimStatus.PARTIALLY_VERIFIED,
        ):
            unresolved.append(claim.text)

    # Fallback: if nothing was verified, carry the most authoritative evidence.
    if not established and not inferences:
        for ev in sorted(evidence, key=_evidence_sort_key, reverse=True):
            if ev.authority_score >= 0.8:
                established.append(ev)

    state = EvidenceState(
        turn=turn,
        established=established[:MAX_CARRIED],
        inferences=inferences[:MAX_CARRIED],
        conflicts=conflicts[:MAX_CARRIED],
        unresolved=unresolved[:MAX_CARRIED],
    )
    return state


def merge_evidence_state(prior: EvidenceState | None, current: EvidenceState) -> EvidenceState:
    """Merge a prior state with the current turn's state.

    - Newer established evidence for the same (metric, geo, period) *supersedes*
      older evidence (older moves to `superseded`).
    - Conflicts and unresolved items are unioned (deduped).
    - Categories are capped to keep context bounded.
    """
    if prior is None or prior.is_empty():
        return current

    prior_established = list(prior.established)
    current_established = list(current.established)

    current_keys = {_evidence_signature(ev) for ev in current_established}
    current_sup_keys = {_supersede_key(ev) for ev in current_established}
    # Anything in prior that is replaced by a current item (same metric/geo/period,
    # even if the newer one has a different status or value) becomes superseded.
    superseded: list[Evidence] = list(prior.superseded)
    kept_prior: list[Evidence] = []
    for ev in prior_established:
        sig = _evidence_signature(ev)
        if sig in current_keys or _supersede_key(ev) in current_sup_keys:
            superseded.append(ev)
        else:
            kept_prior.append(ev)

    established = current_established + kept_prior
    # Drop duplicates by signature.
    seen: set[str] = set()
    deduped_established: list[Evidence] = []
    for ev in sorted(established, key=_evidence_sort_key, reverse=True):
        s = _evidence_signature(ev)
        if s not in seen:
            seen.add(s)
            deduped_established.append(ev)

    inferences = _union_evidence(prior.inferences, current.inferences)
    conflicts = _union_conflicts(prior.conflicts, current.conflicts)
    unresolved = _union_strings(prior.unresolved, current.unresolved)

    return EvidenceState(
        turn=current.turn,
        established=deduped_established[:MAX_CARRIED],
        inferences=inferences[:MAX_CARRIED],
        superseded=superseded[:MAX_CARRIED],
        conflicts=conflicts[:MAX_CARRIED],
        unresolved=unresolved[:MAX_CARRIED],
    )


def _union_evidence(a: list[Evidence], b: list[Evidence]) -> list[Evidence]:
    seen: set[str] = set()
    out: list[Evidence] = []
    for ev in a + b:
        s = _evidence_signature(ev)
        if s not in seen:
            seen.add(s)
            out.append(ev)
    return out


def _union_conflicts(a: list[dict], b: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for c in a + b:
        key = f"{c.get('evidence_a')}|{c.get('evidence_b')}|{c.get('conflict_type')}"
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def _union_strings(a: list[str], b: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for s in a + b:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


# ── Serialization (compact JSON stored with each interaction) ────────────────

def evidence_state_to_json(state: EvidenceState) -> dict:
    return {
        "turn": state.turn,
        "established": [ev.model_dump() for ev in state.established],
        "inferences": [ev.model_dump() for ev in state.inferences],
        "superseded": [ev.model_dump() for ev in state.superseded],
        "conflicts": state.conflicts,
        "unresolved": state.unresolved,
    }


def evidence_state_from_json(payload: dict | None) -> EvidenceState | None:
    if not payload:
        return None
    try:
        return EvidenceState(
            turn=payload.get("turn", 0),
            established=[Evidence(**e) for e in payload.get("established", [])],
            inferences=[Evidence(**e) for e in payload.get("inferences", [])],
            superseded=[Evidence(**e) for e in payload.get("superseded", [])],
            conflicts=payload.get("conflicts", []),
            unresolved=payload.get("unresolved", []),
        )
    except Exception:
        return None


def load_evidence_state_from_text(trajectory: str) -> EvidenceState | None:
    """Parse an EvidenceState JSON block stored after a newline in a trajectory string."""
    if not trajectory or "\n" not in trajectory:
        return None
    _, _, json_part = trajectory.partition("\n")
    json_part = json_part.strip()
    if not json_part.startswith("{"):
        return None
    try:
        return evidence_state_from_json(json.loads(json_part))
    except Exception:
        return None


def serialize_for_storage(state: EvidenceState) -> str:
    """Render the state as a JSON string to append to the trajectory."""
    return json.dumps(evidence_state_to_json(state), default=str)


# ── Context rendering (structured, bounded, provenance-preserving) ───────────

def to_context_block(state: EvidenceState | None) -> str:
    """Render the cross-turn evidence state as a compact context block.

    This is passed to generation/verification so the model sees *established facts*
    and *open questions* without dumping the entire conversation.
    """
    if state is None or state.is_empty():
        return ""

    lines: list[str] = ["[CROSS-TURN EVIDENCE STATE]"]

    if state.established:
        lines.append("ESTABLISHED FACTS (verified in prior turns; re-check before reuse):")
        for ev in state.established:
            meta = _meta_line(ev)
            lines.append(f"  - {ev.text[:180]}  [{meta}]")

    if state.inferences:
        lines.append("PRIOR INFERENCES (derivable, not direct facts):")
        for ev in state.inferences:
            meta = _meta_line(ev)
            lines.append(f"  - {ev.text[:180]}  [{meta}]")

    if state.superseded:
        lines.append("SUPERSEDED (older figures replaced by newer evidence):")
        for ev in state.superseded:
            meta = _meta_line(ev)
            lines.append(f"  - {ev.text[:180]}  [{meta}]")

    if state.conflicts:
        lines.append("OPEN CONFLICTS (classify before asserting):")
        for c in state.conflicts:
            lines.append(f"  - {c.get('conflict_type', 'conflict')}: {c.get('reason', '')}")

    if state.unresolved:
        lines.append("UNRESOLVED CLAIMS:")
        for u in state.unresolved:
            lines.append(f"  - {u[:180]}")

    return "\n".join(lines)


def _meta_line(ev: Evidence) -> str:
    parts = []
    if ev.metric_type != MetricType.UNKNOWN:
        parts.append(ev.metric_type.value.upper())
    if ev.price_basis != PriceBasis.UNKNOWN:
        parts.append(f"{ev.price_basis.value}-price")
    if ev.geography:
        parts.append(ev.geography)
    if ev.year_period:
        parts.append(ev.year_period)
    if ev.temporal_qualifier != TemporalQualifier.UNKNOWN:
        parts.append(ev.temporal_qualifier.value)
    src = ev.source_name or ev.source_url or "unknown"
    parts.append(f"src={src}")
    return " ".join(parts)
