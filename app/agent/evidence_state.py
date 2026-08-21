"""Cross-turn evidence memory: build, merge, serialize, load.

Stored compactly in each interaction's routing_path so the next turn can build
on verified facts. Deliberately stores typed evidence records, never raw
conversation text (that lives in ChatMessage).
"""

from __future__ import annotations

import json
import logging
import re

from app.agent.state import Claim, ClaimStatus, Evidence, EvidenceState

logger = logging.getLogger(__name__)

_MARKER = "EVIDENCE_STATE_JSON:"
MAX_CARRIED_EVIDENCE = 8


def build_evidence_state(
    evidence: list[Evidence],
    claims: list[Claim],
    turn: int,
) -> EvidenceState:
    """Keep only evidence that backed VERIFIED claims; record unresolved texts."""
    supported_ids: set[str] = set()
    for c in claims or []:
        if c.status == ClaimStatus.VERIFIED:
            supported_ids.update(c.evidence_ids or [])

    by_id = {ev.evidence_id: ev for ev in evidence or []}
    established = [by_id[i] for i in supported_ids if i in by_id]

    unresolved: list[str] = []
    for c in claims or []:
        if c.status in (ClaimStatus.UNVERIFIED, ClaimStatus.UNCERTAIN) and c.text:
            unresolved.append(c.text[:200])

    return EvidenceState(
        turn=turn,
        established=established[:MAX_CARRIED_EVIDENCE],
        unresolved=unresolved[:MAX_CARRIED_EVIDENCE],
    )


def merge_evidence_state(prior: EvidenceState | None, current: EvidenceState) -> EvidenceState:
    """Merge this turn's state with the prior one.

    Prior established facts are carried forward unless this turn re-established
    the same content (dedupe by normalized text). Unresolved items reset each
    turn — they describe the previous answer, not durable facts.
    """
    if prior is None or prior.is_empty():
        return current

    def _norm(t: str) -> str:
        return re.sub(r"\s+", " ", (t or "")).strip().lower()[:300]

    seen = {_norm(ev.text) for ev in current.established}
    merged_established = list(current.established)
    for ev in prior.established:
        if _norm(ev.text) not in seen:
            merged_established.append(ev)
            seen.add(_norm(ev.text))

    return EvidenceState(
        turn=current.turn,
        established=merged_established[:MAX_CARRIED_EVIDENCE],
        unresolved=list(current.unresolved),
    )


def serialize_for_storage(state: EvidenceState) -> str:
    """Compact JSON block appended to the interaction's routing_path."""
    try:
        payload = {
            "turn": state.turn,
            "established": [
                {
                    "evidence_id": ev.evidence_id,
                    "text": ev.text[:500],
                    "source_type": ev.source_type.value,
                    "source_name": ev.source_name,
                    "source_url": ev.source_url,
                    "source_date": ev.source_date.isoformat() if ev.source_date else None,
                }
                for ev in state.established
            ],
            "unresolved": state.unresolved,
        }
        return _MARKER + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        logger.exception("Failed to serialize evidence state")
        return ""


def load_evidence_state_from_text(text: str) -> EvidenceState | None:
    """Parse the marker block back into an EvidenceState; None when absent/corrupt."""
    if not text or _MARKER not in text:
        return None
    raw = text.split(_MARKER, 1)[1].strip()
    # The block ends at the first newline (trajectory precedes it) or runs to the end.
    raw = raw.split("\n", 1)[0].strip()
    try:
        data = json.loads(raw)
        established = [
            Evidence(
                evidence_id=e.get("evidence_id", ""),
                text=e.get("text", ""),
                source_type=e.get("source_type", "unknown"),
                source_name=e.get("source_name", ""),
                source_url=e.get("source_url"),
                metadata={"cite_key": None},
            )
            for e in data.get("established", [])
            if e.get("text")
        ]
        return EvidenceState(
            turn=int(data.get("turn", 0)),
            established=established,
            unresolved=[u for u in data.get("unresolved", []) if u],
        )
    except Exception:
        logger.exception("Failed to load evidence state from storage")
        return None
