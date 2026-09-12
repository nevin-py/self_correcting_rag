"""Agent nodes for the lean self-correcting RAG pipeline.

Architecture — one LLM call per job, no heuristic stand-ins:

    classify_and_plan   -> intent + retrieval plan (single structured LLM call)
    gather_evidence     -> parallel document retrieval + web search, rerank, cite keys
    generate_answer     -> cited answer generation from assembled evidence
    verify_answer       -> mechanical citation check + single LLM judge call
    conversational_response / ask_clarification -> short-circuit exits

Self-correction (Reflexion-style): when the judge finds gaps, it serializes a
critique; the graph loops gather_evidence -> generate_answer -> verify_answer
and the regeneration pass sees the critique + its previous draft, so the second
attempt fixes the flagged problems instead of repeating them.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from langgraph.graph import END
import time
from datetime import datetime, timezone
from typing import Any, get_args, get_origin
from urllib.parse import urlparse

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError
from pydantic_core import PydanticUndefined

from app.agent.citation_validator import (
    CITATION_TOKEN_RE,
    CitationValidationResult,
    flag_uncited_in_answer,
    prune_caveats,
    strip_weak_markers,
    validate_answer_citations,
)
from app.agent.context_assembly import GENERATOR_BUDGET
from app.agent.context_provider import get_current_context, format_context_for_llm
from app.agent.reranker import rerank
from app.agent.state import (
    Claim,
    ClaimStatus,
    ComputationPlan,
    Evidence,
    EvidenceState,
    QueryMode,
    QueryUnderstanding,
    SourceType,
    Verdict,
    utc_now,
)
from app.core.config import settings
from app.observability import tracing

logger = logging.getLogger(__name__)

MAX_HISTORY_MESSAGES = 12          # conversation turns fed to LLM calls
MAX_SEARCH_QUERIES = 3             # per gather pass
MAX_EVIDENCE = 12                  # evidence blocks in the generator context
EVIDENCE_SNIPPET_CHARS = 1200      # per evidence block in prompts
REPAIR_RESERVED_SLOTS = 4          # context slots guaranteed to repair-pass evidence
CAVEAT_MAX_BULLETS = 5             # caveat section never grows beyond this


# ── LLM plumbing ─────────────────────────────────────────────────────────────


def _normalize_messages_for_gemini(messages: list) -> list:
    """Gemini rejects system-only payloads ('contents are required')."""
    if not messages:
        return [HumanMessage(content="Continue.")]
    out: list[BaseMessage] = []
    has_human = False
    for msg in messages:
        if isinstance(msg, HumanMessage):
            has_human = True
        out.append(msg)
    if not has_human:
        system_parts = [m.content for m in out if isinstance(m, SystemMessage)]
        other = [m for m in out if not isinstance(m, SystemMessage)]
        merged = "\n\n".join(str(p) for p in system_parts if p)
        return other + [HumanMessage(content=merged or "Follow the instructions.")]
    return out


def _is_google_llm(llm: Any) -> bool:
    return llm is not None and llm.__class__.__name__ == "ChatGoogleGenerativeAI"


def _prepare_messages(llm: Any, messages: list) -> list:
    if _is_google_llm(llm):
        return _normalize_messages_for_gemini(messages)
    return messages


def _strip_json_markers(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _extract_json_object(text: str) -> dict:
    """Best-effort JSON object extraction from model prose / markdown fences."""
    if not text:
        return {}
    cleaned = _strip_json_markers(text)
    try:
        data = json.loads(cleaned)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    if start < 0:
        return {}
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    data = json.loads(cleaned[start : i + 1])
                    return data if isinstance(data, dict) else {}
                except json.JSONDecodeError:
                    return {}
    return {}


def _base_model_type(annotation: Any) -> type[BaseModel] | None:
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    origin = get_origin(annotation)
    if origin is None:
        return None
    for arg in get_args(annotation):
        if isinstance(arg, type) and issubclass(arg, BaseModel):
            return arg
    return None


def _coerce_null_strings(data: Any, model: type[BaseModel]) -> Any:
    """Coerce JSON nulls to schema defaults before Pydantic validation."""
    if not isinstance(data, dict):
        return data
    fixed = dict(data)
    for key, finfo in model.model_fields.items():
        if key not in fixed:
            continue
        val = fixed[key]
        if val is None and finfo.annotation is str:
            fixed[key] = finfo.default if finfo.default is not PydanticUndefined else ""
        elif isinstance(val, dict):
            nested = _base_model_type(finfo.annotation)
            if nested is not None:
                fixed[key] = _coerce_null_strings(val, nested)
    return fixed


def _validate_structured(data: dict, output_schema: type[BaseModel]) -> BaseModel:
    try:
        return output_schema.model_validate(data)
    except ValidationError:
        return output_schema.model_validate(_coerce_null_strings(data, output_schema))


def _response_text(response: Any) -> str:
    """Extract visible text from an LLM response (handles list / reasoning-only)."""
    content = getattr(response, "content", None)
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                parts.append(str(block.get("text") or block.get("content") or ""))
            else:
                parts.append(str(getattr(block, "text", None) or block))
        content = "".join(parts)
    text = str(content or "").strip()
    if text:
        return text
    extra = getattr(response, "additional_kwargs", None) or {}
    for key in ("reasoning_content", "reasoning", "text", "output"):
        val = extra.get(key)
        if isinstance(val, dict):
            val = val.get("text") or val.get("content") or ""
        if isinstance(val, str) and val.strip():
            return val.strip()
    meta = getattr(response, "response_metadata", None) or {}
    for key in ("reasoning", "content"):
        val = meta.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""




# #region agent log
def _agent_dbg(hypothesis_id: str, location: str, message: str, data: dict) -> None:
    try:
        with open(
            "/home/ariva/work/project_self_rag/self_correcting_rag/.cursor/debug-05b494.log",
            "a",
            encoding="utf-8",
        ) as _df:
            _df.write(json.dumps({
                "sessionId": "05b494",
                "hypothesisId": hypothesis_id,
                "location": location,
                "message": message,
                "data": data,
                "timestamp": int(time.time() * 1000),
            }) + "\n")
    except Exception:
        pass
# #endregion


def _llm_with_fallback(primary: Any, fallbacks: Any, messages: list, output_schema: Any,
                       role: str = "", timeout: int = 30, max_attempts: int | None = None):
    """Call primary LLM; on failure walk fallbacks with structured output.

    Every attempt is traced (model, latency, size, outcome) for observability.
    """

    chain = [primary]
    if isinstance(fallbacks, (list, tuple)):
        chain.extend(fallbacks)
    elif fallbacks is not None:
        chain.append(fallbacks)
    if max_attempts is not None:
        chain = chain[: max(1, max_attempts)]
    # #region agent log
    _agent_dbg("B", "nodes.py:_llm_with_fallback", "chain", {
        "role": role, "n": len(chain), "timeout": timeout, "max_attempts": max_attempts,
    })
    # #endregion
    prompt_chars = sum(len(str(getattr(m, "content", ""))) for m in messages)
    last_exc: Exception | None = None
    for idx, llm in enumerate(chain):
        if llm is None:
            continue
        t0 = time.perf_counter()
        try:
            result = _structured_invoke(llm, messages, output_schema, timeout=timeout)
        except Exception as exc:
            status = "timeout" if "timed out" in str(exc).lower() else "error"
            tracing.record_llm_call(
                role=role, llm=llm, attempt=idx + 1, status=status,
                started_at=t0, ended_at=time.perf_counter(),
                prompt_chars=prompt_chars, completion_chars=0,
                error=str(exc)[:300],
            )
            last_exc = exc
            logger.warning(
                "LLM call failed (%s)%s", exc,
                ", trying next" if idx < len(chain) - 1 else "",
            )
            continue
        completion_chars = len(result.model_dump_json()) if isinstance(result, BaseModel) else len(str(result))
        tracing.record_llm_call(
            role=role, llm=llm, attempt=idx + 1, status="ok",
            started_at=t0, ended_at=time.perf_counter(),
            prompt_chars=prompt_chars, completion_chars=completion_chars,
        )
        return result
    if last_exc:
        raise last_exc
    raise RuntimeError("No LLM clients available")


def _structured_invoke(llm: Any, messages: list, output_schema: Any, timeout: int = 30):
    """Invoke with structured output; fall back to raw JSON parsing. Bounded by timeout."""
    import concurrent.futures

    prepared = _prepare_messages(llm, messages)

    def _do_structured():
        try:
            try:
                bound = llm.with_structured_output(output_schema, method="json_schema")
            except TypeError:
                bound = llm.with_structured_output(output_schema)
            return bound.invoke(prepared)
        except Exception:
            return None

    def _do_raw():
        parse_hint = HumanMessage(
            content=(
                "Respond with ONLY a single valid JSON object matching the required schema. "
                "No markdown, no commentary, no code fences. Never return an empty response."
            )
        )
        response = llm.invoke([*prepared, parse_hint])
        text = _response_text(response)
        if not text:
            raise ValueError("Could not parse JSON from model output: (empty response)")
        data = _extract_json_object(text)
        if not data:
            raise ValueError(f"Could not parse JSON from model output: {text[:200]}")
        return _validate_structured(data, output_schema)

    # Do NOT use `with ThreadPoolExecutor`: on TimeoutError the context
    # manager calls shutdown(wait=True) and blocks until the hung HTTP call
    # finishes (observed 344s classify after a "20s" timeout).
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    t_exec = time.perf_counter()
    try:
        future = executor.submit(_do_structured)
        try:
            result = future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            # #region agent log
            _agent_dbg("A", "nodes.py:_structured_invoke", "structured_timeout", {
                "timeout_s": timeout,
                "elapsed_s": round(time.perf_counter() - t_exec, 2),
            })
            # #endregion
            raise ValueError(f"LLM call timed out after {timeout}s (structured attempt)")
        if result is not None:
            return result
        logger.debug("Structured output empty; trying raw JSON parse")

        future = executor.submit(_do_raw)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            raise ValueError(f"LLM call timed out after {timeout}s (raw attempt)")
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"LLM call failed: {exc}")
    finally:
        t_sd = time.perf_counter()
        executor.shutdown(wait=False, cancel_futures=True)
        # #region agent log
        _agent_dbg("A", "nodes.py:_structured_invoke", "executor_shutdown", {
            "shutdown_wait_s": round(time.perf_counter() - t_sd, 3),
        })
        # #endregion


def _invoke_chat(primary: Any, fallbacks: tuple[Any, ...] | list[Any], messages: list, role: str = "generator") -> tuple[str, str]:
    """Invoke chat models with multi-fallback. Returns (text, used_label_suffix).

    Every attempt is traced for observability.
    """
    chain = [primary, *list(fallbacks or ())]
    prompt_chars = sum(len(str(getattr(m, "content", ""))) for m in messages)
    last_exc: Exception | None = None
    for idx, llm in enumerate(chain):
        if llm is None:
            continue
        t0 = time.perf_counter()
        try:
            response = llm.invoke(_prepare_messages(llm, messages))
            text = _response_text(response)
            if not text:
                raise ValueError("Model returned empty response")
            tracing.record_llm_call(
                role=role, llm=llm, attempt=idx + 1, status="ok",
                started_at=t0, ended_at=time.perf_counter(),
                prompt_chars=prompt_chars, completion_chars=len(text),
            )
            return text, ("primary" if idx == 0 else f"fallback-{idx}")
        except Exception as exc:
            status = "timeout" if "timed out" in str(exc).lower() else "error"
            tracing.record_llm_call(
                role=role, llm=llm, attempt=idx + 1, status=status,
                started_at=t0, ended_at=time.perf_counter(),
                prompt_chars=prompt_chars, completion_chars=0,
                error=str(exc)[:300],
            )
            last_exc = exc
            logger.warning(
                "Chat call failed (%s)%s", exc,
                ", trying next" if idx < len(chain) - 1 else "",
            )
    if last_exc:
        raise last_exc
    raise RuntimeError("No LLM clients available")


# ── Shared helpers ───────────────────────────────────────────────────────────


def _resolve_llms(state: dict):
    from app.documents.clients import resolve_llms

    return resolve_llms(
        state.get("provider", "auto"),
        user_credentials=state.get("user_credentials") or {},
    )


def _format_history(state: dict, max_messages: int = MAX_HISTORY_MESSAGES) -> list:
    """Conversation history as LangChain messages (oldest last, trimmed).

    Older messages beyond the verbatim window are represented by the rolling
    ``history_summary`` (a compact system note) so long chats keep their thread
    without blowing the prompt budget.
    """
    msgs = state.get("messages") or []
    trimmed = list(msgs[-max_messages:])
    summary = (state.get("history_summary") or "").strip()
    if not summary or len(msgs) <= max_messages:
        return trimmed
    # Only inject the summary when it adds information (older messages exist).
    omitted = len(msgs) - len(trimmed)
    if omitted <= 0:
        return trimmed
    return [SystemMessage(content=f"Earlier conversation (summary of {omitted} older messages): {summary}"), *trimmed]


def _history_summary_text(state: dict) -> str:
    """Compact rolling summary of older history (beyond the verbatim window).

    Cheap extractive compaction: recent exchanges stay verbatim, older turns are
    reduced to their first sentence(s). Updated per turn and persisted via the
    evidence-state blob so the next turn inherits it.
    """
    msgs = state.get("messages") or []
    older = msgs[:-MAX_HISTORY_MESSAGES] if len(msgs) > MAX_HISTORY_MESSAGES else []
    if not older:
        return (state.get("history_summary") or "").strip()
    lines = []
    for msg in older:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        content = re.sub(r"\s+", " ", (getattr(msg, "content", "") or "")).strip()
        if not content:
            continue
        # Keep the opening sentences of each older message.
        sentences = re.split(r"(?<=[.!?])\s+", content)
        snippet = " ".join(sentences[:2])[:300]
        lines.append(f"{role}: {snippet}")
    return "\n".join(lines[-12:])  # cap the summary itself


def _decompose_queries(query: str, u: QueryUnderstanding | None) -> list[str]:
    """Structural query variants for the sufficiency retry — no word tables.

    Retrieval failed on the current phrasing; the cheapest domain-agnostic
    alternative angles are the ones the LLM planner already produced (the
    rewritten query) plus mechanical clause splits of the original (conjunction
    and punctuation boundaries). If none of those differ from what was already
    searched, the caller skips the retry.
    """
    base = (query or "").strip()
    if not base:
        return []
    candidates: list[str] = []
    rewritten = (u.rewritten_query or "").strip() if u else ""
    if rewritten:
        candidates.append(rewritten)
    for part in re.split(r"(?:\b(?:and|or|vs\.?|versus)\b|[;,.?!—–-])", base):
        p = re.sub(r"\s+", " ", part).strip(" ?.")
        if len(p) >= 8 and p.lower() != base.lower():
            candidates.append(p)
    out: list[str] = []
    seen: set[str] = {base.lower()}
    for c in candidates[:MAX_SEARCH_QUERIES]:
        k = c.lower()
        if k not in seen:
            seen.add(k)
            out.append(c)
    return out


def _temporal_context(state: dict) -> str:
    """Current date/time (+ user locale when provided) for temporal awareness."""
    ctx = get_current_context(
        timezone_str=(state.get("request_context") or {}).get("timezone"),
        user_location=(state.get("request_context") or {}).get("location"),
        device_info=(state.get("request_context") or {}).get("device"),
    )
    return format_context_for_llm(ctx)


def _prior_evidence_block(state: dict) -> str:
    """Compact summary of verified facts carried from prior turns."""
    prior = state.get("prior_evidence_state")
    if not prior or prior.is_empty():
        return ""
    lines = []
    for ev in prior.all_evidence()[:6]:
        src = ev.source_name or ev.source_url or ev.evidence_id
        lines.append(f"- {ev.text[:200]} [{src}]")
    if prior.unresolved:
        lines.append("- Unresolved from earlier: " + "; ".join(prior.unresolved[:3]))
    return "\n".join(lines)


def _prior_evidence_line(state: dict) -> str:
    """Established-facts block for the generate/revise prompts (empty when none)."""
    block = _prior_evidence_block(state)
    if not block:
        return ""
    return (
        "Established in this conversation (verified in earlier turns — prefer "
        "these over re-searching):\n" + block
    )


def _sanitize_evidence_text(text: str) -> str:
    """Strip forged citation tokens so injected fake [E#] markers can't game the verifier."""
    return re.sub(r"\[[Ee]\d{1,3}\]", "", text or "")


def _parse_date(text: str | None) -> datetime | None:
    if not text:
        return None
    m = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", str(text))
    if m:
        try:
            return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)), tzinfo=timezone.utc)
        except ValueError:
            return None
    m = re.search(r"\b(20\d{2})\b", str(text))
    if m:
        return datetime(int(m.group(1)), 1, 1, tzinfo=timezone.utc)
    return None

_TEMPORAL_AUTHORITY_HOSTS = (
    "wikipedia.org", "britannica.com", "reuters.com", "apnews.com", "bbc.com",
)


def _evidence_sort_key(ev: Evidence, *, temporal: bool = False) -> float:
    """Rank by rerank score when available, else retrieval score. Recency breaks ties.

    With ``temporal=True`` (current-events question), stable event/encyclopedic
    sources get a small tie-break nudge — live-news feeds return tangential
    articles that outrank the actual event page on raw keyword overlap.
    """
    base = ev.rerank_score if ev.rerank_score is not None else ev.retrieval_score
    if ev.source_date:
        age_days = max((utc_now() - ev.source_date).days, 0)
        base += min(age_days, 3650) / 365000  # tiny, bounded nudge toward fresher sources
    if temporal:
        host = (urlparse(ev.source_url or "").netloc or "").lower()
        if any(host == h or host.endswith("." + h) for h in _TEMPORAL_AUTHORITY_HOSTS):
            base += 0.02
    return float(base or 0.0)


# ── Deterministic calculator (planned expressions, sandboxed evaluation) ────

_EXPR_LIMIT = 200
import math as _math

_ALLOWED_FUNCS = {
    "sqrt": _math.sqrt, "log": _math.log, "log10": _math.log10, "exp": _math.exp,
    "abs": abs, "round": round, "min": min, "max": max, "pow": pow,
}
_ALLOWED_CONSTS = {"pi": _math.pi, "e": _math.e}


def _needs_computation(query: str) -> bool:
    """Structural trigger: does the question plausibly involve arithmetic?

    Two or more numbers, or any '=' (the user shared worked math). No domain
    word lists — a numeric question in any field qualifies.
    """
    q = query or ""
    return len(re.findall(r"\d+(?:\.\d+)?", q)) >= 2 or "=" in q


def _safe_eval(expression: str) -> float:
    """Evaluate a pure-arithmetic expression. AST-whitelisted, no builtins.

    Allows numbers, + - * / ** % //, unary minus, parentheses, and a small
    math namespace (sqrt/log/exp/abs/round/min/max/pow, pi/e). Everything
    else — names, attributes, strings, calls to anything unlisted — is
    rejected before execution.
    """
    import ast

    expr = (expression or "").strip().rstrip(";").strip()
    if not expr or len(expr) > _EXPR_LIMIT:
        raise ValueError("expression empty or too long")
    tree = ast.parse(expr, mode="eval")
    ok_ops = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.FloorDiv)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Expression, ast.BinOp, ast.UnaryOp, *ok_ops,
                             ast.USub, ast.UAdd, ast.Load)):
            continue
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
                raise ValueError("only numeric constants allowed")
            continue
        if isinstance(node, ast.Name):
            if node.id not in _ALLOWED_FUNCS and node.id not in _ALLOWED_CONSTS:
                raise ValueError(f"name {node.id!r} not allowed")
            continue
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCS:
                raise ValueError("call to non-whitelisted function")
            if node.keywords:
                raise ValueError("keyword arguments not allowed")
            continue
        raise ValueError(f"syntax {type(node).__name__} not allowed")
    namespace = {**_ALLOWED_CONSTS, **_ALLOWED_FUNCS}
    result = float(eval(compile(tree, "<calculator>", "eval"), {"__builtins__": {}}, namespace))
    if not _math.isfinite(result):
        raise ValueError("non-finite result")
    return result


def _format_number(value: float) -> str:
    """Full-precision readable rendering (cents matter; float noise doesn't)."""
    return f"{value:.10f}".rstrip("0").rstrip(".") or "0"


def _plan_computations(state: dict, llms: Any, context: str) -> list[dict]:
    """One small structured call: the LLM plans expressions, code computes them.

    Returns a list of {label, expression, result}; empty when the planner sees
    nothing to compute or anything fails (never fatal to the turn).
    """
    question = state.get("query_original") or state.get("query", "")
    messages = [
        SystemMessage(content=_COMPUTE_PROMPT.format(question=question, evidence=context[:3000])),
        HumanMessage(content="Plan the computations now. Respond with the JSON plan."),
    ]
    try:
        plan: ComputationPlan = _llm_with_fallback(
            llms.planner, llms.planner_fallbacks, messages, ComputationPlan,
            role="compute-planner", timeout=settings.PLANNER_TIMEOUT_SECONDS,
        )
    except Exception as exc:
        logger.warning("compute planner failed (%s); continuing without calculations", exc)
        return []
    out: list[dict] = []
    for c in plan.computations[: settings.COMPUTE_MAX_EXPRESSIONS]:
        expr = (c.expression or "").strip()
        if not expr:
            continue
        try:
            out.append({"label": (c.label or "value").strip()[:80],
                        "expression": expr, "result": _safe_eval(expr)})
        except Exception as exc:
            logger.warning("calc rejected %r: %s", expr[:60], exc)
    return out


def _render_computations(computations: list[dict]) -> str:
    if not computations:
        return ""
    lines = ["Deterministic computations (evaluated exactly — use these values as-is; never re-derive or contradict them):"]
    for i, c in enumerate(computations, start=1):
        lines.append(f"[C{i}] {c['label']}: {c['expression']} = {_format_number(c['result'])}")
    return "\n".join(lines)


_COMPUTE_PROMPT = """You are a computation planner. The user asked a question that involves arithmetic.

User's question (may include their own worked math):
{question}

Supporting evidence (numbers you may use):
{evidence}

List the pure arithmetic expressions (Python syntax) needed to derive or verify the numeric answer.
Rules:
- Only numeric expressions: numbers, + - * / ** % //, parentheses, sqrt/log/exp/abs/round/min/max.
- One expression per item, with a short label (e.g. "periodic rate", "final amount", "interest earned").
- Include each intermediate step the final figure depends on.
- Use ONLY numbers present in the question or the evidence — never invent values.
- If no arithmetic is needed, return an empty list.
"""

_CLASSIFY_PROMPT = """You are the planning module of a research assistant. Analyze the user's message and produce a JSON plan.

Current date and time: {temporal_context}

Documents the user has uploaded into this chat (may be empty):
{document_inventory}

Conversation so far (may be empty):
{conversation}

Verified facts carried over from earlier turns (may be empty):
{prior_evidence}

User message:
{query}

Decide:
1. mode:
   - "conversational": greetings, thanks, small talk, or questions about the assistant itself. No facts need looking up.
   - "clarification": the message is genuinely ambiguous — one term could mean two or more clearly different things and you cannot pick. Write one short question listing the distinct meanings. Unknown acronyms, unfamiliar names, news, and current-event questions are NOT ambiguity — research them instead; search will resolve them.
   - "research": anything that needs evidence to answer.
2. rewritten_query: the message rewritten as a fully standalone search query. Resolve pronouns and implicit references against the conversation ("what about the growth?" -> "growth of <topic from history>"). Keep the user's language and intent; add the time frame only if the conversation implies one.
3. needs_documents: true if the user's own uploaded documents might be relevant.
4. needs_web: true if public/web knowledge is needed. Default true for factual questions about the world.
5. search_queries: 1-3 precise web/document search queries derived from rewritten_query. Each query must be self-contained.
   If temporal_focus is set, make at least one query a bare encyclopedic topic phrase (no question words), e.g.
   "who won the most recent World Cup final" -> "2026 FIFA World Cup final" — event/encyclopedia pages answer these.
6. temporal_focus: the time frame the question is about ("latest", "2023", "Q1 2025", ...). Empty string if no time dimension.
7. geography: the place the question is about, exactly as the user framed it. Empty string if none.

Rules:
- The current date matters: if the user asks for "latest"/"current"/"today", set temporal_focus accordingly and prefer fresh sources.
- Documents listed above are IN this chat. If the user refers to "this paper", "the document", "what I gave you", or names/misspells any of them (e.g. "the zk pfl paper" ≈ "ZK-PFL Paper.pdf"), treat it as RESEARCH about that document: set needs_documents=true, needs_web=false unless the topic also needs public facts, and rewrite the query around the document's title/topic.
- NEVER choose "clarification" to ask WHICH document or paper the user means when the answer is in the uploaded-documents list above — just use it.
- Referential follow-ups are NOT ambiguity. When the conversation already established the topic and the user refines it — "its equations", "all of them", "more detail", even just "yes" confirming your last suggestion — choose RESEARCH and resolve the reference against the conversation. Clarification is ONLY for terms with two or more distinct real-world referents that you genuinely cannot pick between.
- Never invent entities not present in the conversation, the message, or the uploaded documents list.
"""


_GENERATE_PROMPT = """You are a precise research assistant. Answer the user's question using ONLY the evidence below.

Current date and time: {temporal_context}
{temporal_focus_line}
{established_line}
{computed_block}

Documents the user uploaded into this chat (may be empty): {document_inventory}

Evidence (cite with the bracketed keys):
{context}

Rules:
1. EVERY sentence containing a number, date, amount, or named entity MUST end with a citation key from the evidence, e.g. ... [E3]. An uncited factual sentence will be flagged as unverified and shown to the user as a caveat.
2. If the evidence is insufficient, say so plainly — never fabricate facts or citations.
3. Respect time: if the question asks for latest/current figures, prefer the most recent evidence and say which period each figure covers.
4. If two evidence items conflict, report both with citations instead of picking silently.
5. When evidence comes from one of the user's uploaded documents, weave the document's name into the prose naturally (e.g. "According to the uploaded 'ZK-PFL Paper.pdf', ..."). Never refer to uploaded documents as "the evidence" or "the provided context" — name them.
6. Match the user's language. Be direct and concise; structure with short paragraphs or bullets when it helps.
7. Facts listed under "Established in this conversation" were verified in earlier turns — you may rely on them, but cite them when they carry the answer.
8. Correction integrity: if the user claims you previously said something or made an error, CHECK the conversation above first. Quote what you actually said. If their characterization is wrong, say so plainly and correct THEM — never apologize for an error you did not make, and never manufacture a "prior mistake" narrative. If their pushback raises a genuinely new technical claim, evaluate it on its merits — confidence and vocabulary are not evidence; concede only if it holds on inspection. When they are right about a part but wrong about the whole, say exactly which part is which — no blanket "I was wrong" framing. When the user's reply follows your own clarifying question, treat it as sufficient: state your working assumption explicitly and answer — never re-ask.
9. No unconditional guarantees: any "guarantee", "never", or "always" claim that depends on an external system must state its condition in the same sentence (e.g. "guaranteed X as long as Y honors Z; otherwise only detection and reconciliation are possible").
"""

_REVISE_PROMPT = """You previously drafted an answer, and a strict verifier found problems with it.

User's question: {question}

Your previous draft:
{draft}

Verifier's critique of the draft:
{critique}

Current date and time: {temporal_context}
{temporal_focus_line}
{established_line}
{computed_block}

Documents the user uploaded into this chat (may be empty): {document_inventory}

Evidence (cite with the bracketed keys; blocks marked ★ are new evidence gathered to fix the flagged problems):
{context}

Rewrite the answer now:
1. Fix or remove every claim the critique flagged as unsupported, contradicted, or unverified. If new evidence (★) resolves a flagged gap, use it and cite it.
2. Keep what the critique did not flag — do not regress correct content.
3. Every sentence with a number, date, amount, or named entity MUST end with a citation key from the evidence, e.g. ... [E3].
4. If a flagged gap remains unresolvable even with the new evidence, drop the claim or state the uncertainty plainly — never keep an unsupported assertion.
5. Match the user's language. Be direct and concise.
6. If the critique flags a claim because the user challenged it, verify the challenge against the evidence before conceding: quote what the draft actually said, and only change what is actually wrong — never fabricate a concession.
"""


_VERIFY_PROMPT = """You are a strict fact verifier. Given a question, an answer, and the evidence it cites, check every factual assertion.

User's question: {question}
Current date: {temporal_context}

{computed_block}

Evidence:
{context}

Answer to verify:
{answer}

First, judge relevance to the QUESTION above (not just internal consistency):
- addresses_question: true only if the answer actually responds to what was asked. A fluent, well-cited answer about the wrong entity, wrong period, or a tangential aspect is a failure.
- question_gaps: aspects of the question the answer left unanswered.

Then check entity ambiguity:
- referent_ambiguity: true only when the QUESTION names a specific person/organization/acronym AND the evidence describes SEVERAL clearly different real-world referents sharing that name — so the answer would materially differ depending on which one the user meant.
- referents: short identities for each distinct referent found in the evidence (e.g. "Atharva Bhide — investment advisor at MoneyWorks4Me", "Atharva Bhide — physics PhD student at Max Planck Institute"). Empty unless referent_ambiguity is true.
- A conflict over a detail (a credential, a number, which source is fresher) is NOT referent ambiguity. Never set referent_ambiguity for that.

Then, for each factual assertion in the answer (skip greetings, transitions, hedged statements, and questions the answer asks the user). List AT MOST 8 assertions — the central factual claims the answer rests on, not every sentence; the mechanical validator already checks every cited sentence:
- text: the assertion, quoted briefly
- status: "verified" (cited evidence genuinely supports it), "contradicted" (evidence says otherwise), "unverified" (no cited evidence supports it), "uncertain" (evidence is ambiguous, partial, or conflicting)
- evidence_ids: the cite keys (E1, E2, ...) that support or contradict it
- reasoning: one sentence

Also set:
- overall: "supported" if every assertion is verified and the question is addressed, "partial" if some are uncertain/unverified, "unsupported" if any assertion is contradicted or central claims are unverified
- repair_queries: 1-3 web search queries that could fix the gaps — ONLY when gaps exist and are the kind a targeted search could fill. Empty list otherwise.
- clarification_question: only for genuine referent ambiguity — ONE short question listing the interpretations found, e.g. "By CJP do you mean the Chief Justice of Pakistan or the Centre for Justice Policy?". Otherwise empty string.
- explanation: one sentence summary of answer quality

Be strict about numbers, dates, names, and causal claims. Do not fail an assertion merely for informal phrasing.
Numeric assertions that match the deterministic computations block are VERIFIED by definition — arithmetic has already been evaluated exactly. If the answer's figures contradict that block, mark them contradicted.
"""
_CONVERSATIONAL_PROMPT = """You are a friendly research assistant. The user's message is conversational — no lookup is needed.

Current date and time: {temporal_context}

Conversation so far:
{conversation}

Reply naturally in the user's language. If they greet you, greet back and briefly say what you can help with (answering questions from their documents and the web with cited, verified answers). Keep it to 1-3 sentences. If they ask what you are or what you can do, answer honestly and concisely.
"""


# ── Node: classify_and_plan ──────────────────────────────────────────────────


def classify_and_plan(state: dict) -> dict:
    """Classify intent and build a retrieval plan in one structured LLM call.

    Chat-aware (resolves followups against history) and temporally aware
    (current date injected; temporal_focus extracted).
    """
    t0 = time.perf_counter()
    tracing.set_trace_context(node="classify_and_plan", chat_id=state.get("chat_id"), user_id=state.get("user_id"))
    tracing.set_trace_context(node="conversational_response", chat_id=state.get("chat_id"), user_id=state.get("user_id"))
    query = state.get("query_original") or state.get("query", "")
    temporal_context = _temporal_context(state)

    from langchain_core.messages import AIMessage

    convo_lines = []
    for msg in _format_history(state):
        role = "user" if isinstance(msg, HumanMessage) else ("assistant" if isinstance(msg, AIMessage) else "system")
        content = (getattr(msg, "content", "") or "").strip()
        if content:
            convo_lines.append(f"{role}: {content[:500]}")

    messages = [
        SystemMessage(content=_CLASSIFY_PROMPT.format(
            temporal_context=temporal_context,
            document_inventory="\n".join(f"- {name}" for name in state.get("document_inventory") or []) or "(no documents uploaded)",
            conversation="\n".join(convo_lines) or "(no prior conversation)",
            prior_evidence=_prior_evidence_block(state) or "(none)",
            query=query,
        )),
        HumanMessage(content=query),
    ]

    llms = _resolve_llms(state)
    used = llms.label
    try:
        u: QueryUnderstanding = _llm_with_fallback(
            llms.planner, llms.planner_fallbacks, messages, QueryUnderstanding, role="planner",
            timeout=settings.PLANNER_TIMEOUT_SECONDS,
            max_attempts=2,
        )
    except Exception as exc:
        # Honest safe default: treat as research, search everything with the raw query.
        logger.warning("classify_and_plan LLM failed (%s); defaulting to research mode", exc)
        u = QueryUnderstanding(
            mode=QueryMode.RESEARCH,
            rewritten_query=query,
            needs_documents=True,
            needs_web=True,
            search_queries=[query],
        )
        used = f"{llms.label}+fallback-default"
    # Sanitize planner output — never let a malformed plan starve the pipeline.
    if u.mode == QueryMode.CLARIFICATION and not u.clarification_question.strip():
        # Planner asked for clarification but wrote no question: treat as research.
        logger.info("Planner chose clarification without a question; coercing to research")
        u.mode = QueryMode.RESEARCH
        u.needs_web = True
    if u.mode == QueryMode.CLARIFICATION:
        # Mechanical guard: clarification must be GROUNDED (RAC principle) —
        # a planner that has not searched cannot know a fresh name is ambiguous,
        # and a referent the conversation already establishes must never be
        # asked about. Coerce to research when:
        #   (a) there is no prior conversation at all — with nothing visible to
        #       disambiguate between, "who is X" is a lookup, not an ambiguity;
        #       the verify stage enumerates real referents from evidence instead;
        #   (b) the user's message or ANY prior turn references an uploaded
        #       document (by any distinctive filename token);
        #   (c) a referential follow-up ("its equations", "the moneyworks4me
        #       one") after a substantive answer already exists.
        inventory = [n.lower() for n in (state.get("document_inventory") or [])]
        # NOTE: full window — clarification ping-pong can push the doc mention
        # out of a short tail window.
        recent_text = query.lower() + "\n" + "\n".join(convo_lines).lower()
        stems = set()
        for name in inventory:
            # Match on distinctive tokens of the filename (strip extension),
            # skipping generic words like "paper", "the", "final".
            generic = {"paper", "the", "and", "for", "with", "final", "draft", "v2", "doc", "pdf"}
            stems.update(w for w in name.replace(".", " ").replace("-", " ").replace("_", " ").split() if len(w) >= 3 and w not in generic)
        doc_hit = bool(stems) and any(s in recent_text for s in stems)

        reference_words = {"it", "its", "this", "that", "them", "they", "those", "these", "paper", "document", "pdf", "file", "same", "above", "one", "either"}
        query_words = set(re.findall(r"[a-z]+", query.lower()))
        has_substantive_answer = any(
            isinstance(m, AIMessage) and len(getattr(m, "content", "") or "") >= 200
            for m in _format_history(state)
        )
        referent_hit = bool(query_words & reference_words) and has_substantive_answer
        no_visible_referents = not convo_lines  # nothing grounded to disambiguate
        # Rule: clarification never loops. One round was already offered if the
        # previous turn ended with needs_clarification — this turn must proceed
        # on the most reasonable assumption, stated explicitly, never ask again.
        clarification_already_offered = (
            (state.get("prior_evidence_state") or EvidenceState()).last_final_status
            == "needs_clarification"
        )

        if (
            clarification_already_offered
            or no_visible_referents
            or doc_hit
            or referent_hit
        ):
            logger.info(
                "Coercing clarification → research (clarification_already_offered=%s "
                "no_visible_referents=%s doc_hit=%s referent_hit=%s inventory=%s)",
                clarification_already_offered, no_visible_referents, doc_hit, referent_hit, inventory,
            )
            u.mode = QueryMode.RESEARCH
            u.needs_documents = True
            u.clarification_question = ""
            if no_visible_referents:
                # Fresh lookup: ambiguity is unknowable without evidence —
                # search the world, then enumerate real referents at verify.
                u.needs_web = True
            # Anchor retrieval: the user's ask, plus the referenced document
            # itself so chunks from the right file rank first.
            u.search_queries = [u.rewritten_query or query]
            if doc_hit:
                doc_name = next((n for n in inventory if any(s in n for s in stems)), "")
                if doc_name:
                    u.search_queries.append(doc_name)
            u.search_queries = [q for q in u.search_queries if q.strip()][:3]
    if u.mode == QueryMode.RESEARCH:
        if not u.needs_documents and not u.needs_web:
            u.needs_web = True  # research must look somewhere
        if not any(q.strip() for q in u.search_queries):
            u.search_queries = [u.rewritten_query or query]

    logger.info(
        "Node classify_and_plan completed in %.0fms mode=%s docs=%s web=%s queries=%s",
        (time.perf_counter() - t0) * 1000, u.mode.value, u.needs_documents, u.needs_web, u.search_queries,
    )
    return {
        "understanding": u,
        "query": u.rewritten_query or query,
        "provider_used": used,
        "graph_steps": state.get("graph_steps", 0) + 1,
    }


def route_after_classify(state: dict) -> str:
    u = state.get("understanding")
    if u is None:
        return "gather_evidence"
    if u.mode == QueryMode.CONVERSATIONAL:
        return "conversational_response"
    if u.mode == QueryMode.CLARIFICATION and u.clarification_question.strip():
        return "ask_clarification"
    return "gather_evidence"


# ── Node: conversational short-circuits ──────────────────────────────────────


def conversational_response(state: dict) -> dict:
    """Reply to small talk / meta questions directly (one small LLM call)."""
    tracing.set_trace_context(node="conversational_response", chat_id=state.get("chat_id"), user_id=state.get("user_id"))
    query = state.get("query_original") or state.get("query", "")
    temporal_context = _temporal_context(state)

    from langchain_core.messages import AIMessage

    convo_lines = []
    for msg in _format_history(state):
        role = "user" if isinstance(msg, HumanMessage) else ("assistant" if isinstance(msg, AIMessage) else "system")
        content = (getattr(msg, "content", "") or "").strip()
        if content:
            convo_lines.append(f"{role}: {content[:500]}")

    messages = [
        SystemMessage(content=_CONVERSATIONAL_PROMPT.format(
            temporal_context=temporal_context,
            conversation="\n".join(convo_lines) or "(no prior conversation)",
        )),
        HumanMessage(content=query),
    ]
    llms = _resolve_llms(state)
    # Greetings/meta don't need a research-grade model — use the light one when
    # configured, with the full generator as first fallback.
    light = getattr(llms, "light", None)
    if light is not None:
        primary, fallbacks = light, (llms.generator, *tuple(llms.generator_fallbacks))
    else:
        primary, fallbacks = llms.generator, llms.generator_fallbacks
    try:
        answer, suffix = _invoke_chat(primary, fallbacks, messages, role="conversational")
        used = f"{llms.label}+{suffix}"
    except Exception as exc:
        logger.warning("conversational_response LLM failed: %s", exc)
        answer = "Hello! Ask me a question and I'll research it with cited sources."
        used = llms.label
    return {
        "answer": answer,
        "final_status": "conversational",
        "provider_used": used,
        "graph_steps": state.get("graph_steps", 0) + 1,
    }


def ask_clarification(state: dict) -> dict:
    """Ask the user which meaning they intend (question comes from the planner LLM)."""
    u = state.get("understanding")
    question = (u.clarification_question if u else "").strip() or (
        "Could you clarify what you'd like to know?"
    )
    return {
        "answer": question,
        "final_status": "needs_clarification",
        "graph_steps": state.get("graph_steps", 0) + 1,
    }


# ── Node: gather_evidence ────────────────────────────────────────────────────


async def _retrieve_documents(queries: list[str], state: dict) -> list[Evidence]:
    """Vector + BM25 retrieval over the user's knowledge base."""
    from app.documents.service import retrieve_chunks

    user_id = state["user_id"]
    chat_id = state["chat_id"]
    try:
        results = await asyncio.gather(*[
            retrieve_chunks(q, user_id=user_id, top_k=30, scope="chat", chat_id=chat_id)
            for q in queries[:MAX_SEARCH_QUERIES]
        ])
    except Exception as exc:
        logger.exception("Document retrieval failed: %s", exc)
        return []

    evidence: list[Evidence] = []
    for query_result in results:
        for ch in query_result:
            text = (ch.get("text") or "").strip()
            if not text:
                continue
            dist = ch.get("distance")
            score = 0.5 if dist is None else 1.0 - min(1.0, max(0.0, float(dist)))
            meta = dict(ch.get("metadata") or {})
            evidence.append(Evidence(
                text=text,
                source_type=SourceType.DOCUMENT,
                source_name=meta.get("source") or meta.get("filename") or "document",
                source_date=_parse_date(meta.get("date") or meta.get("created_at")),
                retrieval_score=float(score),
                metadata=meta,
            ))
    return evidence


async def _search_web(queries: list[str], state: dict) -> list[Evidence]:
    """Parallel web search (SearXNG / Wikipedia / Tavily) with daily Tavily budget."""
    from app.agent.search_tool import search_structured

    user_id = state.get("user_id")
    allow_tavily = True
    if user_id is not None:
        try:
            from app.core.database import AsyncLocalSession
            from app.core.usage import enforce_tavily_budget

            async with AsyncLocalSession() as session:
                await enforce_tavily_budget(session, user_id)
        except Exception as exc:
            from fastapi import HTTPException

            if isinstance(exc, HTTPException) and exc.status_code == 429:
                logger.warning("Tavily daily budget exhausted for user=%s; SearXNG only", user_id)
                allow_tavily = False
            else:
                logger.exception("Tavily budget check failed")

    async def _one(q: str) -> list[dict]:
        try:
            return await search_structured(q, max_results=6, user_id=user_id, allow_tavily=allow_tavily)
        except Exception as exc:
            logger.warning("Web search failed for query %r: %s", q, exc)
            return []

    results_lists = await asyncio.gather(*[_one(q) for q in queries[:MAX_SEARCH_QUERIES]])

    evidence: list[Evidence] = []
    seen_urls: set[str] = set()
    for results in results_lists:
        for r in results:
            text = (r.get("content") or "").strip()
            if not text:
                continue
            url = r.get("url")
            if url and url in seen_urls:
                continue
            if url:
                seen_urls.add(url)
            evidence.append(Evidence(
                text=text,
                source_type=SourceType.WEB,
                source_name=r.get("title") or r.get("source") or urlparse(url or "").netloc,
                source_url=url,
                source_date=_parse_date(r.get("published_date") or r.get("date")),
                retrieval_score=float(r.get("score", 0.5)),
                metadata=dict(r),
            ))
    return evidence


async def _attach_document_urls(doc_evidence: list[Evidence], state: dict) -> None:
    """Set source_url on DOCUMENT evidence pointing at the stored original.

    URLs are relative signed paths — the frontend prefixes its API base so
    [E#] markers and source cards become real links the user can open.
    Evidence whose file wasn't persisted (legacy uploads) is left untouched.
    """
    from sqlalchemy.future import select as _select
    from app.core.database import AsyncLocalSession
    from app.documents.models import IngestionLog as _IngestionLog
    from app.documents.signing import signed_file_path

    chat_id = state.get("chat_id")
    if chat_id is None:
        return
    # Prefer the graph's session factory (test-overridable); fall back to the
    # app-level factory in production paths.
    session_factory = state.get("session_factory") or AsyncLocalSession
    async with session_factory() as db:
        rows = (
            await db.execute(
                _select(_IngestionLog).where(
                    _IngestionLog.chat_id == chat_id,
                    _IngestionLog.storage_path.isnot(None),
                )
            )
        ).scalars().all()
    if not rows:
        return

    def _norm(name: str) -> str:
        return re.sub(r"\s+", " ", (name or "")).strip().lower()

    by_name: dict[str, str] = {}
    for row in rows:
        by_name[_norm(row.filename)] = str(row.id)
        stem = _norm(Path(row.filename).stem)
        by_name.setdefault(stem, str(row.id))

    for ev in doc_evidence:
        doc_id = by_name.get(_norm(ev.source_name)) or by_name.get(_norm(Path(ev.source_name).stem))
        if doc_id:
            ev.source_url = signed_file_path(doc_id)
            ev.metadata = {**(ev.metadata or {}), "document_id": doc_id}


async def gather_evidence(state: dict) -> dict:
    """Run document retrieval and web search in parallel, rerank, assign cite keys.

    On the repair pass, uses the judge's repair_queries instead of the plan, and
    reranks the new evidence against those queries (not the original query) so
    gap-filling evidence actually makes the context. Repair-pass evidence gets
    reserved context slots so the old pool cannot crowd it out.
    """
    t0 = time.perf_counter()
    u = state.get("understanding") or QueryUnderstanding(
        rewritten_query=state.get("query_original") or state.get("query", ""),
        needs_documents=True, needs_web=True,
    )
    repair_mode = bool(state.get("repair_queries"))
    repair_queries = [q for q in (state.get("repair_queries") or []) if q.strip()][:MAX_SEARCH_QUERIES]
    if repair_mode:
        queries = repair_queries
        needs_web, needs_documents = True, u.needs_documents
    else:
        queries = [q for q in (u.search_queries or [u.rewritten_query]) if q.strip()][:MAX_SEARCH_QUERIES]
        needs_web, needs_documents = u.needs_web, u.needs_documents
    if not queries:
        queries = [state.get("query_original") or state.get("query", "")]

    tasks: dict[str, asyncio.Task] = {}
    if needs_documents and state.get("retrieval_count", 0) < settings.MAX_RETRIEVALS:
        tasks["docs"] = asyncio.ensure_future(_retrieve_documents(queries, state))
    if needs_web and state.get("search_count", 0) < settings.MAX_SEARCHES:
        tasks["web"] = asyncio.ensure_future(_search_web(queries, state))

    evidence: list[Evidence] = list(state.get("evidence") or [])
    new_evidence: list[Evidence] = []
    if tasks:
        done = await asyncio.gather(*tasks.values(), return_exceptions=True)
        for name, result in zip(tasks.keys(), done):
            if isinstance(result, Exception):
                logger.exception("%s failed: %s", name, result)
                continue
            evidence.extend(result)
            new_evidence.extend(result)

    # Hyperlink document citations back to their stored originals: map each
    # DOCUMENT evidence's source_name (chunk metadata carries the upload
    # filename) to a persisted ingestion and attach a signed file URL.
    doc_evs = [ev for ev in evidence if ev.source_type == SourceType.DOCUMENT and not ev.source_url]
    if doc_evs:
        try:
            await _attach_document_urls(doc_evs, state)
        except Exception:  # noqa: BLE001
            logger.exception("document source-url attachment failed; continuing without links")

    query = state.get("query") or u.rewritten_query
    # Prior-turn verified facts are PINNED, not reranked: they were already
    # verified and must not compete with (and lose to) fresh retrieval. But a
    # topic switch must not drag stale facts into context — facts that are not
    # at least minimally relevant to THIS query are dropped here.
    prior = state.get("prior_evidence_state")
    pinned: list[Evidence] = []
    if prior and not repair_mode:
        pinned_candidates = list(prior.all_evidence())
        if pinned_candidates:
            ranked_pinned = await rerank(query, [
                ("chunk", ev.text) for ev in pinned_candidates
            ], top_k=len(pinned_candidates))
            score_by_text = {r.text: r.score for r in ranked_pinned}
            floor = settings.ESTABLISHED_MIN_SCORE
            for ev in pinned_candidates:
                s = score_by_text.get(ev.text)
                if s is None or float(s) >= floor:
                    pinned.append(ev)
            dropped = len(pinned_candidates) - len(pinned)
            if dropped:
                logger.info("Dropped %d pinned fact(s) irrelevant to the current query", dropped)

    # Deduplicate by normalized text prefix (pinned facts win over fresh hits).
    seen_texts: set[str] = {re.sub(r"\s+", " ", ev.text[:300]).lower() for ev in pinned}
    unique: list[Evidence] = []
    for ev in evidence:
        key = re.sub(r"\s+", " ", ev.text[:300]).lower()
        if key in seen_texts:
            continue
        seen_texts.add(key)
        unique.append(ev)

    if unique:
        # Repair evidence must be ranked by how well it answers the GAP, not the
        # original question — otherwise it scores low and never reaches context.
        # The judge's first repair query is its most targeted statement of the gap.
        rank_query = (repair_queries[0] if repair_mode and repair_queries else query)
        ranked = await rerank(rank_query, [
            ("chunk" if ev.source_type == SourceType.DOCUMENT else "search", ev.text)
            for ev in unique
        ], top_k=len(unique))
        score_by_text = {r.text: r.score for r in ranked}
        for ev in unique:
            rs = score_by_text.get(ev.text)
            ev.rerank_score = float(rs) if rs is not None else None
        temporal = bool(getattr(u, "temporal_focus", ""))
        unique.sort(key=lambda ev: _evidence_sort_key(ev, temporal=temporal), reverse=True)

        # Repair pass: guarantee the newest gap-filling evidence context slots.
        new_ids = {id(ev) for ev in new_evidence}
        if repair_mode and new_evidence:
            new_ranked = [ev for ev in unique if id(ev) in new_ids][:REPAIR_RESERVED_SLOTS]
            old_ranked = [ev for ev in unique if id(ev) not in new_ids]
            selected = (new_ranked + old_ranked)[:MAX_EVIDENCE]
        else:
            selected = unique[:MAX_EVIDENCE]

        # Enforce the generator token budget in ranked order.
        budget_chars = GENERATOR_BUDGET.total_tokens * 4
        kept: list[Evidence] = []
        used_chars = 0
        for ev in selected:
            cost = min(len(ev.text), EVIDENCE_SNIPPET_CHARS) + 120
            if used_chars + cost > budget_chars and kept:
                break
            kept.append(ev)
            used_chars += cost
        selected = kept

        # Sufficiency check (CRAG): is the best evidence good enough to answer
        # from at all? A weak, thin pool means retrieval failed — retry search
        # with decomposed queries BEFORE wasting a generate+verify cycle.
        top_score = max((float(ev.rerank_score or 0.0) for ev in selected), default=0.0)
        if (
            settings.SUFFICIENCY_TOP_SCORE > 0
            and not repair_mode
            and state.get("sufficiency_retry_used", False) is False
            and state.get("search_count", 0) < settings.MAX_SEARCHES
            and selected
            and top_score < settings.SUFFICIENCY_TOP_SCORE
            and len(selected) < settings.SUFFICIENCY_MIN_EVIDENCE * 3
        ):
            logger.warning(
                "Evidence pool insufficient (top=%.2f n=%d); retrying with decomposed queries",
                top_score, len(selected),
            )
            retry_state = {**state, "sufficiency_retry_used": True}
            retry_queries = _decompose_queries(query, u)
            if retry_queries and retry_queries != queries:
                tasks2: dict[str, asyncio.Task] = {}
                if needs_documents:
                    tasks2["docs"] = asyncio.ensure_future(_retrieve_documents(retry_queries, retry_state))
                if needs_web:
                    tasks2["web"] = asyncio.ensure_future(_search_web(retry_queries, retry_state))
                if tasks2:
                    done2 = await asyncio.gather(*tasks2.values(), return_exceptions=True)
                    retry_ev: list[Evidence] = []
                    for name, result in zip(tasks2.keys(), done2):
                        if isinstance(result, Exception):
                            logger.exception("%s (sufficiency retry) failed: %s", name, result)
                            continue
                        retry_ev.extend(result)
                    if retry_ev:
                        merged = unique + retry_ev
                        # Deduplicate the grown pool.
                        seen2: set[str] = set()
                        merged_unique: list[Evidence] = []
                        for ev in merged:
                            k2 = re.sub(r"\s+", " ", ev.text[:300]).lower()
                            if k2 in seen2:
                                continue
                            seen2.add(k2)
                            merged_unique.append(ev)
                        ranked2 = await rerank(query, [
                            ("chunk" if ev.source_type == SourceType.DOCUMENT else "search", ev.text)
                            for ev in merged_unique
                        ], top_k=len(merged_unique))
                        score2 = {r.text: r.score for r in ranked2}
                        for ev in merged_unique:
                            rs2 = score2.get(ev.text)
                            ev.rerank_score = float(rs2) if rs2 is not None else ev.rerank_score
                        merged_unique.sort(key=lambda ev: _evidence_sort_key(ev, temporal=temporal), reverse=True)
                        selected = merged_unique[:MAX_EVIDENCE]
                        kept2: list[Evidence] = []
                        used2 = 0
                        for ev in selected:
                            cost = min(len(ev.text), EVIDENCE_SNIPPET_CHARS) + 120
                            if used2 + cost > budget_chars and kept2:
                                break
                            kept2.append(ev)
                            used2 += cost
                        selected = kept2
                        unique = merged_unique
    else:
        selected = []

    # Pinned established facts lead the context, then the reranked pool. Their
    # cite keys are stable across turns (P1, P2, ...) so the generator can cite
    # them and the validator resolves them.
    cite_map: dict[str, str] = {}
    blocks: list[str] = []
    for i, ev in enumerate(pinned, start=1):
        key = f"P{i}"
        ev.metadata["cite_key"] = key
        cite_map[key] = ev.evidence_id
        src = "verified earlier in this conversation"
        if ev.source_name:
            src += f" — {ev.source_name}"
        if ev.source_url:
            src += f" ({ev.source_url})"
        blocks.append(f"[{key}] {src}\n{_sanitize_evidence_text(ev.text[:EVIDENCE_SNIPPET_CHARS])}")
    for i, ev in enumerate(selected, start=1):
        key = f"E{i}"
        ev.metadata["cite_key"] = key
        cite_map[key] = ev.evidence_id
        src = f"document: {ev.source_name}" if ev.source_type == SourceType.DOCUMENT else f"web: {ev.source_name}"
        if ev.source_url:
            src += f" ({ev.source_url})"
        if ev.source_date:
            src += f", published {ev.source_date.date().isoformat()}"
        blocks.append(f"[{key}] {src}\n{_sanitize_evidence_text(ev.text[:EVIDENCE_SNIPPET_CHARS])}")
    context = "\n\n".join(blocks)

    logger.info(
        "Node gather_evidence completed in %.0fms (queries=%s evidence=%d selected=%d pinned=%d repair=%s)",
        (time.perf_counter() - t0) * 1000, queries, len(unique), len(selected), len(pinned), repair_mode,
    )
    updates: dict = {
        "evidence": pinned + selected,
        "cite_map": cite_map,
        "assembled_context": context,
        "history_summary": _history_summary_text(state),
        "graph_steps": state.get("graph_steps", 0) + 1,
    }
    if needs_documents:
        updates["retrieval_count"] = state.get("retrieval_count", 0) + 1
    if needs_web:
        updates["search_count"] = state.get("search_count", 0) + 1
    return updates


# ── Node: generate_answer ────────────────────────────────────────────────────


def generate_answer(state: dict) -> dict:
    """Generate a cited answer from assembled evidence (chat history included).

    On a repair pass (critique present), regenerates from the judge's critique
    plus the previous draft instead of repeating the original prompt.
    """
    t0 = time.perf_counter()
    tracing.set_trace_context(node="generate_answer", chat_id=state.get("chat_id"), user_id=state.get("user_id"))
    context = state.get("assembled_context", "")
    u = state.get("understanding")
    critique = state.get("critique") or ""
    draft = state.get("draft_answer") or ""
    revising = bool(critique.strip() and draft.strip())

    if not context.strip():
        return {
            "answer": "I don't have enough reliable information to answer this question.",
            "claims": [],
            "verification_errors": [],
            "final_status": "answered_with_caveats",
            "graph_steps": state.get("graph_steps", 0) + 1,
        }

    temporal_focus = getattr(u, "temporal_focus", "") if u else ""
    temporal_focus_line = f'The user is asking about: "{temporal_focus}" (current date above). Prefer evidence matching that period; state the period of every figure.' if temporal_focus else ""
    established_line = _prior_evidence_line(state)

    # Deterministic calculator: on numeric questions the LLM plans pure
    # arithmetic expressions and code evaluates them exactly — the answer must
    # match these values instead of trusting model mental arithmetic.
    llms = _resolve_llms(state)
    computations = state.get("computations") or []
    if not computations and settings.COMPUTE_ENABLED and _needs_computation(
        state.get("query_original") or state.get("query", "")
    ):
        computations = _plan_computations(state, llms, context)
    computed_block = _render_computations(computations)

    if revising:
        system_prompt = _REVISE_PROMPT.format(
            question=state.get("query_original") or state.get("query", ""),
            draft=draft[:4000],
            critique=critique[:2500],
            temporal_context=_temporal_context(state),
            temporal_focus_line=temporal_focus_line,
            established_line=established_line,
            computed_block=computed_block,
            document_inventory=", ".join(state.get("document_inventory") or []) or "(none)",
            context=context,
        )
    else:
        system_prompt = _GENERATE_PROMPT.format(
            temporal_context=_temporal_context(state),
            temporal_focus_line=temporal_focus_line,
            established_line=established_line,
            computed_block=computed_block,
            document_inventory=", ".join(state.get("document_inventory") or []) or "(none)",
            context=context,
        )

    messages = [
        SystemMessage(content=system_prompt),
        *_format_history(state),
        HumanMessage(content=state.get("query_original") or state.get("query", "")),
    ]

    try:
        answer, suffix = _invoke_chat(llms.generator, llms.generator_fallbacks, messages)
        answer = answer.replace("【", "[").replace("】", "]")
        # Models sometimes emit markdown links around cite keys — "[E5](url)"
        # (with unbalanced parens when the URL itself has parentheses and the
        # model drops the closer). Cite keys must be bare tokens; the URL
        # belongs to the provenance panel. Pass 1 handles well-formed links
        # (including one nesting level in the URL); pass 2 strips the broken
        # leftover form.
        answer = re.sub(r"\[([EePp]\d{1,3})\]\((?:[^()]|\([^()]*\))*\)", r"[\1]", answer)
        answer = re.sub(r"\[([EePp]\d{1,3})\]\(\S*", r"[\1]", answer)
        used = f"{llms.label}+{suffix}"
    except Exception as exc:
        logger.warning("generate_answer LLM failed: %s", exc)
        # On a failed revise, fall back to the previous draft rather than losing
        # the answer entirely; the verifier has already stripped its bad markers.
        fallback = draft if revising and draft.strip() else (
            "I retrieved evidence but could not generate an answer (model error). Please try again."
        )
        return {
            "answer": fallback,
            "claims": [],
            "verification_errors": [],
            "final_status": "answered_with_caveats",
            "provider_used": llms.label,
        }

    evidence: list[Evidence] = state.get("evidence", [])
    citation_check: CitationValidationResult = validate_answer_citations(
        answer, evidence, cite_map=state.get("cite_map") or {}
    )
    answer = flag_uncited_in_answer(answer, citation_check)

    logger.info(
        "Node generate_answer completed in %.0fms (mode=%s uncited=%d invalid_cites=%d answer_chars=%d)",
        (time.perf_counter() - t0) * 1000,
        "revise" if revising else "generate",
        len(citation_check.uncited_sentences), len(citation_check.invalid_citation_ids), len(answer),
    )
    return {
        "answer": answer,
        "claims": citation_check.claims,
        "verification_errors": [e.to_dict() for e in citation_check.errors],
        "computations": computations,
        "provider_used": used,
        "graph_steps": state.get("graph_steps", 0) + 1,
    }


# ── Node: verify_answer ──────────────────────────────────────────────────────


def _verify_context(evidence: list[Evidence], cite_map: dict[str, str]) -> str:
    """Evidence listing for the judge prompt.

    Deliberately at the same snippet length the generator saw — a verifier that
    sees truncated evidence invents caveats about claims that were supported.
    """
    reverse = {v: k for k, v in (cite_map or {}).items()}
    blocks = []
    used = 0
    max_chars = settings.JUDGE_CONTEXT_CHARS
    for ev in evidence:
        key = reverse.get(ev.evidence_id, ev.metadata.get("cite_key", ev.evidence_id))
        src = ev.source_name or ev.source_url or "unknown"
        date = ev.source_date.date().isoformat() if ev.source_date else "no date"
        block = f"[{key}] {src} ({date}): {_sanitize_evidence_text(ev.text[:settings.JUDGE_SNIPPET_CHARS])}"
        if used + len(block) > max_chars:
            break
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks)


def _norm_claim_text(text: str) -> str:
    """Canonical form of a claim for fuzzy dedup (citations/punct dropped)."""
    t = CITATION_TOKEN_RE.sub("", text or "")
    return re.sub(r"\s+", " ", t).strip().lower().rstrip(" .,;:!?")


def _merge_claims(judge_claims: list[Claim], mechanical_claims: list[Claim]) -> list[Claim]:
    """Judge verdicts win; mechanical claims fill gaps and back the caveats.

    The judge quotes assertions "briefly", so exact-text matching duplicates
    everything (one verified + one unverified copy of the same sentence, both
    reaching the caveats). Match on normalized text with a similarity fallback.
    """
    from difflib import SequenceMatcher

    merged: list[Claim] = []
    norm_keys: list[str] = []
    for c in judge_claims:
        if _norm_claim_text(c.text):
            merged.append(c)
            norm_keys.append(_norm_claim_text(c.text))
    for c in mechanical_claims:
        norm = _norm_claim_text(c.text)
        if not norm:
            continue
        duplicate = False
        for existing in norm_keys:
            if norm == existing or SequenceMatcher(None, norm[:160], existing[:160]).ratio() >= 0.75:
                duplicate = True
                break
        if not duplicate:
            merged.append(c)
            norm_keys.append(norm)
    return merged


def _format_critique(verdict: Verdict, failed: list[Claim]) -> str:
    """Serialize the judge's findings into the revision prompt (Reflexion-style
    verbal feedback: what failed and why, so the rewrite fixes it)."""
    lines: list[str] = []
    for c in failed[:6]:
        text = CITATION_TOKEN_RE.sub("", c.text or "").strip()[:220]
        if not text:
            continue
        lines.append(f"- [{c.status.value}] {text}")
        if c.reasoning:
            lines.append(f"  why: {c.reasoning[:180]}")
    for gap in verdict.question_gaps[:3]:
        lines.append(f"- [question gap] the answer did not address: {gap[:200]}")
    if not lines and verdict.explanation:
        lines.append(f"- {verdict.explanation[:300]}")
    return "\n".join(lines) or "The verifier found no specific failing claims, but judged the answer incomplete."


def _ends_with_user_question(answer: str) -> bool:
    """True when the answer's final non-empty line is a question to the user."""
    for line in reversed((answer or "").strip().splitlines()):
        line = line.strip().lstrip("-*• ")
        if not line:
            continue
        return line.endswith("?")
    return False


def verify_answer(state: dict) -> dict:
    """Mechanical citation check + one LLM judge call over the whole answer.

    Order of resolution:
    1. Fast path — mechanical checks + support gate fully pass → answered,
       judge skipped (saves 30-90s per clean answer).
    2. Referent ambiguity — the query names an entity that matches several
       distinct real-world referents → ask the user which one, enumerating them.
    3. Contradictions with a clarification question → ask instead of caveat-guess.
    4. Gaps a targeted search could fix → repair loop (search or revise-only)
       with the judge's critique carried into regeneration.
    5. Otherwise → caveats (capped, deduped, no meta-sentences).
    """
    t0 = time.perf_counter()
    tracing.set_trace_context(node="verify_answer", chat_id=state.get("chat_id"), user_id=state.get("user_id"))
    answer = state.get("answer", "")
    evidence: list[Evidence] = state.get("evidence", [])
    cite_map: dict[str, str] = state.get("cite_map") or {}

    if not answer or not evidence:
        return {
            "claims": [],
            "verification_errors": state.get("verification_errors", []),
            "final_status": "answered_with_caveats",
            # LangGraph keeps prior values for absent keys — an explicit clear
            # is required or a stale repair_queries loops the graph forever.
            "repair_queries": [],
            "revise_requested": False,
            "critique": "",
            "graph_steps": state.get("graph_steps", 0) + 1,
        }

    mechanical: CitationValidationResult = validate_answer_citations(
        answer, evidence, cite_map=cite_map,
        entailment_gate=True,
    )
    # Unsupported claims must not keep citations in the prose: a marker asserts
    # the evidence backs the statement, which the gate just refuted.
    answer = strip_weak_markers(answer, mechanical)

    repair_count = state.get("repair_count", 0)
    used = ""

    # ── Fast path: mechanical checks fully clean → skip the judge entirely.
    if (
        settings.JUDGE_FAST_PATH
        and not mechanical.errors
        and not mechanical.unsupported_citations
        and not mechanical.invalid_citation_ids
        and mechanical.claims
    ):
        # ...but an answer that ends by asking the user a question is a
        # clarification turn, however clean its citations are.
        if _ends_with_user_question(answer):
            logger.info("Node verify_answer fast-path: answer ends with a question; needs_clarification")
            return {
                "answer": answer,
                "claims": [],
                "verification_errors": [],
                "final_status": "needs_clarification",
                "repair_queries": [],
                "revise_requested": False,
                "critique": "",
                "provider_used": "mechanical-fast-path",
                "graph_steps": state.get("graph_steps", 0) + 1,
            }
        logger.info(
            "Node verify_answer fast-path: mechanical checks clean (%d claims), judge skipped %.0fms",
            len(mechanical.claims), (time.perf_counter() - t0) * 1000,
        )
        return {
            "answer": answer,
            "claims": mechanical.claims,
            "verification_errors": [],
            "final_status": "answered",
            "repair_queries": [],
            "revise_requested": False,
            "critique": "",
            "provider_used": "mechanical-fast-path",
            "graph_steps": state.get("graph_steps", 0) + 1,
        }

    llms = _resolve_llms(state)
    verdict = Verdict(overall="partial", explanation="")
    try:
        messages = [
            SystemMessage(content=_VERIFY_PROMPT.format(
                question=state.get("query_original") or state.get("query", ""),
                temporal_context=_temporal_context(state),
                computed_block=_render_computations(state.get("computations") or []),
                context=_verify_context(evidence, cite_map),
                answer=answer,
            )),
            HumanMessage(content="Verify the answer now. Respond with the JSON verdict."),
        ]
        verdict = _llm_with_fallback(
            llms.verifier, llms.verifier_fallbacks, messages, Verdict, role="verifier",
            timeout=settings.VERIFIER_TIMEOUT_SECONDS,
        )
        used = llms.label
    except Exception as exc:
        logger.warning("verify_answer judge failed: %s; relying on mechanical checks only", exc)
        used = f"{llms.label}+mechanical-only"

    # Merge: judge verdicts win; mechanical uncited assertions stay unverified.
    claims = _merge_claims(verdict.claims, mechanical.claims)

    errors = [e.to_dict() for e in mechanical.errors]
    contradicted = [c for c in claims if c.status == ClaimStatus.CONTRADICTED]
    unverified = [c for c in claims if c.status == ClaimStatus.UNVERIFIED]

    # ── Referent ambiguity: the question names an entity with several distinct
    # real-world matches. Ask which one was meant, enumerating the identities —
    # the user can answer that; a caveat-laden guess they cannot.
    if verdict.referent_ambiguity and len(verdict.referents) >= 2:
        logger.info(
            "Node verify_answer: referent ambiguity (%d referents); asking user %.0fms",
            len(verdict.referents), (time.perf_counter() - t0) * 1000,
        )
        return {
            "answer": _referent_clarification(verdict.referents),
            "claims": [],
            "verification_errors": [],
            "final_status": "needs_clarification",
            "repair_queries": [],
            "revise_requested": False,
            "critique": "",
            "provider_used": used,
            "graph_steps": state.get("graph_steps", 0) + 1,
        }

    # ── Repair decision: pass 1 = targeted search from the judge's repair
    # queries; pass 2 = revise-only (the pool already holds everything search
    # found, so the rewrite integrates it or drops the flagged claims).
    gaps = bool(verdict.repair_queries) or bool(verdict.question_gaps) or bool(unverified)
    fixable = (
        gaps
        and not contradicted  # contradictions route to clarification/caveats
        and verdict.overall != "supported"
    )
    if fixable and repair_count < settings.MAX_REPAIR_PASSES:
        critique = _format_critique(verdict, contradicted + unverified)
        if repair_count >= 1:
            # Pass 2: revise with the critique, no new search.
            logger.info(
                "Node verify_answer: scheduling revise-only pass %d (gaps=%d unverified) %.0fms",
                repair_count + 1, len(unverified), (time.perf_counter() - t0) * 1000,
            )
            return {
                "claims": claims,
                "verification_errors": errors,
                "repair_queries": [],
                "revise_requested": True,
                "repair_count": repair_count + 1,
                "critique": critique,
                "draft_answer": answer,
                "graph_steps": state.get("graph_steps", 0) + 1,
                "provider_used": used,
            }
        if verdict.repair_queries:
            logger.info(
                "Node verify_answer: gaps found (%d unverified, %d question-gaps); scheduling repair search %.0fms",
                len(unverified), len(verdict.question_gaps), (time.perf_counter() - t0) * 1000,
            )
            return {
                "claims": claims,
                "verification_errors": errors,
                "repair_queries": [q for q in verdict.repair_queries if q.strip()][:MAX_SEARCH_QUERIES],
                "revise_requested": False,
                "repair_count": repair_count + 1,
                "critique": critique,
                "draft_answer": answer,
                "graph_steps": state.get("graph_steps", 0) + 1,
                "provider_used": used,
            }
        # Gaps exist but the judge says no search would fix them → finalize.

    # ── Finalize. Order matters: an answer that ends by asking the user a
    # question IS a clarification turn — nothing (caveats included) may be
    # appended after the question.
    if contradicted and verdict.clarification_question.strip():
        # Conflicting sources trace back to an ambiguous term — asking the user
        # which meaning they meant beats presenting a caveat-laden guess.
        answer = _conflict_clarification(contradicted, verdict.clarification_question.strip())
        final_status = "needs_clarification"
    elif _ends_with_user_question(answer):
        logger.info("Node verify_answer: answer ends with a question; promoting to needs_clarification")
        final_status = "needs_clarification"
    elif contradicted or (verdict.overall == "unsupported"):
        final_status = "answered_with_caveats"
        answer = _append_caveats(answer, contradicted + unverified)
    elif unverified or verdict.overall == "partial" or mechanical.errors:
        final_status = "answered_with_caveats"
        answer = _append_caveats(answer, unverified)
    elif not verdict.addresses_question:
        # Judge says the answer missed the question but nothing is fixable —
        # say so honestly instead of passing a wrong-question answer as clean.
        final_status = "answered_with_caveats"
        gaps_text = "; ".join(verdict.question_gaps[:3]) or "the answer may not fully address the question"
        answer += f"\n\nNote: this answer may not fully address your question ({gaps_text})."
    else:
        final_status = "answered"

    logger.info(
        "Node verify_answer completed in %.0fms overall=%s claims=%d status=%s repair_used=%d",
        (time.perf_counter() - t0) * 1000, verdict.overall, len(claims), final_status, repair_count,
    )
    return {
        "answer": answer,
        "claims": claims,
        "verification_errors": errors,
        "final_status": final_status,
        # Same as the repair path: without explicit clears, stale values left in
        # state would re-trigger the repair loop forever.
        "repair_queries": [],
        "revise_requested": False,
        "critique": "",
        "provider_used": used,
        "graph_steps": state.get("graph_steps", 0) + 1,
    }


def _append_caveats(answer: str, failed: list[Claim]) -> str:
    """Honest caveats: atomic factual claims only, capped, no echoes of the answer."""
    pruned = prune_caveats(failed, answer, max_bullets=CAVEAT_MAX_BULLETS)
    if not pruned:
        return answer
    lines = ["", "Caveats:", "The following points could not be fully verified against the retrieved evidence:"]
    for c in pruned:
        t = CITATION_TOKEN_RE.sub("", c.text).strip()[:300]
        t = re.sub(r"\s+([.,;:])", r"\1", t)
        lines.append(f"- {t}")
    return answer.rstrip() + "\n\n" + "\n".join(lines)


def _referent_clarification(referents: list[str]) -> str:
    """Ask which real-world entity the user meant, enumerating those found."""
    lines = [
        "I found several distinct matches for that name and need one detail to answer accurately:",
        "",
        "Which one did you mean?",
        "",
    ]
    seen: set[str] = set()
    for r in referents[:6]:
        identity = (r or "").strip()[:200]
        if identity and identity.lower() not in seen:
            seen.add(identity.lower())
            lines.append(f"- {identity}")
    return "\n".join(lines)


def _conflict_clarification(contradicted: list[Claim], question: str) -> str:
    """Present conflicting findings and ask which interpretation was meant."""
    lines = [
        "I found conflicting information and need one detail to give you an accurate answer:",
        "",
        question,
        "",
        "The conflicting findings were:",
    ]
    seen: set[str] = set()
    for c in contradicted:
        t = c.text.strip()[:300]
        if t and t not in seen:
            seen.add(t)
            lines.append(f"- {t}")
    return "\n".join(lines)


def route_after_verify(state: dict) -> str:
    """Loop back to gathering (repair search) or generating (revise-only pass)."""
    if state.get("repair_queries"):
        return "gather_evidence"
    if state.get("revise_requested"):
        return "generate_answer"
    return END
