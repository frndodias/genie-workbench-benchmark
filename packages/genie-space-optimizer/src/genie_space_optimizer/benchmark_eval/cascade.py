"""Deterministic equivalence cascade + LLM judge for benchmark false-negatives.

Contract: results are passed as "preview" dicts shaped like
``{"columns": [{"name": str}], "data": [[cell, ...]], "error": str | None}`` —
the shape both the app's ``execute_sql`` and the notebook's Spark reader produce.

Tiers (see :func:`compute_equivalence`):
  1. ``sql_exact`` — ``EXCEPT ALL`` both ways on the warehouse (via the injected
     ``run_sql``); exact, full-precision, scales, compares by column POSITION so
     it is robust to column-name diffs and catches entity↔value swaps. Needs the
     two queries to share column count; returns None otherwise → tier 2.
  2. ``numeric_multiset`` — same numbers regardless of shape/order (labels not
     key-verified). Reshape-tolerant fallback.

The LLM judge (tier 3) is advisory: :func:`build_judge_messages` /
:func:`parse_judge_response` let the caller run its own serving endpoint, and
:func:`resolve_llm_verdict` maps the raw verdict to a benchmark assessment.
:func:`final_verdict` resolves the whole cascade into ``(result_final,
decided_by)`` — deterministic tiers can flip a BAD to GOOD; the LLM only lifts
it to NEEDS_REVIEW.
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable


def content_to_str(content: Any) -> str:
    """Normalize an LLM ``message.content`` to text.

    Some serving endpoints (Claude in OpenAI-compat mode) return content as a
    list of blocks (``[{"type": "text", "text": "..."}]``) instead of a plain
    string; join the text parts so downstream string ops don't blow up.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(str(block.get("text") or block.get("content") or ""))
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content or "")


def _extract_json(content: Any) -> dict:
    """Robustly pull a JSON object from an LLM reply (fences / surrounding text)."""
    content = content_to_str(content)
    if not content:
        return {}
    s = content.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass
    m = re.search(r"\{.*\}", s, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}
    return {}

# ── numeric multiset (tier 2) ───────────────────────────────────────────────


def to_float(cell: object) -> float | None:
    """Parse a result cell into a float, or None if it isn't numeric."""
    if cell is None or isinstance(cell, bool):
        return None
    if isinstance(cell, (int, float)):
        return float(cell)
    try:
        return float(str(cell))
    except (ValueError, TypeError):
        return None


def numeric_multiset(preview: dict | None) -> list[float] | None:
    """Sorted multiset of every numeric cell value in a result (rounded 2dp).

    Shape- and order-insensitive; keeps only numbers so text (labels) drops out.
    Returns None when the result is missing or errored.
    """
    if not preview or preview.get("error"):
        return None
    values: list[float] = []
    for row in preview.get("data") or []:
        if not isinstance(row, (list, tuple)):
            continue
        for cell in row:
            f = to_float(cell)
            if f is not None:
                values.append(round(f, 2))
    return sorted(values)


def data_equivalent(generated: dict | None, expected: dict | None) -> bool:
    """True when both results carry the identical multiset of numeric values."""
    a = numeric_multiset(generated)
    b = numeric_multiset(expected)
    if not a or not b:
        return False
    return a == b


# ── SQL exact set-diff (tier 1) ──────────────────────────────────────────────


def strip_sql(sql: str) -> str:
    return sql.strip().rstrip(";").strip()


def sql_setdiff_equivalent(
    a_sql: str | None,
    b_sql: str | None,
    run_sql: Callable[[str], dict],
) -> dict | None:
    """Exact row-level equivalence via SQL ``EXCEPT ALL`` in both directions.

    ``run_sql(sql)`` must return a preview dict (``{columns, data, error}``).
    Returns ``{"a_minus_b": int, "b_minus_a": int}`` or None when the diff query
    errors (e.g. mismatched column count) so the caller falls back to tier 2.
    """
    if not a_sql or not a_sql.strip() or not b_sql or not b_sql.strip():
        return None
    a, b = strip_sql(a_sql), strip_sql(b_sql)
    query = (
        "SELECT "
        f"(SELECT COUNT(*) FROM ((SELECT * FROM ({a})) EXCEPT ALL (SELECT * FROM ({b})))) AS a_minus_b, "
        f"(SELECT COUNT(*) FROM ((SELECT * FROM ({b})) EXCEPT ALL (SELECT * FROM ({a})))) AS b_minus_a"
    )
    try:
        res = run_sql(query)
    except Exception:
        return None
    if not res or res.get("error") or not res.get("data"):
        return None
    try:
        row = res["data"][0]
        return {"a_minus_b": int(float(row[0])), "b_minus_a": int(float(row[1]))}
    except (IndexError, ValueError, TypeError):
        return None


def compute_equivalence(
    a_sql: str | None,
    b_sql: str | None,
    generated: dict | None,
    expected: dict | None,
    run_sql: Callable[[str], dict],
) -> dict:
    """Cascade: exact SQL set-diff first, numeric multiset for reshape, else none."""
    ok = (
        generated and expected
        and not generated.get("error") and not expected.get("error")
    )
    if ok:
        gcols = len(generated.get("columns") or [])
        ecols = len(expected.get("columns") or [])
        if gcols > 0 and gcols == ecols:
            diff = sql_setdiff_equivalent(a_sql, b_sql, run_sql)
            if diff is not None:
                equal = diff["a_minus_b"] == 0 and diff["b_minus_a"] == 0
                return {
                    "method": "sql_exact",
                    "equivalent": equal,
                    "detail": (
                        "Identical rows — exact match (labels and precision verified)."
                        if equal else
                        "Same shape, but some rows/values differ."
                    ),
                }
        equal = data_equivalent(generated, expected)
        return {
            "method": "numeric_multiset",
            "equivalent": equal,
            "detail": (
                "Same values, different shape (labels not key-verified)."
                if equal else
                "Values differ."
            ),
        }
    return {"method": "none", "equivalent": False, "detail": "Could not compare (a query failed)."}


# ── LLM judge (tier 3, advisory) ─────────────────────────────────────────────

JUDGE_SYSTEM_PROMPT = (
    "You are a strict data-analyst judge. You decide whether a GENERATED SQL "
    "answer is semantically equivalent to the EXPECTED (ground-truth) answer "
    "for a business QUESTION. Two answers are EQUIVALENT when they convey the "
    "same information even if the shape differs (long vs wide vs transposed), "
    "columns are named differently (e.g. 'chanel' vs 'canal'), rows are ordered "
    "differently, or numbers are rounded. They are DIFFERENT when the actual "
    "values differ, an entity is mapped to the wrong value, rows/entities are "
    "missing or extra, filters differ, or the metric computed is wrong. If you "
    "cannot tell from the samples, say 'uncertain'. Judge the DATA, not the SQL "
    "style. If an EVALUATION NOTE is provided, it is the grading rubric authored "
    "for this benchmark and TAKES PRECEDENCE — follow it exactly (e.g. it may say "
    "to ignore formatting like '11.000.000' vs '11M', ignore ordering, or compare "
    "only a specific metric). Respond with ONLY a JSON object: "
    '{"verdict": "equivalent" | "different" | "uncertain", '
    '"confidence": 0.0-1.0, "reasoning": "one or two sentences"}.'
)


def _render_preview(preview: dict | None, *, max_rows: int = 30) -> str:
    if not preview:
        return "(no result)"
    if preview.get("error"):
        return f"(query error: {preview['error']})"
    cols = [c.get("name", "") for c in (preview.get("columns") or [])]
    rows = preview.get("data") or []
    lines = [" | ".join(str(c) for c in cols)]
    for row in rows[:max_rows]:
        lines.append(" | ".join("∅" if v is None else str(v) for v in row))
    if len(rows) > max_rows:
        lines.append(f"... (+{len(rows) - max_rows} more rows)")
    return "\n".join(lines)


def build_judge_messages(
    *,
    question: str | None,
    expected_sql: str | None,
    generated_sql: str | None,
    generated: dict | None,
    expected: dict | None,
    evaluation_note: str | None,
) -> list[dict]:
    """Build the chat messages for the LLM judge. Caller runs the endpoint."""
    note_block = (
        f"EVALUATION NOTE (grading rubric — follow it, it takes precedence):\n{evaluation_note}\n\n"
        if evaluation_note and evaluation_note.strip()
        else ""
    )
    user_prompt = (
        f"QUESTION:\n{question or '(not provided)'}\n\n"
        f"{note_block}"
        f"EXPECTED SQL (ground truth):\n{expected_sql or '(not provided)'}\n\n"
        f"EXPECTED RESULT (sample):\n{_render_preview(expected)}\n\n"
        f"GENERATED SQL (Genie):\n{generated_sql or '(not provided)'}\n\n"
        f"GENERATED RESULT (sample):\n{_render_preview(generated)}\n\n"
        "Is the GENERATED answer equivalent to the EXPECTED answer for the QUESTION?"
    )
    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def parse_judge_response(content: str, parse_json: Callable[[str], dict] | None = None) -> dict:
    """Parse the judge's JSON reply into {verdict, confidence, reasoning}."""
    parsed: dict
    try:
        parsed = parse_json(content) if parse_json else _extract_json(content)
    except Exception:
        parsed = _extract_json(content)
    if not isinstance(parsed, dict):
        parsed = {}
    verdict = str(parsed.get("verdict") or "uncertain").strip().lower()
    if verdict not in ("equivalent", "different", "uncertain"):
        verdict = "uncertain"
    confidence = parsed.get("confidence")
    try:
        confidence = round(float(confidence), 2) if confidence is not None else None
    except (TypeError, ValueError):
        confidence = None
    return {
        "verdict": verdict,
        "confidence": confidence,
        "reasoning": str(parsed.get("reasoning") or "").strip(),
    }


# ── final verdict resolution ─────────────────────────────────────────────────


def _norm(a: Any) -> str:
    return str(a or "").strip().upper()


def resolve_llm_verdict(llm_verdict: str | None) -> str | None:
    """Map an LLM verdict to a benchmark assessment lift, or None if it can't help.

    An 'equivalent' verdict lifts a failed row to GOOD (decided_by=llm).
    """
    v = str(llm_verdict or "").strip().lower()
    if v == "equivalent":
        return "GOOD"
    return None


def final_verdict(
    native_assessment: str | None,
    equivalence: dict | None,
    llm_verdict: str | None,
) -> tuple[str, str]:
    """Resolve the cascade into ``(result_final, decided_by)``.

    - native GOOD → (GOOD, "databricks")
    - deterministic equivalent → (GOOD, "python")   [exact/values match]
    - LLM equivalent → (GOOD, "llm")                 [semantic match]
    - otherwise → (native assessment or BAD, "databricks")
    """
    native = _norm(native_assessment)
    if native == "GOOD":
        return "GOOD", "databricks"
    if equivalence and equivalence.get("equivalent"):
        return "GOOD", "python"
    lifted = resolve_llm_verdict(llm_verdict)
    if lifted:
        return lifted, "llm"
    return (native or "BAD"), "databricks"
