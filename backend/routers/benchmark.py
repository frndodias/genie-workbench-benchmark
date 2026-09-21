"""Benchmark tab — job-driven, persisted benchmark evaluation.

Everything triggers a Databricks Job (``BENCHMARK_JOB_ID``) that runs the native
Genie Eval-Run + the false-negative cascade (deterministic SQL/multiset → LLM
review with each benchmark's ``evaluation_note``) and persists every tier plus a
final verdict to Unity Catalog (``<GSO_CATALOG>.<GSO_SCHEMA>.benchmark_runs`` and
``benchmark_results``). This router:

- lists the Agent's configured benchmarks (question + expected SQL + evaluation_note),
- triggers the job with the user-selected questions + model as job parameters,
- and reads the persisted runs/results for the dashboard (runs list + drill-down).

The heavy logic lives in the shared module ``genie_space_optimizer.benchmark_eval``
(used by the job notebook); the app no longer runs the cascade in-process.
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from backend.services.auth import get_service_principal_client, get_workspace_client
from backend.services.genie_client import get_serialized_space, is_scope_error
from backend.sql_executor import execute_sql

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/benchmark", tags=["benchmark"])


def _job_id() -> str | None:
    return (os.environ.get("BENCHMARK_JOB_ID") or "").strip() or None


def _catalog() -> str:
    return (os.environ.get("GSO_CATALOG") or "").strip()


def _schema() -> str:
    return (os.environ.get("GSO_SCHEMA") or "genie_space_optimizer").strip()


def _runs_table() -> str:
    return f"`{_catalog()}`.`{_schema()}`.benchmark_runs"


def _results_table() -> str:
    return f"`{_catalog()}`.`{_schema()}`.benchmark_results"


# ── configured benchmarks (question + expected SQL + evaluation_note) ────────


def _extract_expected_sql(question: dict) -> str | None:
    ans = question.get("answer")
    if isinstance(ans, list) and ans and isinstance(ans[0], dict):
        content = ans[0].get("content")
        if isinstance(content, list):
            joined = "".join(str(c) for c in content).strip()
            if joined:
                return joined
        if isinstance(content, str) and content.strip():
            return content
    for key in ("sql", "expected_sql", "expected_response", "query"):
        val = question.get(key)
        if isinstance(val, list):
            val = val[0] if val else None
        if isinstance(val, str) and val.strip():
            return val
    return None


def _question_text(question: dict) -> str:
    text = question.get("question") or question.get("text")
    if isinstance(text, list):
        text = text[0] if text else ""
    return str(text or "")


def _eval_note(question: dict) -> str | None:
    note = question.get("evaluation_note")
    if isinstance(note, list):
        note = " ".join(str(n) for n in note if n is not None)
    note = str(note or "").strip()
    return note or None


@router.get("/{space_id}")
def list_benchmarks(space_id: str):
    """List the benchmark questions configured on the Genie Agent."""
    try:
        space = get_serialized_space(genie_space_id=space_id) or {}
    except Exception as e:
        if is_scope_error(e):
            raise HTTPException(status_code=403, detail="Missing scope to read this Agent's configuration.")
        logger.exception("Failed to read serialized space for %s", space_id)
        raise HTTPException(status_code=500, detail="Failed to read the Agent configuration.")

    benchmarks = space.get("benchmarks") or {}
    questions = benchmarks.get("questions") if isinstance(benchmarks, dict) else None
    out: list[dict] = []
    for q in questions or []:
        if not isinstance(q, dict):
            continue
        out.append({
            "question_id": str(q.get("id") or ""),
            "question": _question_text(q),
            "expected_sql": _extract_expected_sql(q),
            "evaluation_note": _eval_note(q),
        })
    return {"count": len(out), "benchmarks": out, "job_configured": _job_id() is not None}


# ── trigger the job ──────────────────────────────────────────────────────────


class RunRequest(BaseModel):
    benchmark_question_ids: list[str] | None = None
    llm_model: str | None = None


@router.post("/{space_id}/run")
def run_benchmark(space_id: str, request: Request, body: RunRequest | None = None):
    """Trigger the benchmark job for the selected questions + model."""
    job_id = _job_id()
    if not job_id:
        raise HTTPException(status_code=503, detail="Benchmark job is not configured (BENCHMARK_JOB_ID).")

    ids = (body.benchmark_question_ids if body else None) or []
    ids = [str(i).strip() for i in ids if str(i).strip()]
    llm_model = ((body.llm_model if body else None) or "").strip() or "databricks-claude-sonnet-4-6"
    run_id = str(uuid.uuid4())

    # Best-effort caller identity for the "Quem" column.
    try:
        triggered_by = request.headers.get("x-forwarded-email") or ""
    except Exception:
        triggered_by = ""

    params = {
        "run_id": run_id,
        "space_id": space_id,
        "question_ids": json.dumps(ids),
        "llm_model": llm_model,
        "triggered_by": triggered_by,
        "catalog": _catalog(),
        "schema": _schema(),
    }

    # Trigger as the service principal (it holds CAN_MANAGE on the job).
    sp = get_service_principal_client()
    try:
        res = sp.api_client.do(
            method="POST",
            path="/api/2.1/jobs/run-now",
            body={"job_id": int(job_id), "job_parameters": params},
        )
    except Exception as e:
        logger.exception("Failed to trigger benchmark job for %s", space_id)
        raise HTTPException(status_code=502, detail=f"Failed to start the benchmark job: {str(e)[:200]}")

    return {"run_id": run_id, "job_run_id": str(res.get("run_id") or ""), "status": "RUNNING"}


# ── read persisted runs/results (dashboard) ──────────────────────────────────


def _query(sql: str) -> list[dict]:
    """Run a read query via the OBO warehouse client; return list of row dicts."""
    res = execute_sql(sql, row_limit=1000)
    if res.get("error"):
        raise HTTPException(status_code=502, detail=f"Read failed: {res['error'][:200]}")
    cols = [c.get("name") for c in res.get("columns") or []]
    return [dict(zip(cols, row)) for row in res.get("data") or []]


def _q(v: str) -> str:
    return "'" + str(v).replace("'", "''") + "'"


@router.get("/{space_id}/runs")
def list_runs(space_id: str, limit: int = 50):
    """List recent benchmark runs for an Agent (newest first)."""
    if not _catalog():
        return {"runs": []}
    lim = max(1, min(int(limit), 200))
    try:
        rows = _query(
            f"SELECT run_id, space_name, triggered_by, llm_model, run_at, completed_at, "
            f"status, num_questions, num_good, num_bad, num_needs_review, num_final_good, "
            f"num_final_needs_review, accuracy_native, accuracy_adjusted "
            f"FROM {_runs_table()} WHERE space_id = {_q(space_id)} "
            f"ORDER BY run_at DESC LIMIT {lim}"
        )
    except HTTPException:
        # Table may not exist yet (no run ever) — treat as empty.
        return {"runs": []}
    return {"runs": rows}


@router.get("/runs/{run_id}")
def get_run(run_id: str):
    """One run's header/status (from the persisted table)."""
    try:
        rows = _query(
            f"SELECT run_id, space_id, space_name, triggered_by, llm_model, run_at, completed_at, "
            f"status, num_questions, num_good, num_bad, num_needs_review, num_final_good, "
            f"num_final_needs_review, accuracy_native, accuracy_adjusted, error "
            f"FROM {_runs_table()} WHERE run_id = {_q(run_id)} LIMIT 1"
        )
    except HTTPException:
        rows = []
    if not rows:
        # Not persisted yet (job still starting) — report as running.
        return {"run_id": run_id, "status": "RUNNING", "found": False}
    row = rows[0]
    row["found"] = True
    return row


@router.get("/runs/{run_id}/results")
def get_run_results(run_id: str):
    """Per-question results for a run (drill-down)."""
    try:
        rows = _query(
            f"SELECT question_id, question, evaluation_note, expected_sql, generated_sql, "
            f"result_databricks, assessment_reasons, python_method, python_equivalent, python_detail, "
            f"llm_verdict, llm_confidence, llm_reasoning, llm_model, result_final, decided_by, "
            f"generated_result, expected_result "
            f"FROM {_results_table()} WHERE run_id = {_q(run_id)}"
        )
    except HTTPException:
        rows = []

    def _parse_json(raw: Any, default: Any):
        if isinstance(raw, str) and raw.strip():
            try:
                return json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                return default
        return default

    for r in rows:
        r["assessment_reasons"] = _parse_json(r.get("assessment_reasons"), [])
        r["generated_result"] = _parse_json(r.get("generated_result"), None)
        r["expected_result"] = _parse_json(r.get("expected_result"), None)
        pe = r.get("python_equivalent")
        r["python_equivalent"] = str(pe).lower() == "true" if pe is not None else False
    return {"run_id": run_id, "results": rows}
