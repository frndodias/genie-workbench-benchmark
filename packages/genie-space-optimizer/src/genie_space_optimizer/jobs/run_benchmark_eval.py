# Databricks notebook source
# MAGIC %md
# MAGIC # Benchmark Eval (Genie Workbench)
# MAGIC
# MAGIC | Quick Reference | |
# MAGIC |---|---|
# MAGIC | **Task** | `benchmark_eval` (single-task job) |
# MAGIC | **Reads** | job params, the Agent's native benchmarks + eval-run, the warehouse |
# MAGIC | **Writes** | `<catalog>.<schema>.benchmark_runs`, `<catalog>.<schema>.benchmark_results` |
# MAGIC | **Runs as** | the app service principal |
# MAGIC
# MAGIC Standalone, persisted benchmark evaluation (no optimization, no config
# MAGIC change). Triggers the native Genie Eval-Run, then runs the shared
# MAGIC false-negative cascade per question (native → deterministic SQL/multiset →
# MAGIC LLM review with the benchmark's `evaluation_note`) and persists every tier
# MAGIC plus the final verdict to Unity Catalog. Everything (question selection,
# MAGIC model) arrives as job parameters from the app.

# COMMAND ----------

import datetime as _dt
import json
import logging

import httpx

from genie_space_optimizer._workspace_client import make_workspace_client
from genie_space_optimizer.benchmark_eval import (
    build_judge_messages,
    compute_equivalence,
    final_verdict,
    parse_judge_response,
)
from genie_space_optimizer.optimization.eval_runner import OfficialBenchmarkRunner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("benchmark_eval")

dbutils = globals().get("dbutils")
spark = globals().get("spark")


def _widget(name: str, default: str = "") -> str:
    try:
        dbutils.widgets.text(name, default)
    except Exception:
        pass
    try:
        return (dbutils.widgets.get(name) or "").strip()
    except Exception:
        return default


RUN_ID = _widget("run_id")
SPACE_ID = _widget("space_id")
QUESTION_IDS_RAW = _widget("question_ids", "[]")
LLM_MODEL = _widget("llm_model") or "databricks-claude-sonnet-4-6"
TRIGGERED_BY = _widget("triggered_by")
CATALOG = _widget("catalog")
SCHEMA = _widget("schema") or "genie_space_optimizer"
WAREHOUSE_ID = _widget("warehouse_id")

if not RUN_ID or not SPACE_ID or not CATALOG:
    raise ValueError("run_id, space_id and catalog are required job parameters")

try:
    QUESTION_IDS = [str(q).strip() for q in json.loads(QUESTION_IDS_RAW) if str(q).strip()]
except Exception:
    QUESTION_IDS = []

RUNS_TABLE = f"`{CATALOG}`.`{SCHEMA}`.benchmark_runs"
RESULTS_TABLE = f"`{CATALOG}`.`{SCHEMA}`.benchmark_results"

w = make_workspace_client()

# COMMAND ----------
# MAGIC %md ## Ensure tables (SP owns what it creates)

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{CATALOG}`.`{SCHEMA}`")
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {RUNS_TABLE} (
  run_id STRING, space_id STRING, space_name STRING, triggered_by STRING,
  llm_model STRING, run_at TIMESTAMP, completed_at TIMESTAMP, status STRING,
  job_run_id STRING, eval_run_id STRING,
  num_questions INT, num_good INT, num_bad INT, num_needs_review INT,
  num_final_good INT, num_final_needs_review INT,
  accuracy_native DOUBLE, accuracy_adjusted DOUBLE, error STRING
) USING DELTA
""")
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {RESULTS_TABLE} (
  run_id STRING, space_id STRING, question_id STRING, question STRING,
  evaluation_note STRING, expected_sql STRING, generated_sql STRING,
  result_databricks STRING, assessment_reasons STRING,
  python_method STRING, python_equivalent BOOLEAN, python_detail STRING,
  llm_verdict STRING, llm_confidence DOUBLE, llm_reasoning STRING, llm_model STRING,
  result_final STRING, decided_by STRING,
  generated_result STRING, expected_result STRING, logged_at TIMESTAMP
) USING DELTA
""")
# Backfill the result-sample columns on pre-existing tables (idempotent).
for _col in ("generated_result", "expected_result"):
    try:
        spark.sql(f"ALTER TABLE {RESULTS_TABLE} ADD COLUMNS ({_col} STRING)")
    except Exception:
        pass


def _now():
    return _dt.datetime.now(_dt.timezone.utc)


def _sql_str(v):
    if v is None:
        return "NULL"
    return "'" + str(v).replace("'", "''") + "'"


# Write the run header (RUNNING).
space_name = ""
try:
    meta = w.api_client.do(method="GET", path=f"/api/2.0/genie/spaces/{SPACE_ID}")
    space_name = str((meta or {}).get("display_name") or (meta or {}).get("title") or "")
except Exception:
    pass

job_run_id = ""
try:
    job_run_id = str((dbutils.notebook.entry_point.getDbutils().notebook().getContext()
                      .currentRunId().toString()))
except Exception:
    pass

spark.sql(
    f"INSERT INTO {RUNS_TABLE} (run_id, space_id, space_name, triggered_by, llm_model, "
    f"run_at, status, job_run_id) VALUES ({_sql_str(RUN_ID)}, {_sql_str(SPACE_ID)}, "
    f"{_sql_str(space_name)}, {_sql_str(TRIGGERED_BY)}, {_sql_str(LLM_MODEL)}, "
    f"current_timestamp(), 'RUNNING', {_sql_str(job_run_id)})"
)

# COMMAND ----------
# MAGIC %md ## Native eval-run + per-question cascade

# COMMAND ----------


def run_warehouse_sql(sql: str) -> dict:
    """Execute SQL and return a preview dict {columns, data, error} (Spark)."""
    try:
        df = spark.sql(sql).limit(2000)
        cols = [{"name": f} for f in df.columns]
        data = [[None if v is None else str(v) for v in row] for row in df.collect()]
        return {"columns": cols, "data": data, "error": None}
    except Exception as e:  # noqa: BLE001
        return {"columns": [], "data": [], "error": str(e)[:500]}


def call_llm(messages: list[dict], model: str) -> str:
    host = w.config.host.rstrip("/")
    headers = w.config.authenticate()
    url = f"{host}/serving-endpoints/{model}/invocations"
    resp = httpx.post(url, json={"messages": messages, "max_tokens": 600}, headers=headers, timeout=120)
    resp.raise_for_status()
    body = resp.json()
    return body["choices"][0]["message"]["content"]


# Fetch serialized_space benchmarks → evaluation_note + canonical expected SQL by qid.
notes_by_qid: dict[str, str] = {}
expected_by_qid: dict[str, str] = {}
try:
    space = w.api_client.do(
        method="GET",
        path=f"/api/2.0/genie/spaces/{SPACE_ID}",
        query={"include_serialized_space": "true"},
    )
    ss = space.get("serialized_space")
    if isinstance(ss, str):
        ss = json.loads(ss)
    for q in ((ss or {}).get("benchmarks", {}) or {}).get("questions", []) or []:
        qid = str(q.get("id") or "")
        note = q.get("evaluation_note")
        if isinstance(note, list):
            note = " ".join(str(n) for n in note if n is not None)
        if qid and note and str(note).strip():
            notes_by_qid[qid] = str(note).strip()
        ans = q.get("answer")
        if qid and isinstance(ans, list) and ans and isinstance(ans[0], dict):
            content = ans[0].get("content")
            if isinstance(content, list):
                expected_by_qid[qid] = "".join(str(c) for c in content)
except Exception:
    logger.exception("Could not read serialized_space benchmarks")

# Run the native eval-run (None ⇒ all; else the selected subset).
qids = QUESTION_IDS or None
runner = OfficialBenchmarkRunner(w)
result = runner.run(SPACE_ID, benchmark_question_ids=qids)
eval_run_id = getattr(result, "eval_run_id", "") or ""

num_good = num_bad = num_needs = 0
final_good = final_needs = 0
rows_out: list[dict] = []

for row in result.rows:
    qid = str(row.get("question_id") or "")
    native = str(row.get("assessment") or "").strip().upper() or "NEEDS_REVIEW"
    gen_sql = row.get("generated_sql")
    exp_sql = row.get("expected_sql") or expected_by_qid.get(qid)
    note = notes_by_qid.get(qid)
    reasons = row.get("assessment_reasons") or []

    if native == "GOOD":
        num_good += 1
    elif native == "NEEDS_REVIEW":
        num_needs += 1
    else:
        num_bad += 1

    equivalence = {"method": "none", "equivalent": False, "detail": ""}
    llm = {"verdict": None, "confidence": None, "reasoning": ""}
    generated = expected = None

    if native != "GOOD":
        generated = run_warehouse_sql(gen_sql) if gen_sql else None
        expected = run_warehouse_sql(exp_sql) if exp_sql else None
        equivalence = compute_equivalence(gen_sql, exp_sql, generated, expected, run_warehouse_sql)
        # Always ask the LLM for a justification on a failed row (so every
        # non-GOOD row carries the model's verdict + reasoning), even when the
        # deterministic tier already decided. Retry once on a transient failure
        # and NEVER leave the row silently blank — record the error instead.
        msgs = build_judge_messages(
            question=row.get("question"), expected_sql=exp_sql, generated_sql=gen_sql,
            generated=generated, expected=expected, evaluation_note=note,
        )
        last_err = None
        for _attempt in range(2):
            try:
                llm = parse_judge_response(call_llm(msgs, LLM_MODEL))
                last_err = None
                break
            except Exception as e:  # noqa: BLE001
                last_err = str(e)[:300]
                logger.exception("LLM judge failed for %s (attempt %d)", qid, _attempt + 1)
        if last_err and not llm.get("verdict"):
            llm = {"verdict": "error", "confidence": None, "reasoning": f"LLM review failed: {last_err}"}

    def _sample(preview):
        if not preview or preview.get("error"):
            return None
        return json.dumps({
            "columns": [c.get("name") for c in preview.get("columns") or []],
            "data": (preview.get("data") or [])[:20],
            "error": preview.get("error"),
        }, ensure_ascii=False)

    result_final, decided_by = final_verdict(native, equivalence, llm.get("verdict"))
    if result_final == "GOOD":
        final_good += 1
    elif result_final == "NEEDS_REVIEW":
        final_needs += 1

    rows_out.append({
        "run_id": RUN_ID, "space_id": SPACE_ID, "question_id": qid,
        "question": str(row.get("question") or ""), "evaluation_note": note,
        "expected_sql": exp_sql, "generated_sql": gen_sql,
        "result_databricks": native, "assessment_reasons": json.dumps(reasons, ensure_ascii=False),
        "python_method": equivalence.get("method"), "python_equivalent": bool(equivalence.get("equivalent")),
        "python_detail": equivalence.get("detail"),
        "llm_verdict": llm.get("verdict"), "llm_confidence": llm.get("confidence"),
        "llm_reasoning": llm.get("reasoning"), "llm_model": LLM_MODEL if llm.get("verdict") else None,
        "result_final": result_final, "decided_by": decided_by,
        "generated_result": _sample(generated), "expected_result": _sample(expected),
        "logged_at": _now(),
    })

# COMMAND ----------
# MAGIC %md ## Persist results + finalize run

# COMMAND ----------

if rows_out:
    from pyspark.sql.types import (
        BooleanType, DoubleType, StringType, StructField, StructType, TimestampType,
    )
    # Explicit schema (in table-column order) so a single-row run with an
    # all-None column doesn't hit CANNOT_DETERMINE_TYPE during inference.
    _s = StringType()
    schema = StructType([
        StructField("run_id", _s), StructField("space_id", _s), StructField("question_id", _s),
        StructField("question", _s), StructField("evaluation_note", _s), StructField("expected_sql", _s),
        StructField("generated_sql", _s), StructField("result_databricks", _s),
        StructField("assessment_reasons", _s), StructField("python_method", _s),
        StructField("python_equivalent", BooleanType()), StructField("python_detail", _s),
        StructField("llm_verdict", _s), StructField("llm_confidence", DoubleType()),
        StructField("llm_reasoning", _s), StructField("llm_model", _s),
        StructField("result_final", _s), StructField("decided_by", _s),
        StructField("generated_result", _s), StructField("expected_result", _s),
        StructField("logged_at", TimestampType()),
    ])
    ordered = [{f.name: row.get(f.name) for f in schema.fields} for row in rows_out]
    df = spark.createDataFrame(ordered, schema=schema)
    df.write.mode("append").saveAsTable(f"`{CATALOG}`.`{SCHEMA}`.benchmark_results")

n = len(result.rows)
acc_native = round(100.0 * num_good / n, 1) if n else 0.0
acc_adjusted = round(100.0 * final_good / n, 1) if n else 0.0

spark.sql(
    f"UPDATE {RUNS_TABLE} SET completed_at=current_timestamp(), status='COMPLETED', "
    f"eval_run_id={_sql_str(eval_run_id)}, num_questions={n}, num_good={num_good}, "
    f"num_bad={num_bad}, num_needs_review={num_needs}, num_final_good={final_good}, "
    f"num_final_needs_review={final_needs}, accuracy_native={acc_native}, "
    f"accuracy_adjusted={acc_adjusted} WHERE run_id={_sql_str(RUN_ID)}"
)

logger.info("benchmark_eval done: run=%s n=%d native=%.1f adjusted=%.1f", RUN_ID, n, acc_native, acc_adjusted)
print(json.dumps({"run_id": RUN_ID, "num_questions": n, "accuracy_native": acc_native,
                  "accuracy_adjusted": acc_adjusted, "eval_run_id": eval_run_id}))
