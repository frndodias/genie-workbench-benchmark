# Databricks notebook source
# MAGIC %md
# MAGIC # Run Benchmark Eval for N Genie Agents
# MAGIC
# MAGIC Example notebook that triggers the Genie Workbench **benchmark-eval job** for a
# MAGIC list of Genie Agents (spaces), one job run per space, then (optionally) waits
# MAGIC for them to finish and summarizes the results straight from Unity Catalog.
# MAGIC
# MAGIC What each run does (server-side, in the job): native Genie Eval-Run +
# MAGIC false-negative cascade (exact SQL set-diff -> unbounded numeric-multiset ->
# MAGIC LLM review with each benchmark's `evaluation_note`), persisted to
# MAGIC `<catalog>.<schema>.benchmark_runs` / `benchmark_results`.
# MAGIC
# MAGIC **Requirements**
# MAGIC - The Genie Workbench app is deployed (so the `benchmark-eval` job exists).
# MAGIC - You can trigger that job (CAN_MANAGE / owner) — the app SP has it by default.
# MAGIC - You can `SELECT` the benchmark tables (you own them if you deployed / ran the job).
# MAGIC - The result tables are created by the job on its first run — no manual DDL.

# COMMAND ----------
# MAGIC %md ## 1. Configuration — edit these

# COMMAND ----------

# --- Genie Agents (space ids) to evaluate -----------------------------------
# Get a space id from its URL: .../genie/rooms/<SPACE_ID> (32-char hex).
SPACE_IDS = [
    "01f0000000000000000000000000aaaa",
    "01f0000000000000000000000000bbbb",
]

# Optional: restrict which benchmark questions run per space.
# Map space_id -> list of benchmark question ids. Omit a space (or leave []) = all.
QUESTION_IDS_BY_SPACE: dict[str, list[str]] = {
    # "01f0000000000000000000000000aaaa": ["q_id_1", "q_id_2"],
}

# --- Where the benchmark-eval job + tables live -----------------------------
# Set the job id explicitly, or leave 0 to resolve it by name.
BENCHMARK_JOB_ID = 0
BENCHMARK_JOB_NAME = "genie-workbench-benchmark-eval-job"  # deploy default; notebook-installer uses "<app-name>-benchmark-eval-job"

CATALOG = "fc_catalog"                 # UC catalog that holds the benchmark tables (your GSO_CATALOG)
SCHEMA = "genie_space_optimizer"       # GSO_SCHEMA (default)
LLM_MODEL = "databricks-claude-sonnet-4-6"  # model for the LLM review tier
WAREHOUSE_ID = ""                      # optional; the job defaults from its own config

# --- Behavior ----------------------------------------------------------------
WAIT_FOR_COMPLETION = True
POLL_SECONDS = 20
TIMEOUT_MINUTES = 90

# COMMAND ----------
# MAGIC %md ## 2. Setup

# COMMAND ----------

import json
import time
import uuid

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
me = w.current_user.me().user_name
print(f"Running as: {me}")

# Resolve the job id by name if not provided.
job_id = BENCHMARK_JOB_ID
if not job_id:
    matches = [j for j in w.jobs.list(name=BENCHMARK_JOB_NAME)]
    if not matches:
        raise ValueError(
            f"No job named '{BENCHMARK_JOB_NAME}'. Set BENCHMARK_JOB_ID explicitly "
            f"(find it in the app's app.yaml as BENCHMARK_JOB_ID, or in Workflows)."
        )
    if len(matches) > 1:
        print(f"⚠ {len(matches)} jobs named '{BENCHMARK_JOB_NAME}'; using the first: {matches[0].job_id}")
    job_id = matches[0].job_id

print(f"benchmark-eval job_id: {job_id}")
print(f"Spaces to evaluate: {len(SPACE_IDS)}")

# COMMAND ----------
# MAGIC %md ## 3. Trigger one benchmark run per space
# MAGIC
# MAGIC Each trigger passes a fresh `run_id` (used to look the results up later). The
# MAGIC job allows concurrent runs (extra ones queue), so all N are kicked off here.

# COMMAND ----------

triggered: list[dict] = []
for sid in SPACE_IDS:
    sid = sid.strip()
    if not sid:
        continue
    benchmark_run_id = uuid.uuid4().hex
    qids = QUESTION_IDS_BY_SPACE.get(sid, [])
    params = {
        "run_id": benchmark_run_id,
        "space_id": sid,
        "question_ids": json.dumps(qids),
        "llm_model": LLM_MODEL,
        "triggered_by": me,
        "catalog": CATALOG,
        "schema": SCHEMA,
        "warehouse_id": WAREHOUSE_ID,
    }
    waiter = w.jobs.run_now(job_id=job_id, job_parameters=params)
    triggered.append(
        {"space_id": sid, "benchmark_run_id": benchmark_run_id, "job_run_id": waiter.run_id}
    )
    scope = f"{len(qids)} question(s)" if qids else "all questions"
    print(f"▸ {sid}  →  job_run_id={waiter.run_id}  run_id={benchmark_run_id}  ({scope})")

print(f"\nTriggered {len(triggered)} run(s).")

# COMMAND ----------
# MAGIC %md ## 4. (Optional) Wait for the runs to finish

# COMMAND ----------

TERMINAL = {"TERMINATED", "SKIPPED", "INTERNAL_ERROR"}

if WAIT_FOR_COMPLETION and triggered:
    deadline = time.time() + TIMEOUT_MINUTES * 60
    pending = {t["job_run_id"] for t in triggered}
    print(f"Waiting for {len(pending)} run(s) (poll every {POLL_SECONDS}s, timeout {TIMEOUT_MINUTES}m)...")
    while pending and time.time() < deadline:
        for rid in list(pending):
            run = w.jobs.get_run(run_id=rid)
            life = run.state.life_cycle_state.value if run.state and run.state.life_cycle_state else "UNKNOWN"
            if life in TERMINAL:
                result = run.state.result_state.value if run.state and run.state.result_state else "UNKNOWN"
                print(f"  ✔ job_run_id={rid}: {life} / {result}")
                pending.discard(rid)
        if pending:
            time.sleep(POLL_SECONDS)
    if pending:
        print(f"⚠ Timed out with {len(pending)} run(s) still not terminal: {pending}")
    else:
        print("All runs terminal.")
else:
    print("Not waiting (WAIT_FOR_COMPLETION=False) — check the summary later once runs finish.")

# COMMAND ----------
# MAGIC %md ## 5. Summary from Unity Catalog
# MAGIC
# MAGIC `accuracy_native` is the raw Genie benchmark score; `accuracy_adjusted` is
# MAGIC after false-negative reclassification (data-equivalent BADs flipped to GOOD).

# COMMAND ----------

run_ids = [t["benchmark_run_id"] for t in triggered]
if not run_ids:
    dbutils.notebook.exit("No runs were triggered.")  # noqa: F821

in_list = ", ".join(f"'{r}'" for r in run_ids)

print("Per-space run summary:")
runs_df = spark.sql(  # noqa: F821
    f"""
    SELECT space_name, space_id, status,
           num_questions, num_good        AS native_good,
           num_final_good AS adjusted_good,
           accuracy_native, accuracy_adjusted,
           (accuracy_adjusted - accuracy_native) AS delta,
           run_id, run_at
    FROM {CATALOG}.{SCHEMA}.benchmark_runs
    WHERE run_id IN ({in_list})
    ORDER BY run_at
    """
)
display(runs_df)  # noqa: F821

# COMMAND ----------

print("Per-question detail (only rows the cascade changed from the native verdict):")
results_df = spark.sql(  # noqa: F821
    f"""
    SELECT space_id, question_id, question,
           result_databricks AS native, result_final AS final, decided_by,
           python_method, llm_verdict, llm_confidence
    FROM {CATALOG}.{SCHEMA}.benchmark_results
    WHERE run_id IN ({in_list})
      AND result_databricks <> result_final
    ORDER BY space_id, question_id
    """
)
display(results_df)  # noqa: F821

# COMMAND ----------
# MAGIC %md
# MAGIC ### Notes
# MAGIC - Re-run cell 5 anytime to refresh the summary (it reads the persisted tables).
# MAGIC - To see every question (not just the reclassified ones), remove the
# MAGIC   `result_databricks <> result_final` filter in the last query.
# MAGIC - `generated_result` / `expected_result` columns hold JSON data samples per
# MAGIC   question if you want to inspect the actual rows.
