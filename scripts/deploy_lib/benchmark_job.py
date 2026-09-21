"""SDK/REST-native benchmark-eval job deployment for notebook installs.

A single-task job that runs ``jobs/run_benchmark_eval.py`` (native Genie
Eval-Run + the false-negative cascade — deterministic SQL/multiset then an LLM
review with each benchmark's evaluation_note — persisted to Unity Catalog).

Mirrors ``gso_job.py`` and **reuses** the GSO wheel + jobs notebook directory
(the benchmark notebook lives alongside the GSO task notebooks and imports
``genie_space_optimizer.benchmark_eval`` from the same wheel). The result tables
``<catalog>.<schema>.benchmark_runs`` / ``benchmark_results`` are created by the
notebook itself on first run (``CREATE TABLE IF NOT EXISTS``), so no separate
table provisioning step is needed here.

The app reads ``BENCHMARK_JOB_ID`` (injected into app.yaml by install.py) to
trigger this job from the Benchmark tab.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import InstallConfig
from .gso_job import set_job_permissions, upsert_job
from .workspace_source import upload_source_notebook

BENCHMARK_NOTEBOOK_STEM = "run_benchmark_eval"

# Job parameters the app passes on run-now (question_ids scopes the run; empty
# ⇒ all). catalog/schema/warehouse default from the install config.
BENCHMARK_JOB_PARAMETERS = {
    "run_id": "",
    "space_id": "",
    "question_ids": "[]",
    "llm_model": "",
    "triggered_by": "",
    "catalog": "",
    "schema": "genie_space_optimizer",
    "warehouse_id": "",
}


def _jobs_dir(repo_root: Path) -> Path:
    return repo_root / "packages" / "genie-space-optimizer" / "src" / "genie_space_optimizer" / "jobs"


def benchmark_job_name(cfg: InstallConfig) -> str:
    return f"{cfg.app_name}-benchmark-eval-job"


def build_benchmark_job_settings(cfg: InstallConfig, notebook_path: str, wheel_path: str) -> dict[str, Any]:
    cfg = cfg.normalized()
    params = dict(BENCHMARK_JOB_PARAMETERS)
    params["llm_model"] = cfg.llm_model or ""
    params["catalog"] = cfg.catalog or ""
    params["schema"] = cfg.gso_schema or "genie_space_optimizer"
    params["warehouse_id"] = cfg.warehouse_id or ""
    base_params = {k: f"{{{{job.parameters.{k}}}}}" for k in params}
    return {
        "name": benchmark_job_name(cfg),
        "description": (
            "Genie Workbench benchmark evaluation: native Genie Eval-Run + the "
            "false-negative cascade (deterministic SQL/multiset -> LLM review "
            "with evaluation_note), persisted to Unity Catalog "
            "(<catalog>.<schema>.benchmark_runs / benchmark_results). "
            "Triggered by the app's Benchmark tab."
        ),
        "max_concurrent_runs": 10,
        "queue": {"enabled": True},
        "tags": {
            "app": cfg.app_name,
            "managed-by": "notebook-installer",
            "pattern": "benchmark-eval",
        },
        "parameters": [{"name": name, "default": default} for name, default in params.items()],
        "tasks": [
            {
                "task_key": "benchmark_eval",
                "notebook_task": {
                    "notebook_path": notebook_path,
                    "source": "WORKSPACE",
                    "base_parameters": base_params,
                },
                "environment_key": "default",
                "timeout_seconds": 7200,
                "max_retries": 0,
            }
        ],
        "environments": [
            {
                "environment_key": "default",
                "spec": {"environment_version": "4", "dependencies": [wheel_path]},
            }
        ],
    }


def ensure_benchmark_job(
    w,
    cfg: InstallConfig,
    app_sp_client_id: str,
    deployer_user: str,
    notebooks_path: str,
    wheel_path: str,
) -> int:
    """Upload the benchmark notebook, create/update the job, grant the app SP.

    Reuses the GSO ``notebooks_path`` and ``wheel_path`` (same package). Returns
    the job id, which install.py injects into app.yaml as ``BENCHMARK_JOB_ID``.
    """
    cfg = cfg.normalized()
    repo_root = Path(cfg.repo_root or "").resolve()
    upload_source_notebook(
        w,
        _jobs_dir(repo_root) / f"{BENCHMARK_NOTEBOOK_STEM}.py",
        f"{notebooks_path}/{BENCHMARK_NOTEBOOK_STEM}",
    )
    settings = build_benchmark_job_settings(cfg, f"{notebooks_path}/{BENCHMARK_NOTEBOOK_STEM}", wheel_path)
    job_id = upsert_job(w, settings)
    set_job_permissions(w, job_id, deployer_user, app_sp_client_id)
    return job_id
