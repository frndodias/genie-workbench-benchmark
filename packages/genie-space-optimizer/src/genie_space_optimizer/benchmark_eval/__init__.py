"""Shared benchmark-evaluation cascade.

One home for the false-negative detection logic used by BOTH the Genie
Workbench app (``backend/routers/benchmark.py``) and the benchmark job notebook
(``jobs/run_benchmark_eval.py``): the deterministic equivalence cascade
(``sql_exact`` → ``numeric_multiset``) and the LLM 'review' judge, plus the
final-verdict resolution (``result_final`` + ``decided_by``).

Pure logic + injected callables (``run_sql``, and the LLM call is done by the
caller from ``build_judge_messages`` / ``parse_judge_response``) so each caller
supplies its own SQL execution and LLM client.
"""

from genie_space_optimizer.benchmark_eval.cascade import (  # noqa: F401
    JUDGE_SYSTEM_PROMPT,
    build_judge_messages,
    compute_equivalence,
    data_equivalent,
    final_verdict,
    numeric_multiset,
    parse_judge_response,
    resolve_llm_verdict,
    sql_setdiff_equivalent,
)

__all__ = [
    "JUDGE_SYSTEM_PROMPT",
    "build_judge_messages",
    "compute_equivalence",
    "data_equivalent",
    "final_verdict",
    "numeric_multiset",
    "parse_judge_response",
    "resolve_llm_verdict",
    "sql_setdiff_equivalent",
]
