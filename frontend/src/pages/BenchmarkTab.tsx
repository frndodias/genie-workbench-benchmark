/**
 * BenchmarkTab — job-driven, persisted benchmark evaluation + dashboard.
 *
 * You select the benchmarks + model; "Run benchmark" triggers a Databricks Job
 * that runs the native Eval-Run + the false-negative cascade (deterministic
 * SQL/multiset → LLM review with each benchmark's evaluation_note) and persists
 * every tier + a final verdict to Unity Catalog. The tab polls the persisted
 * run, lists run history, and drills down per question
 * (Databricks / Python / LLM / Final + reasoning + note + SQL).
 */
import { useCallback, useEffect, useRef, useState } from "react"
import {
  Play,
  Loader2,
  ChevronDown,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  ShieldCheck,
  Sparkles,
  ExternalLink,
  RefreshCw,
} from "lucide-react"
import { Checkbox } from "@/components/ui/checkbox"
import { ModelPicker } from "@/components/ModelPicker"
import {
  getBenchmarks,
  runBenchmark,
  listBenchmarkRuns,
  getBenchmarkRun,
  getBenchmarkRunResults,
} from "@/lib/api"
import type {
  BenchmarkListResponse,
  BenchmarkRunHeader,
  BenchmarkQuestionResult,
  BenchmarkResultSample,
} from "@/types"

interface BenchmarkTabProps {
  spaceId: string
  spaceUrl?: string | null
}

const POLL_INTERVAL_MS = 5000
const norm = (a?: string | null) => String(a ?? "").trim().toUpperCase()
const num = (v: unknown): number => (v == null || v === "" ? 0 : Number(v) || 0)

export function BenchmarkTab({ spaceId, spaceUrl }: BenchmarkTabProps) {
  const [list, setList] = useState<BenchmarkListResponse | null>(null)
  const [listLoading, setListLoading] = useState(true)
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set())
  const [llmModel, setLlmModel] = useState<string | null>(null)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const [runs, setRuns] = useState<BenchmarkRunHeader[]>([])
  const [selectedRun, setSelectedRun] = useState<BenchmarkRunHeader | null>(null)
  const [results, setResults] = useState<BenchmarkQuestionResult[] | null>(null)
  const [resultsLoading, setResultsLoading] = useState(false)
  const cancelled = useRef(false)

  const refreshRuns = useCallback(async () => {
    try {
      const r = await listBenchmarkRuns(spaceId)
      setRuns(r.runs || [])
    } catch {
      /* table may not exist yet */
    }
  }, [spaceId])

  useEffect(() => {
    let active = true
    getBenchmarks(spaceId)
      .then((r) => {
        if (active) {
          setList(r)
          setSelectedIds(new Set(r.benchmarks.map((b) => b.question_id).filter(Boolean)))
          setListLoading(false)
        }
      })
      .catch(() => active && setListLoading(false))
    refreshRuns()
    return () => {
      active = false
    }
  }, [spaceId, refreshRuns])

  useEffect(() => {
    cancelled.current = false
    return () => {
      cancelled.current = true
    }
  }, [spaceId])

  const openRun = useCallback(async (run: BenchmarkRunHeader) => {
    setSelectedRun(run)
    setResults(null)
    setResultsLoading(true)
    try {
      const r = await getBenchmarkRunResults(run.run_id)
      setResults(r.results || [])
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load run results.")
    } finally {
      setResultsLoading(false)
    }
  }, [])

  const pollRun = useCallback(
    async (runId: string) => {
      const deadline = Date.now() + 15 * 60 * 1000
      while (!cancelled.current && Date.now() < deadline) {
        await new Promise((r) => setTimeout(r, POLL_INTERVAL_MS))
        let header: BenchmarkRunHeader
        try {
          header = await getBenchmarkRun(runId)
        } catch {
          continue
        }
        if (norm(header.status) === "COMPLETED" || norm(header.status) === "FAILED") {
          setRunning(false)
          await refreshRuns()
          if (norm(header.status) === "FAILED") {
            setError(header.error || "Benchmark run failed.")
          } else {
            void openRun(header)
          }
          return
        }
      }
      setRunning(false)
    },
    [refreshRuns, openRun],
  )

  const selectable = (list?.benchmarks ?? []).filter((b) => b.question_id)
  const allSelected = selectable.length > 0 && selectedIds.size === selectable.length
  const selectedCount = selectedIds.size
  const noteById: Record<string, string> = {}
  for (const b of list?.benchmarks ?? []) if (b.question_id && b.evaluation_note) noteById[b.question_id] = b.evaluation_note

  function toggleOne(id: string) {
    setSelectedIds((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }
  function toggleAll() {
    setSelectedIds((prev) => (prev.size === selectable.length ? new Set() : new Set(selectable.map((b) => b.question_id))))
  }

  async function handleRun() {
    setError(null)
    setResults(null)
    setSelectedRun(null)
    setRunning(true)
    try {
      const ids = allSelected ? undefined : Array.from(selectedIds)
      const { run_id } = await runBenchmark(spaceId, { benchmark_question_ids: ids, llm_model: llmModel })
      await refreshRuns()
      void pollRun(run_id)
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to start the benchmark job.")
      setRunning(false)
    }
  }

  const benchmarkCount = list?.count ?? 0
  const hasBenchmarks = benchmarkCount > 0
  const jobConfigured = list?.job_configured !== false

  return (
    <div className="space-y-6">
      {/* Config + run */}
      <div className="rounded-xl border border-default bg-surface p-5">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div className="max-w-2xl">
            <h3 className="text-sm font-semibold text-primary">Benchmark evaluation</h3>
            <p className="mt-1 text-xs leading-relaxed text-muted">
              Runs the Agent's native Genie benchmark as a Databricks job and persists every result to
              Unity Catalog. Each failed question is re-checked by a deterministic data comparison and,
              if needed, an LLM review that follows the benchmark's evaluation note — the final verdict
              and an adjusted accuracy are stored for audit.
            </p>
            {!listLoading && (
              <p className="mt-2 text-xs text-muted">
                {hasBenchmarks ? (
                  <>
                    <span className="font-semibold text-primary">{benchmarkCount}</span> benchmark
                    {benchmarkCount === 1 ? "" : "s"} configured
                  </>
                ) : (
                  <>No benchmarks configured on this Agent.</>
                )}
                {spaceUrl && (
                  <a href={spaceUrl} target="_blank" rel="noopener noreferrer" className="ml-2 inline-flex items-center gap-1 text-accent hover:underline">
                    Configure in Genie <ExternalLink className="h-3 w-3" />
                  </a>
                )}
              </p>
            )}
            {!jobConfigured && (
              <p className="mt-2 text-xs text-amber-600 dark:text-amber-400">
                Benchmark job not configured (BENCHMARK_JOB_ID) — runs are disabled.
              </p>
            )}
          </div>
          <button
            onClick={handleRun}
            disabled={running || listLoading || !hasBenchmarks || selectedCount === 0 || !jobConfigured}
            className="flex items-center gap-2 rounded-lg bg-accent px-4 py-2.5 text-sm font-semibold text-white transition-colors hover:bg-accent/90 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {running ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
            {running ? "Running job…" : `Run benchmark${selectedCount ? ` (${selectedCount})` : ""}`}
          </button>
        </div>

        <div className="mt-4 max-w-xs">
          <ModelPicker value={llmModel} onChange={setLlmModel} disabled={running} label="Model for LLM review"
            helper="Used by the job's LLM review on a failed row." />
        </div>

        {running && (
          <div className="mt-4 flex items-center gap-3 rounded-lg border border-blue-500/30 bg-blue-500/5 px-4 py-3 text-xs text-muted">
            <Loader2 className="h-4 w-4 animate-spin text-blue-400" />
            Benchmark job running — this takes a few minutes. Results persist automatically.
          </div>
        )}
        {error && <div className="mt-4 rounded-lg border border-danger/20 bg-danger/10 px-4 py-3 text-sm text-danger">{error}</div>}
      </div>

      {/* Selection */}
      {hasBenchmarks && (
        <div className="rounded-xl border border-default bg-surface p-4">
          <div className="flex items-center justify-between border-b border-default pb-2">
            <label className="flex cursor-pointer items-center gap-2 text-xs font-semibold text-primary">
              <Checkbox checked={allSelected} onCheckedChange={toggleAll} disabled={running} />
              Select all
            </label>
            <span className="text-[11px] text-muted">{selectedCount} of {selectable.length} selected</span>
          </div>
          <div className="mt-2 max-h-64 space-y-0.5 overflow-auto">
            {(list?.benchmarks ?? []).map((b, i) => {
              const hasId = Boolean(b.question_id)
              return (
                <label key={b.question_id || i} className={`flex items-start gap-2 rounded px-2 py-1.5 text-xs ${hasId ? "cursor-pointer hover:bg-elevated/60" : "opacity-50"}`}>
                  <Checkbox checked={b.question_id ? selectedIds.has(b.question_id) : false} onCheckedChange={() => b.question_id && toggleOne(b.question_id)} disabled={running || !hasId} className="mt-0.5" />
                  <span className="text-primary">
                    {b.question || b.question_id || "(untitled)"}
                    {b.evaluation_note && (
                      <span className="ml-1.5 rounded-full border border-cyan-500/30 bg-cyan-500/10 px-1.5 py-0.5 text-[9px] font-medium text-cyan-600 dark:text-cyan-400" title={b.evaluation_note}>note</span>
                    )}
                  </span>
                </label>
              )
            })}
          </div>
        </div>
      )}

      {/* Runs dashboard */}
      <div className="rounded-xl border border-default bg-surface p-4">
        <div className="mb-2 flex items-center justify-between">
          <h3 className="text-sm font-semibold text-primary">Run history</h3>
          <button onClick={refreshRuns} className="flex items-center gap-1 text-[11px] text-muted hover:text-accent"><RefreshCw className="h-3 w-3" /> Refresh</button>
        </div>
        {runs.length === 0 ? (
          <p className="py-4 text-center text-xs text-muted">No runs yet. Select benchmarks and run.</p>
        ) : (
          <div className="overflow-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="text-[10px] uppercase tracking-wider text-muted">
                  <th className="px-2 py-1">When</th><th className="px-2 py-1">Who</th><th className="px-2 py-1">Model</th>
                  <th className="px-2 py-1">Status</th><th className="px-2 py-1">Native</th><th className="px-2 py-1">Adjusted</th>
                  <th className="px-2 py-1">Good/Bad/NR</th><th className="px-2 py-1"></th>
                </tr>
              </thead>
              <tbody>
                {runs.map((r) => (
                  <tr key={r.run_id} className={`border-t border-default ${selectedRun?.run_id === r.run_id ? "bg-elevated/40" : ""}`}>
                    <td className="px-2 py-1.5 text-muted">{r.run_at ? new Date(r.run_at).toLocaleString() : "—"}</td>
                    <td className="px-2 py-1.5 text-muted">{(r.triggered_by || "—").split("@")[0]}</td>
                    <td className="px-2 py-1.5 text-muted">{(r.llm_model || "—").replace("databricks-", "")}</td>
                    <td className="px-2 py-1.5"><StatusPill status={r.status} /></td>
                    <td className="px-2 py-1.5 text-muted">{num(r.accuracy_native)}%</td>
                    <td className="px-2 py-1.5 font-semibold text-emerald-600 dark:text-emerald-400">{num(r.accuracy_adjusted)}%</td>
                    <td className="px-2 py-1.5 text-muted">{num(r.num_final_good)}/{num(r.num_bad)}/{num(r.num_needs_review)}</td>
                    <td className="px-2 py-1.5"><button onClick={() => openRun(r)} className="text-accent hover:underline">Details</button></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Drill-down */}
      {selectedRun && (
        <div className="rounded-xl border border-default bg-surface p-4">
          <div className="mb-3 flex items-center justify-between">
            <h3 className="text-sm font-semibold text-primary">
              Run details <span className="ml-1 font-mono text-[11px] text-muted">{selectedRun.run_id.slice(0, 8)}</span>
            </h3>
            <span className="text-[11px] text-muted">
              native {num(selectedRun.accuracy_native)}% → adjusted <span className="font-semibold text-emerald-600 dark:text-emerald-400">{num(selectedRun.accuracy_adjusted)}%</span>
            </span>
          </div>
          {resultsLoading ? (
            <p className="py-4 text-center text-xs text-muted"><Loader2 className="inline h-4 w-4 animate-spin" /> Loading…</p>
          ) : !results || results.length === 0 ? (
            <p className="py-4 text-center text-xs text-muted">No per-question results (run may still be writing).</p>
          ) : (
            <div className="space-y-2">
              {results.map((row, i) => <QuestionResultRow key={`${row.question_id}-${i}`} row={row} note={noteById[row.question_id]} />)}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function StatusPill({ status }: { status: string }) {
  const s = norm(status)
  const cls = s === "COMPLETED" ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-600 dark:text-emerald-400"
    : s === "FAILED" ? "border-red-500/30 bg-red-500/10 text-red-600 dark:text-red-400"
    : "border-blue-500/30 bg-blue-500/10 text-blue-600 dark:text-blue-400"
  return <span className={`rounded-full border px-2 py-0.5 text-[10px] font-medium ${cls}`}>{s || "RUNNING"}</span>
}

function verdictBadge(v: string) {
  const s = norm(v)
  if (s === "GOOD") return { cls: "border-emerald-500/30 bg-emerald-500/10 text-emerald-600 dark:text-emerald-400", icon: CheckCircle2 }
  if (s === "NEEDS_REVIEW") return { cls: "border-amber-500/30 bg-amber-500/10 text-amber-600 dark:text-amber-400", icon: AlertTriangle }
  return { cls: "border-red-500/30 bg-red-500/10 text-red-600 dark:text-red-400", icon: XCircle }
}

function QuestionResultRow({ row, note }: { row: BenchmarkQuestionResult; note?: string }) {
  const [open, setOpen] = useState(false)
  const finalBadge = verdictBadge(row.result_final)
  const FinalIcon = finalBadge.icon
  const reasons = row.assessment_reasons ?? []
  const decidedBy = row.decided_by
  return (
    <div className={`rounded-lg border px-3 py-2.5 text-xs ${norm(row.result_final) === "GOOD" && norm(row.result_databricks) !== "GOOD" ? "border-emerald-500/30 bg-emerald-500/5" : "border-default bg-elevated/30"}`}>
      <div className="flex flex-wrap items-start justify-between gap-2">
        <span className="font-medium text-primary">{row.question || row.question_id}</span>
        <span className={`inline-flex shrink-0 items-center gap-1 rounded-full border px-2 py-0.5 text-[10px] font-medium ${finalBadge.cls}`}>
          <FinalIcon className="h-3 w-3" />
          {norm(row.result_final)}{decidedBy ? ` · ${decidedBy}` : ""}
        </span>
      </div>

      {/* tier chips */}
      <div className="mt-1.5 flex flex-wrap items-center gap-1.5 text-[10px]">
        <span className="rounded-full border border-default bg-surface px-2 py-0.5 text-muted">Databricks: {norm(row.result_databricks)}</span>
        {row.python_method && row.python_method !== "none" && (
          <span className="rounded-full border border-default bg-surface px-2 py-0.5 text-muted">Python: {row.python_equivalent ? "equivalent" : "differs"} ({row.python_method})</span>
        )}
        {row.llm_verdict && (
          <span className="inline-flex items-center gap-1 rounded-full border border-purple-500/30 bg-purple-500/10 px-2 py-0.5 text-purple-600 dark:text-purple-400"><Sparkles className="h-2.5 w-2.5" />LLM: {row.llm_verdict}</span>
        )}
        {note && <span className="rounded-full border border-cyan-500/30 bg-cyan-500/10 px-2 py-0.5 text-cyan-600 dark:text-cyan-400" title={note}>note</span>}
      </div>

      {reasons.length > 0 && (
        <div className="mt-1.5 flex flex-wrap gap-1">
          {reasons.map((r, i) => <span key={i} className="rounded-full border border-default bg-surface px-2 py-0.5 text-[10px] capitalize text-muted">{String(r).toLowerCase().replaceAll("_", " ")}</span>)}
        </div>
      )}

      {norm(row.result_final) === "GOOD" && norm(row.result_databricks) !== "GOOD" && (
        <p className="mt-1.5 flex items-center gap-1 text-[11px] text-emerald-700 dark:text-emerald-300">
          <ShieldCheck className="h-3 w-3" /> Scored {norm(row.result_databricks)} natively, reclassified GOOD by {decidedBy}. {row.python_detail}
        </p>
      )}
      {row.llm_verdict && (
        <p className="mt-1.5 text-[11px] leading-relaxed text-secondary">
          <span className="font-medium text-purple-600 dark:text-purple-400">LLM ({row.llm_verdict}):</span>{" "}
          {row.llm_reasoning || <span className="text-muted">(no reasoning returned)</span>}
        </p>
      )}

      <button onClick={() => setOpen((v) => !v)} className="mt-2 flex items-center gap-1 text-[10px] font-medium text-muted hover:text-primary">
        <ChevronDown className={`h-3 w-3 transition-transform ${open ? "rotate-180" : ""}`} /> {open ? "Hide SQL & data" : "Show SQL & data"}
      </button>
      {open && (
        <div className="mt-2 space-y-3">
          <div className="grid gap-2 md:grid-cols-2">
            <div>
              <p className="text-[10px] font-semibold uppercase tracking-wider text-muted">Generated SQL</p>
              <pre className="mt-1 max-h-40 overflow-auto whitespace-pre-wrap rounded bg-elevated p-2 font-mono text-[10px] text-primary">{row.generated_sql || "—"}</pre>
            </div>
            <div>
              <p className="text-[10px] font-semibold uppercase tracking-wider text-cyan-600 dark:text-cyan-400">Expected SQL</p>
              <pre className="mt-1 max-h-40 overflow-auto whitespace-pre-wrap rounded bg-cyan-500/5 p-2 font-mono text-[10px] text-primary ring-1 ring-cyan-500/20">{row.expected_sql || "—"}</pre>
            </div>
          </div>
          {(row.generated_result || row.expected_result) && (
            <div className="grid gap-2 md:grid-cols-2">
              <SampleTable label="Obtido (generated)" sample={row.generated_result} />
              <SampleTable label="Esperado (ground truth)" sample={row.expected_result} accent />
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function SampleTable({ label, sample, accent }: { label: string; sample: BenchmarkResultSample | null; accent?: boolean }) {
  return (
    <div>
      <p className={`text-[10px] font-semibold uppercase tracking-wider ${accent ? "text-cyan-600 dark:text-cyan-400" : "text-muted"}`}>{label}</p>
      <div className={`mt-1 overflow-auto rounded border border-default ${accent ? "bg-cyan-500/5" : "bg-elevated"}`}>
        {!sample ? (
          <p className="px-2 py-3 text-center text-[10px] text-muted">No data captured</p>
        ) : sample.error ? (
          <p className="px-2 py-3 text-[10px] text-danger">{sample.error}</p>
        ) : (
          <table className="w-full border-collapse text-[10px]">
            <thead>
              <tr>{sample.columns.map((c, i) => <th key={i} className="border-b border-default px-2 py-1 text-left font-semibold text-secondary">{c}</th>)}</tr>
            </thead>
            <tbody>
              {sample.data.slice(0, 20).map((r, ri) => (
                <tr key={ri} className="odd:bg-surface/40">
                  {(r as unknown[]).map((cell, ci) => <td key={ci} className="whitespace-nowrap px-2 py-1 font-mono text-primary">{cell == null ? "∅" : String(cell)}</td>)}
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
      {sample && !sample.error && <p className="mt-1 text-[10px] text-muted">{sample.data.length} row{sample.data.length === 1 ? "" : "s"}{sample.data.length >= 20 ? " (sample)" : ""}</p>}
    </div>
  )
}
