"use client";

import { use, useEffect, useState } from "react";
import {
  ApiError,
  getHistoricalRun,
  getJob,
  historicalChartUrl,
  historicalExportUrl,
  jobChartUrl,
  jobExportUrl,
  type RunDto,
} from "@/lib/api";
import { HardwareCard } from "@/components/HardwareCard";
import { MeasurementsTable } from "@/components/MeasurementsTable";
import { RecommendationCard } from "@/components/RecommendationCard";

/** IDs like "history-m4-sweep-longcontext" route to /api/runs/{name} instead
 * of /api/optimize/{job_id}, since the two live in separate stores on the
 * backend (an in-memory job vs. a committed artifact on disk). */
function parseId(raw: string): { historical: boolean; key: string } {
  if (raw.startsWith("history-")) {
    return { historical: true, key: raw.slice("history-".length) };
  }
  return { historical: false, key: raw };
}

export default function RunPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const { historical, key } = parseId(id);

  const [run, setRun] = useState<RunDto | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    let interval: ReturnType<typeof setInterval> | undefined;

    async function poll() {
      try {
        const data = historical ? await getHistoricalRun(key) : await getJob(key);
        if (cancelled) return;
        setRun(data);
        setError(null);
        // Historical runs are already finished; live jobs stop polling once done.
        if (historical || data.status === "done" || data.status === "error") {
          if (interval) clearInterval(interval);
        }
      } catch (e) {
        if (cancelled) return;
        setError(e instanceof ApiError ? e.message : "Could not load this run.");
        if (interval) clearInterval(interval);
      }
    }

    poll();
    if (!historical) interval = setInterval(poll, 1500);

    return () => {
      cancelled = true;
      if (interval) clearInterval(interval);
    };
  }, [historical, key]);

  if (error) {
    return (
      <div className="rounded-lg border border-warn bg-warn-bg p-4 text-sm">{error}</div>
    );
  }

  if (!run) {
    return <div className="h-32 animate-pulse rounded-lg border border-line bg-code/60" />;
  }

  const chartUrls = run.results.some((r) => r.status === "ok")
    ? {
        prefill: historical ? historicalChartUrl(key, "prefill") : jobChartUrl(key, "prefill"),
        decode: historical ? historicalChartUrl(key, "decode") : jobChartUrl(key, "decode"),
      }
    : undefined;
  const exportUrl = historical ? historicalExportUrl(key) : jobExportUrl(key);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="font-mono text-sm text-muted">{id}</h1>
        <StatusBadge status={run.status} />
      </div>

      <HardwareCard host={run.host} />

      {run.status === "planning" && (
        <div className="rounded-lg border border-line p-5 text-sm text-muted">
          Building the measurement plan from this CPU&apos;s capabilities…
        </div>
      )}

      {run.plan && run.plan.notes.length > 0 && (
        <div className="space-y-1 text-sm text-muted">
          {run.plan.notes.map((n, i) => (
            <p key={i}>· {n}</p>
          ))}
        </div>
      )}

      {(run.status === "running" || run.status === "planning") && (
        <div className="rounded-lg border border-line p-5">
          <div className="flex items-center justify-between text-sm">
            <span className="font-medium">
              {run.progress.total > 0
                ? `${run.progress.index}/${run.progress.total} — ${run.progress.label}`
                : "Preparing…"}
            </span>
            <span className="text-muted">
              {run.results.length} measured
            </span>
          </div>
          {run.progress.total > 0 && (
            <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-code">
              <div
                className="h-full bg-accent transition-all"
                style={{
                  width: `${(run.progress.index / run.progress.total) * 100}%`,
                }}
              />
            </div>
          )}
        </div>
      )}

      {run.status === "error" && (
        <div className="rounded-lg border border-warn bg-warn-bg p-4 text-sm">
          <strong>Sweep failed.</strong>
          <pre className="mt-2 overflow-x-auto whitespace-pre-wrap text-xs">{run.error}</pre>
        </div>
      )}

      {run.plan && run.plan.pruned.length > 0 && (
        <details className="rounded-lg border border-line p-4 text-sm">
          <summary className="cursor-pointer font-medium text-muted">
            {run.plan.pruned.length} configuration
            {run.plan.pruned.length === 1 ? "" : "s"} not measured, and why
          </summary>
          <ul className="mt-2 space-y-1">
            {run.plan.pruned.map((p, i) => (
              <li key={i}>
                <span className="font-medium">{p.label}</span> — {p.reason}
              </li>
            ))}
          </ul>
        </details>
      )}

      {run.recommendation && (
        <RecommendationCard rec={run.recommendation} chartUrls={chartUrls} exportUrl={exportUrl} />
      )}

      {run.results.length > 0 && (
        <div>
          <h2 className="mb-2 text-sm font-semibold uppercase tracking-wide text-muted">
            All measurements
          </h2>
          <MeasurementsTable
            results={run.results}
            winnerLabel={run.recommendation?.winner.label}
          />
        </div>
      )}
    </div>
  );
}

function StatusBadge({ status }: { status: RunDto["status"] }) {
  const styles: Record<RunDto["status"], string> = {
    planning: "text-muted border-line",
    running: "text-accent border-accent",
    done: "text-ok border-ok",
    error: "text-bad border-bad",
  };
  return (
    <span className={`rounded-full border px-2.5 py-0.5 text-xs font-medium ${styles[status]}`}>
      {status}
    </span>
  );
}
