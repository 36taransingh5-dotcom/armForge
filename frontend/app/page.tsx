"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import {
  ApiError,
  getActiveJob,
  getHardware,
  listHistoricalRuns,
  type HistoricalRunSummary,
  type HostProfile,
  type RunDto,
} from "@/lib/api";
import { HardwareCard } from "@/components/HardwareCard";

export default function Dashboard() {
  const [host, setHost] = useState<HostProfile | null>(null);
  const [runs, setRuns] = useState<HistoricalRunSummary[] | null>(null);
  const [active, setActive] = useState<RunDto | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        const [h, r, a] = await Promise.all([
          getHardware(),
          listHistoricalRuns(),
          getActiveJob(),
        ]);
        if (cancelled) return;
        setHost(h);
        setRuns(r);
        setActive(a);
        setError(null);
      } catch (e) {
        if (cancelled) return;
        setError(
          e instanceof ApiError
            ? e.message
            : "Could not reach the ArmForge API. Is the backend running?"
        );
      }
    }

    load();
    const interval = setInterval(load, 4000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  if (error) {
    return (
      <div className="rounded-lg border border-warn bg-warn-bg p-4 text-sm">
        <strong>Could not load the dashboard.</strong> {error}
        <div className="mt-2 text-xs text-muted">
          Start it with:{" "}
          <code>PYTHONPATH=backend uvicorn app.main:app --port 8000</code>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {active && active.status !== "done" && active.status !== "error" && (
        <Link
          href={`/runs/${active.id}`}
          className="block rounded-lg border border-accent bg-accent/10 p-4 text-sm hover:bg-accent/15"
        >
          <span className="font-medium text-accent">Sweep in progress</span> —{" "}
          {active.progress.index}/{active.progress.total} · {active.progress.label}
          <span className="ml-2 text-muted">click to watch</span>
        </Link>
      )}

      <section>
        <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-muted">
          This machine
        </h2>
        {host ? <HardwareCard host={host} /> : <SkeletonCard />}
      </section>

      <section>
        <div className="mb-3 flex items-center justify-between">
          <h2 className="text-sm font-semibold uppercase tracking-wide text-muted">
            Runs
          </h2>
          <Link href="/new" className="text-sm text-accent hover:underline">
            + New optimization
          </Link>
        </div>

        {runs === null ? (
          <SkeletonCard />
        ) : runs.length === 0 ? (
          <p className="rounded-lg border border-line p-5 text-sm text-muted">
            No runs yet.{" "}
            <Link href="/new" className="text-accent hover:underline">
              Start one
            </Link>{" "}
            to sweep configurations on this machine.
          </p>
        ) : (
          <div className="overflow-hidden rounded-lg border border-line">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-line bg-code text-left text-xs uppercase tracking-wide text-muted">
                  <th className="px-4 py-2 font-medium">Run</th>
                  <th className="px-4 py-2 font-medium">CPU</th>
                  <th className="px-4 py-2 font-medium">Measured</th>
                  <th className="px-4 py-2 font-medium">Deploy command</th>
                </tr>
              </thead>
              <tbody>
                {runs.map((r) => (
                  <tr key={r.name} className="border-b border-line/60 last:border-0">
                    <td className="px-4 py-2.5">
                      <Link
                        href={`/runs/history-${r.name}`}
                        className="font-medium hover:text-accent"
                      >
                        {r.name}
                      </Link>
                    </td>
                    <td className="px-4 py-2.5 text-muted">{r.cpu_model}</td>
                    <td className="px-4 py-2.5 num">
                      {r.succeeded}
                      {r.failed > 0 && (
                        <span className="text-warn"> ({r.failed} failed)</span>
                      )}
                    </td>
                    <td className="max-w-xs truncate px-4 py-2.5 font-mono text-xs text-muted">
                      {r.deployment_command ?? "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}

function SkeletonCard() {
  return (
    <div className="h-24 animate-pulse rounded-lg border border-line bg-code/60" />
  );
}
