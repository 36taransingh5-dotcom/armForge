import type { BenchmarkResultDto, MetricStats } from "@/lib/api";

function formatMetric(stats: MetricStats | null): { text: string; noisy: boolean } {
  if (!stats) return { text: "—", noisy: false };
  return {
    text: `${stats.mean.toFixed(1)} ± ${stats.stddev.toFixed(1)}`,
    noisy: stats.relative_stddev > 0.05,
  };
}

function formatBytes(bytes: number | null): string {
  if (!bytes) return "—";
  return `${(bytes / 1024 ** 2).toFixed(0)} MiB`;
}

export function MeasurementsTable({
  results,
  winnerLabel,
}: {
  results: BenchmarkResultDto[];
  winnerLabel?: string;
}) {
  if (results.length === 0) {
    return <p className="text-sm text-muted">No measurements yet.</p>;
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-line text-left text-xs uppercase tracking-wide text-muted">
            <th className="py-2 pr-4 font-medium">Configuration</th>
            <th className="py-2 pr-4 text-right font-medium">Prefill tok/s</th>
            <th className="py-2 pr-4 text-right font-medium">Decode tok/s</th>
            <th className="py-2 text-right font-medium">Peak memory</th>
          </tr>
        </thead>
        <tbody>
          {results.map((r, i) => {
            const isWinner = winnerLabel && r.config.label === winnerLabel;
            if (r.status !== "ok") {
              return (
                <tr key={i} className="border-b border-line/60">
                  <td className="py-2 pr-4">{r.config.label}</td>
                  <td colSpan={3} className="py-2 text-muted italic">
                    {r.status}
                    {r.error ? `: ${r.error}` : ""}
                  </td>
                </tr>
              );
            }
            const prefill = formatMetric(r.metrics.prefill_tps);
            const decode = formatMetric(r.metrics.decode_tps);
            return (
              <tr
                key={i}
                className={`border-b border-line/60 num ${
                  isWinner ? "bg-accent/10" : ""
                }`}
              >
                <td className="py-2 pr-4 font-mono text-[13px]">{r.config.label}</td>
                <td
                  className={`py-2 pr-4 text-right ${prefill.noisy ? "text-warn" : ""}`}
                >
                  {prefill.text}
                  {prefill.noisy && " ⚠"}
                </td>
                <td
                  className={`py-2 pr-4 text-right ${decode.noisy ? "text-warn" : ""}`}
                >
                  {decode.text}
                  {decode.noisy && " ⚠"}
                </td>
                <td className="py-2 text-right">{formatBytes(r.metrics.peak_memory_bytes)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="mt-2 text-xs text-muted">
        ⚠ marks measurements that varied more than 5% between repetitions.
      </p>
    </div>
  );
}
