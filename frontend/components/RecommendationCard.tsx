import type { RecommendationDto } from "@/lib/api";

export function RecommendationCard({
  rec,
  chartUrls,
  exportUrl,
}: {
  rec: RecommendationDto;
  chartUrls?: { prefill: string; decode: string };
  exportUrl?: string;
}) {
  return (
    <div className="rounded-lg border border-line p-5">
      <div className="flex items-center justify-between">
        <h2 className="text-base font-semibold">Recommended configuration</h2>
        <span className="text-xs text-muted mono">{rec.objective}</span>
      </div>

      <div className="mt-4 grid grid-cols-2 gap-4 sm:grid-cols-4">
        <div>
          <div className="text-xs uppercase tracking-wide text-muted">Model</div>
          <div className="mt-0.5 font-semibold">
            {rec.winning_config.model.quantization ?? "unknown"}
          </div>
        </div>
        <div>
          <div className="text-xs uppercase tracking-wide text-muted">Runtime</div>
          <div className="mt-0.5 font-semibold">{rec.winning_config.runtime.label}</div>
        </div>
        {rec.prefill && (
          <div>
            <div className="text-xs uppercase tracking-wide text-muted">
              Prefill threads
            </div>
            <div className="mt-0.5 num">
              <span className="font-semibold">{rec.prefill.threads}</span>{" "}
              <span className="text-xs text-muted">
                {rec.prefill.throughput_tok_s.toFixed(1)} tok/s
              </span>
            </div>
          </div>
        )}
        {rec.decode && (
          <div>
            <div className="text-xs uppercase tracking-wide text-muted">
              Decode threads
            </div>
            <div className="mt-0.5 num">
              <span className="font-semibold">{rec.decode.threads}</span>{" "}
              <span className="text-xs text-muted">
                {rec.decode.throughput_tok_s.toFixed(1)} tok/s
              </span>
            </div>
          </div>
        )}
      </div>

      {rec.reasons.length > 0 && (
        <div className="mt-4">
          <h3 className="text-xs uppercase tracking-wide text-muted">Why</h3>
          <ul className="mt-1.5 space-y-1 text-sm">
            {rec.reasons.map((r, i) => (
              <li key={i} className="flex gap-2">
                <span className="text-ok">·</span>
                <span>{r}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {rec.baseline && (
        <p className="mt-3 text-xs text-muted">
          Compared against the naive default: {rec.baseline} — what <code>nproc</code>{" "}
          and an unmodified build would give you.
        </p>
      )}

      {rec.warnings.length > 0 && (
        <div className="mt-4 space-y-1.5">
          <h3 className="text-xs uppercase tracking-wide text-muted">Caveats</h3>
          {rec.warnings.map((w, i) => (
            <div
              key={i}
              className="rounded border-l-2 border-warn bg-warn-bg px-3 py-1.5 text-sm"
            >
              {w}
            </div>
          ))}
        </div>
      )}

      {chartUrls && (
        <div className="mt-5 flex flex-wrap gap-4">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={chartUrls.prefill}
            alt="Prefill throughput vs threads"
            className="max-w-full rounded border border-line"
          />
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={chartUrls.decode}
            alt="Decode throughput vs threads"
            className="max-w-full rounded border border-line"
          />
        </div>
      )}

      <div className="mt-5">
        <h3 className="text-xs uppercase tracking-wide text-muted">Deploy</h3>
        <pre className="mt-1.5 overflow-x-auto rounded border border-line bg-code p-3 text-[13px]">
          {rec.deployment_command}
        </pre>
      </div>

      {exportUrl && (
        <a
          href={exportUrl}
          download
          className="mt-4 inline-flex items-center gap-1.5 rounded border border-line px-3 py-1.5 text-sm font-medium hover:border-accent hover:text-accent"
        >
          Download deployment package (.zip)
        </a>
      )}
    </div>
  );
}
