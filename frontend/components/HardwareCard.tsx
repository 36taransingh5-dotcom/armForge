import type { HostProfile } from "@/lib/api";

function formatBytes(bytes: number): string {
  const gb = bytes / 1024 ** 3;
  return gb >= 1 ? `${gb.toFixed(1)} GB` : `${(bytes / 1024 ** 2).toFixed(0)} MB`;
}

export function HardwareCard({ host }: { host: HostProfile }) {
  const cpu = host.cpu;
  const topology = cpu.is_heterogeneous ? "heterogeneous" : "uniform";

  return (
    <div className="rounded-lg border border-line p-5">
      <div className="flex items-baseline justify-between gap-4">
        <h2 className="text-base font-semibold">{cpu.model}</h2>
        <span className="text-xs text-muted mono">{cpu.architecture}</span>
      </div>

      <dl className="mt-3 grid grid-cols-2 gap-x-6 gap-y-2 text-sm sm:grid-cols-3">
        <div>
          <dt className="text-xs uppercase tracking-wide text-muted">Cores</dt>
          <dd className="num">
            {cpu.physical_cores} <span className="text-muted">({topology})</span>
          </dd>
        </div>
        <div>
          <dt className="text-xs uppercase tracking-wide text-muted">Memory</dt>
          <dd className="num">{host.total_memory_gb.toFixed(1)} GB</dd>
        </div>
        <div>
          <dt className="text-xs uppercase tracking-wide text-muted">OS</dt>
          <dd>
            {host.os_name} {host.os_release}
          </dd>
        </div>
        {cpu.sve_vector_bits && (
          <div>
            <dt className="text-xs uppercase tracking-wide text-muted">SVE</dt>
            <dd className="num">{cpu.sve_vector_bits}-bit</dd>
          </div>
        )}
        {cpu.sme_vector_bits && (
          <div>
            <dt className="text-xs uppercase tracking-wide text-muted">SME</dt>
            <dd className="num">{cpu.sme_vector_bits}-bit streaming</dd>
          </div>
        )}
      </dl>

      {cpu.clusters.length > 1 && (
        <div className="mt-3 flex flex-wrap gap-2">
          {cpu.clusters.map((c) => (
            <span
              key={c.name}
              className="rounded border border-line px-2 py-0.5 text-xs text-muted"
            >
              {c.physical_cores}× {c.core_name ?? c.name}
              {c.l2_cache_bytes ? ` · L2 ${formatBytes(c.l2_cache_bytes)}` : ""}
            </span>
          ))}
        </div>
      )}

      <div className="mt-3 flex flex-wrap gap-1.5">
        {cpu.features.map((f) => (
          <span
            key={f}
            className="rounded bg-code px-1.5 py-0.5 text-[11px] mono text-muted"
          >
            {f}
          </span>
        ))}
      </div>
    </div>
  );
}
