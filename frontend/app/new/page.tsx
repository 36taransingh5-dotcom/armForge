"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import {
  ApiError,
  getModels,
  getRuntimes,
  startOptimize,
  type GGUFModel,
  type RuntimeSpec,
} from "@/lib/api";

const WORKLOADS = [
  { name: "short", label: "Short chat (128p / 64g)", note: "decode-dominated" },
  { name: "long-context", label: "Long context (2048p / 128g)", note: "prefill-dominated" },
  { name: "code", label: "Code completion (512p / 256g)", note: "balanced" },
  { name: "summarize", label: "Summarize (1024p / 128g)", note: "prefill-leaning" },
];

const OBJECTIVES = [
  { value: "best-balance", label: "Best balance" },
  { value: "fastest", label: "Fastest (prefill + decode)" },
  { value: "fastest-prefill", label: "Fastest prefill" },
  { value: "fastest-decode", label: "Fastest decode" },
  { value: "lowest-memory", label: "Lowest memory" },
];

export default function NewOptimization() {
  const router = useRouter();
  const [models, setModels] = useState<GGUFModel[] | null>(null);
  const [runtimes, setRuntimes] = useState<RuntimeSpec[] | null>(null);
  const [selectedModels, setSelectedModels] = useState<Set<string>>(new Set());
  const [selectedVariants, setSelectedVariants] = useState<Set<string>>(new Set());
  const [workload, setWorkload] = useState("long-context");
  const [objective, setObjective] = useState("best-balance");
  const [iterations, setIterations] = useState(5);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([getModels(), getRuntimes()])
      .then(([m, r]) => {
        setModels(m);
        setRuntimes(r);
        if (r.length > 0) {
          const variant = r[0].build_flags.variant;
          if (typeof variant === "string") setSelectedVariants(new Set([variant]));
        }
      })
      .catch((e) =>
        setError(e instanceof ApiError ? e.message : "Could not reach the ArmForge API.")
      );
  }, []);

  function toggle(set: Set<string>, setter: (s: Set<string>) => void, value: string) {
    const next = new Set(set);
    next.has(value) ? next.delete(value) : next.add(value);
    setter(next);
  }

  async function submit() {
    if (selectedModels.size === 0) {
      setError("Select at least one model.");
      return;
    }
    setSubmitting(true);
    setError(null);
    try {
      const { job_id } = await startOptimize({
        model_paths: [...selectedModels],
        workload,
        objective,
        variants: selectedVariants.size > 0 ? [...selectedVariants] : null,
        iterations,
      });
      router.push(`/runs/${job_id}`);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Could not start the sweep.");
      setSubmitting(false);
    }
  }

  if (error && models === null) {
    return (
      <div className="rounded-lg border border-warn bg-warn-bg p-4 text-sm">
        {error}
      </div>
    );
  }

  return (
    <div className="max-w-2xl space-y-8">
      <div>
        <h1 className="text-lg font-semibold">New optimization</h1>
        <p className="mt-1 text-sm text-muted">
          ArmForge will read this machine&apos;s Arm capabilities, build a
          measurement plan, and benchmark it for real — this runs actual
          inference, so it takes a few minutes.
        </p>
      </div>

      <section>
        <h2 className="mb-2 text-sm font-medium">Models</h2>
        {models === null ? (
          <div className="h-16 animate-pulse rounded border border-line bg-code/60" />
        ) : models.length === 0 ? (
          <p className="rounded border border-line p-3 text-sm text-muted">
            No .gguf models found. Set <code>ARMFORGE_MODELS_DIR</code> or place
            models in <code>~/.cache/armforge/models</code>.
          </p>
        ) : (
          <div className="divide-y divide-line rounded border border-line">
            {models.map((m) => (
              <label
                key={m.path}
                className="flex cursor-pointer items-center justify-between gap-3 px-3 py-2.5 text-sm hover:bg-code/40"
              >
                <div className="flex items-center gap-3">
                  <input
                    type="checkbox"
                    checked={selectedModels.has(m.path)}
                    onChange={() => toggle(selectedModels, setSelectedModels, m.path)}
                    className="h-4 w-4"
                  />
                  <div>
                    <div className="font-medium">{m.name}</div>
                    <div className="text-xs text-muted">
                      {m.quantization ?? "unknown quant"} · {m.file_size_gb.toFixed(2)} GB
                      {m.repackable_for_i8mm && " · i8mm-repackable"}
                    </div>
                  </div>
                </div>
              </label>
            ))}
          </div>
        )}
      </section>

      <section>
        <h2 className="mb-2 text-sm font-medium">Runtime builds</h2>
        {runtimes === null ? (
          <div className="h-10 animate-pulse rounded border border-line bg-code/60" />
        ) : runtimes.length === 0 ? (
          <p className="rounded border border-line p-3 text-sm text-muted">
            No llama.cpp builds found. Run <code>scripts/setup-llama-cpp.sh</code>.
          </p>
        ) : (
          <div className="flex flex-wrap gap-2">
            {runtimes.map((r) => {
              const variant = String(r.build_flags.variant ?? r.name);
              const checked = selectedVariants.has(variant);
              return (
                <label
                  key={variant}
                  className={`cursor-pointer rounded border px-3 py-1.5 text-sm ${
                    checked ? "border-accent text-accent" : "border-line text-muted"
                  }`}
                >
                  <input
                    type="checkbox"
                    checked={checked}
                    onChange={() => toggle(selectedVariants, setSelectedVariants, variant)}
                    className="mr-1.5"
                  />
                  {r.label}
                </label>
              );
            })}
          </div>
        )}
      </section>

      <section>
        <h2 className="mb-2 text-sm font-medium">Workload</h2>
        <div className="grid grid-cols-2 gap-2">
          {WORKLOADS.map((w) => (
            <label
              key={w.name}
              className={`cursor-pointer rounded border px-3 py-2 text-sm ${
                workload === w.name ? "border-accent" : "border-line"
              }`}
            >
              <input
                type="radio"
                name="workload"
                className="mr-1.5"
                checked={workload === w.name}
                onChange={() => setWorkload(w.name)}
              />
              {w.label}
              <div className="ml-5 text-xs text-muted">{w.note}</div>
            </label>
          ))}
        </div>
      </section>

      <section className="flex flex-wrap gap-6">
        <div>
          <h2 className="mb-2 text-sm font-medium">Objective</h2>
          <select
            value={objective}
            onChange={(e) => setObjective(e.target.value)}
            className="rounded border border-line bg-transparent px-3 py-1.5 text-sm"
          >
            {OBJECTIVES.map((o) => (
              <option key={o.value} value={o.value}>
                {o.label}
              </option>
            ))}
          </select>
        </div>
        <div>
          <h2 className="mb-2 text-sm font-medium">Repetitions</h2>
          <input
            type="number"
            min={1}
            max={20}
            value={iterations}
            onChange={(e) => setIterations(Number(e.target.value))}
            className="w-20 rounded border border-line bg-transparent px-3 py-1.5 text-sm num"
          />
        </div>
      </section>

      {error && (
        <div className="rounded border border-warn bg-warn-bg p-3 text-sm">{error}</div>
      )}

      <button
        onClick={submit}
        disabled={submitting || selectedModels.size === 0}
        className="rounded bg-accent px-4 py-2 text-sm font-medium text-white disabled:opacity-50"
      >
        {submitting ? "Starting…" : "Run sweep"}
      </button>
    </div>
  );
}
