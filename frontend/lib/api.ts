/**
 * Typed client for the ArmForge API.
 *
 * Every function here calls a real backend endpoint that runs the actual
 * armforge Python engine -- there is no mock mode. If the backend is down,
 * calls fail loudly rather than falling back to placeholder data.
 */

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
    cache: "no-store",
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new ApiError(res.status, body.detail ?? res.statusText);
  }
  if (res.status === 204) return undefined as T;
  return res.json();
}

// -- hardware ----------------------------------------------------------

export interface CoreCluster {
  name: string;
  kind: "performance" | "efficiency" | "uniform" | "unknown";
  physical_cores: number;
  logical_cores: number;
  max_freq_mhz: number | null;
  l2_cache_bytes: number | null;
  core_name: string | null;
}

export interface CpuProfile {
  architecture: string;
  is_arm64: boolean;
  model: string;
  implementer: string | null;
  physical_cores: number;
  logical_cores: number;
  performance_cores: number;
  is_heterogeneous: boolean;
  clusters: CoreCluster[];
  features: string[];
  sve_vector_bits: number | null;
  sme_vector_bits: number | null;
}

export interface HostProfile {
  cpu: CpuProfile;
  os_name: string;
  os_release: string;
  total_memory_bytes: number;
  total_memory_gb: number;
  available_memory_bytes: number | null;
  detector: string;
  warnings: string[];
}

export const getHardware = () => request<HostProfile>("/api/hardware");

// -- models --------------------------------------------------------------

export interface GGUFModel {
  path: string;
  name: string;
  architecture: string | null;
  quantization: string | null;
  file_size_bytes: number;
  file_size_gb: number;
  parameter_count: number | null;
  context_length: number | null;
  repackable_for_i8mm: boolean;
}

export const getModels = () => request<GGUFModel[]>("/api/models");

// -- runtimes --------------------------------------------------------------

export interface RuntimeSpec {
  name: string;
  label: string;
  version: string;
  binary_path: string;
  build_flags: Record<string, unknown>;
}

export const getRuntimes = () => request<RuntimeSpec[]>("/api/runtimes");

// -- optimize / results ----------------------------------------------------

export interface MetricStats {
  mean: number;
  median: number;
  min: number;
  max: number;
  stddev: number;
  relative_stddev: number;
  samples: number;
  unit: string;
}

export interface BenchmarkResultDto {
  status: "ok" | "unsupported" | "skipped" | "failed" | "timeout";
  config: {
    label: string;
    model: { quantization: string | null; size_bytes: number; n_params: number | null };
    runtime: { label: string; build_flags: Record<string, unknown> };
    workload: { name: string; prompt_tokens: number; generate_tokens: number };
    threads: number;
    iterations: number;
  };
  metrics: {
    prefill_tps: MetricStats | null;
    decode_tps: MetricStats | null;
    ttft_ms: number | null;
    peak_memory_bytes: number | null;
    wall_time_s: number | null;
  };
  error: string | null;
}

export interface PhaseChoiceDto {
  phase: string;
  threads: number;
  throughput_tok_s: number;
  stddev: number;
  decisive: boolean;
  runner_up_threads: number | null;
}

export interface RecommendationDto {
  objective: string;
  winner: { label: string; total: number };
  winning_config: BenchmarkResultDto["config"];
  prefill: PhaseChoiceDto | null;
  decode: PhaseChoiceDto | null;
  baseline: string | null;
  improvements_vs_baseline_pct: Record<string, number | null>;
  deployment_command: string;
  reasons: string[];
  warnings: string[];
  pareto: string[];
  failures: { label: string; status: string; error: string | null }[];
}

export interface PrunedDto {
  label: string;
  reason: string;
}

export interface CandidatePlanDto {
  candidates: { label: string; rationale: string }[];
  pruned: PrunedDto[];
  notes: string[];
}

export interface RunDto {
  id: string;
  status: "planning" | "running" | "done" | "error";
  host: HostProfile;
  objective: string;
  plan: CandidatePlanDto | null;
  progress: { index: number; total: number; label: string };
  results: BenchmarkResultDto[];
  recommendation: RecommendationDto | null;
  error: string | null;
  started_at: string;
  finished_at: string | null;
}

export interface OptimizeRequest {
  model_paths: string[];
  workload: string;
  objective: string;
  variants: string[] | null;
  iterations: number;
}

export const startOptimize = (req: OptimizeRequest) =>
  request<{ job_id: string }>("/api/optimize", {
    method: "POST",
    body: JSON.stringify(req),
  });

export const getActiveJob = () => request<RunDto | null>("/api/optimize/active");

export const getJob = (id: string) => request<RunDto>(`/api/optimize/${id}`);

export const jobChartUrl = (id: string, phase: "prefill" | "decode") =>
  `${API_BASE}/api/optimize/${id}/chart/${phase}.svg`;

export const jobExportUrl = (id: string) => `${API_BASE}/api/optimize/${id}/export.zip`;

// -- historical runs ---------------------------------------------------

export interface HistoricalRunSummary {
  name: string;
  cpu_model: string;
  succeeded: number;
  failed: number;
  finished_at: string | null;
  deployment_command: string | null;
}

export const listHistoricalRuns = () => request<HistoricalRunSummary[]>("/api/runs");

export const getHistoricalRun = (name: string) => request<RunDto>(`/api/runs/${name}`);

export const historicalChartUrl = (name: string, phase: "prefill" | "decode") =>
  `${API_BASE}/api/runs/${name}/chart/${phase}.svg`;

export const historicalExportUrl = (name: string) => `${API_BASE}/api/runs/${name}/export.zip`;
