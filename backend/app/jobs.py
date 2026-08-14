"""In-memory optimization jobs.

A sweep does real benchmarking -- it shells out to llama-bench and measures
wall-clock throughput -- so two sweeps running at once would contend for the
same CPU and corrupt each other's numbers. This module enforces one job at a
time and gives the API something to poll while a sweep runs in the
background.

State lives in a process-local dict, not a database. This is a local
developer tool: the server, the CPU being benchmarked, and the person reading
the results are the same machine. A restart losing in-flight job state is an
acceptable trade for not adding a database dependency to a benchmark runner.
"""

from __future__ import annotations

import threading
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

from armforge.bench.llamacpp import LlamaCppRunner, discover_runtimes
from armforge.bench.types import BenchmarkResult
from armforge.hardware import detect_host
from armforge.hardware.types import HostProfile
from armforge.optimize import Objective, generate, recommend
from armforge.optimize.candidates import CandidatePlan
from armforge.optimize.sweep import run_sweep

Status = Literal["planning", "running", "done", "error"]


@dataclass
class Job:
    id: str
    host: HostProfile
    status: Status = "planning"
    plan: CandidatePlan | None = None
    progress_index: int = 0
    progress_total: int = 0
    progress_label: str = ""
    results: list[BenchmarkResult] = field(default_factory=list)
    objective: Objective = Objective.BEST_BALANCE
    error: str | None = None
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    finished_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        rec = None
        if self.results:
            rec = recommend(self.results, self.host, self.objective)
        return {
            "id": self.id,
            "status": self.status,
            "host": self.host.to_dict(),
            "objective": self.objective.value,
            "plan": self.plan.to_dict() if self.plan else None,
            "progress": {
                "index": self.progress_index,
                "total": self.progress_total,
                "label": self.progress_label,
            },
            "results": [r.to_dict() for r in self.results],
            "recommendation": rec.to_dict() if rec else None,
            "error": self.error,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


class JobStore:
    """Holds at most one running job at a time, plus history of finished ones."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}
        self._active_id: str | None = None

    def active(self) -> Job | None:
        with self._lock:
            return self._jobs.get(self._active_id) if self._active_id else None

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list_recent(self, limit: int = 20) -> list[Job]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.started_at, reverse=True)
            return jobs[:limit]

    def start(
        self,
        *,
        model_paths: list[str],
        workload_name: str,
        objective: Objective,
        variants: list[str] | None,
        iterations: int,
    ) -> Job:
        with self._lock:
            if self._active_id is not None:
                current = self._jobs.get(self._active_id)
                if current and current.status in ("planning", "running"):
                    raise RuntimeError(
                        "a sweep is already running; only one at a time is "
                        "allowed, since two sweeps would contend for the same "
                        "CPU and corrupt each other's timings"
                    )

            host = detect_host()
            job = Job(id=uuid.uuid4().hex[:12], host=host, objective=objective)
            self._jobs[job.id] = job
            self._active_id = job.id

        thread = threading.Thread(
            target=self._run,
            args=(job, model_paths, workload_name, variants, iterations),
            daemon=True,
        )
        thread.start()
        return job

    def _run(
        self,
        job: Job,
        model_paths: list[str],
        workload_name: str,
        variants: list[str] | None,
        iterations: int,
    ) -> None:
        from armforge.bench import workloads as wl

        try:
            runtimes = discover_runtimes()
            if variants:
                wanted = set(variants)
                runtimes = [r for r in runtimes if r.build_flags.get("variant") in wanted]
            if not runtimes:
                raise RuntimeError("no llama.cpp builds found; run scripts/setup-llama-cpp.sh")

            shape = wl.get(workload_name)
            plan = generate(job.host, model_paths, runtimes, shape, iterations=iterations)
            job.plan = plan
            job.progress_total = len(plan.candidates)

            if not plan.candidates:
                job.status = "done"
                job.finished_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
                return

            job.status = "running"

            def on_progress(index: int, total: int, label: str) -> None:
                job.progress_index = index
                job.progress_total = total
                job.progress_label = label

            def on_result(result: BenchmarkResult) -> None:
                job.results.append(result)

            def runner_factory(candidate) -> LlamaCppRunner:
                return LlamaCppRunner(candidate.config.runtime)

            run_sweep(
                plan,
                job.host,
                on_progress=on_progress,
                on_result=on_result,
                runner_factory=runner_factory,
            )
            job.status = "done"
        except Exception as exc:  # noqa: BLE001 - surfaced to the API, not swallowed
            job.status = "error"
            job.error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        finally:
            job.finished_at = datetime.now(timezone.utc).isoformat(timespec="seconds")


store = JobStore()
