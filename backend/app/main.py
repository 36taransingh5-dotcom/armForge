"""ArmForge API.

A thin FastAPI layer over the armforge Python package. It adds nothing the
CLI does not already do -- hardware detection, model analysis, sweeps,
recommendations, export -- it only exposes the same engine over HTTP so the
web UI can drive it. No endpoint here invents data the package itself could
not produce; the "no fabricated numbers" rule applies to this layer too.
"""

from __future__ import annotations

import io
import json
import os
import zipfile
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

from armforge.analyzer import GGUFError, read_gguf
from armforge.bench.deserialize import DeserializationError, sweep_report
from armforge.bench.llamacpp import discover_runtimes
from armforge.hardware import detect_host
from armforge.optimize import Objective, recommend
from armforge.report import export_package
from armforge.report.charts import line_chart

from .jobs import store

app = FastAPI(title="ArmForge API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

#: Where GGUF models are looked for. Overridable so a deployment can point
#: this somewhere other than the developer's own cache directory.
MODELS_DIR = Path(
    os.environ.get("ARMFORGE_MODELS_DIR", Path.home() / ".cache" / "armforge" / "models")
)

#: The repo's results/ directory, read for historical (already-committed) sweeps.
RESULTS_DIR = Path(
    os.environ.get("ARMFORGE_RESULTS_DIR", Path(__file__).resolve().parents[2] / "results")
)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/hardware")
def hardware() -> dict:
    return detect_host().to_dict()


@app.get("/api/models")
def models() -> list[dict]:
    """GGUF files found in MODELS_DIR, with real header-parsed metadata.

    Files that fail to parse are skipped rather than listed with fabricated
    fields -- the picker should only offer models ArmForge can actually use.
    """
    if not MODELS_DIR.is_dir():
        return []
    out = []
    for path in sorted(MODELS_DIR.glob("*.gguf")):
        try:
            info = read_gguf(path)
        except GGUFError:
            continue
        out.append(info.to_dict())
    return out


@app.get("/api/runtimes")
def runtimes() -> list[dict]:
    return [r.to_dict() for r in discover_runtimes()]


class OptimizeRequest(BaseModel):
    model_paths: list[str]
    workload: str = "long-context"
    objective: str = "best-balance"
    variants: list[str] | None = None
    iterations: int = 5


@app.post("/api/optimize")
def start_optimize(req: OptimizeRequest) -> dict:
    if not req.model_paths:
        raise HTTPException(400, "at least one model_paths entry is required")
    try:
        objective = Objective(req.objective)
    except ValueError as exc:
        options = ", ".join(o.value for o in Objective)
        raise HTTPException(400, f"unknown objective {req.objective!r} ({options})") from exc

    try:
        job = store.start(
            model_paths=req.model_paths,
            workload_name=req.workload,
            objective=objective,
            variants=req.variants,
            iterations=req.iterations,
        )
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc

    return {"job_id": job.id}


@app.get("/api/optimize/active")
def active_job() -> dict | None:
    job = store.active()
    return job.to_dict() if job else None


@app.get("/api/optimize/{job_id}")
def get_job(job_id: str) -> dict:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(404, "no such job")
    return job.to_dict()


@app.get("/api/optimize/{job_id}/chart/{phase}.svg")
def job_chart(job_id: str, phase: Literal["prefill", "decode"]) -> Response:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(404, "no such job")
    svg = _phase_chart_svg(job.results, phase)
    return Response(
        content=svg or "<svg xmlns='http://www.w3.org/2000/svg'/>", media_type="image/svg+xml"
    )


@app.get("/api/optimize/{job_id}/export.zip")
def export_job(job_id: str) -> StreamingResponse:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(404, "no such job")
    if not job.results:
        raise HTTPException(400, "nothing measured yet")

    rec = recommend(job.results, job.host, job.objective)
    if rec is None:
        raise HTTPException(400, "no successful measurements to recommend from")

    from armforge.optimize.sweep import SweepReport

    report = SweepReport(host=job.host, plan=job.plan, results=job.results)
    report.finished_at = job.finished_at

    return _zip_export(report, rec)


@app.get("/api/runs")
def list_historical_runs() -> list[dict]:
    """Committed sweep artifacts in results/, summarised.

    The recommendation is recomputed for each rather than trusted from the
    file, the same rule the `armforge export` command follows: measurements
    are what was observed and never change, but scoring logic can improve
    after a file was written.
    """
    if not RESULTS_DIR.is_dir():
        return []

    out = []
    for path in sorted(RESULTS_DIR.glob("*.json")):
        try:
            raw = json.loads(path.read_text())
            report = sweep_report(raw)
        except (OSError, ValueError, DeserializationError):
            continue

        rec = recommend(report.results, report.host, Objective.BEST_BALANCE)
        out.append(
            {
                "name": path.stem,
                "cpu_model": report.host.cpu.model,
                "succeeded": len(report.succeeded),
                "failed": len(report.failed),
                "finished_at": report.finished_at,
                "deployment_command": rec.deployment_command if rec else None,
            }
        )
    return out


def _load_historical(name: str):
    # `name` comes straight from the URL path; resolve and re-check containment
    # so "../../etc/passwd" cannot escape RESULTS_DIR.
    candidate = (RESULTS_DIR / f"{name}.json").resolve()
    try:
        candidate.relative_to(RESULTS_DIR.resolve())
    except ValueError as exc:
        raise HTTPException(400, "invalid run name") from exc

    if not candidate.is_file():
        raise HTTPException(404, "no such run")
    try:
        raw = json.loads(candidate.read_text())
        return sweep_report(raw)
    except (OSError, ValueError, DeserializationError) as exc:
        raise HTTPException(400, f"could not read {candidate.name}: {exc}") from exc


@app.get("/api/runs/{name}")
def get_historical_run(name: str, objective: str = "best-balance") -> dict:
    report = _load_historical(name)
    try:
        goal = Objective(objective)
    except ValueError as exc:
        options = ", ".join(o.value for o in Objective)
        raise HTTPException(400, f"unknown objective {objective!r} ({options})") from exc

    rec = recommend(report.results, report.host, goal)
    return {
        "id": name,
        "status": "done",
        "host": report.host.to_dict(),
        "objective": goal.value,
        "plan": report.plan.to_dict(),
        "progress": {
            "index": len(report.results),
            "total": len(report.results),
            "label": "",
        },
        "results": [r.to_dict() for r in report.results],
        "recommendation": rec.to_dict() if rec else None,
        "error": None,
        "started_at": report.started_at,
        "finished_at": report.finished_at,
    }


@app.get("/api/runs/{name}/chart/{phase}.svg")
def historical_chart(name: str, phase: Literal["prefill", "decode"]) -> Response:
    report = _load_historical(name)
    svg = _phase_chart_svg(report.results, phase)
    return Response(
        content=svg or "<svg xmlns='http://www.w3.org/2000/svg'/>", media_type="image/svg+xml"
    )


@app.get("/api/runs/{name}/export.zip")
def export_historical(name: str, objective: str = "best-balance") -> StreamingResponse:
    report = _load_historical(name)
    try:
        goal = Objective(objective)
    except ValueError as exc:
        raise HTTPException(400, f"unknown objective {objective!r}") from exc

    rec = recommend(report.results, report.host, goal)
    if rec is None:
        raise HTTPException(400, "no successful measurements to recommend from")
    return _zip_export(report, rec)


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------


def _phase_chart_svg(results, phase: str) -> str:
    """Chart the winning model+runtime's throughput across thread counts.

    Falls back to whatever has been measured so far, since a running job's
    winner is not yet known -- the chart still shows real points as they land.
    """
    ok = [r for r in results if r.ok]
    if not ok:
        return ""

    rec = recommend(ok, ok[0].host, Objective.BEST_BALANCE)

    quant = rec.winner.result.config.model.quantization if rec else None
    variant = rec.winner.result.config.runtime.build_flags.get("variant") if rec else None

    points = []
    for result in ok:
        config = result.config
        if quant and config.model.quantization != quant:
            continue
        if variant and config.runtime.build_flags.get("variant") != variant:
            continue
        stats = result.prefill_tps if phase == "prefill" else result.decode_tps
        if stats:
            points.append((config.threads, stats.mean, stats.stddev))

    return line_chart(
        f"{phase.capitalize()} vs threads",
        {phase: points},
        y_label="tok/s",
        width=430,
        height=270,
    )


def _zip_export(report, rec) -> StreamingResponse:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "armforge-export"
        written = export_package(report, rec, out_dir)

        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for file in written:
                zf.write(file, arcname=file.name)
        buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=armforge-export.zip"},
    )
