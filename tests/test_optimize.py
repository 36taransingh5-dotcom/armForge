"""Tests for candidate generation, scoring and recommendation.

Several tests are built from figures actually measured on an Apple M4
(Qwen2.5-0.5B, llama.cpp a94d563). They pin the behaviour that matters: given
the real shape of the data, the engine must reach the conclusion the data
supports.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from armforge.bench.stats import summarize
from armforge.bench.types import BenchConfig, BenchmarkResult, ModelRef, RuntimeSpec, Status
from armforge.bench.workloads import SHORT
from armforge.hardware.types import CoreCluster, CoreKind, CpuProfile
from armforge.optimize import Objective, pareto_frontier, recommend, score_all
from armforge.optimize.candidates import generate, thread_candidates
from armforge.optimize.recommend import pick_baseline

STOCK = RuntimeSpec(
    name="llama.cpp",
    version="a94d563",
    binary_path="/b/cpu/llama-bench",
    build_flags={"variant": "cpu", "kleidiai": False, "accelerate": False},
)
KLEIDI = RuntimeSpec(
    name="llama.cpp",
    version="a94d563",
    binary_path="/b/kle/llama-bench",
    build_flags={"variant": "kleidiai", "kleidiai": True, "accelerate": False},
)

Q4_0 = ModelRef(path="/m/q4_0.gguf", name="q", size_bytes=422782464, quantization="Q4_0")
Q4_K_M = ModelRef(path="/m/q4_k_m.gguf", name="q", size_bytes=485392384, quantization="Q4_K_M")


def make_result(
    host,
    *,
    threads,
    prefill,
    decode,
    model=Q4_0,
    runtime=STOCK,
    memory=1_000_000_000,
    prefill_sd=1.0,
    decode_sd=1.0,
):
    config = BenchConfig(model=model, runtime=runtime, workload=SHORT, threads=threads)
    return BenchmarkResult(
        config=config,
        host=host,
        status=Status.OK,
        prefill_tps=summarize([prefill - prefill_sd, prefill + prefill_sd], "tok/s"),
        decode_tps=summarize([decode - decode_sd, decode + decode_sd], "tok/s"),
        peak_memory_bytes=memory,
    )


#: The real M4 sweep: Q4_0 on the stock CPU build, prefill and decode per
#: thread count. Prefill peaks at 6, decode peaks low, 10 collapses.
M4_Q4_0 = {
    1: (183.78, 65.58),
    2: (372.41, 103.42),
    4: (485.39, 92.00),
    6: (606.36, 80.14),
    8: (572.86, 53.19),
    10: (217.27, 6.73),
}

#: Q4_K_M on the same machine: roughly a third of the prefill throughput.
M4_Q4_K_M = {
    1: (57.93, 41.21),
    2: (101.68, 54.42),
    4: (142.04, 66.89),
    6: (184.16, 61.29),
    8: (163.41, 36.30),
    10: (111.98, 3.79),
}


@pytest.fixture
def m4_results(apple_host):
    results = []
    for threads, (prefill, decode) in M4_Q4_0.items():
        results.append(
            make_result(apple_host, threads=threads, prefill=prefill, decode=decode, model=Q4_0)
        )
    for threads, (prefill, decode) in M4_Q4_K_M.items():
        results.append(
            make_result(
                apple_host, threads=threads, prefill=prefill, decode=decode, model=Q4_K_M
            )
        )
    return results


# -- thread candidate generation -------------------------------------------


def test_heterogeneous_cpu_gets_topology_aware_thread_counts(apple_host):
    counts = [n for n, _ in thread_candidates(apple_host.cpu)]
    assert 1 in counts
    assert 4 in counts, "performance-core count must be tested"
    assert 10 in counts, "nproc must be tested so it can be shown to be worse"
    # The measured optimum for prefill on this CPU was 6 threads; the plan has
    # to contain it or the sweep cannot find it.
    assert 6 in counts


def test_uniform_cpu_gets_a_doubling_ladder(graviton_host):
    counts = [n for n, _ in thread_candidates(graviton_host.cpu)]
    assert counts == sorted(set(counts))
    assert 16 in counts
    assert max(counts) <= graviton_host.cpu.physical_cores


def test_every_thread_candidate_carries_a_reason(apple_host):
    for count, reason in thread_candidates(apple_host.cpu):
        assert reason.strip(), f"{count} threads has no rationale"


def test_thread_counts_never_exceed_physical_cores(apple_host, graviton_host):
    for host in (apple_host, graviton_host):
        counts = [n for n, _ in thread_candidates(host.cpu)]
        assert max(counts) <= host.cpu.physical_cores
        assert min(counts) >= 1


def test_single_core_cpu_yields_exactly_one_candidate():
    cpu = CpuProfile(
        architecture="aarch64",
        model="tiny",
        clusters=(CoreCluster("General", CoreKind.UNIFORM, 1, 1),),
    )
    assert [n for n, _ in thread_candidates(cpu)] == [1]


# -- candidate plan --------------------------------------------------------


def test_plan_prunes_kleidiai_on_a_cpu_without_i8mm_or_sme(tmp_path, graviton_host):
    from tests.helpers import build_gguf

    model = build_gguf(
        tmp_path / "m.gguf",
        metadata={"general.architecture": "llama", "general.file_type": 2},
        tensors=[("w", (16, 16))],
    )
    plain = replace(graviton_host.cpu, features=frozenset({"neon", "dotprod"}))
    host = replace(graviton_host, cpu=plain)

    plan = generate(host, [model], [STOCK, KLEIDI], SHORT)

    assert any("KleidiAI" in p.reason for p in plan.pruned)
    assert all(not c.config.runtime.build_flags.get("kleidiai") for c in plan.candidates)


def test_plan_keeps_kleidiai_when_the_cpu_can_reach_it(tmp_path, apple_host):
    from tests.helpers import build_gguf

    model = build_gguf(
        tmp_path / "m.gguf",
        metadata={"general.architecture": "llama", "general.file_type": 2},
        tensors=[("w", (16, 16))],
    )
    plan = generate(apple_host, [model], [STOCK, KLEIDI], SHORT)
    assert any(c.config.runtime.build_flags.get("kleidiai") for c in plan.candidates)


def test_plan_prunes_a_model_that_would_page(tmp_path, apple_host):
    from tests.helpers import build_gguf

    model = build_gguf(
        tmp_path / "big.gguf",
        metadata={"general.architecture": "llama", "general.file_type": 1},
        tensors=[("w", (16, 16))],
        padding_bytes=200_000,
    )
    cramped = replace(apple_host, available_memory_bytes=1000)

    plan = generate(cramped, [model], [STOCK], SHORT)

    assert plan.candidates == ()
    assert any("paging" in p.reason for p in plan.pruned)


def test_unreadable_model_is_pruned_not_crashed(tmp_path, apple_host):
    bad = tmp_path / "bad.gguf"
    bad.write_bytes(b"NOTGGUF" + b"\x00" * 32)
    plan = generate(apple_host, [bad], [STOCK], SHORT)
    assert plan.candidates == ()
    assert any("unreadable" in p.reason for p in plan.pruned)


def test_every_candidate_explains_the_quantisation_interaction(tmp_path, apple_host):
    from tests.helpers import build_gguf

    model = build_gguf(
        tmp_path / "m.gguf",
        metadata={"general.architecture": "llama", "general.file_type": 2},
        tensors=[("w", (16, 16))],
    )
    plan = generate(apple_host, [model], [STOCK], SHORT)

    assert plan.candidates
    for candidate in plan.candidates:
        assert "SMMLA" in candidate.rationale or "i8mm" in candidate.rationale.lower()


# -- scoring ---------------------------------------------------------------


def test_fastest_prefill_picks_the_prefill_winner(m4_results):
    best = score_all(m4_results, Objective.FASTEST_PREFILL)[0]
    assert best.result.config.threads == 6
    assert best.result.config.model.quantization == "Q4_0"


def test_fastest_decode_picks_a_low_thread_count(m4_results):
    best = score_all(m4_results, Objective.FASTEST_DECODE)[0]
    assert best.result.config.threads == 2
    assert best.result.config.model.quantization == "Q4_0"


def test_scores_expose_their_arithmetic(m4_results):
    best = score_all(m4_results, Objective.BEST_BALANCE)[0]
    assert best.components
    assert 0.0 <= best.total <= 1.0
    for component in best.components:
        assert 0.0 <= component.normalized <= 1.0
        assert component.contribution == pytest.approx(component.normalized * component.weight)


def test_failed_results_are_excluded_from_scoring(apple_host, m4_results):
    broken = BenchmarkResult(
        config=BenchConfig(model=Q4_0, runtime=STOCK, workload=SHORT, threads=4),
        host=apple_host,
        status=Status.FAILED,
        error="boom",
    )
    scores = score_all([*m4_results, broken], Objective.FASTEST)
    assert all(s.result.status is Status.OK for s in scores)


def test_scoring_an_empty_set_returns_nothing():
    assert score_all([], Objective.FASTEST) == []


def test_identical_candidates_do_not_divide_by_zero(apple_host):
    same = [make_result(apple_host, threads=t, prefill=100.0, decode=50.0) for t in (2, 4)]
    scores = score_all(same, Objective.FASTEST)
    assert len(scores) == 2
    assert all(s.total == pytest.approx(1.0) for s in scores)


def test_lowest_memory_prefers_the_smaller_model(apple_host):
    small = make_result(
        apple_host, threads=4, prefill=100, decode=50, model=Q4_0, memory=800_000_000
    )
    large = make_result(
        apple_host, threads=4, prefill=400, decode=90, model=Q4_K_M, memory=1_600_000_000
    )
    best = score_all([small, large], Objective.LOWEST_MEMORY)[0]
    assert best.result.config.model.quantization == "Q4_0"


# -- baseline and recommendation -------------------------------------------


def test_baseline_is_the_nproc_configuration(m4_results, apple_host):
    baseline = pick_baseline(m4_results, apple_host)
    assert baseline.config.threads == 10, "baseline must be what nproc would pick"


def test_baseline_prefers_the_plainest_build(apple_host):
    plain = make_result(apple_host, threads=10, prefill=200, decode=7, runtime=STOCK)
    fancy = make_result(apple_host, threads=10, prefill=250, decode=8, runtime=KLEIDI)
    assert pick_baseline([fancy, plain], apple_host) is plain


def test_recommendation_splits_prefill_and_decode_threads(m4_results, apple_host):
    """The headline behaviour: one config, two thread counts."""
    rec = recommend(m4_results, apple_host, Objective.BEST_BALANCE)

    assert rec is not None
    assert rec.prefill.threads == 6
    assert rec.decode.threads == 2
    assert rec.prefill.threads != rec.decode.threads
    assert any("different resources" in reason for reason in rec.reasons)


def test_deployment_command_expresses_the_split(m4_results, apple_host):
    rec = recommend(m4_results, apple_host, Objective.BEST_BALANCE)
    command = rec.deployment_command
    assert "-t 2" in command
    assert "-tb 6" in command
    assert command.startswith("llama-completion")


def test_improvements_are_measured_against_the_nproc_baseline(m4_results, apple_host):
    rec = recommend(m4_results, apple_host, Objective.BEST_BALANCE)
    # Decode goes from 6.73 tok/s at 10 threads to 103.42 at 2 -- an enormous
    # gain, and precisely the point of the tool.
    assert rec.improvements["decode"] > 1000
    assert rec.improvements["prefill"] > 100


def test_close_results_are_flagged_as_not_decisive(apple_host):
    """A gap swamped by noise must not be presented as a finding."""
    noisy = [
        make_result(apple_host, threads=4, prefill=500, decode=90, prefill_sd=40, decode_sd=40),
        make_result(apple_host, threads=6, prefill=510, decode=92, prefill_sd=40, decode_sd=40),
    ]
    rec = recommend(noisy, apple_host, Objective.FASTEST)
    assert not rec.prefill.decisive
    assert any("noise" in w for w in rec.warnings)


def test_noisy_measurements_produce_a_warning(apple_host):
    noisy = [make_result(apple_host, threads=4, prefill=500, decode=90, decode_sd=30)]
    rec = recommend(noisy, apple_host, Objective.FASTEST)
    assert any("varied by" in w for w in rec.warnings)


def test_recommendation_of_nothing_measurable_is_none(apple_host):
    assert recommend([], apple_host) is None


def test_failures_are_carried_into_the_recommendation(m4_results, apple_host):
    broken = BenchmarkResult(
        config=BenchConfig(model=Q4_0, runtime=STOCK, workload=SHORT, threads=8),
        host=apple_host,
        status=Status.FAILED,
        error="segfault",
    )
    rec = recommend([*m4_results, broken], apple_host)
    assert len(rec.failures) == 1
    assert rec.to_dict()["failures"][0]["error"] == "segfault"


def test_phase_choices_come_from_one_deployable_configuration(m4_results, apple_host):
    """Both thread counts must belong to the same model and runtime."""
    rec = recommend(m4_results, apple_host, Objective.BEST_BALANCE)
    assert rec.prefill.result.config.model.path == rec.decode.result.config.model.path
    assert (
        rec.prefill.result.config.runtime.binary_path
        == rec.decode.result.config.runtime.binary_path
    )


# -- Pareto ----------------------------------------------------------------


def test_pareto_excludes_candidates_beaten_on_every_axis(apple_host):
    good = make_result(apple_host, threads=4, prefill=500, decode=90, memory=800_000_000)
    dominated = make_result(apple_host, threads=8, prefill=300, decode=50, memory=900_000_000)
    frontier = pareto_frontier([good, dominated])
    assert good in frontier
    assert dominated not in frontier


def test_pareto_keeps_a_candidate_that_wins_on_one_axis(apple_host):
    fast = make_result(apple_host, threads=6, prefill=600, decode=80, memory=1_200_000_000)
    small = make_result(apple_host, threads=2, prefill=370, decode=103, memory=700_000_000)
    frontier = pareto_frontier([fast, small])
    assert fast in frontier
    assert small in frontier


def test_trivially_different_memory_does_not_block_domination(apple_host):
    """The bug this tolerance fixes.

    Peak memory barely moves across thread counts of one model -- on a real M4
    sweep it varied under 4%. Comparing exactly left 9 of 12 candidates
    "non-dominated" and made the frontier meaningless.
    """
    strong = make_result(apple_host, threads=4, prefill=584, decode=92, memory=794_000_000)
    weak = make_result(
        apple_host, threads=1, prefill=180, decode=61, memory=768_000_000
    )  # 3% less memory, far slower

    frontier = pareto_frontier([strong, weak])

    assert strong in frontier
    assert weak not in frontier, "a 3% memory edge must not excuse being 3x slower"


def test_a_real_memory_saving_still_earns_a_frontier_place(apple_host):
    fast = make_result(apple_host, threads=4, prefill=584, decode=92, memory=1_600_000_000)
    lean = make_result(
        apple_host, threads=4, prefill=180, decode=61, memory=700_000_000
    )  # 56% less memory
    frontier = pareto_frontier([fast, lean])
    assert fast in frontier
    assert lean in frontier


def test_real_m4_frontier_contains_the_phase_winners(m4_results):
    frontier = pareto_frontier(m4_results)
    threads = {r.config.threads for r in frontier}
    assert 6 in threads, "prefill winner must be on the frontier"
    assert 2 in threads, "decode winner must be on the frontier"
    assert 10 not in threads, "the nproc config is beaten on every axis"
