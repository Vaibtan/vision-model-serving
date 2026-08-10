"""GPU measurement adapters for the L4 optimization campaign."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import nullcontext
from dataclasses import dataclass
import gc
import math
from pathlib import Path
import statistics
from time import perf_counter
import weakref

import numpy as np

from vision_model_serving.validation.evidence import sanitize_error_detail
from vision_model_serving.validation.optimization import (
    CandidateMeasurement,
    OptimizationPolicy,
)


@dataclass(frozen=True, slots=True)
class SwitchTarget:
    model_id: str
    load_model: Callable[[], tuple[object, tuple[object, ...], float]]
    call_model: Callable[[object, tuple[object, ...]], object]


def observe_candidate(
    *,
    torch: object,
    policy: OptimizationPolicy,
    load_model: Callable[[], tuple[object, tuple[object, ...], float]],
    call_model: Callable[[object, tuple[object, ...]], object],
    project_output: Callable[[object], dict[str, np.ndarray]],
    warmup_runs: int,
    measured_runs: int,
    evidence_roots: tuple[Path, ...],
) -> CandidateMeasurement:
    try:
        measurement = _measure_candidate(
            torch=torch,
            policy=policy,
            load_model=load_model,
            call_model=call_model,
            project_output=project_output,
            warmup_runs=warmup_runs,
            measured_runs=measured_runs,
            evidence_roots=evidence_roots,
        )
    except torch.cuda.OutOfMemoryError as error:
        measurement = CandidateMeasurement(
            evidence=_failed_candidate(
                policy,
                "cuda_out_of_memory",
                detail=sanitize_error_detail(error, evidence_roots),
                cuda_oom=True,
            ),
            outputs=None,
        )
    except Exception as error:  # one failed candidate must not hide the matrix
        measurement = CandidateMeasurement(
            evidence=_failed_candidate(
                policy,
                f"candidate_failed:{type(error).__name__}",
                detail=sanitize_error_detail(error, evidence_roots),
                cuda_oom=False,
            ),
            outputs=None,
        )
    finally:
        gc.collect()
        torch.cuda.empty_cache()
        _restore_fp32_policy(torch)
    return measurement


def observe_repeated_switches(
    *,
    torch: object,
    policy: OptimizationPolicy,
    targets: tuple[SwitchTarget, SwitchTarget],
    cycles: int,
    memory_growth_threshold_percent: float,
    evidence_roots: tuple[Path, ...],
) -> dict[str, object]:
    transitions: list[dict[str, object]] = []
    first_reserved: dict[str, int] = {}
    maximum_growth = 0.0
    for cycle in range(1, cycles + 1):
        for target in targets:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats("cuda:0")
            model = None
            inputs = None
            output = None
            model_reference = None
            failure: tuple[str, str] | None = None
            resident_before_load: list[str] = []
            memory_after_inference: dict[str, int] | None = None
            try:
                model, inputs, _ = target.load_model()
                model_reference = weakref.ref(model)
                if policy.compile:
                    torch._dynamo.reset()
                    model = torch.compile(model, fullgraph=True, mode="reduce-overhead")
                _configure_policy(torch, policy)
                with torch.inference_mode(), _autocast_context(torch, policy):
                    output = target.call_model(model, inputs)
                torch.cuda.synchronize("cuda:0")
                memory_after_inference = {
                    "allocated_bytes": int(torch.cuda.memory_allocated("cuda:0")),
                    "reserved_bytes": int(torch.cuda.memory_reserved("cuda:0")),
                    "peak_allocated_bytes": int(torch.cuda.max_memory_allocated("cuda:0")),
                    "peak_reserved_bytes": int(torch.cuda.max_memory_reserved("cuda:0")),
                }
            except Exception as error:
                failure = (
                    (
                        "cuda_out_of_memory"
                        if isinstance(error, torch.cuda.OutOfMemoryError)
                        else f"candidate_switch_failed:{type(error).__name__}"
                    ),
                    sanitize_error_detail(error, evidence_roots),
                )

            del output
            del inputs
            del model
            if policy.compile:
                torch._dynamo.reset()
            gc.collect()
            torch.cuda.synchronize("cuda:0")
            torch.cuda.empty_cache()
            reference_alive = model_reference is not None and model_reference() is not None
            memory_after_release = {
                "allocated_bytes": int(torch.cuda.memory_allocated("cuda:0")),
                "reserved_bytes": int(torch.cuda.memory_reserved("cuda:0")),
            }
            if reference_alive or any(memory_after_release.values()):
                raise RuntimeError(
                    f"{policy.name} {target.model_id} did not release before the next load"
                )
            if failure is not None:
                _restore_fp32_policy(torch)
                return _unmeasured_switch(*failure)
            if memory_after_inference is None:
                raise RuntimeError("switch measurement did not capture loaded VRAM")
            reserved = memory_after_inference["reserved_bytes"]
            if reserved <= 0:
                raise RuntimeError("switch measurement did not observe loaded VRAM")
            initial_reserved = first_reserved.setdefault(target.model_id, reserved)
            maximum_growth = max(
                maximum_growth,
                max(0.0, (reserved - initial_reserved) * 100.0 / initial_reserved),
            )
            transitions.append(
                {
                    "cycle": cycle,
                    "model_id": target.model_id,
                    "resident_before_load": resident_before_load,
                    "resident_after_load": [target.model_id],
                    "resident_after_release": [],
                    "model_reference_alive_after_release": reference_alive,
                    "memory_after_inference": memory_after_inference,
                    "memory_after_release": memory_after_release,
                }
            )
    _restore_fp32_policy(torch)
    return {
        "status": "measured",
        "method": "alternating_detector_classifier_release_v1",
        "cycle_count": cycles,
        "transitions": transitions,
        "maximum_same_model_reserved_growth_percent": maximum_growth,
        "leak_observed": maximum_growth > memory_growth_threshold_percent,
        "failure_code": None,
        "failure_detail": None,
    }


def _unmeasured_switch(failure_code: str, failure_detail: str) -> dict[str, object]:
    return {
        "status": "unmeasured",
        "method": None,
        "cycle_count": 0,
        "transitions": [],
        "maximum_same_model_reserved_growth_percent": None,
        "leak_observed": None,
        "failure_code": failure_code,
        "failure_detail": failure_detail[:1_000],
    }


def _configure_policy(torch: object, policy: OptimizationPolicy) -> None:
    torch.backends.cuda.matmul.allow_tf32 = policy.tf32
    torch.backends.cudnn.allow_tf32 = policy.tf32
    torch.set_float32_matmul_precision("high" if policy.tf32 else "highest")


def _restore_fp32_policy(torch: object) -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def _autocast_context(torch: object, policy: OptimizationPolicy) -> object:
    if policy.precision == "float32":
        return nullcontext()
    return torch.autocast(
        device_type="cuda",
        dtype=getattr(torch, policy.precision),
    )


def _measure_candidate(
    *,
    torch: object,
    policy: OptimizationPolicy,
    load_model: Callable[[], tuple[object, tuple[object, ...], float]],
    call_model: Callable[[object, tuple[object, ...]], object],
    project_output: Callable[[object], dict[str, np.ndarray]],
    warmup_runs: int,
    measured_runs: int,
    evidence_roots: tuple[Path, ...],
) -> CandidateMeasurement:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats("cuda:0")
    model = None
    inputs = None
    output = None
    model_reference = None
    projected: dict[str, np.ndarray] | None = None
    evidence: dict[str, object] | None = None
    failure: tuple[str, str, bool] | None = None
    try:
        model, inputs, load_ms = load_model()
        model_reference = weakref.ref(model)
        compilation_ms: float | None = None
        compilation_started: float | None = None
        if policy.compile:
            torch._dynamo.reset()
            compilation_started = perf_counter()
            model = torch.compile(model, fullgraph=True, mode="reduce-overhead")
        _configure_policy(torch, policy)
        with torch.inference_mode(), _autocast_context(torch, policy):
            for warmup_index in range(warmup_runs):
                output = call_model(model, inputs)
                if policy.compile and warmup_index == 0:
                    torch.cuda.synchronize("cuda:0")
                    compilation_ms = (perf_counter() - compilation_started) * 1_000.0
            torch.cuda.synchronize("cuda:0")
            measured: list[float] = []
            for _ in range(measured_runs):
                started = torch.cuda.Event(enable_timing=True)
                completed = torch.cuda.Event(enable_timing=True)
                started.record()
                output = call_model(model, inputs)
                completed.record()
                torch.cuda.synchronize("cuda:0")
                measured.append(float(started.elapsed_time(completed)))
            projected = project_output(output)
        distribution = _distribution(measured)
        unique_graphs = (
            int(torch._dynamo.utils.counters["stats"]["unique_graphs"]) if policy.compile else 0
        )
        graph_breaks = (
            sum(torch._dynamo.utils.counters["graph_break"].values()) if policy.compile else 0
        )
        evidence = {
            "name": policy.name,
            "status": "passed",
            "load_ms": load_ms,
            "performance": {
                "latency_ms": distribution,
                "throughput_per_second": 1_000.0 / statistics.mean(measured),
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated("cuda:0")),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved("cuda:0")),
            },
            "compile": {
                "enabled": policy.compile,
                "compilation_ms": compilation_ms,
                "recompilation_count": max(0, unique_graphs - 1),
                "graph_break_count": graph_breaks,
                "fullgraph_required": policy.compile,
            },
            "precision": policy.precision,
            "tf32": policy.tf32,
        }
    except Exception as error:
        cuda_oom = isinstance(error, torch.cuda.OutOfMemoryError)
        failure = (
            ("cuda_out_of_memory" if cuda_oom else f"candidate_failed:{type(error).__name__}"),
            sanitize_error_detail(error, evidence_roots),
            cuda_oom,
        )

    del output
    del inputs
    del model
    if policy.compile:
        torch._dynamo.reset()
    gc.collect()
    torch.cuda.synchronize("cuda:0")
    torch.cuda.empty_cache()
    _restore_fp32_policy(torch)
    release = (
        {
            "status": "measured",
            "model_reference_alive_after_release": model_reference() is not None,
            "allocated_after_release_bytes": int(torch.cuda.memory_allocated("cuda:0")),
            "reserved_after_release_bytes": int(torch.cuda.memory_reserved("cuda:0")),
            "method": "weakref_and_cuda_allocator_after_release",
        }
        if model_reference is not None
        else {
            "status": "unmeasured",
            "model_reference_alive_after_release": None,
            "allocated_after_release_bytes": None,
            "reserved_after_release_bytes": None,
            "method": None,
        }
    )
    if failure is not None:
        failure_code, detail, cuda_oom = failure
        return CandidateMeasurement(
            evidence=_failed_candidate(
                policy,
                failure_code,
                detail=detail,
                cuda_oom=cuda_oom,
                release=release,
            ),
            outputs=None,
        )
    if evidence is None or projected is None:
        raise RuntimeError("candidate measurement did not produce evidence")
    evidence["reliability"] = {"cuda_oom": False, "release": release}
    return CandidateMeasurement(evidence=evidence, outputs=projected)


def _failed_candidate(
    policy: OptimizationPolicy,
    failure_code: str,
    *,
    detail: str,
    cuda_oom: bool,
    release: Mapping[str, object] | None = None,
) -> dict[str, object]:
    return {
        "name": policy.name,
        "status": "failed",
        "failure_code": failure_code,
        "failure_detail": detail[:1_000],
        "parity": {"passed": False},
        "performance": {
            "latency_ms": None,
            "throughput_per_second": None,
            "peak_allocated_bytes": None,
            "peak_reserved_bytes": None,
        },
        "reliability": {
            "cuda_oom": cuda_oom,
            "release": dict(release or _unmeasured_release()),
        },
        "compile": {
            "enabled": policy.compile,
            "compilation_ms": None,
            "recompilation_count": None,
            "graph_break_count": None,
            "fullgraph_required": policy.compile,
        },
        "precision": policy.precision,
        "tf32": policy.tf32,
    }


def _unmeasured_release() -> dict[str, object]:
    return {
        "status": "unmeasured",
        "model_reference_alive_after_release": None,
        "allocated_after_release_bytes": None,
        "reserved_after_release_bytes": None,
        "method": None,
    }


def _distribution(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)

    def percentile(proportion: float) -> float:
        return ordered[max(0, math.ceil(proportion * len(ordered)) - 1)]

    return {
        "count": len(ordered),
        "min": ordered[0],
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": ordered[-1],
        "mean": statistics.mean(ordered),
        "population_stddev": statistics.pstdev(ordered),
    }
