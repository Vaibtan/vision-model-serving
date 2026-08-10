"""Bounded NVIDIA resource sampling for benchmark campaigns."""

from __future__ import annotations

import math
import subprocess
from threading import Event, Lock, Thread
import time

from vision_model_serving.validation._benchmark_primitives import (
    BenchmarkContractError,
    latency_distribution,
)


_NVIDIA_SMI_TIMEOUT_SECONDS = 15.0


class NvidiaSampler:
    """Sample bounded `nvidia-smi` metrics while the campaign is active."""

    def __init__(self, interval_ms: int):
        if isinstance(interval_ms, bool) or not isinstance(interval_ms, int):
            raise ValueError("resource sample interval must be an integer")
        if interval_ms < 50:
            raise ValueError("resource sample interval must be at least 50 ms")
        self._interval_seconds = interval_ms / 1_000.0
        self._stop = Event()
        self._thread: Thread | None = None
        self._lock = Lock()
        self._samples: list[tuple[float, float, float, float]] = []
        self._attempt_count = 0
        self._failure_count = 0
        self._started_ns: int | None = None
        self._stopped_ns: int | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("resource sampler can only be started once")
        self._started_ns = time.perf_counter_ns()
        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(_NVIDIA_SMI_TIMEOUT_SECONDS + max(2.0, self._interval_seconds * 4))
            if self._thread.is_alive():
                raise BenchmarkContractError("resource sampler did not stop before serialization")
        self._stopped_ns = time.perf_counter_ns()

    def as_dict(self) -> dict[str, object]:
        with self._lock:
            samples = tuple(self._samples)
            attempt_count = self._attempt_count
            failure_count = self._failure_count
            started_ns = self._started_ns
            stopped_ns = self._stopped_ns
        duration_seconds = (
            0.0
            if started_ns is None
            else ((stopped_ns or time.perf_counter_ns()) - started_ns) / 1_000_000_000.0
        )
        evidence = {
            "interval_ms": int(self._interval_seconds * 1_000),
            "duration_seconds": duration_seconds,
            "attempt_count": attempt_count,
            "success_count": len(samples),
            "failure_count": failure_count,
            "sample_count": len(samples),
        }
        if not samples:
            return {
                "available": False,
                **evidence,
            }
        columns = tuple(zip(*samples, strict=True))
        return {
            "available": True,
            **evidence,
            "gpu_utilization_percent": latency_distribution(columns[0]),
            "memory_used_mib": latency_distribution(columns[1]),
            "power_watts": latency_distribution(columns[2]),
            "temperature_c": latency_distribution(columns[3]),
        }

    def _run(self) -> None:
        while not self._stop.is_set():
            attempt_started_ns = time.perf_counter_ns()
            output = _command(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ]
            )
            values: tuple[float, ...] | None = None
            if output:
                try:
                    parsed = tuple(float(value.strip()) for value in output.split(","))
                    if len(parsed) == 4 and all(
                        value >= 0 and math.isfinite(value) for value in parsed
                    ):
                        values = parsed
                except ValueError:
                    values = None
            with self._lock:
                self._attempt_count += 1
                if values is None:
                    self._failure_count += 1
                else:
                    self._samples.append(values)
            elapsed_seconds = (time.perf_counter_ns() - attempt_started_ns) / 1_000_000_000.0
            self._stop.wait(max(0.0, self._interval_seconds - elapsed_seconds))


def _command(command: list[str]) -> str:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=_NVIDIA_SMI_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip()
