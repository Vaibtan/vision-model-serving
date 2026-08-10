#!/usr/bin/env python3
"""Execute one MMBCD plan without importing PyTorch or model source code."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
from time import perf_counter

import numpy as np


INPUT_CONTRACT = (
    (
        "roi_crops",
        np.dtype(np.float32),
        (1, 8, 3, 224, 224),
        (1, 8, 3, 224, 224),
        (1, 8, 3, 224, 224),
    ),
    ("input_ids", np.dtype(np.int64), (1, 1), (1, 5), (1, 90)),
    ("attention_mask", np.dtype(np.int64), (1, 1), (1, 5), (1, 90)),
)
OUTPUT_CONTRACT = (
    ("logits", np.dtype(np.float32), (1, 2)),
    ("fused_embeddings", np.dtype(np.float32), (1, 768)),
    ("roi_attention", np.dtype(np.float32), (1, 1, 8)),
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--input-bundle", type=Path, required=True)
    parser.add_argument("--output-bundle", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--measured-runs", type=int, default=10)
    args = parser.parse_args()
    if args.warmup_runs < 1 or args.measured_runs < 5:
        parser.error("warmup/measured runs must be at least 1/5")
    forbidden = {"torch", "torch_tensorrt", "transformers"}
    if forbidden.intersection(sys.modules):
        raise RuntimeError("runtime-only verifier imported a forbidden framework")

    import tensorrt as trt
    from cuda.bindings import runtime as cudart

    inputs = _load_inputs(args.input_bundle)
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(args.plan.read_bytes())
    if engine is None:
        raise RuntimeError("TensorRT plan deserialization failed")
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("TensorRT execution-context creation failed")
    engine_inputs: list[str] = []
    engine_outputs: list[str] = []
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            engine_inputs.append(name)
        else:
            engine_outputs.append(name)
    if len(engine_inputs) != 3 or len(engine_outputs) != 3:
        raise RuntimeError("TensorRT plan does not expose the MMBCD I/O count")

    stream = _cuda(cudart.cudaStreamCreate())
    allocations: list[object] = []
    outputs: dict[str, np.ndarray] = {}
    try:
        for engine_name, (logical_name, dtype, minimum, optimum, maximum) in zip(
            engine_inputs,
            INPUT_CONTRACT,
            strict=True,
        ):
            host = inputs[logical_name]
            _verify_input_contract(
                engine,
                context,
                engine_name,
                dtype,
                minimum,
                optimum,
                maximum,
                tuple(host.shape),
                trt,
            )
            device = _cuda(cudart.cudaMalloc(host.nbytes))
            allocations.append(device)
            _cuda(
                cudart.cudaMemcpyAsync(
                    device,
                    host.ctypes.data,
                    host.nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                    stream,
                )
            )
            if not context.set_tensor_address(engine_name, int(device)):
                raise RuntimeError("TensorRT input address binding failed")
        missing_shapes = context.infer_shapes()
        if missing_shapes:
            raise RuntimeError(
                f"TensorRT input shapes remain unspecified: {missing_shapes!r}"
            )
        for engine_name, (logical_name, dtype, shape) in zip(
            engine_outputs,
            OUTPUT_CONTRACT,
            strict=True,
        ):
            _verify_output_contract(context, engine_name, dtype, shape, trt, engine)
            host = np.empty(shape, dtype=dtype)
            outputs[logical_name] = host
            device = _cuda(cudart.cudaMalloc(host.nbytes))
            allocations.append(device)
            if not context.set_tensor_address(engine_name, int(device)):
                raise RuntimeError("TensorRT output address binding failed")

        for _ in range(args.warmup_runs):
            if not context.execute_async_v3(stream_handle=int(stream)):
                raise RuntimeError("TensorRT warmup enqueue failed")
        _cuda(cudart.cudaStreamSynchronize(stream))
        latencies_ms: list[float] = []
        for _ in range(args.measured_runs):
            started = perf_counter()
            if not context.execute_async_v3(stream_handle=int(stream)):
                raise RuntimeError("TensorRT measured enqueue failed")
            _cuda(cudart.cudaStreamSynchronize(stream))
            latencies_ms.append((perf_counter() - started) * 1_000.0)
        for output_index, (_, host) in enumerate(outputs.items()):
            device = allocations[len(INPUT_CONTRACT) + output_index]
            _cuda(
                cudart.cudaMemcpyAsync(
                    host.ctypes.data,
                    device,
                    host.nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                    stream,
                )
            )
        _cuda(cudart.cudaStreamSynchronize(stream))
    finally:
        for allocation in reversed(allocations):
            _cuda(cudart.cudaFree(allocation))
        _cuda(cudart.cudaStreamDestroy(stream))

    args.output_bundle.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_bundle, **outputs)
    report = {
        "schema_version": 1,
        "runtime": {
            "tensorrt": trt.__version__,
            "cuda_python": _package_version("cuda-python"),
            "forbidden_frameworks_imported": sorted(forbidden.intersection(sys.modules)),
        },
        "engine": {
            "input_tensor_names": engine_inputs,
            "output_tensor_names": engine_outputs,
        },
        "performance": {
            "warmup_runs": args.warmup_runs,
            "measured_runs": args.measured_runs,
            "latencies_ms": latencies_ms,
            "p50_ms": statistics.median(latencies_ms),
            "mean_ms": statistics.mean(latencies_ms),
            "population_stddev_ms": statistics.pstdev(latencies_ms),
        },
        "gates": {
            "deserialized": True,
            "dynamic_io_contract": True,
            "executed": True,
            "pytorch_absent": not forbidden.intersection(sys.modules),
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    print("TENSORRT-ONLY MMBCD RUNTIME PASSED")
    return 0


def _load_inputs(path: Path) -> dict[str, np.ndarray]:
    with np.load(path.expanduser().resolve(), allow_pickle=False) as bundle:
        crops = np.ascontiguousarray(bundle["crops"], dtype=np.float32)
        if crops.shape == (8, 3, 224, 224):
            crops = crops[None, ...]
        input_ids = _token_input(bundle["input_ids"])
        attention_mask = _token_input(bundle["attention_mask"])
    values = {
        "roi_crops": crops,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    if input_ids.shape != attention_mask.shape:
        raise RuntimeError("TensorRT token inputs have different shapes")
    for name, dtype, minimum, _, maximum in INPUT_CONTRACT:
        value = values[name]
        if value.dtype != dtype or not _within(tuple(value.shape), minimum, maximum):
            raise RuntimeError(f"TensorRT input {name} differs from the profile")
    return values


def _token_input(values: np.ndarray) -> np.ndarray:
    if (
        values.ndim != 2
        or values.shape[0] != 1
        or values.shape[1] < 1
        or values.shape[1] > 90
    ):
        raise RuntimeError("TensorRT token input is invalid")
    return np.ascontiguousarray(values, dtype=np.int64)


def _verify_input_contract(
    engine: object,
    context: object,
    name: str,
    dtype: np.dtype,
    minimum: tuple[int, ...],
    optimum: tuple[int, ...],
    maximum: tuple[int, ...],
    actual: tuple[int, ...],
    trt: object,
) -> None:
    observed_dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(name)))
    observed_profile = tuple(
        tuple(shape)
        for shape in engine.get_tensor_profile_shape(name, 0)
    )
    if observed_dtype != dtype or observed_profile != (minimum, optimum, maximum):
        raise RuntimeError(
            f"TensorRT input {name} differs: {observed_dtype} {observed_profile}"
        )
    if not _within(actual, minimum, maximum):
        raise RuntimeError(f"TensorRT input {name} is outside its profile")
    engine_shape = tuple(engine.get_tensor_shape(name))
    if -1 in engine_shape and not context.set_input_shape(name, actual):
        raise RuntimeError(f"TensorRT input shape binding failed for {name}")
    if tuple(context.get_tensor_shape(name)) != actual:
        raise RuntimeError(f"TensorRT input shape differs for {name}")


def _verify_output_contract(
    context: object,
    name: str,
    dtype: np.dtype,
    shape: tuple[int, ...],
    trt: object,
    engine: object,
) -> None:
    observed_dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(name)))
    observed_shape = tuple(context.get_tensor_shape(name))
    if observed_dtype != dtype or observed_shape != shape:
        raise RuntimeError(
            f"TensorRT output {name} differs: {observed_dtype} {observed_shape}"
        )


def _within(
    actual: tuple[int, ...],
    minimum: tuple[int, ...],
    maximum: tuple[int, ...],
) -> bool:
    return len(actual) == len(minimum) == len(maximum) and all(
        lower <= observed <= upper
        for observed, lower, upper in zip(actual, minimum, maximum, strict=True)
    )


def _cuda(result: object) -> object:
    values = result if isinstance(result, tuple) else (result,)
    status = values[0]
    if int(status) != 0:
        raise RuntimeError(f"CUDA runtime call failed with status {status}")
    return values[1] if len(values) == 2 else None


def _package_version(name: str) -> str:
    from importlib.metadata import version

    return version(name)


if __name__ == "__main__":
    raise SystemExit(main())
