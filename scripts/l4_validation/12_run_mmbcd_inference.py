#!/usr/bin/env python3
"""Run deterministic FP32 MMBCD inference from the frozen input bundle."""

from __future__ import annotations

import argparse
import gc
import hashlib
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

from _common import (
    DINO_COMMIT,
    MMBCD_COMMIT,
    MMBCD_PREDICTION_SHA256,
    MMBCD_SHA256,
    default_paths,
    load_json,
    sha256_array,
    sha256_file,
    strip_module_prefix,
    verify_git_commit,
    verify_sha256,
    write_json_atomic,
)
from _mmbcd_model import build_mmbcd, verify_alias_values


def main() -> None:
    defaults = default_paths()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dino-repo", type=Path, default=defaults["dino_repo"])
    parser.add_argument("--mmbcd-repo", type=Path, default=defaults["mmbcd_repo"])
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=defaults["artifact_dir"] / "mmbcd_best.pt",
    )
    parser.add_argument(
        "--input-bundle",
        type=Path,
        default=defaults["mmbcd_input_dir"] / "mmbcd-inputs.npz",
    )
    parser.add_argument(
        "--input-manifest",
        type=Path,
        default=defaults["mmbcd_input_dir"] / "mmbcd-input-manifest.json",
    )
    parser.add_argument("--output-dir", type=Path, default=defaults["mmbcd_output_dir"])
    parser.add_argument("--benchmark-runs", type=int, default=5)
    parser.add_argument("--expected-prediction-sha256", default=MMBCD_PREDICTION_SHA256)
    args = parser.parse_args()

    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError(
            "Launch with CUBLAS_WORKSPACE_CONFIG=:4096:8 before Python starts"
        )
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import numpy as np
    import torch
    import torchvision
    import transformers

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    checkpoint_path = args.checkpoint.expanduser().resolve()
    input_bundle_path = args.input_bundle.expanduser().resolve()
    input_manifest_path = args.input_manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    verify_sha256(checkpoint_path, MMBCD_SHA256)
    dino_commit = verify_git_commit(args.dino_repo, DINO_COMMIT)
    mmbcd_commit = verify_git_commit(args.mmbcd_repo, MMBCD_COMMIT)

    input_manifest = load_json(input_manifest_path)
    assert sha256_file(input_bundle_path) == input_manifest["artifacts"]["bundle_sha256"]
    with np.load(input_bundle_path, allow_pickle=False) as bundle:
        crop_array = bundle["crops"]
        boxes = bundle["boxes"]
        input_ids_array = bundle["input_ids"]
        attention_mask_array = bundle["attention_mask"]
    assert crop_array.shape == (8, 3, 224, 224) and crop_array.dtype == np.float32
    assert boxes.shape == (8, 5)
    assert input_ids_array.shape == attention_mask_array.shape
    assert input_ids_array.shape[0] == 1
    assert np.isfinite(crop_array).all() and np.isfinite(boxes).all()
    assert np.all(boxes[:-1, 4] >= boxes[1:, 4])
    assert sha256_array(crop_array) == input_manifest["image_transform"]["tensor_sha256"]
    assert sha256_array(input_ids_array) == input_manifest["text"]["input_ids_sha256"]
    assert (
        sha256_array(attention_mask_array)
        == input_manifest["text"]["attention_mask_sha256"]
    )

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True, warn_only=False)

    model = build_mmbcd(args.dino_repo)
    raw_state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = strip_module_prefix(raw_state)
    verify_alias_values(state_dict)
    load_result = model.load_state_dict(state_dict, strict=True)
    assert not load_result.missing_keys and not load_result.unexpected_keys
    del raw_state, state_dict
    gc.collect()

    device = torch.device("cuda:0")
    model.to(device).eval()
    crops = torch.from_numpy(crop_array).unsqueeze(0).to(device)
    input_ids = torch.from_numpy(input_ids_array).long().to(device)
    attention_mask = torch.from_numpy(attention_mask_array).long().to(device)
    assert crops.shape == (1, 8, 3, 224, 224)

    with torch.inference_mode():
        for _ in range(3):
            model(crops, input_ids, attention_mask)
        torch.cuda.synchronize()
        reference_logits, reference_embeddings = model(
            crops, input_ids, attention_mask
        )
        repeated_logits, repeated_embeddings = model(
            crops, input_ids, attention_mask
        )
        torch.cuda.synchronize()
        logits_max_abs_diff = (
            reference_logits - repeated_logits
        ).abs().max().item()
        embeddings_max_abs_diff = (
            reference_embeddings - repeated_embeddings
        ).abs().max().item()
        logits_equal = torch.equal(reference_logits, repeated_logits)
        embeddings_equal = torch.equal(reference_embeddings, repeated_embeddings)
        if not logits_equal or not embeddings_equal:
            raise RuntimeError(
                "MMBCD output is not bitwise deterministic: "
                f"logits={logits_max_abs_diff}, embeddings={embeddings_max_abs_diff}"
            )

        logits_array = reference_logits.detach().cpu().numpy()
        embeddings_array = reference_embeddings.detach().cpu().numpy()
        probabilities_array = torch.softmax(reference_logits, dim=-1).detach().cpu().numpy()
        del reference_logits, reference_embeddings, repeated_logits, repeated_embeddings

        torch.cuda.reset_peak_memory_stats(device)
        latencies_ms = []
        for _ in range(args.benchmark_runs):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            timed_logits, timed_embeddings = model(crops, input_ids, attention_mask)
            end_event.record()
            torch.cuda.synchronize()
            latencies_ms.append(float(start_event.elapsed_time(end_event)))
            del timed_logits, timed_embeddings

    peak_allocated_mib = torch.cuda.max_memory_allocated(device) / (1024**2)
    peak_reserved_mib = torch.cuda.max_memory_reserved(device) / (1024**2)
    logits_sha256 = sha256_array(logits_array)
    embeddings_sha256 = sha256_array(embeddings_array)
    prediction_digest = hashlib.sha256()
    prediction_digest.update(np.ascontiguousarray(logits_array).tobytes())
    prediction_digest.update(np.ascontiguousarray(embeddings_array).tobytes())
    prediction_sha256 = prediction_digest.hexdigest()
    golden_match = prediction_sha256 == args.expected_prediction_sha256

    output_bundle_path = output_dir / "mmbcd-outputs.npz"
    np.savez_compressed(
        output_bundle_path,
        logits=logits_array,
        probabilities=probabilities_array,
        fused_embeddings=embeddings_array,
    )
    device_properties = torch.cuda.get_device_properties(device)
    manifest = {
        "pipeline": "real-dicom-mmbcd-fp32-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "bundle": str(input_bundle_path),
            "bundle_sha256": sha256_file(input_bundle_path),
            "input_manifest": str(input_manifest_path),
            "input_manifest_sha256": sha256_file(input_manifest_path),
            "crop_shape": list(crops.shape),
            "prompt": input_manifest["text"]["prompt"],
            "label_information_used": False,
        },
        "model": {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": MMBCD_SHA256,
            "strict_load": True,
            "checkpoint_aliases_equal": True,
            "mmbcd_commit": mmbcd_commit,
            "dino_commit": dino_commit,
            "dtype": "float32",
            "evaluation_mode": True,
            "inference_mode": True,
        },
        "determinism": {
            "required": True,
            "bitwise_equal_logits": logits_equal,
            "bitwise_equal_embeddings": embeddings_equal,
            "logits_max_abs_diff": logits_max_abs_diff,
            "embeddings_max_abs_diff": embeddings_max_abs_diff,
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "tf32_enabled": False,
        },
        "outputs": {
            "logits": logits_array.tolist(),
            "probabilities": probabilities_array.tolist(),
            "predicted_class_index": int(np.argmax(probabilities_array[0])),
            "class_1_probability": float(probabilities_array[0, 1]),
            "class_mapping_status": (
                "Class 1 is treated as cancer in upstream evaluation code, but "
                "checkpoint metadata does not independently verify the mapping."
            ),
            "logits_sha256": logits_sha256,
            "fused_embeddings_shape": list(embeddings_array.shape),
            "fused_embeddings_sha256": embeddings_sha256,
            "prediction_sha256": prediction_sha256,
            "expected_prediction_sha256": args.expected_prediction_sha256,
            "golden_prediction_match": golden_match,
            "bundle": output_bundle_path.name,
            "bundle_sha256": sha256_file(output_bundle_path),
        },
        "performance": {
            "warmup_runs": 3,
            "measured_runs": len(latencies_ms),
            "latencies_ms": latencies_ms,
            "median_latency_ms": statistics.median(latencies_ms),
            "peak_allocated_mib": peak_allocated_mib,
            "peak_reserved_mib": peak_reserved_mib,
            "timing_scope": (
                "MMBCD GPU forward only; excludes input preparation, "
                "host-to-device transfer, and detector inference."
            ),
        },
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "transformers": transformers.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": device_properties.name,
            "device_total_memory_mib": device_properties.total_memory / (1024**2),
        },
        "validation_boundary": (
            "This is a serving-pipeline smoke test on one public CBIS-DDSM fixture. "
            "It does not establish medical accuracy, calibration, sensitivity, "
            "specificity, or clinical fitness."
        ),
    }
    manifest_path = output_dir / "inference-manifest.json"
    write_json_atomic(manifest_path, manifest)
    print("Strict load:", load_result)
    print("Input crops:", tuple(crops.shape))
    print("Input IDs:", input_ids.detach().cpu().tolist())
    print("Logits:", logits_array.tolist())
    print("Probabilities:", probabilities_array.tolist())
    print("Predicted class index:", manifest["outputs"]["predicted_class_index"])
    print("Class-1 probability:", manifest["outputs"]["class_1_probability"])
    print("Prediction SHA256:", prediction_sha256)
    print("Golden prediction match:", golden_match)
    print("Determinism logits max abs diff:", logits_max_abs_diff)
    print("Determinism embeddings max abs diff:", embeddings_max_abs_diff)
    print("Latencies ms:", [round(value, 3) for value in latencies_ms])
    print("Median latency ms:", round(statistics.median(latencies_ms), 3))
    print("Peak allocated MiB:", round(peak_allocated_mib, 2))
    print("Peak reserved MiB:", round(peak_reserved_mib, 2))
    print("Output bundle:", output_bundle_path)
    print("Manifest:", manifest_path)
    if not golden_match:
        raise RuntimeError("MMBCD prediction does not match the validated L4 hash")
    print("REAL DICOM MMBCD INFERENCE PASSED")


if __name__ == "__main__":
    main()
