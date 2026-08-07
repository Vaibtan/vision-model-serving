# Vision Model Serving

This repository defines and validates the inference pipeline that will underpin
the planned Django service and containerized deployment.

For detailed task requirements and resource links, see [ASSIGNMENT.md](ASSIGNMENT.md).

The validated NVIDIA L4 reference pipeline, exact source/artifact pins, known
failure modes, and standalone reproduction scripts are documented in
[docs/validation/lightning-l4-fp32-reproduction.md](docs/validation/lightning-l4-fp32-reproduction.md).

The checked-in [model artifact inventory](docs/model-artifact-inventory.md)
records the exact two-model contract, checkpoint structure, source revisions,
tokenizer assets, unresolved licensing and semantic claims, and the checkpoint
trust boundary. Validate it without loading model weights:

```powershell
$env:PYTHONPATH = "src"
python -m vision_model_serving.artifacts config/model-artifacts.json
python -m unittest discover -s tests -v
```

The [pinned L4 FP32 lane](config/l4-fp32-environment.json) and its
[reproduction runbook](docs/validation/lightning-l4-fp32-reproduction.md)
cover exact dependencies, source commits, patch checks, native-operator build
and import validation, offline strict loads, and CPU-only skip semantics:

```bash
python scripts/l4_validation/00_probe_environment.py
python scripts/l4_validation/prepare_focalnet.py --help
```
