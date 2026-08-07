# Lightning L4 FP32 reference-pipeline reproduction

This runbook reproduces the validated research path from a public mammography
DICOM through FocalNet-DINO proposals and the MMBCD multimodal classifier on one
NVIDIA L4. Every Python block used during the original investigation is now a
standalone script under [`scripts/l4_validation/`](../../scripts/l4_validation/).

The reference run completed on 2026-08-07. Its immutable values are recorded in
[`reference-l4-fp32-20260807.json`](reference-l4-fp32-20260807.json), and its
downloaded evidence archive has SHA-256
`5a1311b121edd6c03feb279e5aa7170be5a545d859a1e1fc2f70fc32252f097f`.

## What this run proves

It proves all of the following for the recorded environment and artifacts:

- the CUDA extension compiles against PyTorch 2.8 and agrees with its PyTorch
  reference forward implementation;
- the complete FocalNet-DINO checkpoint strict-loads without the separate
  `focalnet_large_lrf_384.pth` training-time initializer;
- a checksum-pinned CBIS-DDSM DICOM decodes and follows the released MMBCD
  preprocessing path deterministically;
- the detector produces stable top-300 predictions, applies the released
  strict `IoU > 0.1` NMS semantics, and supplies exactly eight proposals;
- the complete MMBCD checkpoint strict-loads using local DINO architecture code
  and a locally pinned RoBERTa tokenizer;
- both real-model FP32 forward passes are deterministic on the L4; and
- the generated manifests and small result bundles can be independently
  verified without loading either model.

It does **not** establish medical accuracy, calibration, sensitivity,
specificity, or clinical fitness. The public DICOM is a pipeline fixture, the
detector scores are extremely low, and the MMBCD class mapping is not confirmed
by checkpoint metadata. Never present class `0` as "benign" or the softmax
values as calibrated confidence.

## Validated environment

| Component | Validated value |
| --- | --- |
| Lightning hardware | NVIDIA L4, compute capability 8.9, 23,034 MiB |
| Driver | 580.173.02 |
| Python | 3.12.11 |
| PyTorch | 2.8.0+cu128 |
| torchvision | 0.23.0+cu128 |
| CUDA runtime/toolkit | 12.8 / 12.8.0 (`nvcc` 12.8.61) |
| NumPy | 1.26.4 |
| OpenCV | 4.11.0.86 headless |
| pydicom | 3.0.2 |
| Pillow | 12.2.0 |
| timm | 1.0.28 |
| Transformers | 5.14.1 |

MMBCD's released environment used PyTorch 2.1.2 and Transformers 4.37.0. The
run below establishes serving compatibility with the newer stack through exact
state-dict loading, pinned token IDs, and deterministic golden outputs. It is
not evidence of bitwise parity with the authors' historical environment.

## Persistent layout

Use the persistent Studio filesystem. Files created elsewhere may disappear
when the Studio is stopped.

```text
/teamspace/studios/this_studio/
├── vision-model-serving/              # this repository
├── vision-model-serving-artifacts/    # supplied weights; never commit
├── src/
│   ├── FocalNet-DINO/
│   ├── MMBCD/
│   └── dino/
├── assets/                             # pinned tokenizer snapshot
├── fixtures/cbis-ddsm/<series-uid>/
└── validation-evidence/
```

Set the common paths after logging in:

```bash
set -euo pipefail

export STUDIO_ROOT="/teamspace/studios/this_studio"
export VMS_REPO="${STUDIO_ROOT}/vision-model-serving"
export ARTIFACT_DIR="${STUDIO_ROOT}/vision-model-serving-artifacts"
export FOCAL_REPO="${STUDIO_ROOT}/src/FocalNet-DINO"
export MMBCD_REPO="${STUDIO_ROOT}/src/MMBCD"
export DINO_REPO="${STUDIO_ROOT}/src/dino"
export SERIES_UID="1.3.6.1.4.1.9590.100.1.2.100131208110604806117271735422083351547"
export FIXTURE_DIR="${STUDIO_ROOT}/fixtures/cbis-ddsm/${SERIES_UID}"
```

## 1. Verify the downloaded reference evidence locally

This step uses only the Python standard library and can be run on Windows before
starting a GPU:

```powershell
python scripts/l4_validation/14_verify_evidence_archive.py `
  vision-model-serving-l4-fp32-20260807.tar.gz `
  vision-model-serving-l4-fp32-20260807.tar.gz.sha256
```

Expected marker:

```text
LIGHTNING L4 EVIDENCE ARCHIVE PASSED
```

The archive and sidecar are intentionally ignored by Git. Track the small
reference JSON, scripts, patches, and documentation instead.

## 2. Activate Lightning's existing environment

Lightning permits one default Conda environment per Studio. Do not run
`conda create`; the platform rejects it, and a second environment is unnecessary
for this compatibility lane.

First try a fresh login shell:

```bash
exec zsh -l
command -v conda
command -v python
python --version
```

If a hardware switch left the shell uninitialized, activate the preserved
environment rather than reinstalling Python:

```bash
if [[ -x /home/zeus/miniconda3/envs/cloudspace/bin/python ]]; then
  export LIGHTNING_CONDA_ROOT="/home/zeus/miniconda3"
elif [[ -x /system/conda/miniconda3/envs/cloudspace/bin/python ]]; then
  export LIGHTNING_CONDA_ROOT="/system/conda/miniconda3"
else
  echo "PRESERVED CLOUDSPACE ENVIRONMENT NOT FOUND"
  exit 1
fi

source "${LIGHTNING_CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${LIGHTNING_CONDA_ROOT}/envs/cloudspace"
hash -r 2>/dev/null || true
rehash 2>/dev/null || true
```

## 3. Align the compiler toolkit with PyTorch

The original machine exposed CUDA toolkit 13.0 while PyTorch was compiled for
CUDA 12.8. The driver can remain newer, but custom extensions must be compiled
with a compatible 12.8 toolkit.

Save a rollback point:

```bash
mkdir -p "${STUDIO_ROOT}/environment-snapshots"
conda list --revisions \
  > "${STUDIO_ROOT}/environment-snapshots/revisions-before-cuda-12.8.txt"
```

Inspect the proposed transaction first:

```bash
conda install \
  --dry-run \
  --freeze-installed \
  --strict-channel-priority \
  --override-channels \
  -c nvidia/label/cuda-12.8.0 \
  -c defaults \
  "nvidia/label/cuda-12.8.0::cuda-toolkit=12.8.0"
```

It must not replace PyTorch, torchvision, Python, or NumPy. Apply it only after
that inspection:

```bash
conda install \
  --freeze-installed \
  --strict-channel-priority \
  --override-channels \
  -c nvidia/label/cuda-12.8.0 \
  -c defaults \
  "nvidia/label/cuda-12.8.0::cuda-toolkit=12.8.0" \
  -y

export CUDA_HOME="${CONDA_PREFIX}"
export CUDACXX="${CUDA_HOME}/bin/nvcc"
export PATH="${CUDA_HOME}/bin:${PATH}"
hash -r

which nvcc
nvcc --version
```

Do not prepend the whole Conda `lib` directory to `LD_LIBRARY_PATH` unless it is
actually required. Doing so produced a non-fatal but noisy `libtinfo.so.6`
version warning in `/bin/bash` during the reference investigation.

## 4. Check out source and install the recorded Python dependencies

```bash
mkdir -p "${STUDIO_ROOT}/src"

if [[ ! -d "${VMS_REPO}/.git" ]]; then
  git clone https://github.com/Vaibtan/vision-model-serving.git "${VMS_REPO}"
fi

if [[ ! -d "${FOCAL_REPO}/.git" ]]; then
  git clone https://github.com/FocalNet/FocalNet-DINO.git "${FOCAL_REPO}"
fi

if [[ ! -d "${MMBCD_REPO}/.git" ]]; then
  git clone https://github.com/adsbansal/MMBCD.git "${MMBCD_REPO}"
fi

if [[ ! -d "${DINO_REPO}/.git" ]]; then
  git clone https://github.com/facebookresearch/dino.git "${DINO_REPO}"
fi

git -C "${FOCAL_REPO}" fetch origin
git -C "${FOCAL_REPO}" checkout --detach \
  23901e021dc6ec8f66bad47983f45a25574452cc

git -C "${MMBCD_REPO}" fetch origin
git -C "${MMBCD_REPO}" checkout --detach \
  14ac5e099c79253b01e0885d2ebefa6f86cfd8f0

git -C "${DINO_REPO}" fetch origin
git -C "${DINO_REPO}" checkout --detach \
  7c446df5b9f45747937fb0d72314eb9f7b66930a
```

Install the exact non-PyTorch packages from one pinned file. This avoids the
failure encountered when unpinned `opencv-python` installed NumPy 2.5.1 and
broke SciPy, pandas, matplotlib, and scikit-learn compatibility.

```bash
python -m pip install -r "${VMS_REPO}/requirements/l4-validation.txt"
python -m pip check
```

Verify the complete runtime:

```bash
python "${VMS_REPO}/scripts/l4_validation/00_probe_environment.py"
```

Expected marker:

```text
LIGHTNING L4 ENVIRONMENT PASSED
```

Save the post-install environment:

```bash
conda list --explicit \
  > "${STUDIO_ROOT}/environment-snapshots/after-cuda-12.8.txt"
```

## 5. Place and verify the supplied model artifacts

From local Windows PowerShell, upload the two evaluator-provided artifacts. The
paths below match the original local artifact directory; adjust only the SSH
alias if necessary.

```powershell
ssh <lightning-alias> `
  "mkdir -p /teamspace/studios/this_studio/vision-model-serving-artifacts"

scp "D:\SWE_DEV_NEW\vision-model-serving-artifacts\focalnet-dino-finetuned.pth" `
  "<lightning-alias>:/teamspace/studios/this_studio/vision-model-serving-artifacts/"

scp "D:\SWE_DEV_NEW\vision-model-serving-artifacts\mmbcd_best.pt" `
  "<lightning-alias>:/teamspace/studios/this_studio/vision-model-serving-artifacts/"
```

Back on Lightning:

```bash
mkdir -p "${ARTIFACT_DIR}"
cp "${VMS_REPO}/config_cfg.py" "${ARTIFACT_DIR}/config_cfg.py"

sha256sum \
  "${ARTIFACT_DIR}/focalnet-dino-finetuned.pth" \
  "${ARTIFACT_DIR}/mmbcd_best.pt"
```

Expected hashes:

```text
67a7b0cd787a3aaba199cf1ff82ed2934c33ffe37544473379d7a837ab1637b4  focalnet-dino-finetuned.pth
2264351216f9fb4945af35e300459ff4ce2e7f5445519348024f3bf1eec721a4  mmbcd_best.pt
```

`focalnet_large_lrf_384.pth` is not needed. The full task checkpoint covers the
backbone, as proven by the strict-load gate below.

## 6. Patch and build MultiScaleDeformableAttention

Apply the two reviewed serving-compatibility patches. The first replaces the
deprecated dispatch argument used by the PyTorch 2.8 extension build. The
second removes the training-only FocalNet backbone preload; the complete task
checkpoint is strict-loaded afterward.

```bash
cd "${FOCAL_REPO}"

for patch in \
  "${VMS_REPO}/patches/focalnet-pytorch-2.8-compat.patch" \
  "${VMS_REPO}/patches/focalnet-serving-no-backbone-preload.patch"
do
  if git apply --check "${patch}"; then
    git apply "${patch}"
  elif git apply --reverse --check "${patch}"; then
    echo "Already applied: ${patch}"
  else
    echo "Patch does not apply cleanly: ${patch}"
    exit 1
  fi
done

git diff --check
git diff -- \
  models/dino/backbone.py \
  models/dino/ops/src/cuda/ms_deform_attn_cuda.cu
```

Build only for the L4's compute capability:

```bash
cd "${FOCAL_REPO}/models/dino/ops"

export CUDA_HOME="${CONDA_PREFIX}"
export CUDACXX="${CUDA_HOME}/bin/nvcc"
export PATH="${CUDA_HOME}/bin:${PATH}"
export TORCH_CUDA_ARCH_LIST="8.9"
export MAX_JOBS=4

python setup.py build_ext --inplace
```

Import `torch` before the extension so `libc10.so` and the other PyTorch native
libraries are loaded. The standalone validator does this correctly:

```bash
python "${VMS_REPO}/scripts/l4_validation/01_validate_cuda_extension.py"
```

Expected marker:

```text
FOCALNET CUDA EXTENSION PASSED
```

If `libc10.so` is still unresolved, add only PyTorch's library directory:

```bash
export TORCH_LIB_DIR="${CONDA_PREFIX}/lib/python3.12/site-packages/torch/lib"
test -d "${TORCH_LIB_DIR}"
export LD_LIBRARY_PATH="${TORCH_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
```

## 7. Strict-load and structurally exercise the detector

```bash
python "${VMS_REPO}/scripts/l4_validation/02_strict_load_detector.py"

python "${VMS_REPO}/scripts/l4_validation/03_make_detector_inference_checkpoint.py"

python "${VMS_REPO}/scripts/l4_validation/04_run_synthetic_detector.py"
```

Required markers:

```text
FOCALNET STRICT LOAD PASSED
FOCALNET INFERENCE CHECKPOINT PASSED
SYNTHETIC DETECTOR INFERENCE PASSED
```

The inference-only checkpoint excludes optimizer and scheduler state. Its
validated SHA-256 is
`a7fed981c7309d12c19532624e148b45087462c56568797cb62b60055bb62f04`.

## 8. Download and validate the public DICOM fixture

The selected series is one `MG` object with `SeriesDescription` equal to
`full mammogram images`. It is a Secondary Capture CBIS-DDSM image licensed by
TCIA under CC BY 3.0. Preserve the `LICENSE` file contained in the downloaded
archive.

```bash
export FIXTURE_ROOT="${STUDIO_ROOT}/fixtures/cbis-ddsm"
export ZIP_PATH="${FIXTURE_ROOT}/${SERIES_UID}.zip"

mkdir -p "${FIXTURE_DIR}"

curl \
  --fail \
  --location \
  --retry 5 \
  --retry-delay 2 \
  --retry-all-errors \
  --output "${ZIP_PATH}.part" \
  "https://nbia.cancerimagingarchive.net/nbia-api/services/v4/getImage?NewFileNames=Yes&SeriesInstanceUID=${SERIES_UID}"

python -m zipfile -t "${ZIP_PATH}.part"
mv "${ZIP_PATH}.part" "${ZIP_PATH}"
python -m zipfile -e "${ZIP_PATH}" "${FIXTURE_DIR}"
```

Validate only `*.dcm`; do not count TCIA's `LICENSE` as a second DICOM:

```bash
python "${VMS_REPO}/scripts/l4_validation/05_validate_dicom.py"
python "${VMS_REPO}/scripts/l4_validation/06_preprocess_dicom.py"
```

Required markers:

```text
REAL DICOM DECODE PASSED
DICOM PREPROCESSING PARITY PASSED
```

For this exact fixture, the largest contour covers the complete 3826×6601
frame, so cropping is a no-op. The released pipeline then distorts the aspect
ratio to 1024×1024. Open `preprocessed/upstream-1024.png` and confirm the breast
tissue is visible, brighter than the background, and not inverted.

## 9. Run the real detector gate

```bash
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTHONHASHSEED=0

python "${VMS_REPO}/scripts/l4_validation/07_run_detector_inference.py"
```

Expected marker:

```text
REAL DICOM DETECTOR INFERENCE PASSED
```

Inspect both generated overlays:

```text
<fixture>/detector/overlay-top8-1024.png
<fixture>/detector/overlay-top8-original.png
```

The boxes must occupy corresponding regions in both coordinate spaces. Do not
introduce an arbitrary score threshold: the validated released handoff is the
ordered top 300, strict normalized-box `IoU > 0.1` NMS, then top eight.

## 10. Audit and strict-load MMBCD without network downloads

```bash
python "${VMS_REPO}/scripts/l4_validation/08_audit_mmbcd_checkpoint.py"

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python "${VMS_REPO}/scripts/l4_validation/09_strict_load_mmbcd.py"
```

Required markers:

```text
MMBCD CHECKPOINT STRUCTURE PASSED
MMBCD STRICT LOAD PASSED
```

The released checkpoint contains DINO, RoBERTa, projection, attention, and
classifier parameters. No separate DINO or RoBERTa base weights are needed.

## 11. Pin tokenizer assets and freeze the MMBCD inputs

The tokenizer download is the final intentional network-dependent model-asset
step:

```bash
python "${VMS_REPO}/scripts/l4_validation/10_download_roberta_tokenizer.py"
```

Then prove the remaining path works offline:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python "${VMS_REPO}/scripts/l4_validation/11_prepare_mmbcd_inputs.py"
```

Expected marker:

```text
MMBCD INPUT CONTRACT PASSED
```

The golden prompt is exactly `Indication:` because this fixture provides no
usable clinical history. No `cancer` or `all_views_cancer` label is allowed to
affect the prompt. Open `mmbcd/input/roi-montage.png` and confirm it contains
eight real, non-blank mammogram regions.

## 12. Run and verify real MMBCD inference

`CUBLAS_WORKSPACE_CONFIG` must be exported before Python starts:

```bash
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTHONHASHSEED=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python "${VMS_REPO}/scripts/l4_validation/12_run_mmbcd_inference.py"
python "${VMS_REPO}/scripts/l4_validation/13_verify_mmbcd_result.py"
```

Expected marker from both commands:

```text
REAL DICOM MMBCD INFERENCE PASSED
```

## 13. Reference results

| Measurement | Observed reference |
| --- | ---: |
| Detector prediction SHA-256 | `4cdd09d986702e8839acff8d7517a63f263ca2a01b0607d78d6b2086c886a9a5` |
| Detector determinism max absolute difference | `0.0` |
| Detector median forward latency | 234.629 ms |
| Detector peak allocated memory | 1242.17 MiB |
| Detector top score | 0.0064186654 |
| Boxes before/after NMS | 300 / 8 |
| MMBCD input tensor SHA-256 | `89cda9694e3696f63eb70706a6ae4cc2dd16e9be214f27ba6f106479eac90155` |
| MMBCD logits | `[3.5829842, -4.8862085]` |
| MMBCD prediction SHA-256 | `43ec1c4593c0549510098ea082ea7092c7fd5631c95d8b912ecf31633185899b` |
| MMBCD determinism max absolute differences | `0.0`, `0.0` |
| MMBCD median forward latency | 91.256 ms |
| MMBCD peak allocated/reserved memory | 870.50 / 948.00 MiB |

The sum of the two isolated GPU-forward medians is approximately 325.885 ms.
That is a lower bound, not API latency: it excludes DICOM decode/preprocessing,
host/device transfers, model loading and unloading, switching, postprocessing,
queueing, and serialization. The separately measured peak-memory values are not
a measurement of simultaneous residency and must not be added as if they were.

## 14. Collect a new evidence archive

The collector intentionally excludes model weights and the raw DICOM:

```bash
python "${VMS_REPO}/scripts/l4_validation/15_collect_evidence.py" \
  --evidence-root "${STUDIO_ROOT}/validation-evidence/l4-fp32-reproduction" \
  --archive "${STUDIO_ROOT}/validation-evidence/vision-model-serving-l4-fp32-reproduction.tar.gz"
```

Download the resulting archive and `.sha256` sidecar. Inspect the evidence
before publishing: manifests and public derived images are suitable for the
assignment evidence path, but model weights and MMBCD source redistribution
must remain external until licensing is clarified.

## Troubleshooting and stop conditions

- **`conda create is not allowed`:** use the existing Studio environment. Do
  not start creating alternate environments inside the same Studio.
- **`nvcc` reports 13.0 while PyTorch reports 12.8:** install the pinned 12.8
  toolkit into the existing environment and point `CUDA_HOME`/`CUDACXX` at it.
- **`AT_DISPATCH_FLOATING_TYPES(value.type(), ...)` fails:** apply only the
  committed PyTorch 2.8 compatibility patch. Do not change PyTorch first.
- **`ImportError: libc10.so`:** import `torch` before the extension. If needed,
  add only PyTorch's own `lib` directory to `LD_LIBRARY_PATH`.
- **`libtinfo.so.6: no version information`:** avoid globally prepending all of
  the Conda `lib` directory. The warning did not indicate a Git failure.
- **OpenCV upgrades NumPy to 2.x:** remove conflicting OpenCV variants and
  reinstall the versions in `requirements/l4-validation.txt`. Run `pip check`.
- **DICOM validator finds two files:** the second file is normally `LICENSE`.
  Preserve it; the validator deliberately selects only `.dcm` files.
- **A timm deprecation warning appears:** the upstream import warning is
  non-fatal. Treat a traceback or failed assertion separately.
- **A golden SHA differs:** do not update the reference automatically or loosen
  tolerances. Preserve the new manifest, compare commits/artifact hashes/package
  versions/build flags, and explain the drift first.
- **An inference script cannot produce its success marker:** stop before FP16,
  BF16, `torch.compile`, ONNX, or TensorRT work.

## Next engineering gate

These scripts are a reference harness, not the production service. The next
phase turns the validated contracts into repository modules, adds a dedicated
single-residency GPU runtime, and proves repeated
`detector -> unload -> MMBCD -> unload` transitions without leaked live tensors
before Django workers or a frontend can invoke the GPU.
