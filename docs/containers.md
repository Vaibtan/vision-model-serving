# Container deployment

The Compose deployment keeps the request plane CPU-only and gives one process
exclusive ownership of CUDA:

```text
localhost:8000 -> Gunicorn/Django -> Redis -> standard RQ worker
                                                |
                                                v
                                  owner-only Unix socket
                                                |
                                                v
                                      persistent L4 executor
```

The web and RQ services use the 145 MB CPU image. The executor uses the CUDA
image and is the only service with an NVIDIA device, checkpoints, tokenizer,
or model-source mounts. Redis is reachable only on the internal Compose
network and runs with RDB and AOF persistence disabled.

## Build and asset boundary

All base images, Python packages, upstream revisions, and uv itself are pinned.
The CUDA build stage checks out the Apache-2.0 FocalNet-DINO revision, applies
the three checksum-pinned serving/build patches, and compiles deformable
attention for compute capability 8.9. `FORCE_CUDA=1` only bypasses upstream's
build-time device-presence check; the toolkit and runtime functional gates
still fail closed. The final image receives the virtual environment,
patched FocalNet source, and native operator, but not the compiler toolchain.

The following inputs remain outside the build context and are mounted read-only
at startup:

- the two evaluator-supplied checkpoints;
- the checksum-pinned RoBERTa tokenizer directory;
- the pinned MMBCD source, whose redistribution grant is unresolved; and
- the pinned DINO source.

`.dockerignore` is a whitelist. It excludes checkpoints, DICOM files, archives,
validation evidence, downloaded source trees, Git metadata, and local virtual
environments before Docker sends the build context.

## CPU-only acceptance

Set `VMS_DICOM_PATH` to the checksum-pinned public CBIS-DDSM file, then run the
real Redis/Django acceptance profile. It has no CUDA device or weight mount.

```powershell
$env:VMS_DICOM_PATH = "C:\fixtures\cbis-ddsm-1-1.dcm"
docker compose --profile test up --build --abort-on-container-exit --exit-code-from test
docker compose --profile test down
```

The test must report Redis 7.4, the expected DICOM hash, real 200/202 API paths,
the bounded error contract, opaque RQ arguments, and private-payload absence.

## L4 smoke profile

Export the five external paths before building. On Lightning Studio they are:

```bash
export VMS_ARTIFACT_ROOT=/teamspace/studios/this_studio/vision-model-serving-artifacts
export VMS_TOKENIZER_ROOT=/teamspace/studios/this_studio/assets/roberta-base-tokenizer-e2da8e2f811d1448a5b465c236feacd80ffbac7b
export VMS_MMBCD_ROOT=/teamspace/studios/this_studio/src/MMBCD
export VMS_DINO_ROOT=/teamspace/studios/this_studio/src/dino
export VMS_DICOM_PATH=/teamspace/studios/this_studio/fixtures/cbis-ddsm/Mass-Training_P_00001_LEFT_CC/07-20-2016-DDSM-NA-74994/1.000000-full-mammogram-images-24515/1-1.dcm

docker compose --profile gpu up --build \
  --abort-on-container-exit --exit-code-from gpu-smoke
docker compose --profile gpu down --volumes
```

The executor health check does not pass on socket existence alone. It requires
the manifest hashes, CUDA device, and functional native-operator probe to pass.
The smoke service then sends a warmup plus one measured full request through
Gunicorn and verifies both exact prediction hashes.

After a successful build, startup can be forced to use only local images and
runtime mounts. The internal network prevents service egress:

```bash
docker compose --profile gpu up --no-build --pull never \
  --abort-on-container-exit --exit-code-from gpu-smoke
```

## Benchmark profile

The benchmark is the same real HTTP path, not a model stub. Its result directory
is an explicit writable bind mount; every other runtime mount remains read-only.

```bash
mkdir -p benchmark-results
export VMS_BENCHMARK_RESULTS="$PWD/benchmark-results"
export VMS_RESULT_UID="$(id -u)"
export VMS_RESULT_GID="$(id -g)"
export VMS_BENCHMARK_RUNS=5
docker compose --profile benchmark up --build \
  --abort-on-container-exit --exit-code-from benchmark
```

`benchmark.json` records the public input hash, exact detector/classifier
hashes, one excluded warmup, and min/median/p95/max end-to-end latency. It does
not record clinical text, DICOM identifiers, filenames, or prediction IDs.

## Runtime limits and inspection

- RQ's standard worker reserves no batch or prefetched jobs and processes one
  job at a time. Queue capacity is one and automatic retry is disabled.
- The shared job store is a 1 GiB tmpfs volume by default; override only with
  `VMS_JOBS_SIZE`. Socket and metrics tmpfs volumes are separately bounded.
- Every service is non-root, drops all Linux capabilities, uses
  `no-new-privileges`, and has a read-only root filesystem.
- Only `127.0.0.1:8000` is published. Redis and the executor socket are never
  published.
- Shutdown gives the RQ work-horse 190 seconds and the executor/Gunicorn 30
  seconds to finish cleanup.

Inspect the built images without starting the model:

```bash
docker image inspect vision-model-serving-web:local \
  --format 'user={{.Config.User}} size={{.Size}}'
docker image inspect vision-model-serving-executor:local \
  --format 'user={{.Config.User}} size={{.Size}}'
docker run --rm --entrypoint python vision-model-serving-web:local -c \
  "import importlib.util; assert importlib.util.find_spec('torch') is None"
docker history --no-trunc vision-model-serving-executor:local
```

Image history and files must contain neither checkpoint filename, public input
hash, clinical text, evidence-archive name, nor a mounted host path. Native
`.so` files are expected; model `.pt` and `.pth` files are not.
