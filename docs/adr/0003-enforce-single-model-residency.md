---
status: accepted
---

# Enforce one accelerator-resident model

The long-lived GPU executor keeps artifact verification, CUDA ownership, and
the private RQ socket for its process lifetime, but retains at most one loaded
model. A same-model request may reuse the sole resident. A request for the
other stage drains inference, unloads and proves the old adapter unreachable,
synchronizes and clears allocator cache, then loads and warms the new stage. A
full request therefore ends with only MMBCD resident.

## Context

ADR 0002 correctly removed per-job pipeline reconstruction, but its later
dual-residency policy contradicted the assignment's explicit load/unload
contract. The lower-latency historical result is not authority to weaken a
hard requirement. `SPEC.md` already reconciles reuse and switching: keep one
runtime per process, reuse only the active model, and evict it when the next
stage needs the other model.

## Consequences

- `SingleResidencyRuntime` is the only residency implementation and exposes no
  retention option.
- Stable status permits zero or one known resident; active and resident model
  identity must match. Impossible socket payloads fail closed.
- Artifact readiness does not imply a model is resident. `inference_warm` is a
  separate model-specific fact.
- Full requests necessarily pay the detector-to-classifier switch; repeated
  full requests also reload the detector because the prior request ended with
  MMBCD.
- Historical dual-resident evidence remains archived and explicitly
  superseded. Current claims require fresh packaged L4 evidence.
