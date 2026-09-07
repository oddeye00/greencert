# 451,008-parameter outcome observation: DGX execution amendment

The user authorized this amendment after the certificate and its independent
replays were complete and before the candidate's future was observed.
The purpose is to complete the already registered observation despite the
source Windows host remaining below its 2,048 MiB free-memory requirement.

## Fixed experiment

The checkpoint, original unscaled velocity, four-block LayerNorm Transformer,
seed 94003, all 173 training and 58 certification examples and their order,
optimizer update, float64 precision, four CPU threads, one interop thread,
disabled MHA fast path, and deterministic-algorithm setting are unchanged.
The anchor is update 3925; the horizon is 64 updates; the event is the first
35-of-58 count crossing sustained for five updates. The issued bracket and
frozen point prediction are both offset 44. No reference path, derivative
envelope, Green query, closure, selector, event threshold, or stopping rule
is recomputed or tuned using the observation.

The optimizer remains CPU float64 momentum:
`v = momentum*v + gradient(theta)`;
`theta = theta - learning_rate*v`.
AdamW and larger models are not part of this amendment.

## Execution delta

The effective observer protocol may differ from the original in exactly
one field: its recorded runtime. Windows/AMD64 with Python 3.12.10 and
PyTorch 2.13.0+cpu becomes Linux/aarch64 with Python 3.12.3 and PyTorch
distribution version 2.13.0 (build 2.13.0+cu130). NumPy remains 2.5.2.
The CUDA-enabled build is used on the CPU; the observation is not a GPU run.
The full runtime capture is hash-bound in the amendment.

This is a hardware/platform/runtime-only amendment, not a claim of bitwise
identity between Windows and ARM floating-point trajectories. The
observation remains a float64 continuation, not an independent exact-real
orbit enclosure or a certificate for preceding floating-point training.

## One-shot execution and disclosure

The original protocol, terminal certificate, full recorded input graph,
completed ARM replay, observer implementation and logger remain unchanged.
A separate wrapper authenticates their hashes and checks that all
non-runtime fields of the effective protocol equal the original.

Before the remote observation, an exclusive delegation reservation occupies
the original Windows `registered_outcome_run` location. This prevents all
original source-host observers from starting a competing run. Its digest is
required by the DGX wrapper. The DGX logger makes its own durable exclusive
reservation before constructing the candidate model and writes an intent
record before each update. No automatic retry or resume is permitted.
These are cooperative filesystem/protocol controls, not an OS sandbox.

During execution, progress reports expose only the number of completed
observations. Count values, logits and crossing comparisons are inspected
only after the full registered window has completed and the final reveal
has been published. Wrong, absent and interrupted outcomes are retained;
none authorizes checkpoint reselection, a changed gate, or a silent retry.

The amendment and its implementation will be published before the candidate
observation. Afterward, the immutable reveal, all row hashes and a separate
comparison audit will be published whether or not the bracket is covered.
The result remains an outcome-sealed development case with a pre-observation
hardware amendment, not a fresh-seed confirmatory cohort.
