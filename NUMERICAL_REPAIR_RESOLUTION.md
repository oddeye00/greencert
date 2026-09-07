# Numerical repair and release resolution

The arithmetic issue recorded on 2026-09-06 is resolved for all 79 affected
historical continuation brackets. The original hold is preserved byte-for-byte
at `audit_history/numerical_release_hold_20260907.json`; the resolution record
is `audit_history/numerical_repair_resolution_20260907.json`. Neither the old
experiment records nor their issuance counts have been changed.

The legacy relative-only matrix-product bound could miss accumulated
subnormal rounding. For a length-16 dot product of `2^-537` and `2^-538`, the
exact result is eight minimum subnormals; the tested legacy upper bounds were
too small. The counterexample is preserved in
`results/legacy_subnormal_matmul_audit_20260907T015437.656557+0000.json`.
No historical neural bracket was shown false by that primitive example.

The corrected implementation includes a dot-length-dependent absolute
underflow allowance, explicit outward arithmetic, and a verified
optimizer-norm calculation. All affected neural continuations were recomputed
under separately identified repair methods. All 79 brackets survived, with
zero pending or nonretained cases. The scope is 63 unique jobs: 56 WDBC events,
7 digits events, and 16 modular-addition appendix events. The appendix events
do not enlarge the paper's 83-event headline population.

The clean public package independently reproduces all output margins and
brackets on Windows and ARM. Full neural replay is available; its three
study interfaces also passed a separate complete-window ARM check. See
`PUBLIC_NUMERICAL_REPLAY.md` for the artifact hashes and commands. The original
fixed-ownership ledger and numerical summary are supplied as provenance;
the path-free public package is the supported replay interface.

The hold was archived only after the full fixed ledger reauthenticated, the
public replays passed, and the reviewed manuscript sources matched their
recorded hashes. Archiving permits a new build, not an automatic publication:
the PDF/source rebuild and clean-repository audit remain mandatory.

The corrected continuation statement is conditional on the documented Arb
and ordinary-binary64 arithmetic assumptions, from an exact stored dyadic
checkpoint. It does not certify the preceding floating-point training
program, arbitrary BLAS implementations, or population generalization. It
does not change the numerical scope of the separate Transformer studies.
