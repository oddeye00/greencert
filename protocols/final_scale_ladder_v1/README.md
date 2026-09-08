# Frozen final-scale production package

Read [the experiment protocol](../../FINAL_SCALE_LADDER_PROTOCOL.md) before
execution. This directory contains immutable candidate experiment inputs,
not a completed scientific result. Registered execution additionally
requires publication at an immutable public commit and passing release/CI
checks.

`protocol.json` pins `source_manifest.json`, `runtime.json`, and the source
bootstrap. `sources/` contains the complete 53-module production bundle;
there are no test or fixture entries. Do not reserialize the JSON: its exact
bytes and numeric representations are part of the seal.

Portable tests and their scope are documented in the linked protocol. The
production `check` command authenticates inputs without instantiating a
model. The `run` command is reserved for the single registered execution;
there is no resume or retry option.
