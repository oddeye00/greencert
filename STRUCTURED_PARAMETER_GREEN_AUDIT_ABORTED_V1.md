# Aborted structured parameter-Green v1 execution

On 2026-08-30, case index 0 was launched under
`STRUCTURED_PARAMETER_GREEN_AUDIT_PROTOCOL.md`. The process completed its
operator queries and then raised a `KeyError` while reading the matched
full-state sweep count from the prefix-panel row rather than the direct-panel
row.

The exception occurred before `save_case`: no bracket, probe norm, cache file,
or aggregate result was printed or written. A filesystem check found neither a
case cache nor `results/structured_parameter_green_transformer_audit.json`.

The v1 run is not reused. Version 2 fixes only that source-row lookup, changes
the cache version and master nonce, and is sealed independently before any v2
probe is generated.

