# Structured causal-row Transformer panel result

Status: frozen post-release systems/theorem audit passed.

This is not a new event-coverage experiment. It asks whether the
forcing-subspace causal row theorem can reproduce the already sealed
four-sweep Transformer brackets with fewer matrix-free linearized operator
sweeps. One row was disclosed for development; the other 14 had not been
evaluated with the structured-row construction before the frozen run. No
future outcome file was read.

## Frozen identity

- Protocol commit:
  `5aff03c2515e92569be81c06f0ec7aa3d2a24b42`
- Aggregate:
  `results/transformer_causal_structured_row_panel_audit.json`
- Aggregate SHA-256:
  `72956D00921CEF63D5720C136E549A25EF57DD0B91C6DC5ABEF820FE780C9BA2`
- Independent scalar verification:
  `results/transformer_causal_structured_row_panel_verification.json`
- Verification SHA-256:
  `9E80C7769F8AD5BD903E86B3A4E3915B9FAB5C2021DB329F636190E738EEECA9`

## Result

- Issued: 15/15 total and 14/14 structured-row holdouts.
- Brackets retained: 15/15; every bracket equals the released sealed bracket.
- Probe stopping: 13 cases at four probes and two cases at eight probes.
- Corrected-path hashes: 15/15 match.
- Derivative-domain checks: 15/15 pass.
- Transpose Green sweeps: zero.
- Future outcome reads: zero.
- Minimum strict event-logic slack: `1.4466087846244381e-6`.
- Maximum radius/domain ratio: `1.2836427679961793e-6`.

The disclosed development coordinate `(373, 0.70, 1280)` required eight probes
and retained `[247,247]`. The other eight-probe case was the previously unseen
structured-row holdout `(373, 0.80, 1760)`, which retained `[214,214]`.

## Matched cost accounting

The frozen promotion gate recorded 83 row-closure sweeps against the released
144-sweep total. That comparison is useful for the gate but is not the fairest
headline because the released total includes one common first-response sweep
per case. Matched accounting gives:

| Scope | Released Gram method | Structured row method | Reduction |
|---|---:|---:|---:|
| Random norm-estimator sweeps | 128 forward/transpose | 68 forward | 1.882x |
| Closure sweeps after the first response | 129 | 83 | 1.554x |
| Full post-reference linearized sweeps | 144 | 98 | 1.469x |
| Transpose sweeps | 64 | 0 | eliminated |

The full count is 98 because the row method uses 15 common first-response
sweeps, 68 random forward probes, and 15 signed deterministic responses. The
released method uses 15 common first responses, 64 forward plus 64 transpose
Gram sweeps, and one deterministic response.

The operator reduction did not yet produce an end-to-end wall-clock speedup in
this implementation. The structured audit took 1085.63 wall seconds versus
927.82 seconds for the released prefix panel; aggregate case time was 2484.47
versus 1945.83 seconds. The current implementation pays for sharper mixed
directional jets and repeats deterministic setup in the two process-isolated
eight-probe extensions. The result therefore supports a lower operator count
and a transpose-free construction, not a present end-to-end speed claim.

## Mechanism

For scaled momentum, every nonlinear Taylor remainder has the form
`Bq=(-eta*q,eta*q)` and depends only on the parameter error. The new corollary
therefore estimates chronological norms of `P K_i B` instead of the full-state
Green operator. It propagates the known signed quadratic and recurrence defect
as a deterministic response and scalarizes only the unresolved mixed
fourth-order term. Velocity-only forcing directions are absent from both the
Gaussian input space and the event radius.

## Reproduction

From the repository root:

```text
python scripts/audit_transformer_causal_structured_row_panel.py --workers 3
python scripts/verify_transformer_causal_structured_row_panel.py
```

The audit enforces a clean frozen commit, reruns five preaudit test programs,
hashes every claim-bearing source, uses fresh worker processes for memory
isolation, and reports the complete denominator. The probabilistic Green claim
has family failure probability at most `1e-6` under the same ideal-PRNG model
as the released Transformer certificate.
