# Transformer certificate theorem-to-code-to-test traceability

Every claim-relevant inequality below has a theorem source, implementation
site, and executable gate.  Missing links are listed at the end.

| Requirement | Theorem / assumption | Implementation | Test / audit |
|---|---|---|---|
| Fixed rectangular output Jacobian | `PROBE_JACOBIAN_THEOREM.md`, setting and fixed-operator assumption | `scripts/probe_jacobian_bound.py:183`, `make_gram_operator` | `scripts/test_probe_jacobian_bound.py:102`, Gate P3 |
| `Jv` by reverse-mode double backward | identity `d_w <J^T w,v> = Jv` | `scripts/probe_jacobian_bound.py:195` | Gate P3 compares against explicit `Jv`, relative error `1.9e-15` |
| `J^T w` and `J^T Jv` | PSD Gram construction | `scripts/probe_jacobian_bound.py:210` and fused apply | Gate P3, relative errors `6.4e-16` and `1.2e-15` |
| Gaussian power upper bound | `PROBE_JACOBIAN_THEOREM.md`, main theorem | `scripts/probe_jacobian_bound.py:129`, `gram_norm_bound` | Gates P1, P2, P4 |
| Exponent `1/(2q)` | `||J||^2=||J^T J||` | `scripts/probe_jacobian_bound.py:166` | Gate P2 known-SVD matrix and Gate P4 real model |
| Folded-normal `c_delta` | `PROBE_JACOBIAN_THEOREM.md`, proof | `scripts/probe_jacobian_bound.py:35` | Gate P1 across `(delta,m)` grid |
| Exogenous RNG independent of training | fixed-operator/random-probe assumption | `scripts/probe_jacobian_bound.py:58`, namespaced master nonce | `scripts/test_transformer_certificate_protocol.py:49`, Gate C2 |
| Collision-free instantiated streams | simultaneous-family assumption | `scripts/probe_jacobian_bound.py:81`, `ProbeRegistry` | Gate C2 enumerates all instantiated identities and rejects collisions |
| No unexpected or duplicate query | predeclared finite family | `ProbeRegistry.claim` | Gate C2 deliberately attempts both failures |
| Candidate selection cannot use probes | operator fixed before probes | candidate seal verified at `scripts/transformer_four_sweep_development_audit.py:73`; registry instantiated only after centerline construction | `scripts/test_transformer_certificate_protocol.py:78`, Gate C2b |
| Complete family-wise count | union bound in `PROBE_JACOBIAN_THEOREM.md` | `scripts/transformer_certificate_protocol.py:66` | Gate C1 checks 31 inclusive scan points, two operator families, and 14,424 maximum |
| Exactly four corrections | iterated recentering identity in `PROJECTED_HVP_SHADOWING_THEOREM.md` | `scripts/transformer_certificate_protocol.py:20`; `build_four_sweep_path` | runtime assertion in `scripts/transformer_four_sweep_development_audit.py:205`; Gate C2b |
| Signed correction residual identity | `VARIATIONAL_SHADOWING_THEOREM.md`, Theorem 4.3 and Corollary 4.4 | `scripts/matrix_free_mlp.py:198`, `signed_variational_recenter` | existing matrix-free tests plus recorded `delta_0,...,delta_4` in every audit row |
| Scaled momentum optimizer JVP | recentered theorem requires `DG(c_j)`; scaled map derived in the independent audit | `scripts/transformer_optimizer_probe.py:20` | `scripts/test_transformer_certificate_protocol.py:109`, Gate C3 adjoint test |
| Optimizer `DG^T DG` upper bound | same PSD-Gram theorem | `scripts/transformer_optimizer_probe.py:59` | Gate C3 PSD identity; complete burned audit |
| Block monomial simplex bound | deterministic monomial Lagrange calculation | `scripts/block_jet_bound.py:64` | `scripts/test_block_jet_bound.py:62`, Gate A, including 200k-direction checks |
| Mixed second/third terms | product and chain rules | `scripts/block_jet_bound.py:126` and `:142` | `scripts/test_block_jet_bound.py:84`, Gate A2, 100 randomized algebraic collapses |
| Scalar reduction | new block polynomial must reproduce shipped scalar assumptions | `BlockJet.scalar` plus `transformer_block_envelope` | Gate B at multiple radii and perturbed parameter points |
| Ball-valid activation values | all derivative bounds must hold on `B(c_j,epsilon_j)` | `scripts/transformer_block_envelope.py:259`, per-stage monotone closure | Gate E checks every named stage, not only the final output |
| Objective Hessian-Lipschitz bound | CE third-derivative composition | `scripts/transformer_block_envelope.py:332` | scalar-reduction and directional stress gates; value recorded at every audited step |
| Scaled optimizer-map Lipschitz modulus | `||DG(y+u)-DG(y)|| <= sqrt(2) eta L_H ||u||` | `scripts/transformer_four_sweep_development_audit.py:435` | recurrence executed on all burned candidates; zero observed violations |
| Recentered state recurrence | `VARIATIONAL_SHADOWING_THEOREM.md`, Theorem 4.3 | `scripts/transformer_four_sweep_development_audit.py:436` | per-step epsilon path plus actual-state audit |
| Margin transport | exact-center output plus first/second derivative radius | `scripts/transformer_four_sweep_development_audit.py:129` and margin radius near `:407` | guaranteed/possible counts recorded at every reached state |
| Persistent first-passage bracket | `PROJECTED_HVP_SHADOWING_THEOREM.md`, persistent corollary | `scripts/transformer_four_sweep_development_audit.py:117` | actual persistent event joined only after construction; aggregate audit |

## Open traceability items

The following are intentionally **not** marked complete:

1. There is no outward-rounded Transformer implementation; all inequalities are
   float64, matching the matrix-free MLP rigor tier.
2. There is no certified finite-window propagator-gain bound for momentum.  The
   current theorem multiplies one-step scalar bounds and therefore abstains by
   steps 26--27.
3. There is no fresh confirmation package because the burned development gate
   failed.  Preparing or running one would violate the prescribed go/no-go.
