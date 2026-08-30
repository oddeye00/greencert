# Modern Transformer matrix-free primitive audit

- Architecture: two-block pre-LayerNorm Transformer, 102,400 parameters.
- Optimizer: bias-corrected AdamW, 307,200-coordinate state.
- AdamW JVP/VJP adjoint relative error: `5.443e-15`.
- Horizon-3 Green adjoint relative error: `4.397e-15`.
- Peak RSS: `0.37 GiB`.
- Scope: exact matrix-free primitives only; no LayerNorm/AdamW jet envelope or event certificate is claimed.
