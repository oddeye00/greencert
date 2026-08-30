# Independent post-seal hardening audit

## Digits exact-real continuation

- Issued/covered: 7/7.
- Brackets identical to GreenCert: 7/7.
- The unique signed-only event remains the singleton `[147,147]`.
- Maximum 192-bit state radius: `5.065764e-09`.
- Minimum strict output-logic slack: `1.456521e-05`.

## Block-batched Transformer probes

- Complete exact replays: 2/2.
- Median measured end-to-end speedup: 2.80x.
- Million-parameter matched projection: 18.28 h serial to 11.37 h batched (1.61x).
- All 16 probes, eight powers, committed streams, and failure budgets are unchanged.

## LayerNorm + AdamW derivative transport

- Two-block pre-LayerNorm Transformer: 102,400 parameters.
- AdamW optimizer state: 307,200 coordinates.
- Optimizer/Green adjoint errors: 5.443e-15/4.397e-15.
- Scope remains matrix-free primitives; global LayerNorm/AdamW jets and event certification are not claimed.
