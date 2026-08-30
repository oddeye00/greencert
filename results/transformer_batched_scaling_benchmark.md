# Block-batched GreenCert probe benchmark

The same 16 Gaussian directions and the same m=16, q=8 theorem are
evaluated as one reverse-mode block rather than 16 serial calls.

| profile | parameters | HVP block speedup | output-Gram block speedup | matched serial H=300 | batched H=300 | matched speedup | peak RSS |
|---|---:|---:|---:|---:|---:|---:|---:|
| paper | 13,792 | 3.00x | 2.69x | 1.66 h | 0.62 h | 2.66x | 0.36 GiB |
| 100k | 115,104 | 3.34x | 1.80x | 5.70 h | 2.24 h | 2.54x | 0.53 GiB |
| 1m | 1,008,864 | 1.58x | 1.79x | 18.28 h | 11.37 h | 1.61x | 1.39 GiB |

The projection retains all 16 probes and eight powers. It is an operation-
matched serial-versus-batched wall-clock estimate, not a measured end-to-end
certificate. The older isolated-primitive projection is retained in JSON
for traceability but is not used as the acceleration denominator. The table
excludes analytic envelopes, orchestration, checkpoint I/O, and cache effects.
