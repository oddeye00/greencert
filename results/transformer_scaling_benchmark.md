# Transformer matrix-free scaling and certificate cost

All profiles use the paper's smooth one-block, no-normalization Transformer
with float64 CPU arithmetic; only width changes.

| profile | parameters | HVP (s) | output Gram (s) | observed peak RSS (GiB) | projected H=300 core cert. | projected 300-step train |
|---|---:|---:|---:|---:|---:|---:|
| paper | 13,792 | 0.1901 | 0.1890 | 0.28 | 6.16 h | 6.68 s |
| 100k | 115,104 | 0.7134 | 0.3233 | 0.31 | 18.98 h | 50.42 s |
| 1m | 1,008,864 | 1.8553 | 2.3851 | 0.42 | 65.88 h | 228.83 s |

Measured on the frozen paper batch, candidate construction consumed
**19.44 aggregate hours**, with a median
of **58.02 minutes per constructed candidate**.
A measured 300-step continuation took a median **5.31 seconds**.

The benchmark establishes that the method's matrix-free HVP and output-Gram
primitives do not require dense Hessians at roughly one million parameters.
It simultaneously makes the current limitation explicit: the number of
probabilistic operator applications, not dense storage, dominates runtime.
The projection omits analytic-envelope and orchestration overhead and should
therefore be read as an operation-count estimate, not a measured full certificate.
