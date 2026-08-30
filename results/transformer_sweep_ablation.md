# Transformer 0--4 sweep ablation

This post-seal diagnostic uses the three candidate coordinates sealed during
development. The 24-seed fresh confirmation and its method remain untouched.

| sweeps | cumulative HVPs/case | median max scaled defect | exact clocks | median |timing error| | median incremental seconds |
|---:|---:|---:|---:|---:|---:|
| 0 | 300 | 8.569e-03 | 0/3 | 3.0 | 26.56 |
| 1 | 600 | 2.231e-03 | 3/3 | 0.0 | 24.58 |
| 2 | 900 | 5.849e-04 | 3/3 | 0.0 | 31.04 |
| 3 | 1200 | 2.582e-05 | 3/3 | 0.0 | 34.70 |
| 4 | 1500 | 5.935e-08 | 3/3 | 0.0 | 37.15 |

The ablation isolates the role of repeated signed correction: each sweep
costs exactly one additional HVP per transition, while the known path defect
contracts superlinearly until the event clock matches the exact rollout.
