# Independent digits confirmation audit

Status: **PASS**

The auditor independently recomputed all 24 trigger-only anchors, verified the three-seal hash chain and every certificate hash, reconstructed persistent events directly from the post-seal accuracy trajectories, and did not import the experiment runner.

- Signed: 7 issued, 7 covered, across 6 seeds.
- Matched unsigned: 6 issued, 6 covered.
- Signed-only: seed 509, 90% gate, bracket and actual lead [147,147].
- Closure: 0.236945 signed versus 13.018189 unsigned.
- Realized family-wise failure bound: 5e-7.
