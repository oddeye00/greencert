# Staged direct-image/Gram Green enclosure

## Corollary (one-sided first release)

Let `T` be fixed independently of iid standard Gaussian probes `g_1,...,g_m`.
For a prescribed failure probability `delta`, define

```text
c = Phi^{-1}((1 + delta^(1/m))/2).
```

Then, with probability at least `1-delta`, the following bounds hold
simultaneously:

```text
||T|| <= max_i ||T g_i|| / c,                              (1)
||T|| <= (max_i ||T^T T g_i|| / c)^(1/2).                 (2)
```

More generally, all prespecified Gram powers are valid on this same event.

### Proof

Let `v` be a unit top right singular vector and write `T v = ||T|| u`.  For
every probe,

```text
||T g_i|| >= |u^T T g_i| = ||T|| |v^T g_i|,
||T^T T g_i|| >= ||T||^2 |v^T g_i|.
```

The Gaussian event `max_i |v^T g_i| >= c` has probability `1-delta` and implies
both inequalities.  Because both releases use the same event, inspecting (1)
before deciding whether to compute (2) incurs no additional probability.

## Prefix-family version

Combine the direct and Gram releases with the prespecified nested-prefix event
from `NESTED_PREFIX_GRAM_THEOREM.md`.  On each valid prefix event, both releases
hold.  Therefore a verifier may:

1. compute and cache `T g_i` for the new prefix vectors;
2. try all downstream closure and event inequalities using (1);
3. stop if they issue;
4. otherwise apply `T^T` to the cached images and use (2);
5. if needed, extend the same block to the next prefix.

One direct success removes every transpose sweep at that prefix.  A failure
wastes no forward work because the cached `T g_i` are exactly the inputs needed
for the Gram fallback.  The theorem changes neither the Gaussian family budget
nor any downstream nonlinear condition.

## Scope

The direct bound can be looser because components outside the top singular
direction contribute to `||Tg||` and because it divides by `c` rather than
`sqrt(c)`.  Its role is therefore computational screening, not replacement of
the Gram release.  Large-slack cases can stop after forward propagation; hard
cases recover the preceding certificate exactly.
