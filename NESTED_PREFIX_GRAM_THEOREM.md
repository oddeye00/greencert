# Nested-prefix Gaussian Green enclosure

## Corollary (valid stopping over probe-count prefixes)

Let `T` be fixed before an iid Gaussian block `g_1,...,g_M` is drawn, put
`A=T^T T`, and fix increasing prefixes

```text
1 <= m_1 < ... < m_L <= M.
```

For each prefix choose `delta_l > 0`, with
`sum_l delta_l <= delta`, and define

```text
Y_l = max_{1 <= i <= m_l} ||A^q g_i||,
c_l = Phi^{-1}((1 + delta_l^(1/m_l))/2).
```

Then, with probability at least `1-delta`, simultaneously for every prefix,

```text
||T|| <= (Y_l/c_l)^(1/(2q)).
```

Consequently an algorithm may stop at the first prefix whose downstream
certificate closes. It pays only `m_l*q` Gram applications, and optional
stopping does not alter the failure guarantee.

### Proof

For a unit top eigenvector `v` of `A`, prefix `l` fails only if every one of
the first `m_l` absolute projections `|v^T g_i|` is smaller than `c_l`. By the
definition of `c_l`, that event has probability `delta_l`. On its complement,
some probe in the prefix gives

```text
Y_l >= ||A||^q c_l.
```

A union bound over the prespecified prefixes proves simultaneous validity.
Stopping is a measurable function of already valid prefix bounds and therefore
needs no additional penalty.

## Family-wise version

For a predeclared family of operators `T_n`, give operator `n` prefix budgets
`delta_{n,l}` and require

```text
sum_n sum_l delta_{n,l} <= delta_family.
```

Independent domain-separated Gaussian blocks are sufficient. The same union
argument validates every inspected prefix of every queried operator with
family-wise probability at least `1-delta_family`.

Each `T_n` must be fixed independently of its assigned Gaussian block before
that block is queried. The operators may share training data and need not be
mutually independent. Independence of the domain-separated blocks is not
needed by the union bound itself, but supplies the required marginal Gaussian
law conditional on the already fixed operator family under the audit's
ideal-PRNG model.

If downstream output-Jacobian enclosures were produced by an earlier random
family with failure upper bound `delta_output`, combining those enclosures with
this Green family gives the explicit bound

```text
delta_total <= delta_output + delta_family.
```

The cohort audit therefore reports both its new `10^-6` Green budget and the
combined `2*10^-6` output-plus-Green upper bound.

## Practical role in GREENCERT

The corrected-path cohort audit fixes prefixes `(4,8,16)`, Gram power one, and
equal family-wise spending before its nonce-derived vectors are generated. A
candidate stops only when both nonlinear closure and persistent output-event
transport succeed. Later prefix vectors are never generated after stopping.
