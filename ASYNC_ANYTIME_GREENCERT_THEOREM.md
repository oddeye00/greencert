# Asynchronous anytime GreenCert theorem

## 1. Simultaneous power bounds

Let \(T_r:\mathbb R^{d_r}\to\mathbb R^{k_r}\), \(r\in\mathcal R\), be a
finite family of matrix-free operators and let \(A_r=T_r^*T_r\).  The family
may depend on the training data and checkpoint, but it must be fixed before,
and independent of, the Gaussian blocks used to certify it.  Conditional on
the realized operator family, draw for each \(r\) one committed block of
independent standard Gaussian vectors \(g_{r,1},\ldots,g_{r,m}\).  Independence
between different operators' blocks is convenient but is not required for the
union bound.  For every integer
\(q\geq1\), define

\[
Y_{r,q}=\max_{1\le i\le m}\|A_r^q g_{r,i}\|_2,
\qquad
U_{r,q}=\left(\frac{Y_{r,q}}{c_{\delta_r,m}}\right)^{1/(2q)},
\]

where \(0<\delta_r<1\) and

\[
c_{\delta,m}=\Phi^{-1}\!\left(\frac{1+\delta^{1/m}}2\right).
\]

**Theorem 1 (operator-specific anytime powers).** Conditional on the fixed
operator family, with probability at least
\(1-\sum_{r\in\mathcal R}\delta_r\), simultaneously for every operator
\(r\) and every inspected positive power \(q\),

\[
\boxed{\|T_r\|_2\le U_{r,q}.}
\]

Consequently, an algorithm may choose a different stopping power \(Q_r\) for
every operator, reveal powers in any adaptive order, and stop as soon as a
deterministic downstream certificate closes.  There is no union penalty over
power levels, stopping times, or deterministic closure tests.

### Proof

Fix \(r\) and a unit top eigenvector \(v_r\) of the positive semidefinite
matrix \(A_r\), with eigenvalue \(\lambda_r=\|T_r\|_2^2\).  The single event

\[
E_r=\left\{\max_i|\langle v_r,g_{r,i}\rangle|\ge c_{\delta_r,m}\right\}
\]

has probability \(1-\delta_r\).  On \(E_r\), for every \(q\ge1\),

\[
Y_{r,q}\ge
\max_i|\langle v_r,A_r^qg_{r,i}\rangle|
=\lambda_r^q\max_i|\langle v_r,g_{r,i}\rangle|
\ge\lambda_r^q c_{\delta_r,m}.
\]

Taking the \(2q\)-th root yields \(\|T_r\|_2\le U_{r,q}\) for all powers on
the same event.  A union bound over the predeclared operator identities proves
the family statement.  Since every adaptively selected bound is valid on that
single family event, optional stopping introduces no additional failure
probability.  Integrating the conditional statement over a random training
path gives the same unconditional guarantee. \(\square\)

The independence boundary is important.  Query order, power, and deterministic
certificate logic may adapt to revealed probe values.  An operator may not be
constructed from the same Gaussian block that is then used to certify that
operator under Theorem 1.

## 2. Predictable online operator families

A large predeclared universe is not necessary when every newly chosen operator
receives a fresh block.

**Theorem 2 (predictable confidence spending).** Let \((\mathcal F_n)\) be the
history after the first \(n\) operator queries.  Before query \(n\), let the
operator \(T_n\) and failure allocation \(\delta_n\) be
\(\mathcal F_{n-1}\)-measurable, with \(0<\delta_n<1\) for every queried
operator and

\[
\sum_{n\ge1}\delta_n\le\delta
\quad\text{almost surely}.
\]

Conditional on \(\mathcal F_{n-1}\), draw a fresh iid standard-Gaussian block
independent of \(T_n\), and form every power bound for \(T_n\) from that one
block.  Then, with probability at least \(1-\delta\), every bound at every
queried operator and every adaptively inspected power is valid.

**Proof.** Conditional on \(\mathcal F_{n-1}\), Theorem 1's single-operator
event fails with probability at most \(\delta_n\).  Hence

\[
\Pr\!\left(\bigcup_n E_n^c\right)
\le \sum_n\mathbb E[\Pr(E_n^c\mid\mathcal F_{n-1})]
\le \mathbb E\sum_n\delta_n\le\delta.
\]

The same-event argument within each query is simultaneous over all powers, so
there is still no stopping-time penalty. \(\square\)

For an indefinitely running monitor, one example is
\(\delta_n=6\delta/(\pi^2n^2)\).  For a sealed finite candidate manifest, a
more efficient allocation can be stratified by role.  If candidate \(c\) has
budget \(\delta_c\), one Green operator, and \(H_c\) output operators, the
predeclared assignment

\[
\delta_{c,K}=\frac{\delta_c}{2},\qquad
\delta_{c,G_j}=\frac{\delta_c}{2H_c}
\]

spends exactly \(\delta_c\).  This avoids calibrating the expensive Green
gain as though all \(H_c+1\) operators played identical roles.

## 3. Anchor-fixed drift and asynchronous GreenCert

For an anchor-fixed recurrence, \(h_0=z_0=e_0=0\).  Therefore the first
nonlinear injection is exactly

\[
N_0(h_0)=N_0(z_0)=0.
\]

Only transition-input states \(c_1,\ldots,c_{H-1}\) require Jacobian-drift
envelopes; \(c_H\) is not a transition input.  This explains the apparently
shifted indexing in the executable Transformer records and removes one
unnecessary derivative query.

GreenCert uses one causal Green operator \(K_H\) and output-Jacobian operators
\(G_j\) along the reference path.  Theorem 1 permits powers

\[
q_K,\quad q_{G,1},\ldots,q_{G,H}
\]

to differ.  Substitute the resulting upper bounds into the deterministic
one-shot closure and persistent-event margin inequalities.  Every bracket
issued by those inequalities is valid on the same prebudgeted family event.

A concrete online cascade is therefore valid:

1. evaluate every required output operator at power one;
2. advance the Green operator one power at a time and stop if the event closes;
3. if unresolved at the maximum Green power, advance output operators one
   power at a time, rechecking all stored Green bounds;
4. issue at the first successful pair or abstain after the frozen grid is
   exhausted.

The cascade changes cost, not the theorem's event.  Unqueried powers consume no
operator applications.  A stateful implementation stores current probe blocks
and resumes them; a checkpoint/recompute implementation trades additional
applications for lower memory.

## 4. Sparse persistent-event witnesses

Full output tubes at every time are sufficient but not logically necessary for
a first-passage bracket.  Let \(P_j\) denote the certified statement that the
gate holds at time \(j\), and \(F_j\) the certified statement that it fails.

**Proposition 3 (witness-sparse bracket).** A proposed bracket \([L,U]\) is
valid if

1. \(P_j\) holds for every \(j=U,\ldots,U+K-1\); and
2. there is a set \(W^-\) of certified failure times such that, for every
   \(t<L\), \(W^-\cap[t,t+K-1]\ne\varnothing\).

Indeed, condition 1 proves a persistent event by \(U\), while condition 2
exhibits a failed checkpoint in every earlier persistence window.  Thus no
persistent event occurs before \(L\).  A greedy interval-stabbing pass gives a
minimum-cardinality \(W^-\) for a fixed set of candidate failure times.

For singleton timing, event transport can therefore require only the \(K\)
success times and a sparse set of pre-event failure witnesses.  The executable
Transformer audit below separates the optimizer-drift and event-margin roles,
then fuses their shared traversals; it is post-seal implementation evidence and
does not change v3 issuance.

**Proposition 4 (role-separated derivative transport).** Suppose the optimizer
map depends on a training-output map \(f_{\rm tr}\), while the certified event
depends on a disjoint evaluation-output map \(f_{\rm ev}\).  It is sufficient to
certify:

1. derivative bounds for \(f_{\rm tr}\) only at the transition-input states
   \(1,\ldots,H-1\), for construction of the optimizer drift envelopes; and
2. derivative bounds for \(f_{\rm ev}\) only at the \(K\) success times and the
   chosen failure-witness times from Proposition 3, for event transport.

No trigger-output derivative operator and no joint
\(f_{\rm tr}\oplus f_{\rm trigger}\oplus f_{\rm ev}\) operator is required after
candidate selection.  If each role receives a valid independent or prebudgeted
operator-norm event, the response-centered state closure and witness-sparse
first-passage bracket hold on their intersection.

**Proof.** The Jacobian drift of the optimizer map factors only through
derivatives of the objective and hence through \(f_{\rm tr}\).  The output-margin
Taylor inequalities factor only through \(f_{\rm ev}\), and Proposition 3 uses
those inequalities only at its witness times.  The trigger map has no role once
the anchor is frozen.  Intersecting the role-specific confidence events and
applying Theorem 2 proves the claim. \(\square\)

**Corollary 4.1 (predictable acquisition and role fusion).** The witness times
in Proposition 3 need not be known before the certificate begins.  At query
\(n\), choose the next output time from the centerline margins and previously
returned certificates, then give that operator a fresh block and a predictable
allocation \(\delta_n\).  Stop when the success window is certified and the
acquired failures hit every earlier persistence window.  If
\(\sum_n\delta_n\le\delta\), the returned bracket is valid with probability at
least \(1-\delta\).

Moreover, when time \(j\) needs both training and event derivatives, one bound
on the direct-sum output map

\[
f_{\rm tr}\oplus f_{\rm ev}
\]

may replace two separate operator queries: coordinate projection gives
\(\|Df_{\rm tr}\|,\|Df_{\rm ev}\|\le
\|D(f_{\rm tr}\oplus f_{\rm ev})\|\).  Thus deterministic centerline slacks may
preplan likely witness times, fuse both roles there, and reserve separate event
queries for predictable fallbacks.

**Proof.** The next operator is measurable with respect to the pre-query
history, so Theorem 2 makes every acquired bound valid on one family event.
On that event, the stopping condition is exactly Proposition 3.  The direct-sum
claim follows because each component Jacobian is an orthogonal output
projection of the stacked Jacobian. \(\square\)

**Corollary 4.2 (affine launch-cost break-even).** Let derivative transport be
required at output times \(1,\ldots,H\), with training derivatives at
\(1,\ldots,H-1\) on \(n_{\rm tr}\) examples and event derivatives on a query
set \(Q\subseteq\{1,\ldots,H\}\) of size \(q\), using \(n_{\rm ev}\) examples.
Let the legacy dense operator use \(n_{\rm all}\) examples at every time.  If
one operator call on \(n\) examples costs \(C(n)=a+bn\), then

\[
\begin{aligned}
C_{\rm dense}&=Ha+bHn_{\rm all},\\
C_{\rm separate}&=(H-1+q)a
 +b\{(H-1)n_{\rm tr}+qn_{\rm ev}\},\\
C_{\rm fused}&=(H-1+\mathbf 1_{\{H\in Q\}})a
 +b\{(H-1)n_{\rm tr}+qn_{\rm ev}\}.
\end{aligned}
\]

Consequently fusion removes exactly
\((q-\mathbf 1_{\{H\in Q\}})a\) from the separate-role path.  Separate roles
beat the dense baseline only when

\[
b\bigl[Hn_{\rm all}-(H-1)n_{\rm tr}-qn_{\rm ev}\bigr]>(q-1)a,
\]

whereas fusion also removes the repeated-launch term.  This is a cost-model
statement, not a hardware-independent runtime theorem.

**Proof.** Count calls and example evaluations.  Direct-sum fusion replaces
the separate event call at every overlap by extra output coordinates in the
already-required training call; only a possible event-only call at time \(H\)
remains.  Subtraction gives the identities and inequality. \(\square\)

If one operator application scales linearly with the number of evaluated
examples, the corresponding example-level work is

\[
(H-1)n_{\rm tr}+|W|n_{\rm ev}
\]

instead of \(H(n_{\rm tr}+n_{\rm trigger}+n_{\rm ev})\), where \(W\) is the
non-anchor event-witness set.  This is an accounting identity, not a wall-time
claim; fixed framework overhead and batching determine the realized speedup.

A post-seal reconstruction of all 11 issued v3 brackets verifies the sparse
logic.  The minimum witness sets reduce event-output query times by a median
\(4.73\times\) (range \(1.00\)--\(8.31\times\)).  A predictable acquisition
policy using only centerline slacks and prior query outcomes also reconstructs
all 11, reducing 1,882 full output-time queries to 349, with median reduction
\(4.73\times\) (range \(1.00\)--\(8.08\times\)).  With the sealed 173/58/58
train/trigger/evaluation split, role-separated accounting reduces example-level
output-operator work by a median 36.3%, or \(1.57\times\).

A matched wall-time panel at sealed horizons 26, 52, 94, and 142 shows why the
fusion clause matters.  Preplanning event times, stacking training and event
outputs at overlaps, and reserving predictable fallbacks preserves all four
singleton brackets with no fallback query.  Dense event queries fall from 314
to 112 and example-pair work by 33.7%.  Aggregate output-operator and
output-phase wall speedups are 1.37x and 1.35x (per-case wall range
1.19--1.40x).  Fusion beats the dense and naive separate-role paths in all
four cases; the naive path itself regresses to 0.88x on the longest case.
Including shared centerline time gives a 1.11x aggregate replay speedup; the
sealed Green solve is shared and excluded.  These are post-seal method audits,
not changes to the prospective 11/19 result.  Records are
`results/transformer_v3_adaptive_witness_postseal_audit.json` and
`results/transformer_v3_role_sparse_panel_audit.json`.

## 5. Numerical-padding corollary

Suppose an exact-real certificate encloses a mathematical trajectory in a
pointwise radius \(R_j\), while a verified implementation-error calculation
encloses the deployed trajectory within \(\tau_j\) of that mathematical
trajectory.  Then the deployed trajectory lies in radius

\[
\boxed{R_j^{\rm deploy}=R_j+\tau_j.}
\]

If output arithmetic contributes an additional verified scalar error
\(\nu_j\), the event margin at step \(j\) is reduced by

\[
L_j(R_j+\tau_j)
+\frac12B_j(R_j+\tau_j)^2
+\nu_j.
\]

This corollary is only a triangle inequality, but it cleanly separates the
modal/shadowing theorem from outward-rounded implementation verification.  A
numerical padding sensitivity experiment does **not** establish a bound on
\(\tau_j\); it measures how large such a future verified bound could be before
the event certificate changes.

A subsequent 256-bit Arb audit closes two scalar pieces of this interface:
the folded-normal calibration and every
\((Y/c_\delta)^{1/(2q)}\) root are outward-enclosed conditional on each stored
binary64 probe norm \(Y\), and the result is maximized with the original stored
bound.  It repairs 13,052 scalar rows by at most a factor
\(1+8.9\times10^{-16}\) and preserves all 11 brackets.  This does not enclose
the HVP/VJP producing \(Y\), norm accumulation, neural jets, or margins, so it
is a strict partial hardening rather than a complete \(\tau_j\) construction.

### Residual-corrected approximate operators and sweeps

Approximate Gram products need not be hidden inside one global padding term.
If computed powers obey

\[
\max_i\|\widetilde v_{i,\ell+1}-A\widetilde v_{i,\ell}\|\le\xi_\ell,
\]

then the operator bound is the square root of the unique positive root of

\[
c_{\delta,m}R^q=\widetilde Y_q+
\sum_{\ell=0}^{q-1}\xi_\ell R^{q-1-\ell}.
\]

The Gaussian event remains simultaneous over every inspected power.  A
256-bit Arb implementation returns an outward binary64 supersolution from
verified residual norms; a strict 0.99 interior replay at the measured
binary64 sensitivity boundary preserves the sealed `[2,2]` event.

The signed response has the local identity

\[
K_Hs-\widetilde z=K_H(s-\widetilde s-d),\qquad
d_j=\widetilde z_{j+1}-J_j\widetilde z_j-\widetilde s_j,
\]

so \(\alpha=\kappa(\sigma+\tau)\) follows from verified defect and recurrence
residuals.  The same calculation bounds a computed second-order response by
\(\beta=\|\widetilde y\|_X+\kappa(\sigma_q+\tau_q)\), supplying both numerical
interfaces in the nonlinear closure from local residuals.  An approximate
recentering sweep has the companion identity

\[
r'_j=N_j(\widetilde z_j)+(r_j-\widetilde r_j)-d_j,
\]

and therefore

\[
\|r'_j\|\le\tfrac12M_j\|\widetilde z_j\|^2+\sigma_j+\tau_j.
\]

Thus reduced precision can be certified locally at the centerline, response,
second-order response, and randomized norm layers without discarding the signed
correction.  Arithmetic precision may be refined adaptively on the same fixed
probe event: valid local residuals change only the deterministic scalar root
and consume no additional failure probability.  Property
tests cover 960 inexact sweeps, 1,280 causal response cases, and 6,000
randomized inexact Gram cases.

## 6. Frozen v3 diagnostic boundary

The Transformer-v3 q-grid and padding audits are post-seal diagnostics.  They
do not alter the 11/19 frozen issuance result.  The stored traces show that an
operator-specific q grid could reduce measured cumulative operator time by
2.24--7.41x on the 11 issued cases (median 4.04x), and all 11 event brackets
survive an added \(10^{-8}\) radius floor.  Realized online wall-time claims
require an executable stopping implementation, supplied separately, rather
than treating this retrospective grid optimum as prospective evidence.

The stateful implementation has now been run in matched online and forced-q8
modes on the same sealed horizon-26 candidate.  It reproduces the same
\([2,2]\) bracket and every queried trace exactly while reducing logical Gram
applications by \(8.0\)x, measured operator time by \(8.03\)x, and measured
end-to-end time by \(2.19\)x.  This is one-candidate implementation evidence,
not a population-wide wall-time estimate.

A monolithic post-seal replay composes online Green stopping with direct-sum
role fusion on that same immutable candidate.  It stops both operator families
at q=1, uses no fallback, preserves `[2,2]`, and reduces example-pair work by
22.37%.  The same-process output phase improves 1.29x, but the matched complete
replay improves only 1.018x because centerline construction and the Green solve
dominate the short horizon.  This measured composition closes the earlier
systems-interface question and localizes the remaining cost.

An actual float32 implementation of every q=1 Green/output Gram product on the
same replay has maximum measured residual ratio \(3.36\times10^{-7}\), versus
the independently computed common admissible ratio 0.5999.  Residual-corrected
outward scalar roots preserve `[2,2]`; across four separately launched,
five-repeat alternating-order invocations, all 20 paired kernel timings improve,
with pooled median 1.83x and range 1.04--2.38x.
The residual comparison is binary64, not an outward exact-real neural-kernel
bound; it demonstrates practical mixed-precision headroom rather than a new
prospective or computer-assisted certificate.

A separate post-seal audit combines the time-resolved closure from
`HETEROGENEOUS_RECENTER_THEOREM.md` with the role-stratified budget above.  On
the hardest fresh issuance (seed 372, 70% gate, 246-step lead), the same sealed
probe traces and the same total \(10^{-6}\) family budget certify the unchanged
\([246,246]\) bracket at power three rather than power four.  The refinement
therefore removes 25% of Gram depth on this case.  Neither this recalibration
nor the time-resolved theorem changes the frozen 11/19 issuance result; both
are method-development evidence for a future sealed protocol.

A diagonally time-weighted Green theorem was also implemented and tested.  On
the immutable q=1 issuance it reduces the final radius only after requiring
q=2, doubling Green depth.  It is therefore retained as a mathematically valid
preconditioning option but rejected as a practical headline improvement.
