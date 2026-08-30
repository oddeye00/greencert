# Fresh-probe residual and Green-image theorems

## 1. Residual norm from scalar intervals

Let \(e\in\mathbb R^D\) be any vector fixed before a fresh probe block, and
draw \(g_1,\ldots,g_m\stackrel{\rm iid}{\sim}\mathcal N(0,I_D)\).  Suppose an
outward scalar evaluator proves

\[
g_i^\top e\in[\ell_i,u_i]
\]

for every probe.  Put

\[
B=\max_i\max\{|\ell_i|,|u_i|\},\qquad
c_\delta=\Phi^{-1}\!\left(\frac{1+\delta^{1/m}}2\right).
\]

**Theorem 1 (fresh-probe residual enclosure).**  With probability at least
\(1-\delta\), conditional on all pre-probe computation,

\[
\boxed{\|e\|_2\le B/c_\delta.}
\]

**Proof.**  For \(e\ne0\),
\(g_i^\top e/\|e\|\) are independent standard normals.  The probability that
all \(m\) absolute projections are below \(c_\delta\) is exactly
\((2\Phi(c_\delta)-1)^m=\delta\).  Outside that event,
\(B\ge\max_i|g_i^\top e|\ge c_\delta\|e\|\).  The zero case is immediate.
\(\square\)

The theorem remains valid when \(e\) depends on training, candidate selection,
earlier Green probes, and a computed secant, provided the residual block is
fresh and independent after those quantities are fixed.  A prespecified union
allocation covers multiple residuals.

## 2. Probe the propagated error, not its worst-case norm

Let \(K:U\to X\) be the finite-window Green operator, fixed before a fresh
block \(g_i\sim\mathcal N(0,I_X)\), and let \(e\in U\) be a fixed forcing
error.  If outward intervals contain

\[
\langle K^\top g_i,e\rangle_U=\langle g_i,Ke\rangle_X,
\]

then Theorem 1 applied to \(Ke\) gives

\[
\boxed{\|Ke\|_X\le B/c_\delta}
\]

with probability at least \(1-\delta\).  This is strictly more informative
than first proving \(\|e\|_U\le R\) and then paying \(\kappa R\) whenever the
realized residual avoids the worst-amplified Green direction.  A verifier can
always retain the smaller of the direct image bound and \(\kappa R\).

For the amplified response, write

\[
e_{\rm ar}=q^{[\lambda]}-\widetilde q^{[\lambda]}-d^y,
\qquad
d^y_j=\widetilde y_{j+1}-J_j\widetilde y_j-
\widetilde q_j^{[\lambda]}.
\]

Then

\[
Kq-\widetilde y
=K(q-q^{[\lambda]})+Ke_{\rm ar},
\]

so a direct image-probe bound \(\Gamma_{\rm ar}\ge\|Ke_{\rm ar}\|_X\)
permits

\[
\boxed{\beta=\|\widetilde y\|_X+\kappa\sigma_{\rm sec}
+\Gamma_{\rm ar}.}
\]

Unlike the forcing-space interface, the numerical term is not multiplied by
the worst-case gain \(\kappa\).  The exact adjoint products and scalar
intervals must themselves be enclosed; an approximate adjoint can be used only
after its recurrence residual is budgeted.

## 3. Two useful amplified-secant interfaces

Let \(q^{[\lambda]}\) be the exact amplified secant and
\(\widetilde q^{[\lambda]}\) its computed vector.  Apply Theorem 1 to

\[
e_q=q^{[\lambda]}-\widetilde q^{[\lambda]}.
\]

Then, with the allocated probability,

\[
\sigma_{\rm ar}=B_q/c_\delta
\]

is valid in the amplified-secant response corollary.  The same construction
applied independently to the causal recurrence residual supplies \(\tau_y\).
Thus a full-vector interval implementation is sufficient but not necessary.

There is an even simpler response-free option.  Apply Theorem 1 directly to
the exact amplified forcing \(q^{[\lambda]}\), rather than to its arithmetic
error.  If \(Q_\delta=B_q/c_\delta\), then

\[
\boxed{\beta=\kappa(\sigma_{\rm sec}+Q_\delta)}.
\]

This discards the directional gain of the second causal response, but removes
that response, its recurrence audit, and any full-vector interval.  It is
attractive when the amplified forcing is tiny compared with the analytic ray
discrepancy; a verifier may compute both bounds and retain the smaller under a
preallocated failure budget.

## 4. Scalar neural obligation

For scaled momentum, write the parameter response as \(a_j\) and split a
state probe as \(g_j=(g_{\theta,j},g_{w,j})\).  Set
\(w_j=g_{w,j}-g_{\theta,j}\).  The exact scalar projection of the amplified
forcing is

\[
g_j^\top q_j^{[\lambda]}
=\frac{\eta}{\lambda^2}\left[
D_{w_j}F(\theta_j+\lambda a_j)-D_{w_j}F(\theta_j)
-\lambda D^2_{w_j,a_j}F(\theta_j)
\right].
\]

For a sequence probe, its projection is the sum of these checkpoint
contributions.  Therefore each forcing-space residual projection can be
enclosed with scalar bivariate neural jets plus outward dot products with the
stored computed forcing.  For a Green-image probe, replace \(g\) by the exact
adjoint response \(K^\top g\).  No interval for all \(d\) gradient or HVP
coordinates is required.

## 5. Concrete proof budget

On the sealed horizon-52 amplified audit, \(\lambda=4096\) leaves
\(6.014744428490773\times10^{-20}\) of injection-space arithmetic and
recurrence budget after the analytic ray discrepancy.  For \(m=16\) fresh
probes and \(\delta=10^{-6}\),

\[
c_\delta=0.5558644877400047.
\]

If the entire remaining budget were assigned to secant arithmetic, each
outward scalar state-forcing projection could have radius up to

\[
3.3433828306300715\times10^{-20}.
\]

Before multiplication by \(\eta/\lambda^2\) (here \(\eta=1\)), this is an
outward scalar neural-jet tolerance of

\[
\boxed{5.609265592017213\times10^{-13}.}
\]

This does not complete the outward Transformer proof.  It identifies a small,
quantitative scalar target that replaces the previous full-vector interval
obligation.

A second nonce was then frozen after choosing a four-probe policy but before
seeing its draws.  With the same \(\delta=10^{-6}\), the fresh block gives
\(c_\delta=0.039643654651021064\).  Its point-projection value bounds the
stored forcing at 32.1347 times its observed norm, yet the response-free closure
retains \([28,28]\) with 16.2530 times forcing headroom.  Thus the demonstrated
policy needs four scalar sequence jets rather than a second Green solve.  Those
float64 points were a geometry-only development audit.  The frozen v2 execution
subsequently evaluated all \(51\times4=204\) scalar sequence jets with 192-bit
Arb intervals, formed the probe difference inside Arb, and independently
recomputed every interval and closure decision.  It gives

\[
\|q^{[\lambda]}\|_2
\le 1.3054822832359867\times10^{-29},
\]

retains \([28,28]\) with \(24.4136\times\) forcing headroom, and takes 9.80
minutes wall time on four CPU workers.  A superseded v1 run had formed one
probe subtraction in float64 before interval evaluation; it is retained only
as provenance and is not claim-bearing.  This closes the amplified-secant
scalar arithmetic conditional on the stored dyadic center, response, and probe
inputs.  Upstream Green/HVP/JVP/VJP products, derivative envelopes, state-norm
accumulation, and output margins remain outside this outward audit.

Probing the propagated residual directly moves the same closure slack into
state space, where it is

\[
\kappa(6.014744428490773\times10^{-20})
=2.889867730079382\times10^{-16}.
\]

With the same \(m,\delta,\lambda\), the corresponding aggregate weighted
scalar-jet radius before \(\eta/\lambda^2\) is

\[
\boxed{2.6950497758526715\times10^{-9}},
\]

a \(4804.64\times\) relaxation.  Its directions are adjoint Green responses,
so this is not a per-checkpoint or unweighted tolerance.  It quantifies the
benefit available once those adjoint products are outward-audited.
