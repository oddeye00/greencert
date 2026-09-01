# Directional block remainder theorem, v2

## Status and relation to v1

This note makes explicit the slot symmetry used by the directional block
remainder proof. The frozen cohort audit remains tied to the immutable v1
theorem record with SHA-256
`F9C680DD7E6C47EFA8AE91753612464DF8622456A0F22179BF5174A6134B6AAF`.
Version 2 changes no bound, implementation, protocol, or result. It adds the
symmetrization lemma below so the factor `1/4` follows from stated assumptions
rather than an implicit convention.

## Block-majorant definition

Let the Euclidean parameter space be an orthogonal direct sum
\(\Theta=\bigoplus_{b=1}^B\Theta_b\), with projections \(\pi_b\). On a set
\(S\), suppose the symmetric fourth derivative of \(F\) has a nonnegative
polarized block majorant \(\mathcal C\): for every \(x\in S\) and every
\(h_1,\ldots,h_4\),

\[
 |D^4F(x)[h_1,h_2,h_3,h_4]|
 \le \mathcal C(s(h_1),s(h_2),s(h_3),s(h_4)),
 \qquad s(h)_b=\|\pi_bh\|_2,
\]

where \(\mathcal C\) is four-linear on the nonnegative block-radius cone and
has nonnegative coefficients.

## Lemma 1: symmetrization without loss

Define

\[
 \overline{\mathcal C}(v_1,v_2,v_3,v_4)
 =\frac1{4!}\sum_{\rho\in S_4}
 \mathcal C(v_{\rho(1)},v_{\rho(2)},v_{\rho(3)},v_{\rho(4)}).
\]

Then \(\overline{\mathcal C}\) is a nonnegative symmetric four-linear block
majorant for \(D^4F\). Moreover,

\[
 \overline{\mathcal C}(s,s,s,s)=\mathcal C(s,s,s,s).
\]

### Proof

For each permutation \(\rho\), symmetry of \(D^4F\) and validity of the
original majorant give

\[
 |D^4F(x)[h_1,h_2,h_3,h_4]|
 \le \mathcal C(s(h_{\rho(1)}),\ldots,s(h_{\rho(4)})).
\]

The left side is therefore bounded by the average of the 24 right sides.
Nonnegativity and four-linearity are preserved by averaging. On the diagonal,
all 24 summands are identical. \(\square\)

Henceforth write \(\mathcal P_4=\overline{\mathcal C}\) and

\[
 P_4(s)=\mathcal P_4(s,s,s,s).
\]

## Theorem 1: three-known, one-free contraction

Let \(r_b=\|\pi_bz\|_2\). Under the block-majorant assumption,

\[
 \sup_{x\in S}\|D^4F(x)[z,z,z,\cdot]\|_{2\to\mathbb R}
 \le \frac14\|\nabla P_4(r)\|_2.
\]

### Proof

For a unit dual direction \(u\), put \(t_b=\|\pi_bu\|_2\), so
\(t_b\ge0\) and \(\|t\|_2=1\). The symmetric four-linearity of
\(\mathcal P_4\) gives

\[
 \mathcal P_4(r,r,r,t)=\frac14\nabla P_4(r)^\top t.
\]

Applying the majorant and then Cauchy-Schwarz yields

\[
 |D^4F(x)[z,z,z,u]|
 \le \frac14\nabla P_4(r)^\top t
 \le \frac14\|\nabla P_4(r)\|_2.
\]

Taking the supremum over unit \(u\) proves the result. \(\square\)

## Corollary 1: directional gradient Taylor remainder

Let the segment \([\theta,\theta+z]\) lie in \(S\). Then

\[
\begin{aligned}
 \nabla F(\theta+z)
 &=\nabla F(\theta)+\nabla^2F(\theta)z
   +\tfrac12D^3F(\theta)[z,z]+R_3,\\
 \|R_3\|_2
 &\le \frac1{24}\|\nabla P_4(r)\|_2.
\end{aligned}
\]

### Proof

Taylor's integral remainder is

\[
 R_3=\frac12\int_0^1(1-t)^2
 D^4F(\theta+tz)[z,z,z,\cdot],dt.
\]

Theorem 1 and \(\frac12\int_0^1(1-t)^2dt=1/6\) give the claim.
\(\square\)

For the scaled-momentum state used by GREENCERT, the corresponding local
forcing contribution is at most

\[
 \sqrt2\,\eta\,\|\nabla P_4(r)\|_2/24.
\]

## Implementation correspondence

The implementation stores each derivative polynomial by sorted block-index
monomials. This is the diagonal polynomial of the symmetric polarization above;
the multinomial factors in product and composition rules count all slot
placements. The independent mixed jet computes the same
\(D_t^3D_\epsilon\) contraction directly. Thus v2 clarifies the mathematical
interface without changing any coefficient or cohort calculation.

## Scope

The theorem changes only the deterministic Taylor enclosure. It does not
alter Green calibration, the familywise probability budget, centerlines,
event logic, or the future-outcome firewall. Its current implementation covers
the one-block normalization-free Transformer used by the frozen audit.
