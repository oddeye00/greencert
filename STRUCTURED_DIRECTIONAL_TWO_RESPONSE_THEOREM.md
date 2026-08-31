# Forcing-subspace directional two-response closure

Status: post-v1.1.0 theorem development. This result combines the signed
second-response interface with the scaled-momentum forcing factorization. It
does not alter any prospective count or released certificate.

## Sequence convention

Let \(E\) be the optimizer-state space and \(Q\) the parameter space. For
Jacobians \(\bar J_0,\ldots,\bar J_{H-1}:E\to E\), the anchor-fixed causal
Green operator \(\bar K_H:E^H\to E^H\) maps a forcing sequence
\(f=(f_0,\ldots,f_{H-1})\) to \(y=(y_1,\ldots,y_H)\), where

\[
y_0=0,\qquad y_{j+1}=\bar J_jy_j+f_j.
\]

Let \(P:E\to Q\) be the parameter projection and let
\(\mathcal P:E^H\to Q^H\) apply \(P\) blockwise. For scaled momentum,

\[
G(\theta,w)=(\theta-r,r),\qquad
r=\mu w+\eta\nabla F(\theta),\qquad \eta>0,
\]

define two state-injection channels

\[
Bq=(-\eta q,\eta q),\qquad Ch=(h,h),
\]

and their block-diagonal sequence maps \(\mathcal B,\mathcal C\). Every
state block \((u,v)\) has the unique decomposition

\[
(u,v)=B\frac{v-u}{2\eta}+C\frac{u+v}{2}.
\]

Moreover, the channels are orthogonal in the scaled Euclidean coordinates:

\[
\|Bq\|^2=2\eta^2\|q\|^2,\qquad
\|Ch\|^2=2\|h\|^2,\qquad \langle Bq,Ch\rangle=0.
\]

Every exact nonlinear remainder of the scaled-momentum map lies in the
\(B\)-channel. The \(C\)-channel is needed only for a general path-construction
or arithmetic residual.

## Structured directional response theorem

Suppose a corrected reference path has defect

\[
\bar s=\mathcal Bq+\mathcal Ch,
\]

where \(q,h\in Q^H\). Let \(\widetilde q\) approximate \(q\), and let an
anchor-fixed computed response
\(\widetilde y=(\widetilde y_1,\ldots,\widetilde y_H)\) have
\(\widetilde y_0=0\) and recurrence residual

\[
\rho_j=\widetilde y_{j+1}-\bar J_j\widetilde y_j-B\widetilde q_j.
\]

Split this residual exactly as

\[
\rho=\mathcal Br+\mathcal Cc,\qquad
r_j=\frac{\rho_j^{(w)}-\rho_j^{(\theta)}}{2\eta},\qquad
c_j=\frac{\rho_j^{(\theta)}+\rho_j^{(w)}}2.
\]

Assume certified bounds

\[
\|q-\widetilde q\|\le\sigma_q,\qquad
\kappa_B\ge\|\mathcal P\bar K_H\mathcal B\|,\qquad
\kappa_C\ge\|\mathcal P\bar K_H\mathcal C\|.
\]

Then the parameter response to the exact corrected defect satisfies

\[
\boxed{
\|\mathcal P\bar K_H\bar s\|
\le Y_\theta :=
\|\mathcal P\widetilde y\|
+\kappa_B(\sigma_q+\|r\|)
+\kappa_C(\|h\|+\|c\|).
}
\]

### Proof

The residual identity and the zero anchor give

\[
\widetilde y
=\bar K_H(\mathcal B\widetilde q+\rho)
=\bar K_H\{\mathcal B(\widetilde q+r)+\mathcal Cc\}.
\]

Consequently,

\[
\mathcal P\bar K_H\bar s-\mathcal P\widetilde y
=\mathcal P\bar K_H
 \{\mathcal B(q-\widetilde q-r)+\mathcal C(h-c)\}.
\]

The triangle inequality and the two declared operator bounds prove the
displayed result. \(\square\)

## Profiled parameter closure

Let \(b_0,\ldots,b_H\) be the corrected path, let
\(e_j=x_{a+j}-b_j\) be the realized anchored error, and suppose

\[
e_{j+1}=\bar J_je_j+\bar s_j+B R_j(Pe_j),\qquad e_0=0,
\]

with

\[
\|R_j(u)\|\le \frac{L_j}{2}\|u\|^2
\]

on declared parameter domains. Let

\[
p=(Pe_1,\ldots,Pe_H),\qquad
\mathcal S(p_1,\ldots,p_H)=(0,p_1,\ldots,p_{H-1}).
\]

Write \(Q_0:Q^{H-1}\to Q^H\) for the injection that prepends the exact
zero update-zero forcing block, and let
\(\mathcal D_L=\operatorname{diag}(L_0I,\ldots,L_{H-1}I)\). Certify

\[
\kappa_{L,0}\ge
\|\mathcal P\bar K_H\mathcal B\mathcal D_LQ_0\|.
\]

If

\[
D=1-2\kappa_{L,0}Y_\theta\ge0,\qquad
E_\theta=\frac{2Y_\theta}{1+\sqrt D},
\]

and every pointwise radius-\(E_\theta\) parameter ball stays in the declared
derivative domain, then

\[
\boxed{
\left(\sum_{j=1}^H\|P(x_{a+j}-b_j)\|^2\right)^{1/2}
\le E_\theta.
}
\]

### Proof

The response theorem gives a bound \(Y_\theta\) on the affine term
\(\mathcal P\bar K_H\bar s\). After writing
\(R_j(u)=L_j\widehat R_j(u)\) (and \(\widehat R_j=0\) when \(L_j=0\)),
the parameter error obeys

\[
p=\mathcal P\bar K_H\bar s+
\mathcal P\bar K_H\mathcal B\mathcal D_LQ_0
\widehat R_+(\mathcal Sp).
\]

As in the forcing-subspace theorem,

\[
\|\widehat R_+(\mathcal Sp)\|
\le \frac12\|p\|^2.
\]

Thus the right-hand side maps the radius-\(E_\theta\) ball into itself because
\(Y_\theta+\kappa_{L,0}E_\theta^2/2=E_\theta\). Brouwer gives a fixed
point; lifting it through the causal recurrence and using uniqueness of the
forward optimizer trajectory identifies it with the realized error. \(\square\)

## Cancellation-safe scaled-momentum specialization

Let \(c\) be a pre-correction path and let \(z_j=(a_j,b_j)\) be its first
signed variational correction. If \(z_{j+1}=J_jz_j+s_j\), then the defect of
the recentered path \(c+z\) is exactly

\[
G(c_j+z_j)-G(c_j)-J_jz_j=Bq_j,
\]

where

\[
q_j=\nabla F(\theta_j+a_j)-\nabla F(\theta_j)-H_ja_j.
\]

Use the cancellation-safe center quadratic approximation

\[
\widetilde q_j=\frac12D^3F(\theta_j)[a_j,a_j,\cdot].
\]

If \(\|D^4F\|\le L_{F,4,j}\) on the parameter segment, Taylor's theorem gives

\[
\boxed{
\sigma_q\le
\left(\sum_{j=0}^{H-1}
\left(\frac{L_{F,4,j}}6\|a_j\|^3\right)^2\right)^{1/2}.
}
\]

The update-zero term vanishes exactly because \(a_0=0\). Unlike the
full-state two-response bound, the expression contains no
\(\sqrt2\eta\): that scaling is already part of
\(\mathcal P\bar K_H\mathcal B\).

In exact arithmetic, with \(h=r=c=0\) and matched Taylor information, this
structured response interface cannot be worse than the full-state interface:

\[
\|\mathcal P\widetilde y\|\le\|\widetilde y\|,\qquad
\|\mathcal P\bar K_H\mathcal B\|
\le\sqrt2\eta\|\bar K_H\|.
\]

Under finite precision, the comparison also depends on the certified
arithmetic budgets. The exact \(B/C\) split prevents a tiny complementary
residual from forcing the entire known response back through a full-state
norm.

## Computational consequence

The signed second response costs one causal Jacobian sweep, but only its
parameter output must be accumulated. Randomized bounds for
\(\mathcal P\bar K_H\mathcal B\),
\(\mathcal P\bar K_H\mathcal C\), and the profiled nonlinear operator use
parameter-sized probe blocks and may share a streamed Jacobian pass under a
predeclared joint probability budget. If the implementation proves the
recurrence residual lies entirely in the \(B\)-channel (or encloses it as
zero), no \(C\)-channel query is needed.

The theorem therefore combines two gains that were previously separate:
directional cancellation in the known response and forcing-subspace
restriction in the unknown nonlinear closure. Whether either gain changes a
stopping decision is empirical; the matched Transformer audit must report
operator work and issuance together.
