# Primary-source novelty audit

Last updated: 2026-08-30.

This record makes the paper's novelty search inspectable. It is not a proof
that no related paper exists. It records the queries, primary sources, and
object-level distinctions used to avoid relying on a brittle “first known”
claim.

## Search protocol

The search combined exact-phrase and concept queries across arXiv, PMLR,
OpenReview, JMLR, Springer, IEEE, and publisher/DOI records. Representative
queries were:

- `optimizer trajectory first-passage certificate neural network training`
- `training trajectory shadowing certificate neural network optimizer`
- `verified numerical continuation neural network training trajectory`
- `grokking certificate first-passage training event`
- `signed defect Green operator neural training`
- `trajectory-adaptive stopping rule stochastic optimization`
- `certified neural approximation nonlinear dynamics`

Search results were followed to the primary paper or publisher record before a
comparison was entered below. Searches were repeated with “persistent event,”
“realized checkpoint,” “pseudo-orbit,” “radii polynomial,” “HVP,” and
“endogenous defect.” No result located on 2026-08-30 delivered the complete
GREENCERT object: an anchor-fixed realized optimizer orbit, signed endogenous
defect propagation, a finite-window nonlinear tube, and a persistent empirical
first-passage bracket or abstention.

## Closest objects checked

| Source | Object certified or predicted | Difference from GREENCERT |
| --- | --- | --- |
| Chow, Lin, and Palmer (1989–1994); Hammel et al. (1987); Sauer and Yorke (1991) | Shadowing or existence near pseudo-orbits | Classical dynamical-systems foundation; not neural-output first-passage transport from a realized training checkpoint |
| Orvieto and Lucchi (2019) | Shadowing viewpoint for optimization dynamics | Optimization interpretation rather than an executable event certificate |
| Wicker et al. (UAI 2021), [primary record](https://proceedings.mlr.press/v161/wicker21a.html) | Reach/avoid probabilities for iterative Bayesian-network predictions | Certifies model rollout under predictive uncertainty, not the optimizer training orbit |
| Mathiesen et al. (SAIV 2026), [publisher record](https://link.springer.com/chapter/10.1007/978-3-032-32357-6_4) | Error of a neural approximation to nonlinear dynamics | The network is a surrogate for external dynamics; the endogenous optimizer trajectory is not the certified object |
| Sosnin et al. (JMLR 2026), [primary record](https://www.jmlr.org/papers/v27/25-2206.html) | Parameter reachable sets under training-data perturbations | Robustness to exogenous data changes, not a directional certificate for one realized orbit and future event |
| Taheri et al. (IEEE OJCSYS 2026), [arXiv record](https://arxiv.org/abs/2512.20865) | Terminal robustness over poisoned training trajectories | Barrier/scenario guarantee over perturbation families rather than local first-passage timing |
| Aolaritei et al. (2026), [arXiv record](https://arxiv.org/abs/2608.25551) | Time-uniform stopping certificates for strongly convex stochastic optimization | Certifies distance/suboptimality from stochastic observations, not an empirical neural-output event along nonconvex training |
| Naughton (2026), [archival record](https://doi.org/10.5281/zenodo.21893060) | Exact-real endpoint and circuit properties of a fixed grokked Transformer checkpoint | A complementary fixed-checkpoint certificate, not a certificate of when the training trajectory reaches the event |
| Khanh et al. (2026), [arXiv record](https://arxiv.org/abs/2605.18845) | Calibrated grokking-delay forecast | Predictive timing law without a finite-window enclosure of the realized nonlinear trajectory |
| Meterez et al. (2026), [arXiv record](https://arxiv.org/abs/2607.21716) | Empirical validity of local quadratic training models at LLM scale | Establishes predictive reach of a local model; does not convert it to a first-passage certificate |
| Altıntaş et al. (ICML 2025), [primary record](https://proceedings.mlr.press/v267/altintas25a.html) | Sensitivity of neural-training trajectories to tiny perturbations | Motivates preserving the realized anchor; it does not provide a trajectory/event certificate |

## Claim boundary used in the paper

The paper does not claim new Green operators, Newton correction, Gaussian norm
probes, pseudo-orbit shadowing, HVPs, or interval arithmetic. Its contribution
is the verification object and executable construction: causal anchor-fixed
continuation, signed finite-window defect propagation, neural derivative
transport, and persistent first-passage bracketing with abstention. This is the
comparison a novelty claim should survive; a vague assertion that the
ingredients are individually unprecedented would not.

## Bibliographic corrections made during this audit

- The Mathiesen et al. citation was updated from its OpenReview manuscript to
  the peer-reviewed SAIV 2026 Springer chapter, DOI
  `10.1007/978-3-032-32357-6_4`.
- Kodali et al., arXiv:2503.22652, is cited as a 2025 preprint (its initial
  submission year) rather than by the year of its 2026 revision.
- The ICML 2025 trajectory-sensitivity paper by Altıntaş et al. was added to
  support the realized-anchor motivation.

## 2026-08-30 forcing-subspace follow-up

The parameter-only Green theorem prompted a narrower search for projected
Newton--Kantorovich arguments, product-space radii polynomials, componentwise
validated bounds, and goal-oriented adjoint estimators. Added queries included:

- "projected Newton Kantorovich nonlinear remainder Green operator"
- "radii polynomial finite dimensional projection nonlinear fixed point"
- "componentwise radii polynomials product space"
- "goal-oriented a posteriori nonlinear dynamical system dual weighted residual"
- "forcing subspace projected Green operator trajectory enclosure"

Primary sources checked include Lessard and Reinhardt's radii-polynomial
validation, van den Berg and Jaquette's componentwise product-space
Newton--Kantorovich construction
([publisher record](https://doi.org/10.1016/j.jde.2018.02.018)), and Becker and
Rannacher's dual-weighted-residual framework for quantities of interest
([publisher record](https://doi.org/10.1017/S0962492901000010)). These sources
occupy any broad claim that quadratic self-map tests, anisotropic product-space
enclosures, projection, or adjoint targeting are new.

The surviving claim is optimizer-specific. For scaled momentum, the nonlinear
Taylor remainder has the exact range $Bq=(-eta q,eta q)$, depends only on
parameter error, and the neural event also depends only on parameters. This
gives a closed, anchor-fixed parameter-sequence equation with causal operator
$P_theta K_H B$, and it can absorb checkpointwise curvature on the input side.
No searched source delivered that factorization as a finite-window certificate
for the realized neural-optimizer orbit and then transported it to a persistent
first-passage event.

The proof audit also exposed a suppressed causal shift in the first theorem
note: $K_H$ returns errors $e_1,...,e_H$, while update $j$ uses $e_j$. The norm
bound and completed audit remain valid because the shift has operator norm one.
The corrected theorem writes the shift explicitly and uses the stronger fact
that update-zero nonlinear forcing vanishes at the fixed anchor. A separately
sealed 15-case audit preserved every bracket but did not reduce the 96-sweep
structured baseline, so that refinement is reported as a negative systems
result rather than a speedup.
