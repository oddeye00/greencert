# Primary-source novelty audit

Last updated: 2026-09-02.

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
| Chow, Lin, and Palmer (1989-1994); Hammel et al. (1987); Sauer and Yorke (1991) | Shadowing or existence near pseudo-orbits | Classical dynamical-systems foundation; not neural-output first-passage transport from a realized training checkpoint |
| Van Vleck (2000), [primary record](https://doi.org/10.1137/S1064827599353452) | Componentwise shadowing after rotation into finite-time growth/decay directions | A close antecedent for nonuniform tubes; it controls ODE shadowing distance rather than chronological optimizer-output rows from a fixed realized anchor |
| Hayes and Jackson (2003), [primary record](https://doi.org/10.1137/S0036142901399100) | Rigorous finite-time ODE shadows by inductive containment boxes | A close antecedent for local chronological closure; the shadow may perturb the initial condition and is not transported to a neural persistent-event margin |
| Chaudhry et al. (2021), [primary record](https://doi.org/10.1007/s10543-020-00825-0) | Adjoint a posteriori estimate of error in the first time a differential-equation functional crosses a threshold | Direct first-event antecedent; it estimates event-time error through a local linearization rather than enclosing the realized discrete path and returning a strict persistent bracket or abstention |
| Zou, Lie, and Marzouk (2026), [primary record](https://arxiv.org/abs/2603.20467) | Goal-oriented learning with error bounds for path-space observables, including mean first-hitting times | Learns a surrogate for external stochastic dynamics and bounds expected observables; it does not verify the optimizer trajectory that trains a neural network |
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
| Hadou et al. (CPAL 2026), [primary record](https://proceedings.mlr.press/v328/hadou26a.html) | Convergence and distribution-shift analysis for learned, fixed-depth stochastic unrolled optimizers | The optimizer itself is learned and analyzed for near-stationarity; it does not enclose a realized future training orbit or certify an event time |
| Shi et al. (L4DC 2026), [primary record](https://proceedings.mlr.press/v331/shi26a.html) | Branch-and-bound certified training of Lyapunov-stable neural controllers | Certifies a controller's Lyapunov condition over an input region, not the optimizer trajectory that trains the network |
| Kazanskii (2026), [arXiv record](https://arxiv.org/abs/2607.11666) | Representation-geometry intervention that changes grokking time | Empirical mechanism and control of delayed generalization, without a finite-window certificate for the realized crossing |
| Manir and Rupa (2026), [arXiv record](https://arxiv.org/abs/2603.25009) | Controlled empirical comparison of grokking across depth, architecture, activation, and regularization | Characterizes when grokking occurs; it does not certify a future first passage from a checkpoint |
| Thomas (2014), [primary record](https://arxiv.org/abs/1309.1275) | Polarization identity for symmetric multilinear maps | Supplies classical multilinear algebra; it does not construct blockwise neural derivative majorants or optimizer-event certificates |
| Berz and Hoffstatter (1998), [publisher record](https://doi.org/10.1023/A:1009958918582) | Taylor polynomials with interval remainder bounds | Establishes validated Taylor-model arithmetic for factorable maps; the paper does not study neural training trajectories |
| Schilling, Forets, and Guadalupe (AAAI 2022), [primary record](https://ojs.aaai.org/index.php/AAAI/article/view/20790) | Taylor-model/zonotope reachability for neural-network control systems | Preserves dependencies for exogenous plant/controller reachability, not a realized optimizer correction or training-event first passage |
| Sharifi and Fazlyab (2024), [arXiv record](https://arxiv.org/abs/2406.04476) | Derivative-preserving Hessian bounds for smooth neural networks | Bounds local input reachability through first- and second-order information, not an endogenous fourth-order optimizer remainder |
| Entesari and Fazlyab (L4DC 2026), [primary record](https://proceedings.mlr.press/v331/entesari26a.html) | Hierarchical Taylor bounds using Hessian Lipschitz constants | Closest neural derivative-transport antecedent; its object is input-set output reachability rather than an anchor-fixed optimizer orbit and persistent event |

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

## 2026-08-31 freshness pass

Immediately before the public preprint build, the search was repeated against
arXiv, PMLR, and OpenReview with the following queries:

- `neural network training trajectory certification shadowing optimizer orbit first passage`
- `grokking certified verification training dynamics certificate`
- `certified optimizer trajectory neural network training shadowing`
- `radii polynomial neural training trajectory Green operator certification`

The four 2026 records added above were opened at their primary PMLR or arXiv
pages and compared by delivered guarantee rather than keyword overlap. Hadou et
al. analyze a learned unrolled optimizer; Shi et al. certify Lyapunov stability
of the trained controller; Kazanskii intervenes on representation geometry;
and Manir and Rupa perform a controlled empirical grokking study. None encloses
the unique optimizer orbit continuing from a realized checkpoint and transports
that enclosure to a persistent first-passage bracket. The refresh therefore
does not change the paper's claim boundary, but it expands the inspectable set
of negative comparisons.

## 2026-08-31 directional-remainder follow-up

The polarized block theorem prompted a narrower search using:

- `neural network verification fourth derivative blockwise Taylor bound`
- `mixed directional derivative neural verification Taylor remainder`
- `polarization symmetric multilinear neural network verification`
- `Taylor model optimizer trajectory first passage certificate`

The search recovered the classical polarization identity, validated Taylor
models, neural-controller Taylor-model reachability, derivative-preserving
Hessian bounds, and the recent HiTaB hierarchy. These sources occupy broad
claims that polarization, Taylor remainders, dependency-preserving polynomial
arithmetic, or compositional neural derivative bounds are new. No opened
primary source kept three realized optimizer-correction directions while
maximizing only one free dual direction, inserted that bound into an
anchor-fixed nonlinear training tube, and transported it to a persistent event
bracket. The manuscript therefore claims the optimizer/event construction and
its measured sweep reduction, not new polarization or Taylor-model theory.

## 2026-09-02 chronological-row and first-event follow-up

The causal-row theorem prompted a targeted search for prior work that could
subsume either its nonuniform tube or its event output. Queries included:

- `causal Green operator rowwise shadowing finite time trajectory error`
- `componentwise shadowing finite time fixed point`
- `goal-oriented shadowing adjoint trajectory error dynamical systems`
- `a posteriori ODE trajectory quantity of interest first threshold time`
- `validated ODE event detection first crossing interval arithmetic`
- `row-wise Green operator trajectory error bound`

This pass found two closer lines than the earlier audit recorded. Van Vleck's
componentwise shadowing theorem rotates an ODE into finite-time Lyapunov
directions and uses direction-dependent local tolerances inside a sharper fixed
point result. Hayes and Jackson's containment method builds rigorous local boxes
inductively along a finite-time ODE shadow. These papers occupy any broad claim
that nonuniform, componentwise, or chronological trajectory enclosures are new.

Chaudhry, Estep, Stevens, and Tavener directly analyze error in the first time a
differential-equation functional reaches a threshold. Their adjoint estimator is
therefore a direct event-time antecedent, not merely generic goal-oriented error
analysis. Zou, Lie, and Marzouk additionally train stochastic dynamical-system
surrogates against error bounds for path-space observables, including mean
first-hitting times. Related validated hybrid simulation also localizes guard crossings
from interval trajectory enclosures. These results occupy any broad claim that
certified or a posteriori event timing is itself new.

The surviving distinction is narrower and more concrete. GREENCERT fixes the
initial state to the realized discrete optimizer checkpoint; propagates the
signed endogenous defect; uses the exact scaled-momentum forcing range to close
checkpoint-specific parameter rows; transports the resulting nonlinear tube
through every fixed-evaluation neural margin; and returns a persistent
first-passage bracket or abstention. No primary source opened in this pass
delivered that complete object. The manuscript now cites the closer antecedents
and states the difference by delivered guarantee rather than by a categorical
priority claim.
