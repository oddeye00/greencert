#!/usr/bin/env python3
"""Probabilistic centre-Jacobian enclosure via the PSD Gram operator A = J^T J.

Implements `PROBE_JACOBIAN_THEOREM.md`:

    ||J||_2 <= (Y / c_delta)^{1/(2q)}    with probability >= 1 - delta,
    Y = max_i ||A^q g_i||,   c_delta = Phi^{-1}((1 + delta^{1/m}) / 2).

`J` is never formed.  One application of `A` is one JVP followed by one VJP.

Adaptivity control
------------------
Probes are derived from a precommitted deterministic map from *operator
identity* to RNG seed (`probe_seed`).  The operator identity is built from
protocol coordinates only -- seed, gate, anchor, step, sweep -- never from any
probe outcome.  ``m``, ``q`` and ``delta`` are protocol constants passed in by
the caller and are not chosen per operator.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from math import sqrt
from statistics import NormalDist
from typing import Callable, Iterable

import torch
from torch import Tensor

from transformer_hvp_grokking import logits, unflatten_parameters

_NORMAL = NormalDist()


def c_delta(delta: float, m: int) -> float:
    """Phi^{-1}((1 + delta^{1/m}) / 2)."""
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0,1)")
    if m < 1:
        raise ValueError("m must be positive")
    return _NORMAL.inv_cdf(0.5 * (1.0 + delta ** (1.0 / m)))


def probe_seed(*identity: int) -> int:
    """Precommitted deterministic operator-identity -> RNG seed map (SHA-256).

    An ad-hoc integer mixer was tried first and collided within the enumerated
    protocol coordinates (``(321,2,0)`` against ``(321,1,65)``), which would have
    made two distinct operators share a probe stream and silently break the
    independence the union bound assumes.  SHA-256 of the canonical identity
    string is used instead.
    """
    payload = "|".join(str(int(part)) for part in identity).encode("ascii")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def namespaced_probe_seed(master_nonce: str, identity: tuple[int, ...]) -> int:
    """Derive one protocol probe stream from an independent committed nonce.

    ``probe_seed`` is retained for reproduction of the burned development gate.
    A probabilistic confirmation must additionally contain exogenous randomness
    that is independent of training and candidate selection.  The fresh
    protocol therefore commits a master nonce and domain-separates every full
    operator identity below it.
    """
    if not master_nonce:
        raise ValueError("master_nonce must be a non-empty committed string")
    if not identity:
        raise ValueError("operator identity must be non-empty")
    payload = (
        "certified-local-training-events/probe-v1\0"
        + master_nonce
        + "\0"
        + "|".join(str(int(part)) for part in identity)
    ).encode("ascii")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


class ProbeRegistry:
    """Runtime guard for a finite, predeclared probabilistic-operator family."""

    def __init__(self, allowed: Iterable[tuple[int, ...]], master_nonce: str):
        self.allowed = frozenset(tuple(int(v) for v in row) for row in allowed)
        if not self.allowed:
            raise ValueError("the allowed operator universe must be non-empty")
        self.master_nonce = master_nonce
        self.used: set[tuple[int, ...]] = set()
        streams: dict[int, tuple[int, ...]] = {}
        for identity in sorted(self.allowed):
            seed = namespaced_probe_seed(master_nonce, identity)
            if seed in streams and streams[seed] != identity:
                raise RuntimeError(
                    f"probe-stream collision: {streams[seed]} and {identity} -> {seed}"
                )
            streams[seed] = identity
        self.stream_count = len(streams)

    def claim(self, identity: tuple[int, ...]) -> int:
        identity = tuple(int(v) for v in identity)
        if identity not in self.allowed:
            raise RuntimeError(f"probabilistic operator is outside the frozen universe: {identity}")
        if identity in self.used:
            raise RuntimeError(f"probabilistic operator queried twice instead of cached: {identity}")
        self.used.add(identity)
        return namespaced_probe_seed(self.master_nonce, identity)

    def summary(self) -> dict:
        return {
            "allowed_operator_count": len(self.allowed),
            "collision_free_stream_count": self.stream_count,
            "queried_operator_count": len(self.used),
            "all_queries_predeclared": self.used <= self.allowed,
        }


@dataclass(frozen=True)
class ProbeConfig:
    probes: int          # m
    power: int           # q
    delta: float         # per-operator failure probability

    def c_delta(self) -> float:
        return c_delta(self.delta, self.probes)


@torch.no_grad()
def gram_norm_bound(
    apply_gram: Callable[[Tensor], Tensor],
    *,
    dimension: int,
    dtype: torch.dtype,
    device: torch.device,
    config: ProbeConfig,
    identity: tuple[int, ...],
    registry: ProbeRegistry | None = None,
) -> dict:
    """Apply the PSD-Gram theorem to any predeclared matrix-free operator."""
    if dimension < 1:
        raise ValueError("dimension must be positive")
    if registry is None:
        seed = probe_seed(*identity)
    else:
        seed = registry.claim(identity)
    generator = torch.Generator(device=device).manual_seed(seed)

    best = 0.0
    lower = 0.0
    applications = 0
    for _ in range(config.probes):
        vector = torch.randn(
            dimension, generator=generator, dtype=dtype, device=device
        )
        initial_norm = float(torch.linalg.vector_norm(vector))
        for _ in range(config.power):
            vector = apply_gram(vector)
            applications += 1
        final_norm = float(torch.linalg.vector_norm(vector))
        best = max(best, final_norm)
        if initial_norm > 0.0 and final_norm > 0.0:
            lower = max(
                lower,
                (final_norm / initial_norm) ** (1.0 / (2.0 * config.power)),
            )

    cd = config.c_delta()
    bound = 0.0 if best <= 0.0 else (best / cd) ** (1.0 / (2.0 * config.power))
    return {
        "identity": list(identity),
        "rng_seed": int(seed),
        "Y": best,
        "c_delta": cd,
        "operator_norm_upper_bound": bound,
        "operator_norm_lower_estimate": lower,
        "gram_applications": applications,
        "delta": config.delta,
        "probes": config.probes,
        "power": config.power,
    }


def make_gram_operator(parameter: Tensor, pairs: Tensor, template, spec):
    """Return ``v -> J^T (J v)`` using reverse mode only.  ``J`` is never formed.

    The fused multi-head-attention kernel has no forward-mode rule, so ``J v`` is
    obtained by the double-backward identity: with ``u(w) = J^T w`` (linear in
    ``w``), ``d<u(w), v> / dw = J v``.  One application therefore costs one
    forward pass and three reverse passes.
    """

    def forward(theta: Tensor) -> Tensor:
        return logits(theta, pairs, template, spec).reshape(-1)

    with torch.enable_grad():
        n_out = int(forward(parameter.detach()).numel())

    def jvp(vector: Tensor) -> Tensor:
        with torch.enable_grad():
            theta = parameter.detach().clone().requires_grad_(True)
            out = forward(theta)
            cotangent = torch.zeros(
                n_out, dtype=parameter.dtype, device=parameter.device, requires_grad=True
            )
            (transposed,) = torch.autograd.grad(
                out, theta, grad_outputs=cotangent, create_graph=True
            )
            (product,) = torch.autograd.grad(
                transposed, cotangent, grad_outputs=vector
            )
        return product.detach()

    def vjp(cotangent: Tensor) -> Tensor:
        with torch.enable_grad():
            theta = parameter.detach().clone().requires_grad_(True)
            out = forward(theta)
            (product,) = torch.autograd.grad(
                out, theta, grad_outputs=cotangent.detach()
            )
        return product.detach()

    def apply(vector: Tensor) -> Tensor:
        with torch.enable_grad():
            theta = parameter.detach().clone().requires_grad_(True)
            out = forward(theta)
            w = torch.zeros(n_out, dtype=parameter.dtype, requires_grad=True)
            (u,) = torch.autograd.grad(out, theta, grad_outputs=w, create_graph=True)
            (jv,) = torch.autograd.grad(
                u, w, grad_outputs=vector, retain_graph=True
            )
            (av,) = torch.autograd.grad(
                out, theta, grad_outputs=jv.detach(), retain_graph=False
            )
        return av.detach()

    return apply, {"jvp": jvp, "vjp": vjp, "output_dimension": n_out}


@torch.no_grad()
def jacobian_norm_bound(
    parameter: Tensor,
    pairs: Tensor,
    template,
    spec,
    config: ProbeConfig,
    identity: tuple[int, ...],
    registry: ProbeRegistry | None = None,
) -> dict:
    """Certified upper bound on ||J||_2 with failure probability <= config.delta."""
    apply, _ = make_gram_operator(parameter, pairs, template, spec)
    result = gram_norm_bound(
        apply,
        dimension=parameter.numel(),
        dtype=parameter.dtype,
        device=parameter.device,
        config=config,
        identity=identity,
        registry=registry,
    )
    result.update({
        "jacobian_norm_upper_bound": result["operator_norm_upper_bound"],
        "jvp_calls": result["gram_applications"],
        "vjp_calls": result["gram_applications"],
    })
    return result


@torch.no_grad()
def power_iteration_lower_estimate(
    parameter: Tensor, pairs: Tensor, template, spec, iterations: int = 40
) -> float:
    """NON-RIGOROUS development diagnostic: a lower estimate of ||J||_2."""
    apply, _ = make_gram_operator(parameter, pairs, template, spec)
    generator = torch.Generator().manual_seed(11)
    v = torch.randn(parameter.numel(), generator=generator, dtype=parameter.dtype)
    v = v / torch.linalg.vector_norm(v)
    value = 0.0
    for _ in range(iterations):
        w = apply(v)
        norm = float(torch.linalg.vector_norm(w))
        if norm <= 0.0:
            return 0.0
        value = norm
        v = w / norm
    return sqrt(value)


# --------------------------------------------------------------------------
# mechanical operator-count enumeration
# --------------------------------------------------------------------------
def permitted_operator_count(
    *,
    seeds: int,
    gates_per_seed: int,
    candidates_per_gate: int,
    horizon_steps: int,
    operators_per_step: int,
    screen_anchors_per_gate: int = 0,
    operators_per_screen_anchor: int = 0,
) -> dict:
    """Maximum number of probe-enclosed operators the protocol PERMITS.

    Derived from protocol constants only -- never from realised candidates.
    """
    certification = (
        seeds * gates_per_seed * candidates_per_gate * horizon_steps * operators_per_step
    )
    screening = (
        seeds * gates_per_seed * screen_anchors_per_gate * operators_per_screen_anchor
    )
    total = certification + screening
    return {
        "seeds": seeds,
        "gates_per_seed": gates_per_seed,
        "candidates_per_gate": candidates_per_gate,
        "horizon_steps": horizon_steps,
        "operators_per_step": operators_per_step,
        "screen_anchors_per_gate": screen_anchors_per_gate,
        "operators_per_screen_anchor": operators_per_screen_anchor,
        "certification_operators": certification,
        "screening_operators": screening,
        "permitted_operator_count": total,
    }


def allocate_budget(family_wise: float, operator_count: int) -> ProbeConfig | float:
    """Uniform per-operator budget delta_i = Delta / n_ops."""
    if operator_count < 1:
        raise ValueError("operator count must be positive")
    return family_wise / operator_count
