"""Case-independent full-batch momentum engine with a one-way anchor barrier.

This is not a standalone experiment launcher. The production caller must
authenticate the public protocol/source/runtime seal before constructing an
engine, enforce resource limits, and durably store its records and checkpoint.
Only the separate disposition-authorized observer may continue a frozen
checkpoint. This engine has no thaw, resume, or post-selection update API.
"""
from dataclasses import asdict
import hashlib
import math

import numpy as np
import torch

from final_scale_protocol_v1 import first_event, validate_counts, validate_settings
from streaming_variational_centerline import build_streaming_transformer_centerline
from transformer_hvp_grokking import (TransformerConfig, flat_spec, flatten_parameters,
                                      gradient, logits, make_disjoint_split, make_template)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def array_sha(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def production_config(settings, rung_index):
    """Resolve a registered rung without instantiating a model or any data."""
    validate_settings(settings)
    require(type(rung_index) is int and 0 <= rung_index < len(settings["rungs"]), "invalid rung")
    rung, architecture, optimizer = settings["rungs"][rung_index], settings["architecture"], settings["optimizer"]
    rule, execution = settings["selection"], settings["execution"]
    return TransformerConfig(modulus=architecture["modulus"], model_dim=rung["model_dim"],
        hidden_dim=rung["hidden_dim"], heads=architecture["heads"], depth=architecture["depth"],
        train_fraction=settings["data"]["train_fraction"], learning_rate=optimizer["learning_rate"],
        momentum=optimizer["momentum"], weight_decay=optimizer["weight_decay"], loss=optimizer["loss"],
        normalization=architecture["normalization"], seed=rung["seed"], threads=execution["training_threads"],
        dtype=execution["dtype"], steps=rule["maximum_updates"], log_every=rule["inspection_stride"],
        checkpoint_every=rule["inspection_stride"])


class Engine:
    def __init__(self, config, *, expected_parameters, expected_counts, maximum_updates, guard, record):
        require(isinstance(config, TransformerConfig) and config.dtype == "float64" and
                config.loss == "cross_entropy" and config.normalization == "layernorm", "unsupported engine model")
        require(type(expected_parameters) is int and expected_parameters > 0 and
                type(maximum_updates) is int and maximum_updates >= 0, "invalid engine budget")
        require(type(expected_counts) is dict and set(expected_counts) == {"training", "trigger", "certification"}
                and all(type(v) is int and v > 0 for v in expected_counts.values()), "invalid dataset counts")
        require(callable(guard) and callable(record), "source/resource and durable record callbacks required")
        require(math.isfinite(config.learning_rate) and config.learning_rate > 0 and
                math.isfinite(config.momentum) and 0 <= config.momentum < 1 and
                math.isfinite(config.weight_decay) and config.weight_decay >= 0, "invalid optimizer constants")
        require(torch.are_deterministic_algorithms_enabled() and not torch.backends.mha.get_fastpath_enabled(),
                "deterministic runtime must be configured before engine construction")
        self.guard, self.record, self.config = guard, record, config
        self.maximum_updates, self.failed, self.frozen, self.step = maximum_updates, False, False, 0
        self.last_clock = None
        self._check()
        self.template = make_template(config)
        self.spec = flat_spec(self.template)
        self.parameter = flatten_parameters(self.template).detach().clone()
        self.velocity = torch.zeros_like(self.parameter)
        self.data = make_disjoint_split(config)
        require(self.parameter.device.type == "cpu" and self.parameter.dtype == torch.float64 and
                self.parameter.shape == (expected_parameters,), "registered architecture metadata differs")
        self.populations = {"training": (self.data[0], self.data[1]),
                            "trigger": (self.data[2], self.data[3]), "certification": (self.data[4], self.data[5])}
        require({name: len(pair[0]) for name, pair in self.populations.items()} == expected_counts,
                "registered dataset sizes differ")
        joined = torch.cat([pairs for pairs, _ in self.populations.values()])
        require(len(joined) == config.modulus**2 and len(set(map(tuple, joined.tolist()))) == len(joined),
                "dataset populations overlap or omit modular inputs")
        self.expected_counts = dict(expected_counts)
        self._check()
        self._record("engine_ready", {"config": asdict(config), "parameters": expected_parameters,
            "population_counts": expected_counts, "population_sha256": {
                name: {"pairs": array_sha(pairs), "labels": array_sha(labels)}
                for name, (pairs, labels) in self.populations.items()}, **self.state_identity()})

    def _check(self):
        require(not self.failed, "failed engine cannot be reused")
        require(not self.frozen, "frozen anchor cannot be continued by selector")
        try:
            self.guard()
        except BaseException:
            self.failed = True
            raise

    def _record(self, kind, payload):
        try:
            self.record(kind, payload)
        except BaseException:
            self.failed = True
            raise

    def state_identity(self):
        return {"step": self.step, "parameter_sha256": array_sha(self.parameter),
                "unscaled_velocity_sha256": array_sha(self.velocity)}

    def _count(self, parameter, population):
        pairs, labels = self.populations[population]
        with torch.no_grad():
            values = logits(parameter, pairs, self.template, self.spec)
        require(bool(torch.isfinite(values).all()), "nonfinite neural logits")
        return int((values.argmax(1) == labels).sum())

    def current_counts(self):
        self._check()
        try:
            result = {name: self._count(self.parameter, name) for name in self.populations}
            self._check()
            return result
        except BaseException:
            self.failed = True
            raise

    def clock_counts(self, horizon, sweeps, numeric_cap):
        self._check()
        require(type(horizon) is int and horizon > 0 and type(sweeps) is int and sweeps > 0
                and math.isfinite(numeric_cap) and numeric_cap > 0, "invalid clock policy")
        # Retire a preceding inspection's reference before allocating a new one.
        self.last_clock = None
        identity = self.state_identity()
        try:
            def check_step(j, state):
                self._check()
                return False
            path = build_streaming_transformer_centerline(self.config, self.template, self.spec,
                self.data[0], self.data[1], self.parameter, self.velocity, maximum_horizon=horizon,
                sweeps=sweeps, numeric_cap=numeric_cap, stop_when=check_step)
            reference = path["scaled_center"]
            require(path["horizon_reached"] == horizon and reference.shape == (horizon+1, 2*len(self.parameter)),
                    "incomplete clock reference")
            counts = []
            for row in reference:
                self._check()
                counts.append(self._count(row[:len(self.parameter)], "certification"))
            validate_counts(counts, evaluation_count=self.expected_counts["certification"], expected_length=horizon+1)
            require(self.state_identity() == identity and torch.equal(reference[0, :len(self.parameter)], self.parameter),
                    "clock modified the physical checkpoint")
            require(array_sha(reference) == path["centerline_sha256"].lower(), "clock content digest differs")
            self.last_clock = {"scaled_reference": reference.detach(), "counts": counts,
                "horizon": horizon, "sweeps": sweeps, "numeric_cap": numeric_cap,
                "diagnostics": path["diagnostics"], "reference_sha256": array_sha(reference), **identity}
            self._check()
            return list(counts)
        except BaseException:
            self.failed = True
            raise

    def advance(self):
        self._check()
        require(self.step < self.maximum_updates, "true-update budget exhausted")
        # The reference is not an optimizer state and is never used as one.
        self.last_clock = None
        self._record("update_intent", {"next_step": self.step+1, **self.state_identity()})
        self._check()
        try:
            next_velocity = self.config.momentum*self.velocity + gradient(self.parameter, self.data[0],
                self.data[1], self.template, self.spec, self.config)
            next_parameter = self.parameter - self.config.learning_rate*next_velocity
            require(bool(torch.isfinite(next_parameter).all()) and bool(torch.isfinite(next_velocity).all()),
                    "nonfinite optimizer state")
            self.parameter, self.velocity = next_parameter.detach(), next_velocity.detach()
            self.step += 1
            self._record("update_completed", self.state_identity())
            self._check()
        except BaseException:
            self.failed = True
            raise

    def freeze_candidate(self, selection, *, target, persistence):
        """Close the one-way barrier before making a selected snapshot available."""
        self._check()
        require(type(selection) is dict and selection.get("status") == "candidate_selected" and
                type(selection.get("anchor")) is int and selection["anchor"] == self.step and
                type(selection.get("predicted_offset")) is int and selection["predicted_offset"] > 0 and
                selection.get("future_continuation_executed") is False, "invalid selected checkpoint")
        require(self.last_clock is not None and self.last_clock["step"] == self.step and
                self.last_clock["parameter_sha256"] == array_sha(self.parameter) and
                self.last_clock["unscaled_velocity_sha256"] == array_sha(self.velocity), "selected clock absent or stale")
        require(first_event(self.last_clock["counts"], target=target, persistence=persistence) ==
                selection["predicted_offset"], "selected event differs from the recorded clock")
        self._record("candidate_frozen", {"selection": selection, **self.state_identity(),
            "clock_reference_sha256": self.last_clock["reference_sha256"]})
        self._check()
        self.frozen = True
        return {"parameter": self.parameter.detach().numpy().copy(),
                "unscaled_velocity": self.velocity.detach().numpy().copy(),
                "clock": self.last_clock, "config": asdict(self.config), "data": self.data,
                "normalization_eps": [(b.norm1.eps, b.norm2.eps) for b in self.template.blocks],
                "spec": {"names": list(self.spec.names), "shapes": [list(s) for s in self.spec.shapes],
                         "sizes": list(self.spec.sizes)}, "selection": dict(selection)}
