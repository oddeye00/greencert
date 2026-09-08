"""Case-independent neural producers for the final-scale preparation.

Uses the already audited arbitrary-depth LayerNorm kernels. The caller must
bind parameter rows, dataset populations, source bytes and runtime before
using these outputs in a certificate. No future trajectory is evaluated.
"""
import hashlib
import math

import numpy as np
import torch
from flint import arb, ctx

from arb_deep_point_gram_bound_strategy import network_point_gains
from arb_native_row_bounds import psd_native_row_gain_upper
from arb_psd_moment_bounds import psd_moment_gain_upper
from arb_deep_envelope import neural_envelope, verified_weight_norms, saturation_constants, objective_drift
from arb_deep_transformer_forward import arb_deep_logits
from arb_matrix_bounds import upper_float
from transformer_hvp_grokking import gradient
from transformer_optimizer_probe import make_scaled_optimizer_jvp_vjp


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_population(pairs, labels, classes):
    pairs, labels = np.asarray(pairs), np.asarray(labels)
    require(pairs.ndim == 2 and pairs.shape[1] == 2 and len(pairs) > 0
            and np.issubdtype(pairs.dtype, np.integer) and labels.shape == (len(pairs),)
            and np.issubdtype(labels.dtype, np.integer), "invalid population layout")
    require(((pairs >= 0) & (pairs < classes)).all() and
            ((labels >= 0) & (labels < classes)).all(), "invalid token or label")
    return pairs, labels


def validate_parameter(parameter, spec):
    require(isinstance(parameter, np.ndarray) and parameter.dtype == np.float64 and
            parameter.shape == (sum(spec.sizes),) and np.isfinite(parameter).all(), "invalid parameter row")
    return parameter


def margins_from_logits(values, labels):
    require(values.nrows() == len(labels), "logit population differs")
    result = []
    for index, label in enumerate(labels):
        differences = [values[index, int(label)] - values[index, k]
                       for k in range(values.ncols()) if k != label]
        require(differences and all(v.is_finite() for v in differences), "nonfinite margin")
        lower = math.nextafter(float(min(v.lower() for v in differences)), -math.inf)
        upper = upper_float(min(v.upper() for v in differences))
        require(math.isfinite(lower) and math.isfinite(upper) and lower <= upper, "unresolved margin")
        result.append({"index": index, "label": int(label), "lower": lower, "upper": upper})
    return result


def anchor_margins(parameter, pairs, labels, spec, config, *, normalization_eps, precision_bits=192):
    validate_parameter(parameter, spec)
    pairs, labels = validate_population(pairs, labels, config.modulus)
    require(type(precision_bits) is int and precision_bits >= 64, "invalid point precision")
    old = ctx.prec
    ctx.prec = precision_bits
    try:
        values = arb_deep_logits(parameter, pairs, spec, config, normalization_eps=normalization_eps)
        return margins_from_logits(values, labels)
    finally:
        ctx.prec = old


def recenter_scaled(clock, train_pairs, train_labels, template, spec, config, *, guard):
    """The original extra float64 signed correction, with no case constants."""
    n = sum(spec.sizes)
    require(isinstance(clock, torch.Tensor) and clock.device.type == "cpu" and clock.dtype == torch.float64
            and clock.ndim == 2 and clock.shape[1] == 2*n and len(clock) >= 2
            and bool(torch.isfinite(clock).all()), "invalid scaled reference")
    response = torch.zeros_like(clock)
    for j in range(len(clock) - 1):
        guard()
        moved_velocity = config.momentum * clock[j, n:] + config.learning_rate * gradient(
            clock[j, :n], train_pairs, train_labels, template, spec, config)
        defect = torch.cat((clock[j, :n] - moved_velocity, moved_velocity)) - clock[j+1]
        jvp, _ = make_scaled_optimizer_jvp_vjp(clock[j, :n], train_pairs, train_labels, template, spec, config)
        response[j+1] = jvp(response[j]) + defect
        require(bool(torch.isfinite(response[j+1]).all()), "nonfinite float64 correction")
    corrected = clock + response
    require(bool(torch.isfinite(corrected).all()) and torch.equal(corrected[0], clock[0]), "reference anchor changed")
    guard()
    return corrected, response


def envelope_row(parameter, train_pairs, train_labels, evaluation_pairs, evaluation_labels,
                 spec, config, *, normalization_eps, include_training, radius=1e-8,
                 geometry_bits=(64, 96, 192), transport_bits=192, guard, persist):
    """Compute every registered example; partial results never return a row.

persist(kind, index, record) must durably bind records to the caller's row
identity. Caches are created locally for this exact parameter only. Geometry
fallbacks follow the supplied, prospectively fixed precision ladder.
"""
    validate_parameter(parameter, spec)
    require(config.loss == "cross_entropy" and getattr(config, "dropout", 0) == 0,
            "only deterministic cross-entropy training is supported")
    parameter = parameter.copy()
    parameter.setflags(write=False)
    train_pairs, train_labels = validate_population(train_pairs, train_labels, config.modulus)
    evaluation_pairs, evaluation_labels = validate_population(evaluation_pairs, evaluation_labels, config.modulus)
    require(type(include_training) is bool and type(radius) in (float, int) and math.isfinite(radius)
            and radius > 0 and type(transport_bits) is int and transport_bits >= 64, "invalid row policy")
    require(isinstance(geometry_bits, tuple) and geometry_bits and
            all(type(v) is int and v >= 64 for v in geometry_bits)
            and list(geometry_bits) == sorted(set(geometry_bits)), "invalid precision ladder")
    guard()
    old = ctx.prec
    caches, rows, outputs, drifts = {}, [], [], []
    try:
        populations = ([] if not include_training else [("training", train_pairs, train_labels)])
        populations += [("evaluation", evaluation_pairs, evaluation_labels)]
        for role, pairs, labels in populations:
            for index, (pair, label) in enumerate(zip(pairs, labels)):
                guard()
                attempts, chosen = [], None
                for bits in geometry_bits:
                    ctx.prec = bits
                    if bits not in caches:
                        try:
                            weights = verified_weight_norms(parameter, spec, config)
                        except (ValueError, OverflowError, ZeroDivisionError) as error:
                            attempts.append({"geometry_bits": bits, "stage": "weights",
                                             "error_type": type(error).__name__, "error": str(error)})
                            continue
                        guard()
                        persist("weights", bits, {"geometry_bits": bits, "weights": weights})
                        caches[bits] = weights
                    ctx.prec = bits
                    try:
                        gains, local, _ = network_point_gains(parameter, pair, spec, config,
                            normalization_eps=normalization_eps, gain_bound=psd_moment_gain_upper,
                            local_gain_bound=psd_native_row_gain_upper)
                        ctx.prec = transport_bits
                        jet, trace = neural_envelope(parameter, pair, spec, config, radius,
                            normalization_eps=normalization_eps, point_gains=gains, local_gains=local,
                            weight_norms=caches[bits])
                        constants = saturation_constants(jet.center.entries(), int(label), jet.distance)
                        margin = margins_from_logits(jet.center, [int(label)])[0]
                        chosen = {"geometry_bits": bits, "transport_bits": transport_bits,
                            "point_gains": gains, "local_gains": local, "logit_jet": jet.scalars(),
                            "loss_constants": {k: upper_float(v) for k, v in constants.items()},
                            "drift_upper": upper_float(objective_drift(jet, constants, config.learning_rate))
                            if role == "training" else None,
                            "margin_lower": margin["lower"], "margin_upper": margin["upper"], "trace": trace}
                        break
                    except (ValueError, OverflowError, ZeroDivisionError) as error:
                        attempts.append({"geometry_bits": bits, "error_type": type(error).__name__, "error": str(error)})
                guard()
                if chosen is None:
                    persist(role, index, {"status": "unresolved", "role": role, "index": index,
                                          "attempts": attempts})
                    raise ValueError("fixed geometry precision ladder exhausted")
                row = {"status": "verified", "role": role, "index": index, "pair": pair.tolist(), "label": int(label),
                       "attempts": attempts, **chosen}
                persist(role, index, row)
                rows.append(row)
                if role == "training":
                    drifts.append(row["drift_upper"])
                else:
                    outputs.append({"index": index, "label": int(label), "lower": row["margin_lower"],
                                    "upper": row["margin_upper"], "point_jacobian_upper": gains["logits"],
                                    "ball_hessian_upper": row["logit_jet"]["second"]})
        ctx.prec = max(256, transport_bits)
        mean = upper_float(sum((arb(v) for v in drifts), arb(0)) / len(drifts)) if drifts else None
        guard()
        return {"status": "complete_neural_row", "training_count": len(drifts),
                "evaluation_count": len(outputs), "mean_training_drift_upper": mean,
                "radius": radius, "output_bounds": outputs, "examples": len(rows),
                "parameter_sha256": hashlib.sha256(parameter.tobytes()).hexdigest(),
                "event_certificate_issued": False}
    finally:
        ctx.prec = old
