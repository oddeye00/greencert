#!/usr/bin/env python3
"""Certify the strict modular-addition accuracy gate from one checkpoint.

The train-state tube is the same exact-path Hessian tube used for the held-out
risk certificate.  Deployment is handled example by example.  For every
incorrect class q, the true-class margin

    m_iq(theta) = f_yi(theta) - f_q(theta)

is expanded to second order at the checkpoint.  The quadratic network has a
cubic output, so the frozen-path Taylor tail is available exactly.  A uniform
Jacobian bound then transports that prediction across the unknown state tube.
Counting definitely-correct and possibly-correct examples gives a rigorous
bracket for the first 95% accuracy event, or an abstention.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from modular_checkpoint_certificate import (
    CHECKPOINTS,
    RUN_JSON,
    DerivativeEnvelope,
    LocalGeometry,
    analytic_least_squares_hessians,
    canonical_config,
    derivative_envelope,
    exact_centered_state_tube,
)
from quadratic_modular_grokking import (
    Config,
    analytic_gradient,
    initialize,
    logits,
    make_split,
    objective,
    unpack,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "modular_accuracy_certificate.json"
FIGURE = ROOT / "figures" / "modular_accuracy_certificate.png"

ANCHOR = 152_000
HORIZON = 200


@torch.no_grad()
def output_taylor_paths(
    parameter: torch.Tensor,
    displacement: torch.Tensor,
    pairs: torch.Tensor,
    config: Config,
) -> dict[str, torch.Tensor]:
    """Return linear, quadratic, and exact logits on a displacement path.

    For ``f = V (W x)^2``, the only term beyond second order at an anchor is
    ``delta_V (delta_W x)^2``.  The identity is exact, not an approximation.
    """
    p, h = config.modulus, config.width
    w, v = unpack(parameter, config)
    x = torch.zeros((len(pairs), 2 * p), dtype=parameter.dtype)
    rows = torch.arange(len(pairs))
    x[rows, pairs[:, 0]] = 1.0
    x[rows, p + pairs[:, 1]] = 1.0
    anchor_activation = x @ w.T
    anchor_logits = config.output_scale * anchor_activation.square() @ v.T

    linear_rows: list[torch.Tensor] = []
    quadratic_rows: list[torch.Tensor] = []
    exact_rows: list[torch.Tensor] = []
    for delta in displacement:
        delta_w, delta_v = unpack(delta, config)
        delta_activation = x @ delta_w.T
        linear = anchor_logits + config.output_scale * (
            (2.0 * anchor_activation * delta_activation) @ v.T
            + anchor_activation.square() @ delta_v.T
        )
        second_order = config.output_scale * (
            delta_activation.square() @ v.T
            + (2.0 * anchor_activation * delta_activation) @ delta_v.T
        )
        cubic = config.output_scale * delta_activation.square() @ delta_v.T
        linear_rows.append(linear)
        quadratic_rows.append(linear + second_order)
        exact_rows.append(linear + second_order + cubic)
    return {
        "linear": torch.stack(linear_rows),
        "quadratic": torch.stack(quadratic_rows),
        "exact": torch.stack(exact_rows),
    }


def accuracy_from_logits(values: torch.Tensor, labels: torch.Tensor) -> np.ndarray:
    return (
        (torch.argmax(values, dim=-1) == labels[None, :])
        .to(torch.float64)
        .mean(dim=1)
        .numpy()
    )


def certified_accuracy_envelopes(
    model_logits: torch.Tensor,
    exact_frozen_logits: torch.Tensor,
    labels: torch.Tensor,
    epsilon: np.ndarray,
    path_norm: np.ndarray,
    envelope: DerivativeEnvelope,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return guaranteed-correct, possibly-correct, and margin-error arrays."""
    horizon, sample_count, class_count = model_logits.shape
    true_index = labels[None, :, None].expand(horizon, sample_count, 1)
    model_true = torch.gather(model_logits, 2, true_index)
    frozen_true = torch.gather(exact_frozen_logits, 2, true_index)
    model_margin = model_true - model_logits
    frozen_margin = frozen_true - exact_frozen_logits
    path_tail = torch.abs(frozen_margin - model_margin).numpy()

    common = np.zeros(horizon, dtype=np.float64)
    for step in range(horizon):
        radius = float(path_norm[step] + epsilon[step])
        jacobian_bound, _, _ = envelope.output_derivatives(radius)
        common[step] = np.sqrt(2.0) * jacobian_bound * epsilon[step]
    margin_error = path_tail + common[:, None, None]
    lower_margin = model_margin.numpy() - margin_error
    upper_margin = model_margin.numpy() + margin_error

    rows = np.arange(sample_count)
    lower_margin[:, rows, labels.numpy()] = np.inf
    upper_margin[:, rows, labels.numpy()] = np.inf
    definitely_correct = np.all(lower_margin > 0.0, axis=2)
    definitely_incorrect = np.any(upper_margin < 0.0, axis=2)
    guaranteed_correct = definitely_correct.sum(axis=1)
    possibly_correct = sample_count - definitely_incorrect.sum(axis=1)
    return guaranteed_correct, possibly_correct, margin_error


def counts_from_margin_error(
    model_logits: torch.Tensor,
    labels: torch.Tensor,
    margin_error: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Count certified and possible correct predictions from margin radii."""
    horizon, sample_count, _ = model_logits.shape
    true_index = labels[None, :, None].expand(horizon, sample_count, 1)
    true_value = torch.gather(model_logits, 2, true_index)
    margin = true_value - model_logits
    lower = margin.numpy() - margin_error
    upper = margin.numpy() + margin_error
    rows = np.arange(sample_count)
    lower[:, rows, labels.numpy()] = np.inf
    upper[:, rows, labels.numpy()] = np.inf
    definitely_correct = np.all(lower > 0.0, axis=2)
    definitely_incorrect = np.any(upper < 0.0, axis=2)
    return definitely_correct.sum(axis=1), sample_count - definitely_incorrect.sum(axis=1)


@torch.no_grad()
def margin_gradient_path(
    parameter: torch.Tensor,
    displacement: torch.Tensor,
    pair: torch.Tensor,
    label: int,
    competitor: int,
    config: Config,
    quadratic: bool,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray, float]:
    """Analytic target gradients and derivative norms for one logit margin."""
    p, h = config.modulus, config.width
    scale = float(config.output_scale)
    w, v = unpack(parameter, config)
    x = torch.zeros(2 * p, dtype=parameter.dtype)
    x[pair[0]] = 1.0
    x[p + pair[1]] = 1.0
    activation = w @ x
    readout_difference = v[label] - v[competitor]

    gradients: list[torch.Tensor] = []
    gradient_tail_norm = np.zeros(len(displacement), dtype=np.float64)
    hessian_mismatch = np.zeros(len(displacement), dtype=np.float64)
    for step, delta in enumerate(displacement):
        delta_w, delta_v = unpack(delta, config)
        delta_activation = delta_w @ x
        delta_readout_difference = delta_v[label] - delta_v[competitor]

        exact_w_coefficient = (
            2.0
            * scale
            * (readout_difference + delta_readout_difference)
            * (activation + delta_activation)
        )
        exact_v_coefficient = scale * (activation + delta_activation).square()
        exact_w_gradient = exact_w_coefficient[:, None] * x[None, :]
        exact_v_gradient = torch.zeros((p, h), dtype=parameter.dtype)
        exact_v_gradient[label] = exact_v_coefficient
        exact_v_gradient[competitor] = -exact_v_coefficient
        exact_gradient = torch.cat((exact_w_gradient.reshape(-1), exact_v_gradient.reshape(-1)))

        if quadratic:
            model_w_coefficient = 2.0 * scale * (
                readout_difference * (activation + delta_activation)
                + delta_readout_difference * activation
            )
            model_v_coefficient = scale * (
                activation.square() + 2.0 * activation * delta_activation
            )
        else:
            model_w_coefficient = 2.0 * scale * readout_difference * activation
            model_v_coefficient = scale * activation.square()
        model_w_gradient = model_w_coefficient[:, None] * x[None, :]
        model_v_gradient = torch.zeros((p, h), dtype=parameter.dtype)
        model_v_gradient[label] = model_v_coefficient
        model_v_gradient[competitor] = -model_v_coefficient
        model_gradient = torch.cat((model_w_gradient.reshape(-1), model_v_gradient.reshape(-1)))
        gradients.append(model_gradient)
        gradient_tail_norm[step] = float(torch.linalg.vector_norm(exact_gradient - model_gradient))

        # Frobenius norms are rigorous operator-norm upper bounds.  Since one
        # modular input has ||x||^2=2, the block formulas simplify exactly.
        moved_readout = readout_difference + delta_readout_difference
        moved_activation = activation + delta_activation
        moved_hessian_norm = abs(scale) * np.sqrt(
            16.0 * float(torch.sum(moved_readout.square()))
            + 32.0 * float(torch.sum(moved_activation.square()))
        )
        if quadratic:
            hessian_mismatch[step] = abs(scale) * np.sqrt(
                16.0 * float(torch.sum(delta_readout_difference.square()))
                + 32.0 * float(torch.sum(delta_activation.square()))
            )
        else:
            hessian_mismatch[step] = moved_hessian_norm

    anchor_hessian_norm = abs(scale) * np.sqrt(
        16.0 * float(torch.sum(readout_difference.square()))
        + 32.0 * float(torch.sum(activation.square()))
    )
    return (
        torch.stack(gradients),
        gradient_tail_norm,
        hessian_mismatch,
        anchor_hessian_norm if quadratic else 0.0,
    )


@torch.no_grad()
def target_projected_margin_remainder(
    parameter: torch.Tensor,
    displacement: torch.Tensor,
    epsilon: np.ndarray,
    injected: np.ndarray,
    eigenvectors: torch.Tensor,
    factors: torch.Tensor,
    model_logits: torch.Tensor,
    exact_frozen_logits: torch.Tensor,
    pair: torch.Tensor,
    label: int,
    competitor: int,
    config: Config,
    quadratic: bool,
) -> np.ndarray:
    """Theorem-3 remainder with the margin-aligned Duhamel propagator."""
    gradients, gradient_tail, mismatch, model_hessian_norm = margin_gradient_path(
        parameter,
        displacement,
        pair,
        label,
        competitor,
        config,
        quadratic,
    )
    modal_gradients = gradients @ eigenvectors
    absolute_factors = torch.abs(factors)
    powers_squared = torch.pow(
        absolute_factors[None, :],
        2.0 * torch.arange(len(displacement), dtype=torch.float64)[:, None],
    )
    sigma = np.zeros(len(displacement), dtype=np.float64)
    for step in range(1, len(displacement)):
        propagated_norm = torch.sqrt(
            powers_squared[:step] @ modal_gradients[step].square()
        ).numpy()
        sigma[step] = config.learning_rate * float(
            np.dot(propagated_norm, injected[step - 1 :: -1])
        )

    model_margin = (model_logits[:, label] - model_logits[:, competitor]).numpy()
    exact_margin = (
        exact_frozen_logits[:, label] - exact_frozen_logits[:, competitor]
    ).numpy()
    path_tail = np.abs(exact_margin - model_margin)
    third_derivative_norm = 12.0 * np.sqrt(2.0) * abs(config.output_scale)
    omega = mismatch + third_derivative_norm * epsilon
    return (
        sigma
        + 0.5 * model_hessian_norm * epsilon**2
        + path_tail
        + gradient_tail * epsilon
        + 0.5 * omega * epsilon**2
    )


def unresolved_margin_pairs(
    model_logits: torch.Tensor,
    labels: torch.Tensor,
    margin_error: np.ndarray,
) -> list[tuple[int, int]]:
    horizon, sample_count, class_count = model_logits.shape
    true_index = labels[None, :, None].expand(horizon, sample_count, 1)
    margin = (torch.gather(model_logits, 2, true_index) - model_logits).numpy()
    unresolved: list[tuple[int, int]] = []
    for sample in range(sample_count):
        for competitor in range(class_count):
            if competitor == int(labels[sample]):
                continue
            lower = margin[:, sample, competitor] - margin_error[:, sample, competitor]
            upper = margin[:, sample, competitor] + margin_error[:, sample, competitor]
            if np.any((lower <= 0.0) & (upper >= 0.0)):
                unresolved.append((sample, competitor))
    return unresolved


def event_bracket(
    guaranteed_correct: np.ndarray,
    possibly_correct: np.ndarray,
    required_correct: int,
) -> list[int] | None:
    """Bracket the first event ``correct_count >= required_correct``."""
    lower = 0
    while lower < len(possibly_correct) and possibly_correct[lower] < required_correct:
        lower += 1
    upper_candidates = np.flatnonzero(guaranteed_correct >= required_correct)
    if len(upper_candidates) == 0:
        return None
    upper = int(upper_candidates[0])
    return None if lower > upper else [int(lower), upper]


def persistent_event_bracket(
    guaranteed_correct: np.ndarray,
    possibly_correct: np.ndarray,
    required_correct: int,
    persistence: int,
) -> list[int] | None:
    """Bracket the first threshold crossing sustained for ``persistence`` steps.

    A start is possible only if every upper count in its persistence block
    reaches the threshold, and is guaranteed if every lower count does.  The
    first possible and first guaranteed starts therefore form a rigorous
    first-passage bracket.  ``persistence=1`` reduces to :func:`event_bracket`.
    """

    if persistence < 1:
        raise ValueError("persistence must be positive")
    if len(guaranteed_correct) != len(possibly_correct):
        raise ValueError("guaranteed and possible paths must have equal length")
    if persistence == 1:
        return event_bracket(
            guaranteed_correct, possibly_correct, required_correct
        )
    starts = len(guaranteed_correct) - persistence + 1
    if starts <= 0:
        return None
    possible_start = np.asarray([
        np.all(possibly_correct[j : j + persistence] >= required_correct)
        for j in range(starts)
    ])
    guaranteed_start = np.asarray([
        np.all(guaranteed_correct[j : j + persistence] >= required_correct)
        for j in range(starts)
    ])
    lower_candidates = np.flatnonzero(possible_start)
    upper_candidates = np.flatnonzero(guaranteed_start)
    if len(lower_candidates) == 0 or len(upper_candidates) == 0:
        return None
    lower = int(lower_candidates[0])
    upper = int(upper_candidates[0])
    return None if lower > upper else [lower, upper]


@torch.no_grad()
def refine_accuracy_crossing(
    start: int,
    stop: int,
    parameter: torch.Tensor,
    train_pairs: torch.Tensor,
    train_labels: torch.Tensor,
    test_pairs: torch.Tensor,
    test_labels: torch.Tensor,
    config: Config,
) -> tuple[int, np.ndarray, np.ndarray]:
    required = int(np.ceil(config.test_accuracy_gate * len(test_pairs)))
    accuracies: list[float] = []
    parameters: list[np.ndarray] = []
    for absolute_step in range(start, stop + 1):
        test_logits = logits(parameter, test_pairs, config)
        correct = int(torch.sum(torch.argmax(test_logits, dim=1) == test_labels))
        accuracies.append(correct / len(test_pairs))
        parameters.append(parameter.numpy().copy())
        if correct >= required:
            return absolute_step, np.asarray(accuracies), np.stack(parameters)
        gradient = analytic_gradient(parameter, train_pairs, train_labels, config)
        parameter.add_(gradient, alpha=-config.learning_rate)
    raise RuntimeError("the accuracy gate was not reached inside the refinement interval")


@torch.no_grad()
def audit_global_accuracy_crossing(
    stop: int,
    train_pairs: torch.Tensor,
    train_labels: torch.Tensor,
    test_pairs: torch.Tensor,
    test_labels: torch.Tensor,
    checkpoints: np.lib.npyio.NpzFile,
    config: Config,
) -> tuple[int, float]:
    """Replay every GD iterate to exclude an unlogged earlier gate crossing."""
    required = int(np.ceil(config.test_accuracy_gate * len(test_pairs)))
    parameter = initialize(config)
    maximum_checkpoint_error = 0.0
    for step in range(stop + 1):
        if step % config.checkpoint_every == 0:
            key = f"step_{step}"
            if key in checkpoints.files:
                maximum_checkpoint_error = max(
                    maximum_checkpoint_error,
                    float(
                        torch.linalg.vector_norm(
                            parameter - torch.from_numpy(checkpoints[key])
                        )
                    ),
                )
        prediction = torch.argmax(logits(parameter, test_pairs, config), dim=1)
        if int(torch.sum(prediction == test_labels)) >= required:
            return step, maximum_checkpoint_error
        gradient = analytic_gradient(parameter, train_pairs, train_labels, config)
        parameter.add_(gradient, alpha=-config.learning_rate)
    raise RuntimeError("no exact accuracy crossing found in the global replay")


def first_gate(values: np.ndarray, gate: float) -> int | None:
    hits = np.flatnonzero(values >= gate)
    return None if len(hits) == 0 else int(hits[0])


def render(result: dict, model_paths: dict[str, dict], actual_accuracy: np.ndarray) -> None:
    relative = np.arange(len(actual_accuracy))
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2), constrained_layout=True)
    axes[0].step(relative, actual_accuracy, where="post", color="#374151", label="actual GD")
    for name, color in (("full_quadratic", "#087e8b"), ("jacobian_only", "#d1495b")):
        axes[0].step(
            np.arange(len(model_paths[name]["accuracy"])),
            model_paths[name]["accuracy"],
            where="post",
            color=color,
            alpha=0.85,
            label=name.replace("_", " "),
        )
    axes[0].axhline(result["accuracy_gate"], color="#6b7280", linestyle="--")
    axes[0].set(xlabel=f"steps after {result['anchor']}", ylabel="test accuracy", title="Strict grokking gate")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.2)

    for name, color in (("full_quadratic", "#087e8b"), ("jacobian_only", "#d1495b")):
        axes[1].plot(model_paths[name]["guaranteed"], color=color, label=f"{name}: guaranteed")
        axes[1].plot(model_paths[name]["possible"], color=color, linestyle=":", label=f"{name}: possible")
    axes[1].axhline(result["required_correct"], color="#6b7280", linestyle="--")
    axes[1].set(xlabel=f"steps after {result['anchor']}", ylabel="number correct", title="Certified accuracy envelope")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(alpha=0.2)

    axes[2].semilogy(result["state_error_bound"], color="#087e8b", label="state-tube bound")
    axes[2].semilogy(result["observed_state_error"], color="#374151", label="observed error (audit)")
    axes[2].set(xlabel=f"steps after {result['anchor']}", ylabel="parameter-space norm", title="Checkpoint-only shadowing tube")
    axes[2].legend(frameon=False)
    axes[2].grid(alpha=0.2)
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=220)
    plt.close(fig)


def main() -> None:
    torch.set_num_threads(1)
    config = canonical_config()
    run = json.loads(RUN_JSON.read_text(encoding="utf-8"))
    checkpoints = np.load(CHECKPOINTS)
    parameter = torch.from_numpy(checkpoints[f"step_{ANCHOR}"]).clone()
    train_pairs, train_labels, test_pairs, test_labels = make_split(config)

    train_hessian, _ = analytic_least_squares_hessians(
        parameter,
        train_pairs,
        train_labels,
        config,
        include_weight_decay=True,
    )
    train_geometry = LocalGeometry(
        objective(parameter, train_pairs, train_labels, config),
        analytic_gradient(parameter, train_pairs, train_labels, config),
        train_hessian,
    )
    eigenvalues, eigenvectors = torch.linalg.eigh(train_hessian)
    factors = 1.0 - config.learning_rate * eigenvalues
    displacement, epsilon, defect, injected = exact_centered_state_tube(
        parameter,
        train_geometry,
        eigenvalues,
        eigenvectors,
        train_pairs,
        train_labels,
        config,
        HORIZON,
    )
    path_norm = torch.linalg.vector_norm(displacement, dim=1).numpy()
    output_paths = output_taylor_paths(parameter, displacement, test_pairs, config)
    test_envelope = derivative_envelope(parameter, test_pairs, test_labels, config)

    required = int(np.ceil(config.test_accuracy_gate * len(test_pairs)))
    variants: dict[str, dict] = {}
    plot_paths: dict[str, dict] = {}
    for name, key in (("full_quadratic", "quadratic"), ("jacobian_only", "linear")):
        guaranteed, possible, margin_error = certified_accuracy_envelopes(
            output_paths[key],
            output_paths["exact"],
            test_labels,
            epsilon,
            path_norm,
            test_envelope,
        )
        unresolved = unresolved_margin_pairs(
            output_paths[key],
            test_labels,
            margin_error,
        )
        for sample, competitor in unresolved:
            margin_error[:, sample, competitor] = target_projected_margin_remainder(
                parameter,
                displacement,
                epsilon,
                injected,
                eigenvectors,
                factors,
                output_paths[key][:, sample, :],
                output_paths["exact"][:, sample, :],
                test_pairs[sample],
                int(test_labels[sample]),
                competitor,
                config,
                quadratic=key == "quadratic",
            )
        guaranteed, possible = counts_from_margin_error(
            output_paths[key],
            test_labels,
            margin_error,
        )
        model_accuracy = accuracy_from_logits(output_paths[key], test_labels)
        bracket = event_bracket(guaranteed, possible, required)
        variants[name] = {
            "raw_predicted_crossing_relative": first_gate(model_accuracy, config.test_accuracy_gate),
            "certified_crossing_bracket": bracket,
            "certificate_issued": bracket is not None,
            "target_projected_margin_pairs": len(unresolved),
            "maximum_margin_error_at_actual_crossing": None,
        }
        plot_paths[name] = {
            "accuracy": model_accuracy,
            "guaranteed": guaranteed,
            "possible": possible,
            "margin_error": margin_error,
        }

    actual_crossing, actual_accuracy, actual_parameters = refine_accuracy_crossing(
        ANCHOR,
        ANCHOR + HORIZON,
        parameter.clone(),
        train_pairs,
        train_labels,
        test_pairs,
        test_labels,
        config,
    )
    global_crossing, replay_checkpoint_error = audit_global_accuracy_crossing(
        ANCHOR + HORIZON,
        train_pairs,
        train_labels,
        test_pairs,
        test_labels,
        checkpoints,
        config,
    )
    if global_crossing != actual_crossing:
        raise RuntimeError(
            f"local crossing {actual_crossing} is not the first global crossing {global_crossing}"
        )
    actual_relative = actual_crossing - ANCHOR
    observed_state_error = np.asarray(
        [
            np.linalg.norm(actual_parameters[step] - parameter.numpy() - displacement[step].numpy())
            for step in range(len(actual_parameters))
        ]
    )
    violations = int(np.sum(observed_state_error > epsilon[: len(observed_state_error)] * (1.0 + 1e-8) + 1e-10))
    for name in variants:
        bracket = variants[name]["certified_crossing_bracket"]
        variants[name]["bracket_covers_actual"] = (
            None if bracket is None else bracket[0] <= actual_relative <= bracket[1]
        )
        variants[name]["timing_error_steps"] = (
            None
            if variants[name]["raw_predicted_crossing_relative"] is None
            else variants[name]["raw_predicted_crossing_relative"] - actual_relative
        )
        variants[name]["maximum_margin_error_at_actual_crossing"] = float(
            np.max(plot_paths[name]["margin_error"][actual_relative])
        )

    positive_bounds = epsilon[: len(observed_state_error)] > 1e-14
    audited_ratios = observed_state_error[positive_bounds] / epsilon[: len(observed_state_error)][positive_bounds]
    result = {
        "experiment": "checkpoint-only certificate for the strict modular-addition accuracy gate",
        "config": asdict(config),
        "anchor": ANCHOR,
        "horizon": HORIZON,
        "accuracy_gate": config.test_accuracy_gate,
        "test_examples": len(test_pairs),
        "required_correct": required,
        "actual_crossing_absolute": actual_crossing,
        "actual_crossing_relative": actual_relative,
        "global_every-step_replay_verified": True,
        "global_replay_maximum_checkpoint_parameter_error": replay_checkpoint_error,
        "fit_step": run["summary"]["fit_step"],
        "grokking_delay_to_exact_accuracy_crossing": actual_crossing - run["summary"]["fit_step"],
        "minimum_train_hessian_eigenvalue": float(eigenvalues[0]),
        "maximum_train_hessian_eigenvalue": float(eigenvalues[-1]),
        "state_error_bound_at_actual_crossing": float(epsilon[actual_relative]),
        "observed_state_error_at_actual_crossing": float(observed_state_error[actual_relative]),
        "maximum_observed_state_error_to_bound_ratio": float(
            np.max(audited_ratios) if len(audited_ratios) else 0.0
        ),
        "state_tube_bound_violations": violations,
        "maximum_deterministic_path_defect": float(np.max(defect)),
        "maximum_injected_defect_bound": float(np.max(injected)),
        "variants": variants,
        "state_error_bound": epsilon.tolist(),
        "observed_state_error": observed_state_error.tolist(),
    }
    render(result, plot_paths, actual_accuracy)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUT), "figure": str(FIGURE), "summary": {k: result[k] for k in (
        "actual_crossing_absolute",
        "actual_crossing_relative",
        "state_tube_bound_violations",
        "variants",
    )}}, indent=2))


if __name__ == "__main__":
    main()
