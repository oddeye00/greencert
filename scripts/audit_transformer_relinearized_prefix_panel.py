#!/usr/bin/env python3
"""Outcome-blind 15-case corrected-path 4/8/16-prefix Green audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

from batched_green_operator import make_batched_transformer_green_products
from cost_aware_forcing import cost_aware_forcing_upper
from prefix_gram_enclosure import (
    equal_family_stage_delta,
    family_failure_upper,
    prefix_increment,
    prefix_gram_rows,
)
from probe_jacobian_bound import ProbeRegistry
from relinearized_green_closure import exact_relinearized_closure
from transformer_certificate_protocol import Candidate
from transformer_four_sweep_development_audit import to_scaled
from transformer_fourth_jet_bound import objective_fourth_derivative_bound
from transformer_green_development_audit import build_frozen_centerline
from transformer_green_operator import make_causal_green_products
from transformer_hvp_grokking import logits
from transformer_optimizer_probe import make_scaled_optimizer_jvp_vjp
from transformer_two_response import optimizer_center_quadratic_defect
from transformer_v3_certificate import (
    _gate_raw_slacks,
    _logic_slack,
    _persistent_bracket,
    load_candidate,
    output_path,
    safe_json,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT = RESULTS / "transformer_v3_relinearized_prefix_panel_audit.json"
CACHE = RESULTS / "transformer_v3_relinearized_prefix_panel_cache"
PROTOCOL = ROOT / "RELINEARIZED_PREFIX_PANEL_PROTOCOL.md"
TWO_RESPONSE = RESULTS / "transformer_v3_two_response_postseal_audit.json"

VERSION = 2
SWEEPS = 4
POWER = 1
PREFIXES = (4, 8, 16)
FAMILY_FAILURE = 1.0e-6
INHERITED_OUTPUT_FAMILY_FAILURE = 1.0e-6
MASTER_NONCE = "b3fe3a46aafe29d1ea08c5d1e24a547932e0dff73c67be8e4899011f328fdd05"
IDENTITY_NAMESPACE = 94

# Frozen from the 15 Green-evaluable rows of the outcome-blind two-response
# audit. Certificate hashes and horizons make accidental set drift fatal.
CASE_ROWS = (
    (360, 0.7, 3480, 131, "BF9A46F67BD8AD23DA7C77945B0203BB8B4CA4A99CEDF297E23ED4BF64B3CBBA"),
    (361, 0.7, 1880, 275, "D26ECDAFA3568040C30790090B286CCF9925AEA2FC76ED3A1D1881C7149F6F30"),
    (361, 0.8, 2800, 289, "896292CEF8EDB70B5B2ECFB8241F737154576E99B20CCAD7745BF7F8683624A2"),
    (366, 0.7, 1040, 52, "63DA40D4989669A1D1349B25323732C6800FD05E7EF793FBDE9C624E709CE24B"),
    (366, 0.8, 1120, 26, "9F773A441C44988A876BB4A707BA7F2C5B98C407A12F474C6D314F10862D639C"),
    (366, 0.9, 1360, 94, "2AE37E3B9E914F9E38EA6EDF813887AB4C9CD41FA845DDC7E1958A2DEF947A95"),
    (369, 0.7, 4160, 181, "06EFAC13FA25E054C69EFDCCA4776124684999982C4EB88CF7ADC18BD4E723CF"),
    (369, 0.8, 4480, 142, "557661FE278E39AEB1E066B539212A1A25C33F2DE5ADBBD972718DC76DE865BF"),
    (369, 0.9, 5760, 256, "AD8D4896899E4B6C719634260347BE0BCD072C7552C229C3C5F65257E122F30D"),
    (370, 0.7, 2280, 299, "D1CE3675187805FCF94911432036A41792C43939918163F2AA9B6C243935594D"),
    (372, 0.7, 3440, 270, "0E0323259DFD5A8C4A87870B42CF95D9EF991660C9DA0BA5905AB4D92DE54383"),
    (373, 0.7, 1280, 271, "E9A33BD6F367468238649A81D0CA91ED89AB732AAEA47FDFD4E017B7FB4782D0"),
    (373, 0.8, 1760, 238, "2C7FCD2C64990577BBAF143C716FD09F55A3FC9E3B847491AE99D9D363AEE90F"),
    (375, 0.8, 1800, 142, "43E6ADEF120EDEBE35063C83F12CD7CB2B94F07C945665DB1F42DD0B63BA3738"),
    (378, 0.7, 3640, 285, "262950FDE6B2C8E7CD4F6B4F9945A4FAF5D1FD90EE3245AF470BBEF64C62876C"),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(
        value.detach().cpu().contiguous().numpy().tobytes(order="C")
    ).hexdigest().upper()


def case_set_sha256() -> str:
    payload = json.dumps(CASE_ROWS, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("ascii")).hexdigest().upper()


def dependency_paths() -> tuple[Path, ...]:
    return (
        Path(__file__),
        ROOT / "scripts" / "batched_green_operator.py",
        ROOT / "scripts" / "cost_aware_forcing.py",
        ROOT / "scripts" / "prefix_gram_enclosure.py",
        ROOT / "scripts" / "probe_jacobian_bound.py",
        ROOT / "scripts" / "relinearized_green_closure.py",
        ROOT / "scripts" / "transformer_four_sweep_development_audit.py",
        ROOT / "scripts" / "transformer_fourth_jet_bound.py",
        ROOT / "scripts" / "transformer_green_development_audit.py",
        ROOT / "scripts" / "transformer_green_operator.py",
        ROOT / "scripts" / "transformer_optimizer_probe.py",
        ROOT / "scripts" / "transformer_two_response.py",
        ROOT / "scripts" / "transformer_v3_certificate.py",
        ROOT / "NESTED_PREFIX_GRAM_THEOREM.md",
        ROOT / "COST_AWARE_FORCING_THEOREM.md",
    )


def dependency_hashes() -> dict[str, str]:
    return {
        path.relative_to(ROOT).as_posix(): sha256(path)
        for path in dependency_paths()
    }


def assert_protocol_frozen() -> None:
    """Refuse to generate probes unless the current implementation is sealed."""

    if not PROTOCOL.exists():
        raise RuntimeError("prefix-panel protocol is absent; refusing to query probes")
    protocol = PROTOCOL.read_text(encoding="utf-8").upper()
    required = {
        "MASTER_NONCE": MASTER_NONCE.upper(),
        "CASE_SET_SHA256": case_set_sha256(),
        **{f"DEPENDENCY:{name}": value for name, value in dependency_hashes().items()},
        "TWO_RESPONSE_SHA256": sha256(TWO_RESPONSE),
    }
    missing = [name for name, value in required.items() if value not in protocol]
    if missing:
        raise RuntimeError(
            "prefix-panel protocol does not seal the current inputs: "
            + ", ".join(missing)
        )


def relative_error(left: float, right: float) -> float:
    return abs(float(left) - float(right)) / max(
        abs(float(right)), torch.finfo(torch.float64).tiny
    )


def from_scaled(path: torch.Tensor, dimension: int, eta: float) -> torch.Tensor:
    return torch.cat((path[..., :dimension], path[..., dimension:] / eta), dim=-1)


def identity(candidate: Candidate, horizon: int) -> tuple[int, ...]:
    return (
        IDENTITY_NAMESPACE,
        candidate.seed,
        candidate.gate_index,
        candidate.anchor,
        int(horizon),
        SWEEPS,
        POWER,
    )


def cache_path(candidate: Candidate) -> Path:
    return CACHE / (
        f"seed_{candidate.seed}_gate_{candidate.gate_index}_"
        f"anchor_{candidate.anchor}_v{VERSION}.json"
    )


def maximum_injection_forcing(
    *, kappa: float, derivative_drift: float, correction_max: float, domain: float
) -> dict:
    available = max(0.0, float(domain) - float(correction_max))
    coefficient = float(kappa) * float(derivative_drift)
    if kappa <= 0.0:
        return {"radius": available, "response_cap": math.inf, "injection_cap": math.inf}
    if coefficient <= 0.0:
        return {
            "radius": available,
            "response_cap": available,
            "injection_cap": available / kappa,
        }
    radius = min(available, 1.0 / coefficient)
    response_cap = max(0.0, radius - 0.5 * coefficient * radius * radius)
    return {
        "radius": radius,
        "response_cap": response_cap,
        "injection_cap": response_cap / kappa,
    }


def directional_row_index() -> dict[tuple[int, float, int], dict]:
    payload = safe_json(TWO_RESPONSE)
    rows = {}
    for row in payload["rows"]:
        if not row.get("evaluable"):
            continue
        candidate = row["candidate"]
        key = (
            int(candidate["seed"]),
            float(candidate["threshold"]),
            int(candidate["anchor"]),
        )
        rows[key] = row
    return rows


def output_bracket(
    *,
    certificate: dict,
    corrected: torch.Tensor,
    correction: torch.Tensor,
    dimension: int,
    cert_pairs: torch.Tensor,
    cert_labels: torch.Tensor,
    template,
    spec,
    radius: float,
) -> dict:
    required = int(certificate["required_correct"])
    raw = [
        _gate_raw_slacks(
            logits(corrected[step, :dimension], cert_pairs, template, spec),
            cert_labels,
            required,
        )
        for step in range(len(corrected))
    ]
    maximum_power = min(
        len(row["trace"]["rows"]) for row in certificate["output_rows"]
    )
    for output_power in range(1, maximum_power + 1):
        guarantee_slacks = []
        exclusion_slacks = []
        margins = []
        for step, pair in enumerate(raw):
            if step == 0:
                margin = 0.0
            else:
                output = certificate["output_rows"][step - 1]
                output_upper = float(
                    output["trace"]["rows"][output_power - 1][
                        "operator_norm_upper_bound"
                    ]
                )
                second = float(output["block_second"])
                shift = float(torch.linalg.vector_norm(correction[step, :dimension]))
                margin = math.sqrt(2.0) * (
                    (output_upper + second * shift) * radius
                    + 0.5 * second * radius * radius
                )
            margins.append(margin)
            guarantee_slacks.append(pair[0] - margin)
            exclusion_slacks.append(pair[1] - margin)
        bracket = _persistent_bracket(guarantee_slacks, exclusion_slacks)
        if bracket is not None:
            return {
                "bracket": bracket,
                "output_power": output_power,
                "logic_slack": _logic_slack(
                    bracket, guarantee_slacks, exclusion_slacks
                ),
                "maximum_margin_radius": max(margins),
            }
    return {
        "bracket": None,
        "output_power": None,
        "logic_slack": None,
        "maximum_margin_radius": None,
    }


def audit_case(case: tuple[int, float, int, int, str]) -> dict:
    assert_protocol_frozen()
    started = time.perf_counter()
    seed, threshold, anchor, horizon, expected_certificate_sha = case
    candidate = Candidate(int(seed), float(threshold), int(anchor))
    certificate_path = output_path(candidate)
    if sha256(certificate_path) != expected_certificate_sha:
        raise RuntimeError(f"certificate hash mismatch for {candidate}")
    certificate = safe_json(certificate_path)
    directional = directional_row_index()[
        (candidate.seed, candidate.threshold, candidate.anchor)
    ]
    if int(directional["horizon"]) != int(horizon):
        raise RuntimeError(f"directional horizon mismatch for {candidate}")

    config, template, spec, data, parameter, velocity = load_candidate(candidate)
    torch.set_num_threads(config.threads)
    torch.use_deterministic_algorithms(True)
    train_pairs, train_labels, _, _, cert_pairs, cert_labels = data
    dimension = int(parameter.numel())
    timings: dict[str, float] = {}

    phase = time.perf_counter()
    path = build_frozen_centerline(
        config,
        template,
        spec,
        train_pairs,
        train_labels,
        parameter,
        velocity,
    )
    timings["centerline"] = time.perf_counter() - phase
    if path["centerline_sha256"] != certificate["centerline_sha256"]:
        raise RuntimeError(f"centerline hash mismatch for {candidate}")
    center = path["center"][: horizon + 1]
    scaled_center = path["scaled_center"][: horizon + 1]

    phase = time.perf_counter()
    mapped = [path["map_step"](center[step]) for step in range(horizon)]
    residual = torch.stack(
        [
            to_scaled(mapped[step], dimension, config.learning_rate)
            - scaled_center[step + 1]
            for step in range(horizon)
        ]
    )
    old_products = [
        make_scaled_optimizer_jvp_vjp(
            center[step, :dimension],
            train_pairs,
            train_labels,
            template,
            spec,
            config,
        )
        for step in range(horizon)
    ]
    old_jvps = [row[0] for row in old_products]
    old_apply, _ = make_causal_green_products(
        old_jvps, [row[1] for row in old_products], 2 * dimension
    )
    correction_rows = old_apply(residual.reshape(-1)).reshape(horizon, -1)
    correction = torch.cat(
        (torch.zeros_like(correction_rows[:1]), correction_rows), dim=0
    )
    timings["first_response"] = time.perf_counter() - phase
    correction_norm = float(torch.linalg.vector_norm(correction_rows))
    correction_max = float(torch.linalg.vector_norm(correction, dim=1).max())
    if relative_error(correction_norm, directional["response_sequence_norm"]) > 2.0e-12:
        raise RuntimeError(f"response sequence mismatch for {candidate}")
    if relative_error(correction_max, directional["response_max_state_norm"]) > 2.0e-12:
        raise RuntimeError(f"response maximum mismatch for {candidate}")

    phase = time.perf_counter()
    recurrence_rows = []
    prior = torch.zeros_like(correction_rows[0])
    for step in range(horizon):
        recomputed = old_jvps[step](prior) + residual[step]
        recurrence_rows.append(correction_rows[step] - recomputed)
        prior = correction_rows[step]
    recurrence = torch.stack(recurrence_rows)
    recurrence_norm = float(torch.linalg.vector_norm(recurrence))
    timings["recurrence_replay"] = time.perf_counter() - phase

    phase = time.perf_counter()
    q_rows = [torch.zeros_like(correction_rows[0])]
    taylor_terms = []
    fourth_bounds = []
    direction_norms = []
    for step in range(1, horizon):
        direction = correction[step, :dimension]
        direction_norm = float(torch.linalg.vector_norm(direction))
        direction_norms.append(direction_norm)
        q_rows.append(
            optimizer_center_quadratic_defect(
                center[step, :dimension],
                correction[step],
                train_pairs,
                train_labels,
                template,
                spec,
                config,
            )
        )
        fourth = objective_fourth_derivative_bound(
            center[step, :dimension],
            template,
            spec,
            config,
            radius=direction_norm,
        )
        fourth_bounds.append(fourth)
        taylor_terms.append(
            math.sqrt(2.0)
            * float(config.learning_rate)
            * fourth
            * direction_norm**3
            / 6.0
        )
    q_surrogate = torch.stack(q_rows)
    q_norm = float(torch.linalg.vector_norm(q_surrogate))
    taylor_error = math.sqrt(sum(value * value for value in taylor_terms))
    injection_upper = q_norm + taylor_error + recurrence_norm
    timings["directional_forcing"] = time.perf_counter() - phase
    if relative_error(q_norm, directional["quadratic_surrogate_injection_norm"]) > 2.0e-11:
        raise RuntimeError(f"quadratic forcing mismatch for {candidate}")

    corrected_scaled = scaled_center + correction
    corrected = from_scaled(corrected_scaled, dimension, config.learning_rate)
    phase = time.perf_counter()
    corrected_products = [
        make_scaled_optimizer_jvp_vjp(
            corrected[step, :dimension],
            train_pairs,
            train_labels,
            template,
            spec,
            config,
        )
        for step in range(horizon)
    ]
    corrected_apply, _ = make_causal_green_products(
        [row[0] for row in corrected_products],
        [row[1] for row in corrected_products],
        2 * dimension,
    )
    corrected_batch_apply, corrected_batch_transpose = (
        make_batched_transformer_green_products(
            corrected[:horizon, :dimension],
            train_pairs,
            train_labels,
            template,
            spec,
            config,
        )
    )
    timings["corrected_operator_setup"] = time.perf_counter() - phase

    operator_identity = identity(candidate, horizon)
    registry = ProbeRegistry(
        [identity(Candidate(s, g, a), h) for s, g, a, h, _ in CASE_ROWS],
        MASTER_NONCE,
    )
    probe_seed = registry.claim(operator_identity)
    generator = torch.Generator(device=corrected.device).manual_seed(probe_seed)
    stage_delta = equal_family_stage_delta(
        family_failure=FAMILY_FAILURE, operators=len(CASE_ROWS), prefixes=PREFIXES
    )
    initial_norms: list[float] = []
    final_norms: list[float] = []
    probe_hashes: list[str] = []
    stage_rows = []
    first_row = certificate["power_rows"][0]
    drift = float(first_row["maximum_optimizer_derivative_drift_upper"])
    domain = float(first_row["one_shot_closure"]["domain_radius"])
    probe_started = time.perf_counter()
    issued = False
    final_event = None
    direct_response_rows = None
    direct_response_norm = None
    direct_response_recurrence_residual_norm = None
    batched_gram_calls = 0
    surrogate_error = taylor_error + recurrence_norm

    def evaluate_attempt(kappa: float, response_bound: float) -> dict:
        closure = exact_relinearized_closure(
            kappa=kappa,
            derivative_drift=drift,
            corrected_defect_response_bound=response_bound,
            correction_max_state_norm=correction_max,
            domain_radius=domain,
        )
        event = {
            "bracket": None,
            "output_power": None,
            "logic_slack": None,
            "maximum_margin_radius": None,
        }
        if closure.closure_passed:
            event = output_bracket(
                certificate=certificate,
                corrected=corrected,
                correction=correction,
                dimension=dimension,
                cert_pairs=cert_pairs,
                cert_labels=cert_labels,
                template=template,
                spec=spec,
                radius=float(closure.remainder_radius),
            )
        return {
            "corrected_defect_response_bound": response_bound,
            "closure": closure.as_dict(),
            **event,
            "issued": event["bracket"] is not None,
        }

    for prefix in PREFIXES:
        new_vectors = []
        new_count = prefix_increment(
            initial_count=len(initial_norms),
            final_count=len(final_norms),
            target=prefix,
        )
        for _ in range(new_count):
            vector = torch.randn(
                horizon * 2 * dimension,
                generator=generator,
                dtype=corrected.dtype,
                device=corrected.device,
            )
            probe_hashes.append(
                hashlib.sha256(vector.detach().cpu().numpy().tobytes()).hexdigest().upper()
            )
            initial_norms.append(float(torch.linalg.vector_norm(vector)))
            new_vectors.append(vector)
        if new_vectors:
            block = torch.stack(new_vectors)
            final_block = corrected_batch_transpose(corrected_batch_apply(block))
            final_norms.extend(
                float(value)
                for value in torch.linalg.vector_norm(final_block, dim=1)
            )
            batched_gram_calls += 1
        if len(initial_norms) != prefix or len(final_norms) != prefix:
            raise RuntimeError("probe-prefix accounting failed after batched query")
        prefix_row = prefix_gram_rows(
            final_norms=final_norms,
            initial_norms=initial_norms,
            prefixes=(prefix,),
            power=POWER,
            stage_delta=stage_delta,
        )[0]
        kappa = float(prefix_row["operator_norm_upper_bound"])
        norm_release = cost_aware_forcing_upper(
            kappa=kappa,
            surrogate_injection_norm=q_norm,
            surrogate_error_norm=surrogate_error,
        )
        norm_attempt = evaluate_attempt(
            kappa, norm_release.norm_only_response_upper
        )
        selected_release = norm_release
        selected_attempt = norm_attempt
        response_attempt = None

        # A single direct response is a deterministic, cost-aware fallback.  It
        # is paid only when the current norm-only certificate does not issue and
        # is then reused at every later prefix.
        if not norm_attempt["issued"]:
            if direct_response_rows is None:
                response_started = time.perf_counter()
                direct_response_rows = corrected_apply(
                    q_surrogate.reshape(-1)
                ).reshape(horizon, -1)
                direct_response_norm = float(
                    torch.linalg.vector_norm(direct_response_rows)
                )
                response_recurrence_rows = []
                prior_response = torch.zeros_like(direct_response_rows[0])
                for step in range(horizon):
                    replayed = (
                        corrected_products[step][0](prior_response)
                        + q_surrogate[step]
                    )
                    response_recurrence_rows.append(
                        direct_response_rows[step] - replayed
                    )
                    prior_response = direct_response_rows[step]
                direct_response_recurrence_residual_norm = float(
                    torch.linalg.vector_norm(
                        torch.stack(response_recurrence_rows)
                    )
                )
                timings["direct_forcing_response_and_replay"] = (
                    time.perf_counter() - response_started
                )
            selected_release = cost_aware_forcing_upper(
                kappa=kappa,
                surrogate_injection_norm=q_norm,
                surrogate_error_norm=surrogate_error,
                direct_response_norm=direct_response_norm,
                direct_response_recurrence_residual_norm=(
                    direct_response_recurrence_residual_norm
                ),
            )
            response_attempt = evaluate_attempt(
                kappa, selected_release.selected_response_upper
            )
            selected_attempt = response_attempt

        capacity = maximum_injection_forcing(
            kappa=kappa,
            derivative_drift=drift,
            correction_max=correction_max,
            domain=domain,
        )
        stage_rows.append(
            {
                **prefix_row,
                "cumulative_batched_gram_calls": batched_gram_calls,
                "forcing_capacity": capacity,
                "norm_only_injection_headroom_ratio": (
                    math.inf
                    if injection_upper == 0.0
                    else capacity["injection_cap"] / injection_upper
                ),
                "selected_forcing_headroom_ratio": (
                    math.inf
                    if selected_release.selected_response_upper == 0.0
                    else capacity["response_cap"]
                    / selected_release.selected_response_upper
                ),
                "forcing_release": selected_release.as_dict(),
                "norm_only_attempt": norm_attempt,
                "response_aware_attempt": response_attempt,
                "selected_method": selected_release.selected_method,
                **selected_attempt,
            }
        )
        if selected_attempt["issued"]:
            issued = True
            final_event = selected_attempt
            break
    timings.setdefault("direct_forcing_response_and_replay", 0.0)
    timings["prefix_green_query"] = time.perf_counter() - probe_started

    final_stage = stage_rows[-1]
    directional_q = int(directional["surrogate_earliest_power"])
    old_gram = 16 * directional_q
    new_gram = int(final_stage["gram_applications"])
    direct_response_used = direct_response_rows is not None
    old_linearized_sweeps = 2 + 2 * old_gram
    new_linearized_sweeps = 1 + int(direct_response_used) + 2 * new_gram
    row = {
        "version": VERSION,
        "candidate": candidate.__dict__,
        "certificate_path": certificate_path.relative_to(ROOT).as_posix(),
        "certificate_sha256": expected_certificate_sha,
        "horizon": horizon,
        "centerline_sha256": path["centerline_sha256"],
        "corrected_path_sha256": tensor_sha256(corrected_scaled),
        "quadratic_surrogate_sha256": tensor_sha256(q_surrogate),
        "operator_identity": list(operator_identity),
        "probe_seed": int(probe_seed),
        "probe_hashes": probe_hashes,
        "initial_probe_norms": initial_norms,
        "final_probe_norms": final_norms,
        "prefixes_computed": len(final_norms),
        "batched_gram_calls": batched_gram_calls,
        "stage_delta": stage_delta,
        "new_green_family_failure_upper": FAMILY_FAILURE,
        "inherited_output_family_failure_upper": (
            INHERITED_OUTPUT_FAMILY_FAILURE
        ),
        "combined_family_failure_upper": (
            FAMILY_FAILURE + INHERITED_OUTPUT_FAMILY_FAILURE
        ),
        "correction_sequence_norm": correction_norm,
        "correction_max_state_norm": correction_max,
        "measured_response_recurrence_residual_norm": recurrence_norm,
        "quadratic_surrogate_injection_norm": q_norm,
        "directional_quadratic_taylor_error_upper": taylor_error,
        "total_corrected_injection_upper": injection_upper,
        "direct_forcing_response_used": direct_response_used,
        "direct_forcing_response_norm": direct_response_norm,
        "direct_forcing_response_recurrence_residual_norm": (
            direct_response_recurrence_residual_norm
        ),
        "direct_forcing_response_sha256": (
            None
            if direct_response_rows is None
            else tensor_sha256(direct_response_rows)
        ),
        "maximum_local_objective_fourth_derivative_upper": max(fourth_bounds, default=0.0),
        "maximum_parameter_direction_norm": max(direction_norms, default=0.0),
        "derivative_drift_upper": drift,
        "domain_radius": domain,
        "stage_rows": stage_rows,
        "issued": issued,
        "bracket": None if final_event is None else final_event["bracket"],
        "directional_baseline_bracket": directional["surrogate_bracket"],
        "same_as_directional_bracket": (
            issued and final_event["bracket"] == directional["surrogate_bracket"]
        ),
        "directional_baseline_power": directional_q,
        "directional_baseline_green_gram_applications": old_gram,
        "relinearized_green_gram_applications": new_gram,
        "green_gram_application_reduction": old_gram / new_gram,
        "green_gram_applications_saved": old_gram - new_gram,
        "directional_baseline_theoretical_linearized_sweeps": old_linearized_sweeps,
        "relinearized_theoretical_linearized_sweeps": new_linearized_sweeps,
        "theoretical_linearized_sweep_reduction": (
            old_linearized_sweeps / new_linearized_sweeps
        ),
        "recurrence_replay_sweeps_for_float64_audit": (
            1 + int(direct_response_used)
        ),
        "second_causal_response_sweeps_avoided": (
            0 if direct_response_used else 1
        ),
        "outcome_files_read": 0,
        "timings_seconds": timings,
        "elapsed_seconds": time.perf_counter() - started,
        "protocol_sha256": sha256(PROTOCOL),
        "source_sha256": sha256(Path(__file__)),
        "two_response_source_sha256": sha256(TWO_RESPONSE),
        "case_set_sha256": case_set_sha256(),
        "dependency_sha256": dependency_hashes(),
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "platform": platform.platform(),
            "torch_threads": int(config.threads),
            "deterministic_algorithms": True,
        },
    }
    return row


def valid_cache(case: tuple[int, float, int, int, str], row: dict) -> bool:
    candidate = Candidate(case[0], case[1], case[2])
    return (
        int(row.get("version", -1)) == VERSION
        and row.get("candidate") == candidate.__dict__
        and row.get("certificate_sha256") == case[4]
        and row.get("protocol_sha256") == sha256(PROTOCOL)
        and row.get("source_sha256") == sha256(Path(__file__))
        and row.get("two_response_source_sha256") == sha256(TWO_RESPONSE)
        and row.get("case_set_sha256") == case_set_sha256()
        and row.get("dependency_sha256") == dependency_hashes()
        and int(row.get("outcome_files_read", -1)) == 0
    )


def write_cache(case: tuple[int, float, int, int, str], row: dict) -> None:
    candidate = Candidate(case[0], case[1], case[2])
    CACHE.mkdir(parents=True, exist_ok=True)
    destination = cache_path(candidate)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)


def aggregate(rows: list[dict], elapsed: float) -> dict:
    rows.sort(
        key=lambda row: (
            int(row["candidate"]["seed"]),
            float(row["candidate"]["threshold"]),
            int(row["candidate"]["anchor"]),
        )
    )
    issued = [row for row in rows if row["issued"]]
    prefix_distribution = {
        str(prefix): sum(
            row["issued"] and row["relinearized_green_gram_applications"] == prefix
            for row in rows
        )
        for prefix in PREFIXES
    }
    total_old = sum(row["directional_baseline_green_gram_applications"] for row in rows)
    total_new = sum(row["relinearized_green_gram_applications"] for row in rows)
    old_sweeps = sum(
        row["directional_baseline_theoretical_linearized_sweeps"] for row in rows
    )
    new_sweeps = sum(
        row["relinearized_theoretical_linearized_sweeps"] for row in rows
    )
    return {
        "status": (
            "OUTCOME-BLIND RELINEARIZED PREFIX PANEL COMPLETED"
            if len(rows) == len(CASE_ROWS)
            else "INCOMPLETE RELINEARIZED PREFIX PANEL"
        ),
        "evidence_boundary": (
            "Post-seal method-development audit over the 15 pre-existing "
            "Green-evaluable v3 records. Candidate set, 4/8/16 prefix rule, "
            "family-wise spending, forcing construction, output rule, code, and "
            "nonce were fixed before probe generation. No future outcomes are "
            "read. Neural products, recurrence residuals, directional products, "
            "derivative envelopes, and margins remain float64/high-confidence."
        ),
        "cases": len(rows),
        "issued": len(issued),
        "same_as_directional_bracket": sum(row["same_as_directional_bracket"] for row in rows),
        "prefix_distribution": prefix_distribution,
        "nonissued_after_prefix_16": sum(not row["issued"] for row in rows),
        "selected_method_distribution": {
            method: sum(
                row["stage_rows"][-1]["selected_method"] == method
                for row in rows
            )
            for method in ("norm_only", "direct_response")
        },
        "direct_forcing_response_cases": sum(
            row["direct_forcing_response_used"] for row in rows
        ),
        "prefixes": list(PREFIXES),
        "power": POWER,
        "new_green_family_failure_budget": FAMILY_FAILURE,
        "inherited_output_family_failure_upper": (
            INHERITED_OUTPUT_FAMILY_FAILURE
        ),
        "combined_family_failure_upper": (
            FAMILY_FAILURE + INHERITED_OUTPUT_FAMILY_FAILURE
        ),
        "stage_delta": equal_family_stage_delta(
            family_failure=FAMILY_FAILURE,
            operators=len(CASE_ROWS),
            prefixes=PREFIXES,
        ),
        "family_failure_upper": family_failure_upper(
            stage_delta=equal_family_stage_delta(
                family_failure=FAMILY_FAILURE,
                operators=len(CASE_ROWS),
                prefixes=PREFIXES,
            ),
            operators=len(CASE_ROWS),
            prefixes=PREFIXES,
        ),
        "old_total_green_gram_applications": total_old,
        "new_total_green_gram_applications": total_new,
        "aggregate_green_gram_reduction": total_old / total_new,
        "green_gram_applications_saved": total_old - total_new,
        "old_total_theoretical_linearized_sweeps": old_sweeps,
        "new_total_theoretical_linearized_sweeps": new_sweeps,
        "aggregate_theoretical_linearized_sweep_reduction": (
            old_sweeps / new_sweeps
        ),
        "median_pairwise_green_gram_reduction": statistics.median(
            row["green_gram_application_reduction"] for row in rows
        ),
        "minimum_pairwise_green_gram_reduction": min(
            row["green_gram_application_reduction"] for row in rows
        ),
        "maximum_pairwise_green_gram_reduction": max(
            row["green_gram_application_reduction"] for row in rows
        ),
        "minimum_issued_forcing_headroom": min(
            row["stage_rows"][-1]["selected_forcing_headroom_ratio"]
            for row in issued
        ) if issued else None,
        "maximum_measured_recurrence_residual_norm": max(
            row["measured_response_recurrence_residual_norm"] for row in rows
        ),
        "maximum_direct_response_recurrence_residual_norm": max(
            (
                row["direct_forcing_response_recurrence_residual_norm"]
                for row in rows
                if row["direct_forcing_response_recurrence_residual_norm"]
                is not None
            ),
            default=0.0,
        ),
        "total_second_causal_response_sweeps_avoided": sum(
            row["second_causal_response_sweeps_avoided"] for row in rows
        ),
        "aggregate_case_elapsed_seconds": sum(row["elapsed_seconds"] for row in rows),
        "wall_elapsed_seconds": elapsed,
        "outcome_files_read": sum(row["outcome_files_read"] for row in rows),
        "protocol_sha256": sha256(PROTOCOL),
        "source_sha256": sha256(Path(__file__)),
        "theorem_sha256": {
            "nested_prefix": sha256(ROOT / "NESTED_PREFIX_GRAM_THEOREM.md"),
            "cost_aware_forcing": sha256(ROOT / "COST_AWARE_FORCING_THEOREM.md"),
        },
        "two_response_source_sha256": sha256(TWO_RESPONSE),
        "case_set_sha256": case_set_sha256(),
        "dependency_sha256": dependency_hashes(),
        "rows": rows,
    }


def validate_case_manifest() -> dict:
    keys = [(seed, gate, anchor) for seed, gate, anchor, _, _ in CASE_ROWS]
    if len(keys) != len(set(keys)) or len(keys) != 15:
        raise RuntimeError("the frozen panel must contain 15 unique candidates")
    directional = directional_row_index()
    if set(keys) != set(directional):
        raise RuntimeError("the panel and two-response candidate sets differ")
    for seed, gate, anchor, horizon, expected_hash in CASE_ROWS:
        candidate = Candidate(seed, gate, anchor)
        if anchor % 40:
            raise RuntimeError(f"off-grid panel anchor: {candidate}")
        path = output_path(candidate)
        if not path.exists() or sha256(path) != expected_hash:
            raise RuntimeError(f"certificate artifact mismatch: {candidate}")
        row = directional[(seed, gate, anchor)]
        if int(row["horizon"]) != int(horizon):
            raise RuntimeError(f"two-response horizon mismatch: {candidate}")
    return {
        "cases": len(keys),
        "case_set_sha256": case_set_sha256(),
        "two_response_sha256": sha256(TWO_RESPONSE),
        "dependency_sha256": dependency_hashes(),
        "protocol_sha256": sha256(PROTOCOL),
        "outcome_files_read": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()

    assert_protocol_frozen()
    validation = validate_case_manifest()
    if args.validate_only:
        print(json.dumps({"status": "FROZEN INPUT VALIDATION PASSED", **validation}, indent=2))
        return

    cases = list(CASE_ROWS)
    if args.seed is not None:
        cases = [row for row in cases if int(row[0]) == int(args.seed)]
        if not cases:
            raise ValueError("requested seed is outside the frozen panel")
    allowed = [identity(Candidate(s, g, a), h) for s, g, a, h, _ in CASE_ROWS]
    registry = ProbeRegistry(allowed, MASTER_NONCE)
    if registry.summary()["collision_free_stream_count"] != len(CASE_ROWS):
        raise RuntimeError("panel probe streams are not collision free")

    rows = []
    pending = []
    for case in cases:
        candidate = Candidate(case[0], case[1], case[2])
        destination = cache_path(candidate)
        if not args.refresh and destination.exists():
            cached = safe_json(destination)
            if valid_cache(case, cached):
                rows.append(cached)
                print(
                    f"reused seed={candidate.seed} gate={candidate.threshold:.1f} "
                    f"anchor={candidate.anchor}",
                    flush=True,
                )
                continue
        pending.append(case)

    with ProcessPoolExecutor(max_workers=min(max(1, args.workers), max(1, len(pending)))) as pool:
        futures = {pool.submit(audit_case, case): case for case in pending}
        for future in as_completed(futures):
            case = futures[future]
            row = future.result()
            rows.append(row)
            write_cache(case, row)
            candidate = row["candidate"]
            print(
                f"audited seed={candidate['seed']} gate={candidate['threshold']:.1f} "
                f"anchor={candidate['anchor']} issued={row['issued']} "
                f"m={row['relinearized_green_gram_applications']}",
                flush=True,
            )

    if args.seed is not None:
        print(json.dumps({"seed": args.seed, "rows": rows}, indent=2))
        return
    payload = aggregate(rows, time.perf_counter() - started)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
