#!/usr/bin/env python3
"""Frozen prospective audit for the disjoint-set matrix-free certificate.

Only training and trigger-set histories open an audit window.  Within that
window, a deterministic two-sweep centerline screen is evaluated on the
separate certificate set at the stored 250-step checkpoints.  Expensive
probabilistic geometry is constructed only when the centerline predicts a
future 25-step-persistent threshold event inside the next 250 updates.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from disjoint_large_mlp import DisjointConfig, artifact_paths, make_disjoint_split
from hvp_projected_mlp_certificate import recentered_hvp_reference
from replay_smooth_mlp_thresholds import THRESHOLDS, required_counts
from run_disjoint_hvp_certificate import run as run_certificate
from smooth_mlp_modular_grokking import logits


ROOT = Path(__file__).resolve().parents[1]
FROZEN_SEEDS = (202, 203, 204, 205)
HORIZON = 250
PERSISTENCE = 25
RANK = 64
MARGIN_STARTS = 1
GEOMETRY_STRIDE = 5
POWER = 12
PROBES = 4
RECENTER_SWEEPS = 2
PER_CERTIFICATE_FAILURE = 1e-9
MAXIMUM_TRIGGER_LAG = 5_000
CHECKPOINT_STRIDE = 250
MAXIMUM_PROSPECTIVE_TESTS = (
    len(FROZEN_SEEDS)
    * len(THRESHOLDS)
    * (MAXIMUM_TRIGGER_LAG // CHECKPOINT_STRIDE + 1)
)
FAMILYWISE_FAILURE_UPPER = MAXIMUM_PROSPECTIVE_TESTS * PER_CERTIFICATE_FAILURE


def first_persistent(values: np.ndarray, required: int, persistence: int) -> int | None:
    starts = len(values) - persistence + 1
    for start in range(max(starts, 0)):
        if bool(np.all(values[start : start + persistence] >= required)):
            return start
    return None


@torch.no_grad()
def center_count_path(
    reference: torch.Tensor,
    pairs: torch.Tensor,
    labels: torch.Tensor,
    config,
) -> np.ndarray:
    return np.asarray([
        int(torch.sum(torch.argmax(logits(point, pairs, config), dim=1) == labels))
        for point in reference
    ], dtype=np.int64)


def load_model(seed: int, *, development: bool):
    result_path, checkpoint_path = artifact_paths(seed, development=development)
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    config = DisjointConfig(**payload["config"])
    if config.seed != seed:
        raise ValueError("artifact seed mismatch")
    checkpoints = np.load(checkpoint_path)
    return payload, config, checkpoints


def first_trigger_eligibility(payload: dict, threshold: float) -> int | None:
    columns = payload["trajectory_columns"]
    train_index = columns.index("train_accuracy")
    trigger_index = columns.index("trigger_accuracy")
    gate = payload["config"]["train_accuracy_gate"]
    for row in payload["trajectory"]:
        if row[train_index] >= gate and row[trigger_index] >= threshold:
            step = int(row[0])
            return ((step + CHECKPOINT_STRIDE - 1) // CHECKPOINT_STRIDE) * CHECKPOINT_STRIDE
    return None


def audit_seed(seed: int, *, development: bool) -> dict:
    payload, config, checkpoints = load_model(seed, development=development)
    model_config = config.model_config()
    train_pairs, train_labels, _, _, cert_pairs, cert_labels = make_disjoint_split(config)
    required = required_counts(len(cert_pairs))
    threshold_rows: dict[str, dict] = {}
    candidate_certificates = 0

    for threshold in THRESHOLDS:
        key = f"{threshold:.2f}"
        eligibility = first_trigger_eligibility(payload, threshold)
        row = {
            "threshold": threshold,
            "required_correct": required[threshold],
            "trigger_eligibility_checkpoint": eligibility,
            "screens": [],
            "status": "trigger never became eligible",
            "issued_anchor": None,
            "certified_bracket": None,
            "actual_crossing": None,
            "covered": None,
        }
        threshold_rows[key] = row
        if eligibility is None:
            continue

        last_anchor = min(
            eligibility + MAXIMUM_TRIGGER_LAG,
            config.steps - HORIZON,
        )
        for anchor in range(eligibility, last_anchor + 1, CHECKPOINT_STRIDE):
            checkpoint_key = f"step_{anchor}"
            if checkpoint_key not in checkpoints:
                raise KeyError(f"missing frozen checkpoint {checkpoint_key}")
            parameter = torch.from_numpy(checkpoints[checkpoint_key]).clone()
            current_count = int(torch.sum(
                torch.argmax(logits(parameter, cert_pairs, model_config), dim=1)
                == cert_labels
            ))
            screen_row = {
                "anchor": anchor,
                "current_certificate_count": current_count,
                "predicted_persistent_crossing": None,
                "candidate": False,
            }
            row["screens"].append(screen_row)
            if current_count >= required[threshold]:
                row["status"] = "certificate threshold already present at eligible checkpoint"
                break

            _, corrected, diagnostic = recentered_hvp_reference(
                parameter,
                train_pairs,
                train_labels,
                model_config,
                horizon=HORIZON,
                recenter_sweeps=RECENTER_SWEEPS,
            )
            counts = center_count_path(corrected, cert_pairs, cert_labels, model_config)
            predicted = first_persistent(
                counts, required[threshold], PERSISTENCE
            )
            screen_row["predicted_persistent_crossing"] = predicted
            screen_row["recenter_diagnostics"] = diagnostic["recenter_diagnostics"]
            if predicted is None or predicted == 0:
                row["status"] = "screened without a future centerline event"
                continue

            screen_row["candidate"] = True
            candidate_certificates += 1
            certificate = run_certificate(
                seed,
                anchor,
                HORIZON,
                development=development,
                rank=RANK,
                margin_starts=MARGIN_STARTS,
                geometry_stride=GEOMETRY_STRIDE,
                power=POWER,
                probes=PROBES,
                failure_probability=PER_CERTIFICATE_FAILURE,
                recenter_sweeps=RECENTER_SWEEPS,
                persistence=PERSISTENCE,
                use_cache=True,
            )
            event = certificate["events"][key]
            screen_row["reached_horizon"] = certificate["reached_horizon"]
            screen_row["certified_bracket"] = event["certified_bracket"]
            if event["certified_bracket"] is None:
                row["status"] = "candidate certificate abstained"
                continue

            row.update({
                "status": "certificate issued",
                "issued_anchor": anchor,
                "certified_bracket": event["certified_bracket"],
                "actual_crossing": event["local_actual_crossing"],
                "covered": event["covered_local_crossing"],
                "certificate_result": (
                    f"seed_{seed}_anchor_{anchor}_h{HORIZON}"
                    f"_r{RANK}_m{MARGIN_STARTS}_g{GEOMETRY_STRIDE}"
                    f"_q{POWER}_p{PROBES}_s{RECENTER_SWEEPS}_k{PERSISTENCE}.json"
                ),
            })
            break

    issued = [row for row in threshold_rows.values() if row["status"] == "certificate issued"]
    return {
        "seed": seed,
        "config": asdict(config),
        "thresholds": threshold_rows,
        "candidate_certificates": candidate_certificates,
        "issued_certificates": len(issued),
        "covered_certificates": sum(row["covered"] is True for row in issued),
    }


def run_audit(seeds: tuple[int, ...], *, development: bool, overwrite: bool) -> dict:
    torch.set_num_threads(1)
    output = ROOT / "results" / (
        "disjoint_hvp_development_audit.json"
        if development else "disjoint_hvp_prospective_audit.json"
    )
    if output.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {output}")
    seed_rows = [audit_seed(seed, development=development) for seed in seeds]
    issued_rows = [
        row
        for seed_row in seed_rows
        for row in seed_row["thresholds"].values()
        if row["status"] == "certificate issued"
    ]
    result = {
        "status": "development" if development else "frozen prospective audit",
        "protocol": {
            "seeds": list(seeds),
            "horizon": HORIZON,
            "persistence": PERSISTENCE,
            "active_rank": RANK,
            "margin_starts": MARGIN_STARTS,
            "geometry_stride": GEOMETRY_STRIDE,
            "power": POWER,
            "probes": PROBES,
            "recenter_sweeps": RECENTER_SWEEPS,
            "per_certificate_failure_probability": PER_CERTIFICATE_FAILURE,
            "maximum_trigger_lag": MAXIMUM_TRIGGER_LAG,
            "checkpoint_stride": CHECKPOINT_STRIDE,
            "maximum_prospective_tests": MAXIMUM_PROSPECTIVE_TESTS,
            "familywise_failure_probability_upper": FAMILYWISE_FAILURE_UPPER,
        },
        "seeds": seed_rows,
        "summary": {
            "seed_count": len(seed_rows),
            "candidate_certificates": sum(row["candidate_certificates"] for row in seed_rows),
            "issued_certificates": len(issued_rows),
            "covered_certificates": sum(row["covered"] is True for row in issued_rows),
            "issuing_seeds": len({
                seed_row["seed"]
                for seed_row in seed_rows
                if seed_row["issued_certificates"] > 0
            }),
        },
    }
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--overwrite-development", action="store_true")
    args = parser.parse_args()
    seeds = tuple(args.seeds) if args.seeds else FROZEN_SEEDS
    if not args.development and seeds != FROZEN_SEEDS:
        raise ValueError("prospective mode requires the frozen seed tuple")
    result = run_audit(
        seeds,
        development=args.development,
        overwrite=args.development and args.overwrite_development,
    )
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
