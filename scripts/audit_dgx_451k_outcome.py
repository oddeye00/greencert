"""Independent read-only audit of the completed, hardware-amended observation.

Reconstructs argmax decisions from stored dyadic logits and first-passage
timing independently of the observer. It does not rerun neural training.
"""
import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np


def require(value, message):
    if not value:
        raise ValueError(message)


def checked(path, expected):
    raw = Path(path).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected, "record checksum mismatch")
    return parse(raw)


def parse(raw):
    def pairs(items):
        result = {}
        for key,value in items:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    def reject(value):
        raise ValueError("nonfinite JSON constant")
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=reject)


def first_streak(counts, target, persistence):
    require(type(target) is int and target > 0 and type(persistence) is int and persistence > 0 and
            all(type(v) is int and v >= 0 for v in counts), "invalid count path or event")
    streak = 0
    for index, count in enumerate(counts):
        streak = streak+1 if count >= target else 0
        if streak == persistence:
            return index-persistence+1
    return None


def audit(root, bundle, reveal_sha, amendment_sha, publication_commit):
    require(isinstance(publication_commit,str) and re.fullmatch("[0-9a-f]{40}",publication_commit), "commit reference must be a SHA1")
    amendment = checked(bundle/"amendment.json", amendment_sha)
    original = checked(bundle/"original_protocol.json", amendment["original_protocol_sha256"])
    effective = checked(bundle/"effective_protocol.json", amendment["effective_protocol_sha256"])
    require(set(original) == set(effective) and original["runtime"] != effective["runtime"] and
            all(original[k] == effective[k] for k in original if k != "runtime"), "non-runtime protocol amendment")
    source = root/"results/larger_transformer_third_pass"
    reveal = checked(source/"reveal.json", reveal_sha)
    run = source/"registered_outcome_run"
    require(not (run/"interrupted.json").exists(), "interrupted outcome cannot be a complete run")
    started = checked(run/"started.json", reveal["started_sha256"])
    engine = parse((run/"engine_ready.json").read_bytes())
    expected_names = {"started.json", "engine_ready.json"} | {f"row_{j:03d}.json" for j in range(effective["horizon"]+1)} | {
        f"intent_{j:03d}.json" for j in range(1,effective["horizon"]+1)}
    actual_names = {p.name for p in run.iterdir()}
    require(actual_names in (expected_names, expected_names | {"dgx_delegation.json"}), "unexpected ledger population")
    require(reveal["protocol_sha256"] == started["protocol_sha256"] == amendment["effective_protocol_sha256"],
            "wrong effective protocol")
    require(reveal["disposition_sha256"] == amendment["original_terminal_sha256"] and
            reveal["gate"]["hardware_amendment_sha256"] == amendment_sha and
            started["gate"] == reveal["gate"], "amendment/terminal lineage mismatch")
    require(reveal["runtime"] == engine["runtime"] and engine["future_updates_executed"] == 0,
            "runtime changed within observation")
    require(all(reveal["runtime"][key] == value for key,value in effective["runtime"].items()),
            "execution runtime differs from sealed amendment")
    settings = {"device":"cpu", "threads":original["config"]["threads"],
        "interop_threads":original["torch_interop_threads"], "mha_fastpath":original["torch_mha_fastpath"],
        "deterministic_algorithms":original["deterministic_algorithms"],
        "torch_build":amendment["runtime_capture"]["torch_build"], "hardware_amendment_sha256":amendment_sha}
    require(all(type(reveal["runtime"].get(k)) is type(v) and reveal["runtime"][k] == v for k,v in settings.items()),
            "registered execution settings differ")
    H, K, target = effective["horizon"], effective["persistence"], effective["target"]
    require(all(type(v) is int for v in (H,K,target,reveal["completed_updates"],reveal["anchor"])) and
            H > 0 and 1 <= K <= H+1 and 1 <= target <= len(original["evaluation_examples"]), "invalid event integers")
    require(reveal["horizon"] == H and reveal["completed_updates"] == H and
            reveal["target"] == target and reveal["persistence"] == K and
            reveal["anchor"] == original["anchor"], "registered window changed")
    require(set(reveal["row_sha256"]) == {f"row_{j:03d}.json" for j in range(H+1)}, "incomplete row population")
    require(reveal["exact_real_continuation_verified"] is False and
            reveal["exact_real_state_tube_tested"] is False and
            reveal["whole_prior_training_program_certified"] is False, "numerical scope relabeled")
    labels = [row["label"] for row in original["evaluation_examples"]]
    counts, all_ties, total_logits = [], 0, 0
    last_time = datetime.fromisoformat(engine["utc"])
    require(datetime.fromisoformat(started["utc"]) <= last_time, "engine preceded reservation")
    initial = None
    for step in range(H+1):
        name = f"row_{step:03d}.json"
        row = checked(run/name, reveal["row_sha256"][name])
        require(last_time <= datetime.fromisoformat(row["utc"]), "observation time moved backward")
        require(type(row["step"]) is int and type(row["absolute_update"]) is int and
                row["step"] == step and row["absolute_update"] == original["anchor"]+step and
                row["started_sha256"] == reveal["started_sha256"], "wrong row identity")
        if step:
            intent = parse((run/f"intent_{step:03d}.json").read_bytes())
            require(type(intent["step"]) is int and intent["step"] == step and intent["started_sha256"] == reveal["started_sha256"],
                    "wrong update intent")
            intent_time = datetime.fromisoformat(intent["utc"])
            require(last_time <= intent_time <= datetime.fromisoformat(row["utc"]), "intent/observation order differs")
        observation = row["observation"]
        require(all(type(observation[k]) is int for k in ("evaluation_count","correct_count","argmax_ties")) and
                all(type(v) is int for v in observation["predictions"]), "invalid observation integer types")
        require(len(observation["logits_float64_hex"]) == len(labels) == observation["evaluation_count"],
                "evaluation population changed")
        predictions, ties = [], 0
        for values in observation["logits_float64_hex"]:
            logits = [float.fromhex(v) for v in values]
            require(len(logits) == original["config"]["modulus"] and all(math.isfinite(v) for v in logits),
                    "invalid full logit vector")
            # Python's max returns the first index on a tie; independent of Torch.
            winner = max(range(len(logits)), key=logits.__getitem__)
            predictions.append(winner)
            ties += int(sum(v == logits[winner] for v in logits) > 1)
            total_logits += len(logits)
        correct = sum(a == b for a,b in zip(predictions, labels, strict=True))
        require(predictions == observation["predictions"] and correct == observation["correct_count"] and
                ties == observation["argmax_ties"], "logit-derived decisions differ")
        require(observation["argmax_tie_rule"] == "lowest class index (torch.argmax)", "tie convention changed")
        counts.append(correct)
        all_ties += ties
        last_time = datetime.fromisoformat(row["utc"])
        if step == 0:
            initial = observation
    checkpoint = source/"anchor.npz"
    require(hashlib.sha256(checkpoint.read_bytes()).hexdigest() == original["checkpoint_sha256"], "checkpoint changed")
    with np.load(checkpoint, allow_pickle=False) as stored:
        for key, output_key in (("parameter","parameter_sha256"), ("velocity","unscaled_velocity_sha256")):
            values = stored[key]
            require(values.dtype == np.dtype("float64") and values.shape == (original["identity"]["parameters"],),
                    "checkpoint representation differs")
            require(hashlib.sha256(values.tobytes(order="C")).hexdigest() == initial[output_key],
                    "initial observed state differs from physical checkpoint")
    event = first_streak(counts, target, K)
    require(last_time <= datetime.fromisoformat(reveal["utc"]), "reveal preceded final observation")
    bracket = amendment["issued_bracket"]
    covered = event is not None and bracket[0] <= event <= bracket[1]
    require(counts == reveal["counts"] and event == reveal["observed_first_persistent_offset"] and
            covered is reveal["observed_float64_crossing_in_bracket"] and
            reveal["issued_bracket"] == bracket, "independent event calculation differs")
    replay = checked(bundle/"completed_replay.json", amendment["completed_replay_sha256"])
    require(replay["status"] == "PASS" and replay["complete_recorded_graph_rehashed"] is True and
            replay["future_outcome_access"] is False and replay["identity"] == original["identity"] and
            replay["manifest_sha256"] == amendment["recorded_manifest_sha256"], "completed replay binding differs")
    assembly = replay["assembly"]
    require(assembly["inputs_compatible"] is True and assembly["bracket"] == bracket, "completed assembly differs")
    in_count_bounds = all(lo <= c <= hi for lo,c,hi in zip(
        assembly["lower_counts"], counts, assembly["upper_counts"], strict=True))
    return {
        "schema":"independent_dgx_451k_outcome_audit_v2", "status":"PASS",
        "amendment_sha256":amendment_sha, "reveal_sha256":reveal_sha,
        "pre_observation_publication_commit":publication_commit,
        "parameters":original["identity"]["parameters"], "depth":original["config"]["depth"],
        "normalization":original["config"]["normalization"], "observed_updates":H,
        "observed_rows":H+1, "logits_independently_reconstructed":total_logits,
        "argmax_ties":all_ties, "physical_initial_state_hashes_match":True,
        "counts":counts, "issued_bracket":bracket, "observed_first_persistent_offset":event,
        "observed_absolute_update":None if event is None else original["anchor"]+event,
        "bracket_covered":covered, "all_observed_counts_inside_certified_bounds":in_count_bounds,
        "original_prediction_error_updates":None if event is None else original["predicted_offset"]-event,
        "optimizer_window_seconds":(last_time-datetime.fromisoformat(engine["utc"])).total_seconds(),
        "ledger_population_checked":True, "runtime_only_delta_independently_checked":True,
        "timestamp_order_checked":True, "timestamps_externally_authenticated":False,
        "publication_commit_is_external_provenance_reference":True,
        "audit_source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "separate_source_runtime_amendment":True, "exact_real_continuation_verified":False,
        "new_certificate_issued":False, "neural_optimizer_reexecuted":False,
        "scope":"Independent hash/logit/event audit of the single completed CPU-float64 observation, not an exact-real state-tube replay.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--reveal-sha256", required=True)
    parser.add_argument("--amendment-sha256", required=True)
    parser.add_argument("--publication-commit", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root, args.bundle, args.reveal_sha256, args.amendment_sha256, args.publication_commit)
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+"\n"
    with args.report.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
    print(json.dumps({k:v for k,v in result.items() if k != "counts"}, indent=2))
