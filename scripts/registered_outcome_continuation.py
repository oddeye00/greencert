"""One-shot outcome logging after a separate authenticated disposition gate.

This module does not evaluate a model or prove an event. Interrupted runs stay
closed to automatic retry, including failures before the first update.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

from verified_artifact_io import IntegrityError, require_sha256


def require(condition, message):
    if not condition:
        raise IntegrityError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def publish_json(path, value):
    """Fsync complete bytes and install without replacing an existing record."""
    path = Path(path)
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".outcome-", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    require(path.read_bytes() == payload, "published outcome bytes differ")
    return hashlib.sha256(payload).hexdigest()


def first_persistent(counts, target, persistence):
    require(type(target) is int and target > 0 and type(persistence) is int and
            1 <= persistence <= len(counts), "invalid persistent event")
    require(all(type(v) is int and v >= 0 for v in counts), "invalid count path")
    return next((j for j in range(len(counts)-persistence+1)
                 if all(v >= target for v in counts[j:j+persistence])), None)


def unopened(reader, run_folder, outcome_path):
    # Even an empty directory can be the remnant of a crash after reservation.
    require(not reader.resolve(run_folder).exists(),
            "outcome run already reserved; no automatic retry or resume")
    require(not reader.resolve(outcome_path).exists(), "outcome already published")


def execute_once(reader, *, gate, protocol, protocol_sha256, run_folder,
                 outcome_path, prepare, recheck, progress=None):
    """Run a registered observer; only the caller authenticates the full gate.

`prepare` returns observe/advance methods. It is not called until the gate is
authorized, the inputs are rechecked, and an exclusive reservation is durable.
Production passes the frozen neural adapter; tests pass synthetic observers.
"""
    require_sha256(protocol_sha256)
    require(gate.get("outcome_join_authorized") is True and
            gate.get("future_outcome_access") is False and
            gate.get("decision") in ("certificate", "route_abstention"), "unauthorized outcome gate")
    require_sha256(gate["disposition_sha256"])
    H, target, K = protocol["horizon"], protocol["target"], protocol["persistence"]
    N = protocol["evaluation_count"]
    require(type(H) is int and H > 0 and type(N) is int and N > 0 and
            type(target) is int and 1 <= target <= N and type(K) is int and 1 <= K <= H+1,
            "invalid registered event")
    bracket = gate.get("bracket")
    if gate["decision"] == "certificate":
        require(isinstance(bracket, (list, tuple)) and len(bracket) == 2 and
                all(type(v) is int for v in bracket) and 0 < bracket[0] <= bracket[1] <= H-K+1,
                "invalid issued bracket")
    else:
        require(bracket is None, "abstention carries an issued bracket")
    unopened(reader, run_folder, outcome_path)
    recheck()
    unopened(reader, run_folder, outcome_path)
    folder = reader.resolve(run_folder)
    folder.parent.mkdir(parents=True, exist_ok=True)
    folder.mkdir(exist_ok=False)
    started_sha = publish_json(folder/"started.json", {
        "schema": "registered_outcome_start_v1", "utc": utc(),
        "protocol_sha256": protocol_sha256, "gate": gate,
        "future_access_must_be_assumed_possible_after_this_reservation": True,
        "automatic_retry_or_resume": False,
    })
    counts, rows, actual_updates = [], {}, 0
    try:
        engine = prepare()
        publish_json(folder/"engine_ready.json", {
            "utc": utc(), "runtime": engine.runtime, "future_updates_executed": 0,
        })
        for step in range(H+1):
            if step:
                # Intent precedes execution. A crash cannot look outcome-blind.
                publish_json(folder/f"intent_{step:03d}.json", {
                    "step": step, "utc": utc(), "started_sha256": started_sha,
                })
                engine.advance()
                actual_updates = step
            row = engine.observe()
            require(type(row.get("correct_count")) is int and 0 <= row["correct_count"] <= N,
                    "observer returned invalid correct count")
            require(row.get("evaluation_count") == N, "observer changed evaluation population")
            for key in ("parameter_sha256", "unscaled_velocity_sha256"):
                require_sha256(row[key])
            record = {"schema": "registered_float64_outcome_row_v1", "step": step,
                      "absolute_update": protocol["anchor"]+step, "utc": utc(),
                      "started_sha256": started_sha, "observation": row}
            rows[f"row_{step:03d}.json"] = publish_json(folder/f"row_{step:03d}.json", record)
            counts.append(row["correct_count"])
            if progress:
                progress(step, row["correct_count"])
        # Verify source/input identity once more before publishing any comparison.
        recheck()
        event = first_persistent(counts, target, K)
        result = {
            "schema": "registered_float64_outcome_v1", "utc": utc(),
            "protocol_sha256": protocol_sha256, "started_sha256": started_sha,
            "disposition_sha256": gate["disposition_sha256"], "gate": gate,
            "horizon": H, "target": target, "persistence": K, "anchor": protocol["anchor"],
            "counts": counts, "observed_first_persistent_offset": event,
            "predicted_offset": protocol["predicted_offset"],
            "prediction_error_updates": None if event is None else protocol["predicted_offset"]-event,
            "issued_bracket": bracket,
            "observed_float64_crossing_in_bracket": None if bracket is None else
                (event is not None and bracket[0] <= event <= bracket[1]),
            "future_outcome_access": True, "completed_updates": actual_updates,
            "observation_arithmetic": "registered CPU float64 optimizer continuation",
            "exact_real_continuation_verified": False, "exact_real_state_tube_tested": False,
            "whole_prior_training_program_certified": False,
            "row_folder": str(run_folder), "row_sha256": rows, "runtime": engine.runtime,
        }
        sha = publish_json(reader.resolve(outcome_path), result)
        return {"status": "float64_outcome_recorded", "outcome_sha256": sha,
                "outcome_path": str(outcome_path), "future_outcome_access": True,
                "exact_real_continuation_verified": False}
    except BaseException as error:
        # Counts already observed are retained. Do not erase the reservation.
        publish_json(folder/"interrupted.json", {
            "utc": utc(), "error_type": type(error).__name__, "error": str(error),
            "completed_updates": actual_updates, "completed_observations": len(counts),
            "future_access_must_be_assumed_possible": True, "automatic_retry_or_resume": False,
        })
        raise
