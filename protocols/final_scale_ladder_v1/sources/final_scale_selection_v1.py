"""Recorded, one-shot selection phase and independent control-flow replay.

The production launcher must authenticate the public source/runtime/protocol
seal before calling this layer. This module has no standalone training CLI.
Replay checks the recorded policy, population, states and selection order;
it does not independently retrain the prefix or recompute every past logit.
"""
from dataclasses import asdict
import hashlib

import numpy as np

from final_scale_artifacts_v1 import PhaseReader, PhaseWriter
from final_scale_engine_v1 import production_config
from final_scale_protocol_v1 import (eligible, first_event, select_without_continuation,
                                     validate_counts, validate_settings)
from prospective_ledger_v1 import encode, is_hash, sync_directory


SCHEMA = "final_scale_selection_phase_v1"
ROLES = ("training", "trigger", "certification")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def same(a, b):
    return encode(a) == encode(b)


def sha(value):
    return hashlib.sha256(value).hexdigest()


def raw_sha(value):
    return sha(memoryview(np.ascontiguousarray(value)).cast("B"))


def registered_policy(settings, rung_index):
    return {"schema": SCHEMA, "scope": "registered_scale", "settings": settings,
            "rung_index": rung_index, "config": asdict(production_config(settings, rung_index))}


def validate_policy(policy, *, allow_fixture=False):
    require(type(policy) is dict and set(policy) == {"schema", "scope", "settings", "rung_index", "config"}
            and policy["schema"] == SCHEMA, "invalid selection policy")
    settings, index, cfg = policy["settings"], policy["rung_index"], policy["config"]
    require(type(index) is int and 0 <= index < len(settings["rungs"]), "invalid selection rung")
    if policy["scope"] == "registered_scale":
        validate_settings(settings)
        require(same(cfg, asdict(production_config(settings, index))), "model config differs from registered rung")
    else:
        # No production CLI may expose this unit-test-only mode.
        require(allow_fixture is True and policy["scope"] == "tiny_development_fixture", "fixture execution is not authorized")
        required = {"modulus": 3, "model_dim": 4, "hidden_dim": 16, "heads": 2, "depth": 2,
                    "seed": 0, "dtype": "float64", "normalization": "layernorm", "loss": "cross_entropy",
                    "train_fraction": .6, "learning_rate": .003, "momentum": .9, "weight_decay": .01}
        require(all(same(cfg.get(k), v) for k, v in required.items()), "fixture model must remain tiny and seed zero")
        require(settings["rungs"][index]["parameters"] == 528 and
                settings["data"] == {"training_count": 5, "trigger_count": 2, "certification_count": 2},
                "fixture dimension or data counts differ")
        rule = settings["selection"]
        require(type(rule["maximum_updates"]) is int and 0 <= rule["maximum_updates"] <= 3 and
                rule["inspection_stride"] == 1 and type(rule["horizon"]) is int and 1 <= rule["horizon"] <= 4,
                "fixture update/window budget exceeded")
    sizes, rule = settings["data"], settings["selection"]
    require(type(rule["target_correct"]) is int and 1 <= rule["target_correct"] <= sizes["certification_count"]
            and type(rule["trigger_minimum_correct"]) is int and 0 <= rule["trigger_minimum_correct"] <= sizes["trigger_count"]
            and type(rule["persistence"]) is int and 1 <= rule["persistence"] <= rule["horizon"]+1,
            "invalid finite-set event policy")
    return policy


class RecordChain:
    def __init__(self, writer):
        self.writer, self.count, self.head = writer, 0, None

    def append(self, kind, payload):
        row = {"sequence": self.count, "previous_sha256": self.head, "kind": kind, "payload": payload}
        identity = self.writer.record(f"records/record_{self.count:06d}.json", row)
        self.count += 1
        self.head = identity["sha256"]


def run_phase(root, *, policy, runtime_record, bindings, engine_factory, guard,
              maximum_bytes, sync=sync_directory, allow_fixture=False):
    """Save all fixed data before selection, then freeze and seal its result.

engine_factory(record_callback) must construct the source-authenticated
Engine under already installed worker limits. This injection seam is for
the production launcher and small unit fixtures, not outcome-based tuning.
"""
    validate_policy(policy, allow_fixture=allow_fixture)
    require(bindings["phase_input_sha256"] == sha(encode(policy)) and
            bindings["runtime_manifest_sha256"] == sha(encode(runtime_record)), "selection input/runtime digest differs")
    writer = PhaseWriter(root, role="selection", bindings=bindings, guard=guard, sync=sync)
    writer.record("policy.json", policy)
    writer.record("runtime.json", runtime_record)
    chain = RecordChain(writer)
    guard()
    engine = engine_factory(chain.append)
    require(same(asdict(engine.config), policy["config"]), "initialized engine config differs")
    settings, index = policy["settings"], policy["rung_index"]
    require(len(engine.parameter) == settings["rungs"][index]["parameters"] and
            engine.maximum_updates == settings["selection"]["maximum_updates"] and engine.step == 0,
            "initialized engine state/budget differs")
    data = {}
    for role in ROLES:
        pairs, labels = engine.populations[role]
        data[role] = {
            "pairs": writer.array(f"data/{role}_pairs.npy", pairs.detach().cpu().numpy()),
            "labels": writer.array(f"data/{role}_labels.npy", labels.detach().cpu().numpy())}
    writer.record("data.json", data)
    result = select_without_continuation(engine, settings,
        record=lambda row: chain.append("inspection", row), guard=guard)
    clock_record = None
    if result["status"] == "candidate_selected":
        snapshot = engine.freeze_candidate(result, target=settings["selection"]["target_correct"],
                                            persistence=settings["selection"]["persistence"])
        parameter, velocity = snapshot["parameter"], snapshot["unscaled_velocity"]
        clock = dict(snapshot["clock"])
        reference = clock.pop("scaled_reference").cpu().numpy()
        clock_record = writer.array("clock/scaled_reference.npy", reference)
        require(clock_record["raw_sha256"] == clock["reference_sha256"], "saved clock identity differs")
        clock.update(normalization_eps=[list(pair) for pair in snapshot["normalization_eps"]], spec=snapshot["spec"])
        writer.record("clock/metadata.json", clock)
    else:
        require(result["status"] == "no_candidate" and engine.step == engine.maximum_updates,
                "selection ended without a registered terminal state")
        parameter = engine.parameter.detach().cpu().numpy()
        velocity = engine.velocity.detach().cpu().numpy()
    checkpoint = {"parameter": writer.array("checkpoint/parameter.npy", parameter),
                  "unscaled_velocity": writer.array("checkpoint/unscaled_velocity.npy", velocity)}
    selected = writer.record("selection.json", {"result": result, "checkpoint": checkpoint, "clock": clock_record})
    guard()
    seal = writer.seal({"status": result["status"], "selection_record_sha256": selected["sha256"],
        "record_count": chain.count, "record_head_sha256": chain.head}, maximum_bytes=maximum_bytes)
    return {**seal, "selection": result, "future_observation_executed": False}


def load_array_reference(reader, reference, *, path, shape, dtype):
    require(type(reference) is dict and set(reference) == {"path", "bytes", "sha256", "shape", "dtype", "raw_sha256"},
            "invalid typed array reference")
    require(reference["path"] == path and same(reference["shape"], list(shape)) and reference["dtype"] == np.dtype(dtype).str
            and is_hash(reference["raw_sha256"]), "array reference layout differs")
    require({key: reference[key] for key in ("path", "bytes", "sha256")} == reader.files[path], "array file identity differs")
    value = reader.array(path, shape=shape, dtype=dtype)
    require(raw_sha(value) == reference["raw_sha256"], "raw array identity differs")
    return value


def audit_phase(root, *, expected_manifest_sha256, policy, runtime_record, bindings, guard, allow_fixture=False):
    """Replay all recorded selection decisions before admitting a checkpoint.

The returned candidate has authenticated arrays and a consistent causal
record. Prefix optimizer steps/logits and source execution are not independently
recomputed here. The future observation is neither read nor performed.
"""
    validate_policy(policy, allow_fixture=allow_fixture)
    require(bindings["phase_input_sha256"] == sha(encode(policy)) and
            bindings["runtime_manifest_sha256"] == sha(encode(runtime_record)), "expected selection binding differs")
    reader = PhaseReader(root, expected_manifest_sha256=expected_manifest_sha256,
        expected_role="selection", expected_bindings=bindings, guard=guard)
    require(same(reader.record("policy.json"), policy) and same(reader.record("runtime.json"), runtime_record),
            "stored policy or runtime differs")
    settings, index, cfg = policy["settings"], policy["rung_index"], policy["config"]
    rule, sizes, n = settings["selection"], settings["data"], settings["rungs"][index]["parameters"]
    payload = reader.manifest["payload"]
    require(type(payload) is dict and set(payload) == {"status", "selection_record_sha256", "record_count", "record_head_sha256"}
            and type(payload["record_count"]) is int and 1 <= payload["record_count"] <= 3*rule["maximum_updates"]+5
            and is_hash(payload["record_head_sha256"]), "invalid selection terminal metadata")
    require(reader.files["selection.json"]["sha256"] == payload["selection_record_sha256"], "selection record digest differs")
    selected = reader.record("selection.json")
    require(set(selected) == {"result", "checkpoint", "clock"}, "invalid selection fields")
    populations, data = {}, reader.record("data.json")
    require(set(data) == set(ROLES), "population roles differ")
    for role in ROLES:
        require(set(data[role]) == {"pairs", "labels"}, "population fields differ")
        count = sizes[role + "_count"]
        pairs = load_array_reference(reader, data[role]["pairs"], path=f"data/{role}_pairs.npy", shape=(count, 2), dtype="<i8")
        labels = load_array_reference(reader, data[role]["labels"], path=f"data/{role}_labels.npy", shape=(count,), dtype="<i8")
        require(((pairs >= 0) & (pairs < cfg["modulus"])).all() and np.array_equal(labels, pairs.sum(1) % cfg["modulus"]),
                "modular population labels/tokens differ")
        populations[role] = (pairs, labels)
    pairs = np.concatenate([v[0] for v in populations.values()])
    require(len(pairs) == cfg["modulus"]**2 and len(np.unique(pairs, axis=0)) == len(pairs), "populations overlap or are incomplete")
    require(set(selected["checkpoint"]) == {"parameter", "unscaled_velocity"}, "checkpoint fields differ")
    parameter = load_array_reference(reader, selected["checkpoint"]["parameter"], path="checkpoint/parameter.npy", shape=(n,), dtype="<f8")
    velocity = load_array_reference(reader, selected["checkpoint"]["unscaled_velocity"], path="checkpoint/unscaled_velocity.npy", shape=(n,), dtype="<f8")
    records, head = [], None
    for seq in range(payload["record_count"]):
        path = f"records/record_{seq:06d}.json"
        record = reader.record(path)
        require(set(record) == {"sequence", "previous_sha256", "kind", "payload"} and
                type(record["sequence"]) is int and record["sequence"] == seq and record["previous_sha256"] == head,
                "selection record chain differs")
        require(type(record["kind"]) is str and type(record["payload"]) is dict, "invalid selection record")
        records.append(record)
        head = reader.files[path]["sha256"]
    require(head == payload["record_head_sha256"], "selection chain head differs")
    cursor = 0

    def take(kind):
        nonlocal cursor
        require(cursor < len(records) and records[cursor]["kind"] == kind, "unexpected selection record phase")
        value = records[cursor]["payload"]
        cursor += 1
        return value

    def state(value, step):
        require(type(value) is dict and set(value) == {"step", "parameter_sha256", "unscaled_velocity_sha256"} and
                type(value["step"]) is int and value["step"] == step and
                is_hash(value["parameter_sha256"]) and is_hash(value["unscaled_velocity_sha256"]), "invalid state identity")
        return value

    ready = take("engine_ready")
    require(set(ready) == {"config", "parameters", "population_counts", "population_sha256", "step",
                          "parameter_sha256", "unscaled_velocity_sha256"} and same(ready["config"], cfg) and
            type(ready["parameters"]) is int and ready["parameters"] == n and
            same(ready["population_counts"], {role: sizes[role+"_count"] for role in ROLES}) and
            ready["population_sha256"] == {role: {"pairs": raw_sha(p), "labels": raw_sha(y)} for role, (p, y) in populations.items()},
            "engine initialization record differs")
    current = state({key: ready[key] for key in ("step", "parameter_sha256", "unscaled_velocity_sha256")}, 0)
    chosen_counts, frozen, expected_result = None, None, None
    for step in range(rule["maximum_updates"]+1):
        if step % rule["inspection_stride"] == 0:
            inspection = take("inspection")
            qualifies = eligible(inspection["current"], settings)
            require(type(inspection.get("anchor")) is int and inspection["anchor"] == step and
                    type(inspection.get("eligible")) is bool and inspection["eligible"] == qualifies,
                    "inspection eligibility differs")
            expected_fields = {"anchor", "current", "eligible"}
            if qualifies:
                expected_fields |= {"clock_counts", "predicted_offset"}
                counts = validate_counts(inspection["clock_counts"], evaluation_count=sizes["certification_count"], expected_length=rule["horizon"]+1)
                require(counts[0] == inspection["current"]["certification"], "clock count anchor differs")
                offset = first_event(counts, target=rule["target_correct"], persistence=rule["persistence"])
                require(same(offset, inspection["predicted_offset"]), "recorded first event differs")
                if offset is not None and offset > 0:
                    chosen_counts = counts
                    expected_result = {"status": "candidate_selected", "anchor": step, "predicted_offset": offset,
                                       "future_continuation_executed": False}
            require(set(inspection) == expected_fields, "inspection fields differ")
            if expected_result is not None:
                frozen = take("candidate_frozen")
                require(set(frozen) == {"selection", "step", "parameter_sha256", "unscaled_velocity_sha256", "clock_reference_sha256"}
                        and same(frozen["selection"], expected_result) and
                        state({key: frozen[key] for key in current}, step) == current and is_hash(frozen["clock_reference_sha256"]),
                        "frozen anchor differs")
                break
        if step == rule["maximum_updates"]:
            expected_result = {"status": "no_candidate", "completed_updates": step, "future_continuation_executed": False}
            break
        intent = take("update_intent")
        require(set(intent) == set(current) | {"next_step"} and type(intent["next_step"]) is int and
                intent["next_step"] == step+1 and same({key: intent[key] for key in current}, current), "update intent differs")
        current = state(take("update_completed"), step+1)
    require(cursor == len(records), "real update or extra record after terminal selection")
    require(same(selected["result"], expected_result) and payload["status"] == expected_result["status"], "selected result differs from first eligible event")
    require(raw_sha(parameter) == current["parameter_sha256"] and raw_sha(velocity) == current["unscaled_velocity_sha256"],
            "stored physical checkpoint differs from selection chain")
    expected_files = {"method.json", "policy.json", "runtime.json", "data.json", "selection.json",
                      "checkpoint/parameter.npy", "checkpoint/unscaled_velocity.npy"}
    expected_files |= {f"data/{role}_{part}.npy" for role in ROLES for part in ("pairs", "labels")}
    expected_files |= {f"records/record_{j:06d}.json" for j in range(len(records))}
    reference, metadata = None, None
    if frozen is not None:
        expected_files |= {"clock/scaled_reference.npy", "clock/metadata.json"}
        reference = load_array_reference(reader, selected["clock"], path="clock/scaled_reference.npy",
                                          shape=(rule["horizon"]+1, 2*n), dtype="<f8")
        metadata = reader.record("clock/metadata.json")
        require(metadata["reference_sha256"] == raw_sha(reference) == frozen["clock_reference_sha256"] and
                same(metadata["counts"], chosen_counts) and metadata["horizon"] == rule["horizon"] and
                metadata["sweeps"] == rule["clock_sweeps"] and metadata["numeric_cap"] == rule["numeric_cap"] and
                same({key: metadata[key] for key in current}, current), "stored clock metadata differs")
        require(np.array_equal(reference[0, :n], parameter) and
                np.array_equal(reference[0, n:], cfg["learning_rate"]*velocity), "stored reference anchor differs")
    else:
        require(selected["clock"] is None, "no-candidate result contains a selected clock")
    require(set(reader.files) == expected_files, "unregistered selection artifact")
    guard()
    return {"status": "selection_control_and_artifacts_replayed", "selection": expected_result,
            "manifest_sha256": reader.identity, "record_count": len(records), "record_head_sha256": head,
            "prefix_training_and_logits_independently_recomputed": False,
            "future_outcome_accessed": False, "parameter": parameter, "unscaled_velocity": velocity,
            "reference": reference, "clock_metadata": metadata, "populations": populations}
