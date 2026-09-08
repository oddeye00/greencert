"""Disposition-gated one-shot observation and independent output replay.

The observer runs CPU float64 momentum once, from the stored dyadic physical
checkpoint. The auditor re-evaluates logits at recorded states; it does not
rerun optimizer updates or claim exact-real enclosure of the floating-point
training program. All runtime, update-intent, state and completion records
enter a durable one-shot ledger. There is no resume or retry operation.
"""
import numpy as np
import torch

from final_scale_artifacts_v1 import PhaseReader, PhaseWriter
from final_scale_construction_v1 import (audit_disposition, phase_bindings, selected_context,
    require, same, sha)
from final_scale_protocol_v1 import first_event
from final_scale_selection_v1 import load_array_reference, raw_sha, validate_policy
from prospective_ledger_v1 import Ledger, audit_complete, is_hash, sync_directory
from transformer_hvp_grokking import gradient, logits


SCHEMA = "final_scale_one_shot_observation_v1"
CASE_FIELDS = {"selection_root", "selection_manifest", "construction_root", "construction_manifest",
    "disposition_root", "disposition_manifest", "policy", "selection_runtime", "construction_runtime",
    "disposition_runtime", "selection_bindings"}


def validate_case(case, allow_fixture):
    require(type(case) is dict and set(case) == CASE_FIELDS, "incomplete observation case")
    validate_policy(case["policy"], allow_fixture=allow_fixture)
    require(all(is_hash(case[key]) for key in ("selection_manifest", "construction_manifest", "disposition_manifest")),
            "external phase pins required before observation")


def observation_input(case):
    return {"schema": SCHEMA, "selection_policy_sha256": sha(case["policy"]),
        **{key+"_sha256": case[key] for key in ("selection_manifest", "construction_manifest", "disposition_manifest")}}


def admit(case, guard, allow_fixture):
    validate_case(case, allow_fixture)
    audit = audit_disposition(case["disposition_root"],
        **{k: v for k, v in case.items() if k != "disposition_root"}, guard=guard, allow_fixture=allow_fixture)
    require(audit["status"] == "complete_disposition_recomputed" and
            audit["disposition"]["future_observation_authorized"] is True, "no complete disposition authorizes observation")
    return audit


def selected(case, guard, allow_fixture):
    return selected_context(**{k: case[k] for k in
        ("selection_root", "selection_manifest", "policy", "selection_runtime", "selection_bindings")},
        guard=guard, allow_fixture=allow_fixture)


def ledger_bindings(case, admission):
    return {"protocol_sha256": case["selection_bindings"]["protocol_sha256"],
        "source_manifest_sha256": case["selection_bindings"]["source_manifest_sha256"],
        "disposition_sha256": admission["disposition_sha256"],
        "checkpoint_sha256": admission["disposition"]["identity"]["candidate_sha256"]}


def ready_record(context, bindings, runtime_record, admission):
    return {"schema": SCHEMA, "config": context["expected"]["config"], "phase_bindings": bindings,
        "runtime_record_sha256": sha(runtime_record), "disposition_sha256": admission["disposition_sha256"],
        "anchor_update": context["selected"]["selection"]["anchor"],
        "parameter_sha256": raw_sha(context["inputs"]["parameter"]),
        "unscaled_velocity_sha256": raw_sha(context["inputs"]["unscaled_velocity"]),
        "numeric_scope": "one CPU float64 continuation, not independently validated exact-real states"}


def observed_summary(counts, ties, admission, context, rule):
    require(len(counts) == rule["horizon"]+1 and len(ties) == len(counts), "incomplete observed window")
    offset = first_event(counts, target=rule["target_correct"], persistence=rule["persistence"])
    disposition, assembly = admission["disposition"], admission["numerical_replay"]["assembly"]
    bracket = disposition["bracket"]
    covered = None if bracket is None else (offset is not None and bracket[0] <= offset <= bracket[1])
    count_check = None
    if "lower_counts" in assembly and "upper_counts" in assembly:
        require(len(assembly["lower_counts"]) == len(counts) == len(assembly["upper_counts"]), "count enclosure horizon differs")
        count_check = [lo <= actual <= hi for lo, actual, hi in zip(assembly["lower_counts"], counts, assembly["upper_counts"])]
    return {"schema": SCHEMA, "status": "one_shot_float64_observation_complete",
        "anchor_update": context["selected"]["selection"]["anchor"], "horizon": rule["horizon"],
        "observations": len(counts), "optimizer_updates": rule["horizon"], "counts": counts, "top_ties": ties,
        "first_persistent_offset": offset, "target_correct": rule["target_correct"], "persistence": rule["persistence"],
        "disposition_sha256": admission["disposition_sha256"], "certificate_issued": disposition["event_certificate_issued"],
        "bracket": bracket, "issued_bracket_covered": covered, "count_bounds_contained_by_step": count_check,
        "all_count_bounds_contained": None if count_check is None else all(count_check),
        "independent_exact_real_state_continuation": False}


def observe_phase(root, *, case, runtime_record, guard, maximum_bytes, sync=sync_directory, allow_fixture=False):
    validate_case(case, allow_fixture)
    inputs = observation_input(case)
    bindings = phase_bindings(case["selection_bindings"], runtime_record, inputs)
    writer = PhaseWriter(root, role="observation", bindings=bindings, guard=guard, sync=sync)
    writer.record("observation_input.json", inputs)
    writer.record("runtime.json", runtime_record)
    admission = admit(case, guard, allow_fixture)
    ledger = None
    try:
        with selected(case, guard, allow_fixture) as context:
            cfg, template, spec = context["config"], context["template"], context["spec"]
            rule = case["policy"]["settings"]["selection"]
            parameter = torch.tensor(np.asarray(context["inputs"]["parameter"]), dtype=torch.float64)
            velocity = torch.tensor(np.asarray(context["inputs"]["unscaled_velocity"]), dtype=torch.float64)
            tx = torch.tensor(np.asarray(context["inputs"]["train_pairs"]), dtype=torch.long)
            ty = torch.tensor(np.asarray(context["inputs"]["train_labels"]), dtype=torch.long)
            ex = torch.tensor(np.asarray(context["inputs"]["evaluation_pairs"]), dtype=torch.long)
            ey = torch.tensor(np.asarray(context["inputs"]["evaluation_labels"]), dtype=torch.long)
            ledger = Ledger.create(writer.root / "ledger", horizon=rule["horizon"], bindings=ledger_bindings(case, admission), sync=sync)
            ledger.append("engine_ready", ready_record(context, bindings, runtime_record, admission))
            counts, ties, previous = [], [], None
            for j in range(rule["horizon"]+1):
                guard()
                if j:
                    ledger.append("update_intent", {"step": j, "previous_step": j-1,
                        "parameter_sha256": previous["parameter"]["raw_sha256"],
                        "unscaled_velocity_sha256": previous["unscaled_velocity"]["raw_sha256"]})
                    guard()  # No optimizer update may precede its durable intent.
                    velocity = (cfg.momentum*velocity + gradient(parameter, tx, ty, template, spec, cfg)).detach()
                    parameter = (parameter - cfg.learning_rate*velocity).detach()
                with torch.no_grad():
                    values = logits(parameter, ex, template, spec).detach()
                require(bool(torch.isfinite(parameter).all()) and bool(torch.isfinite(velocity).all()) and
                        bool(torch.isfinite(values).all()), "nonfinite observed state or output")
                count = int((values.argmax(1) == ey).sum())
                tie = int(((values == values.max(1, keepdim=True).values).sum(1) > 1).sum())
                prefix = f"states/step_{j:03d}"
                previous = {"step": j, "absolute_update": context["selected"]["selection"]["anchor"]+j,
                    "parameter": writer.array(prefix+"_parameter.npy", parameter.numpy()),
                    "unscaled_velocity": writer.array(prefix+"_unscaled_velocity.npy", velocity.numpy()),
                    "logits": writer.array(prefix+"_logits.npy", values.numpy()), "correct": count, "top_ties": tie}
                ledger.append("observation", previous)
                counts.append(count)
                ties.append(tie)
            summary = observed_summary(counts, ties, admission, context, rule)
            record = writer.record("observation_summary.json", summary)
            ledger.append("completed", {"summary_sha256": record["sha256"], "optimizer_updates": rule["horizon"],
                                        "observations": rule["horizon"]+1})
            seal = writer.seal({"ledger_head_sha256": ledger.head, "summary_sha256": record["sha256"],
                "disposition_sha256": admission["disposition_sha256"]}, maximum_bytes=maximum_bytes)
            return {**seal, "ledger_head_sha256": ledger.head, "summary": summary}
    except BaseException as error:
        if ledger is not None and not ledger.failed and ledger.phase.expected != "terminal":
            ledger.append("interrupted", {"error_type": type(error).__name__, "automatic_retry_or_resume": False})
        raise


def audit_observation(root, *, observation_manifest, case, observation_runtime, guard, allow_fixture=False):
    """Recompute the disposition, ledger, all point logits, counts and event.

Stored optimizer states and their update intents are authenticated. The
optimizer updates themselves are NOT re-executed by this output auditor.
"""
    validate_case(case, allow_fixture)
    admission = admit(case, guard, allow_fixture)
    inputs = observation_input(case)
    bindings = phase_bindings(case["selection_bindings"], observation_runtime, inputs)
    reader = PhaseReader(root, expected_manifest_sha256=observation_manifest, expected_role="observation",
                         expected_bindings=bindings, guard=guard)
    same(reader.record("observation_input.json"), inputs, "observer phase input differs")
    same(reader.record("runtime.json"), observation_runtime, "observer runtime differs")
    payload = reader.manifest["payload"]
    same(payload, {"ledger_head_sha256": payload["ledger_head_sha256"],
        "summary_sha256": reader.files["observation_summary.json"]["sha256"],
        "disposition_sha256": admission["disposition_sha256"]}, "observer completion differs")
    ledger_audit = audit_complete(reader.root / "ledger", expected_head_sha256=payload["ledger_head_sha256"])
    rule = case["policy"]["settings"]["selection"]
    require(ledger_audit["horizon"] == rule["horizon"] and ledger_audit["records"] == 2*rule["horizon"]+4,
            "observation ledger length differs")
    records = [reader.record(f"ledger/record_{j:06d}.json") for j in range(ledger_audit["records"])]
    same(records[0]["payload"], {"horizon": rule["horizon"], "bindings": ledger_bindings(case, admission),
        "automatic_retry_or_resume": False}, "observer ledger admission differs")
    expected_files = {"method.json", "runtime.json", "observation_input.json", "observation_summary.json"}
    expected_files.update(f"ledger/record_{j:06d}.json" for j in range(len(records)))
    with selected(case, guard, allow_fixture) as context:
        same(records[1]["payload"], ready_record(context, bindings, observation_runtime, admission), "observer engine identity differs")
        pairs = torch.tensor(np.asarray(context["inputs"]["evaluation_pairs"]), dtype=torch.long)
        labels = context["inputs"]["evaluation_labels"]
        n, ne, classes = context["expected"]["parameters"], len(labels), context["config"].modulus
        counts, ties, previous = [], [], None
        for j in range(rule["horizon"]+1):
            guard()
            row = records[2+2*j]["payload"]
            require(set(row) == {"step", "absolute_update", "parameter", "unscaled_velocity", "logits", "correct", "top_ties"}
                    and type(row["step"]) is int and row["step"] == j and
                    row["absolute_update"] == context["selected"]["selection"]["anchor"]+j, "observed row identity differs")
            if j:
                same(records[1+2*j]["payload"], {"step": j, "previous_step": j-1,
                    "parameter_sha256": previous["parameter"]["raw_sha256"],
                    "unscaled_velocity_sha256": previous["unscaled_velocity"]["raw_sha256"]}, "observed update intent differs")
            mapped = {}
            try:
                for name, shape in (("parameter", (n,)), ("unscaled_velocity", (n,)), ("logits", (ne, classes))):
                    path = f"states/step_{j:03d}_{name}.npy"
                    mapped[name] = load_array_reference(reader, row[name], path=path, shape=shape, dtype="<f8")
                    expected_files.add(path)
                if j == 0:
                    require(np.array_equal(mapped["parameter"], context["inputs"]["parameter"]) and
                            np.array_equal(mapped["unscaled_velocity"], context["inputs"]["unscaled_velocity"]),
                            "observer did not start from physical checkpoint")
                with torch.no_grad():
                    independently_computed = logits(torch.tensor(np.asarray(mapped["parameter"])), pairs,
                        context["template"], context["spec"]).numpy()
                require(np.array_equal(independently_computed, mapped["logits"]), "independent observed logits differ")
                values = mapped["logits"]
                count = int((values.argmax(1) == labels).sum())
                tie = int(((values == values.max(1, keepdims=True)).sum(1) > 1).sum())
                require(type(row["correct"]) is int and row["correct"] == count and
                        type(row["top_ties"]) is int and row["top_ties"] == tie, "observed count or tie differs")
                counts.append(count)
                ties.append(tie)
            finally:
                for value in mapped.values():
                    value._mmap.close()
            previous = row
        summary = observed_summary(counts, ties, admission, context, rule)
        same(reader.record("observation_summary.json"), summary, "observed first-passage or coverage differs")
    same(records[-1]["payload"], {"summary_sha256": payload["summary_sha256"],
        "optimizer_updates": rule["horizon"], "observations": rule["horizon"]+1}, "observer ledger terminal differs")
    require(set(reader.files) == expected_files, "unregistered observation artifact")
    return {"status": "complete_observation_and_point_outputs_replayed", "manifest_sha256": reader.identity,
        "ledger_head_sha256": ledger_audit["head_sha256"], "summary": summary,
        "point_logits_independently_recomputed": (rule["horizon"]+1)*ne*classes,
        "optimizer_updates_reexecuted": False, "exact_real_state_tubes_independently_validated": False,
        "timestamps_externally_attested": False}
