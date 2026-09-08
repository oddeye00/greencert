"""Selected-checkpoint construction and a separate numerical disposition.

No true optimizer continuation is performed. The caller must authenticate
the public source/runtime/protocol seal and install worker limits before
calling this module. The restricted unit-fixture option must never be exposed
by a production launcher; it is validated by the selection policy layer.
"""
from contextlib import contextmanager
from dataclasses import asdict
import hashlib

import numpy as np
import torch

from final_scale_artifacts_v1 import PhaseReader, PhaseWriter
from final_scale_numeric_core_v1 import build
from final_scale_numeric_replay_v1 import audit_numeric
from final_scale_protocol_v1 import first_event
from final_scale_selection_v1 import audit_phase, raw_sha, validate_policy
from prospective_ledger_v1 import encode, is_hash, sync_directory
from transformer_hvp_grokking import TransformerConfig, flat_spec, logits, make_disjoint_split, make_template


SCHEMA = "final_scale_construction_phases_v1"


def require(value, message):
    if not value:
        raise ValueError(message)


def same(actual, expected, message):
    require(encode(actual) == encode(expected), message)


def sha(value):
    return hashlib.sha256(encode(value)).hexdigest()


def phase_input(stage, policy, selection_manifest, construction_manifest=None):
    require(stage in ("construction", "disposition") and is_hash(selection_manifest), "invalid phase input")
    require((stage == "construction" and construction_manifest is None) or
            (stage == "disposition" and is_hash(construction_manifest)), "missing or unexpected construction pin")
    return {"schema": SCHEMA, "stage": stage, "selection_policy_sha256": sha(policy),
        "selection_manifest_sha256": selection_manifest, "construction_manifest_sha256": construction_manifest}


def phase_bindings(selection_bindings, runtime_record, inputs):
    return {"protocol_sha256": selection_bindings["protocol_sha256"],
        "source_manifest_sha256": selection_bindings["source_manifest_sha256"],
        "runtime_manifest_sha256": sha(runtime_record), "phase_input_sha256": sha(inputs)}


def replay_clock_counts(reference, pairs, labels, template, spec, guard):
    """Re-evaluate saved reference outputs, never true optimizer updates."""
    counts = []
    with torch.no_grad():
        for row in reference:
            guard()
            values = logits(torch.tensor(np.asarray(row[:sum(spec.sizes)])),
                            torch.tensor(np.asarray(pairs)), template, spec)
            require(bool(torch.isfinite(values).all()), "nonfinite clock replay")
            counts.append(int((values.argmax(1) == torch.tensor(np.asarray(labels))).sum()))
    return counts


@contextmanager
def selected_context(selection_root, *, selection_manifest, policy, selection_runtime,
                     selection_bindings, guard, allow_fixture=False):
    """Admit only the externally pinned selected checkpoint and actual model."""
    selected = audit_phase(selection_root, expected_manifest_sha256=selection_manifest,
        policy=policy, runtime_record=selection_runtime, bindings=selection_bindings,
        guard=guard, allow_fixture=allow_fixture)
    try:
        require(selected["selection"]["status"] == "candidate_selected", "no selected candidate for construction")
        guard()
        cfg = TransformerConfig(**policy["config"])
        template = make_template(cfg)
        spec = flat_spec(template)
        eps = [[block.norm1.eps, block.norm2.eps] for block in template.blocks]
        metadata = {"names": list(spec.names), "sizes": list(spec.sizes), "shapes": [list(s) for s in spec.shapes]}
        same(selected["clock_metadata"]["spec"], metadata, "actual parameter specification differs")
        same(selected["clock_metadata"]["normalization_eps"], eps, "actual LayerNorm epsilon differs")
        settings, index = policy["settings"], policy["rung_index"]
        n = settings["rungs"][index]["parameters"]
        require(sum(spec.sizes) == n, "actual parameter count differs")
        data = make_disjoint_split(cfg)
        for j, role in enumerate(("training", "trigger", "certification")):
            for observed, rebuilt in zip(selected["populations"][role], data[2*j:2*j+2]):
                require(np.array_equal(observed, rebuilt.numpy()), "regenerated registered population differs")
        pairs, labels = selected["populations"]["certification"]
        counts = replay_clock_counts(selected["reference"], pairs, labels, template, spec, guard)
        same(counts, selected["clock_metadata"]["counts"], "independent clock count replay differs")
        offset = first_event(counts, target=settings["selection"]["target_correct"],
                             persistence=settings["selection"]["persistence"])
        same(offset, selected["selection"]["predicted_offset"], "independent clock first event differs")
        values = {"parameter": selected["parameter"], "unscaled_velocity": selected["unscaled_velocity"],
            "clock": selected["reference"], "train_pairs": selected["populations"]["training"][0],
            "train_labels": selected["populations"]["training"][1], "evaluation_pairs": pairs, "evaluation_labels": labels}
        options = settings["construction"]
        expected = {"parameters": n, "horizon": settings["selection"]["horizon"], "config": asdict(cfg), "spec": metadata,
            "normalization_eps": eps, "options": options, "probe_seed": options["probe_seed_base"]+index,
            "probability_ratio": options["per_rung_failure_probability_ratio"],
            "input_raw_sha256": {key: raw_sha(value) for key, value in values.items()},
            "training_count": len(values["train_pairs"]), "evaluation_count": len(pairs)}
        clock_replay = {"selection_manifest_sha256": selection_manifest, "reference_sha256": raw_sha(selected["reference"]),
            "counts": counts, "predicted_offset": offset, "saved_reference_outputs_reexecuted": True,
            "reference_construction_reexecuted": False, "future_optimizer_updates_executed": False}
        yield {"selected": selected, "config": cfg, "template": template, "spec": spec, "normalization_eps": eps,
               "inputs": values, "expected": expected, "clock_replay": clock_replay}
    finally:
        for key in ("parameter", "unscaled_velocity", "reference"):
            if isinstance(selected[key], np.memmap):
                selected[key]._mmap.close()
        for pair in selected["populations"].values():
            for value in pair:
                value._mmap.close()


def construct_phase(root, *, selection_root, selection_manifest, policy, selection_runtime,
                    runtime_record, selection_bindings, guard, maximum_bytes,
                    sync=sync_directory, allow_fixture=False):
    validate_policy(policy, allow_fixture=allow_fixture)
    inputs = phase_input("construction", policy, selection_manifest)
    bindings = phase_bindings(selection_bindings, runtime_record, inputs)
    # Reserve before admitting the model: any later failure leaves a one-shot
    # failed phase, never a directory that a controller may resume.
    writer = PhaseWriter(root, role="construction", bindings=bindings, guard=guard, sync=sync)
    writer.record("construction_input.json", inputs)
    writer.record("runtime.json", runtime_record)
    with selected_context(selection_root, selection_manifest=selection_manifest, policy=policy,
            selection_runtime=selection_runtime, selection_bindings=selection_bindings, guard=guard,
            allow_fixture=allow_fixture) as context:
        writer.record("clock_replay.json", context["clock_replay"])
        result = build(writer, **context["inputs"], template=context["template"], spec=context["spec"],
            config=context["config"], normalization_eps=context["normalization_eps"],
            options=context["expected"]["options"], probe_seed=context["expected"]["probe_seed"],
            probability_ratio=context["expected"]["probability_ratio"], guard=guard)
        payload = {"status": "complete_numeric_evidence_constructed", "event_certificate_issued": False,
            "future_observation_authorized": False, "numeric_result_sha256": sha(result),
            "external_numeric_contract_sha256": sha(context["expected"])}
        seal = writer.seal(payload, maximum_bytes=maximum_bytes)
    return {**seal, **payload}


def replay_construction(root, *, construction_manifest, selection_root, selection_manifest, policy,
                        selection_runtime, construction_runtime, selection_bindings, guard, allow_fixture=False):
    inputs = phase_input("construction", policy, selection_manifest)
    bindings = phase_bindings(selection_bindings, construction_runtime, inputs)
    reader = PhaseReader(root, expected_manifest_sha256=construction_manifest, expected_role="construction",
                         expected_bindings=bindings, guard=guard)
    same(reader.record("construction_input.json"), inputs, "construction selection binding differs")
    same(reader.record("runtime.json"), construction_runtime, "construction runtime differs")
    with selected_context(selection_root, selection_manifest=selection_manifest, policy=policy,
            selection_runtime=selection_runtime, selection_bindings=selection_bindings,
            guard=guard, allow_fixture=allow_fixture) as context:
        same(reader.record("clock_replay.json"), context["clock_replay"], "saved clock replay differs")
        same(reader.manifest["payload"], {"status": "complete_numeric_evidence_constructed", "event_certificate_issued": False,
            "future_observation_authorized": False, "numeric_result_sha256": reader.files["numeric_result.json"]["sha256"],
            "external_numeric_contract_sha256": sha(context["expected"])}, "construction completion differs")
        rule = policy["settings"]["selection"]
        replay = audit_numeric(reader, expected=context["expected"], target=rule["target_correct"],
                               persistence=rule["persistence"], guard=guard)
    root_files = {"method.json", "construction_input.json", "runtime.json", "clock_replay.json",
        "numeric_method.json", "identity.json", "response/summary.json", "green_capture.json", "numeric_result.json"}
    prefixes = ("inputs/", "reference/", "response/", "response_kernels/", "green/", "green_kernels/", "neural/", "point_logits/")
    require(all(path in root_files or path.startswith(prefixes) for path in reader.files), "unregistered construction artifact")
    return replay


def disposition_record(replay, *, inputs, policy):
    assembly = replay["assembly"]
    bracket = assembly["bracket"]
    issued = bracket is not None
    if issued:
        require(assembly["inputs_compatible"] is True and assembly["state_closure"]["closure"] is True,
                "issued bracket lacks complete closure")
        require(type(bracket) is list and len(bracket) == 2 and all(type(v) is int for v in bracket) and
                0 < bracket[0] <= bracket[1] <= policy["settings"]["selection"]["horizon"], "invalid issued bracket")
    return {"schema": SCHEMA, "inputs": inputs, "status": "certificate_issued" if issued else "scientific_abstention",
        "event_certificate_issued": issued, "bracket": bracket, "reason": assembly.get("reason"),
        "target_correct": policy["settings"]["selection"]["target_correct"],
        "persistence": policy["settings"]["selection"]["persistence"], "identity": replay["identity"],
        "numerical_replay_sha256": sha(replay), "future_observation_authorized": True,
        "authorization_scope": "one new observation from this pinned checkpoint; never retry or resume",
        "numerical_scope": "source-authenticated outward neural producers and local arithmetic; ideal Gaussian event",
        "neural_kernels_independently_reexecuted": False, "preceding_float64_training_program_certified": False}


def disposition_phase(root, *, construction_root, construction_manifest, selection_root, selection_manifest,
                       policy, selection_runtime, construction_runtime, runtime_record, selection_bindings,
                       guard, maximum_bytes, sync=sync_directory, allow_fixture=False):
    validate_policy(policy, allow_fixture=allow_fixture)
    inputs = phase_input("disposition", policy, selection_manifest, construction_manifest)
    writer = PhaseWriter(root, role="audit", bindings=phase_bindings(selection_bindings, runtime_record, inputs),
                         guard=guard, sync=sync)
    writer.record("disposition_input.json", inputs)
    writer.record("runtime.json", runtime_record)
    replay = replay_construction(construction_root, construction_manifest=construction_manifest,
        selection_root=selection_root, selection_manifest=selection_manifest, policy=policy,
        selection_runtime=selection_runtime, construction_runtime=construction_runtime,
        selection_bindings=selection_bindings, guard=guard, allow_fixture=allow_fixture)
    writer.record("numerical_replay.json", replay)
    disposition = disposition_record(replay, inputs=inputs, policy=policy)
    record = writer.record("disposition.json", disposition)
    seal = writer.seal({"disposition_sha256": record["sha256"], "status": disposition["status"]}, maximum_bytes=maximum_bytes)
    return {**seal, "disposition_sha256": record["sha256"], "disposition": disposition}


def audit_disposition(root, *, disposition_manifest, construction_root, construction_manifest,
                      selection_root, selection_manifest, policy, selection_runtime, construction_runtime,
                      disposition_runtime, selection_bindings, guard, allow_fixture=False):
    """Recompute the numerical decision; do not admit a saved 'issued' flag."""
    validate_policy(policy, allow_fixture=allow_fixture)
    inputs = phase_input("disposition", policy, selection_manifest, construction_manifest)
    reader = PhaseReader(root, expected_manifest_sha256=disposition_manifest, expected_role="audit",
        expected_bindings=phase_bindings(selection_bindings, disposition_runtime, inputs), guard=guard)
    require(set(reader.files) == {"method.json", "disposition_input.json", "runtime.json", "numerical_replay.json", "disposition.json"},
            "disposition file population differs")
    same(reader.record("disposition_input.json"), inputs, "disposition input differs")
    same(reader.record("runtime.json"), disposition_runtime, "disposition runtime differs")
    replay = replay_construction(construction_root, construction_manifest=construction_manifest,
        selection_root=selection_root, selection_manifest=selection_manifest, policy=policy,
        selection_runtime=selection_runtime, construction_runtime=construction_runtime,
        selection_bindings=selection_bindings, guard=guard, allow_fixture=allow_fixture)
    same(reader.record("numerical_replay.json"), replay, "disposition numerical replay differs")
    disposition = disposition_record(replay, inputs=inputs, policy=policy)
    same(reader.record("disposition.json"), disposition, "disposition decision differs")
    same(reader.manifest["payload"], {"disposition_sha256": reader.files["disposition.json"]["sha256"],
        "status": disposition["status"]}, "disposition terminal differs")
    return {"status": "complete_disposition_recomputed", "manifest_sha256": reader.identity,
            "disposition_sha256": reader.files["disposition.json"]["sha256"], "disposition": disposition,
            "numerical_replay": replay, "future_observation_executed": False}
