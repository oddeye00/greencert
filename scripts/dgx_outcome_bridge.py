"""One-shot, hardware-amended observation of an already certified window.

This wrapper authenticates a pre-outcome amendment. It uses the unchanged
registered NeuralObserver and durable logger, and never constructs a new
reference, derivative envelope, Green bound, or certificate.
"""
import argparse
from dataclasses import asdict
import hashlib
import importlib
import importlib.util
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_pinned(path, expected):
    from verified_artifact_io import unique_object, invalid_constant, finite_json_float
    raw = Path(path).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected, "pinned amendment input changed")
    return json.loads(raw, object_pairs_hook=unique_object,
                      parse_constant=invalid_constant, parse_float=finite_json_float)


def runtime_record():
    import torch
    return {
        "runtime": {"python": platform.python_version(), "numpy": version("numpy"),
                    "torch": version("torch"), "platform": platform.platform(),
                    "machine": platform.machine()},
        "torch_build": str(torch.__version__), "torch_cuda_build": torch.version.cuda,
        "execution_device": "cpu", "candidate_model_constructed": False,
        "future_outcome_access": False,
    }


def validate_runtime_only(original, effective, amendment, actual):
    require(amendment["schema"] == "dgx_hardware_only_outcome_amendment_v1", "wrong amendment schema")
    require(amendment["allowed_changed_protocol_fields"] == ["runtime"], "unregistered amendment scope")
    require(set(original) == set(effective), "protocol fields added or removed")
    require({k:v for k,v in original.items() if k != "runtime"} ==
            {k:v for k,v in effective.items() if k != "runtime"}, "non-runtime protocol change")
    require(effective["runtime"] == actual["runtime"] == amendment["runtime_capture"]["runtime"],
            "execution runtime differs from amendment")
    for field in ("torch_build", "torch_cuda_build", "execution_device"):
        require(actual[field] == amendment["runtime_capture"][field], "execution build/device changed")
    require(actual["execution_device"] == "cpu", "GPU optimizer execution not registered")
    require(amendment["future_outcome_access"] is False and
            amendment["automatic_retry_or_resume"] is False and
            amendment["user_authorized_hardware_amendment"] is True, "missing sealed authority")
    require(original["future_outcome_access"] is False and
            original["automatic_retry_or_resume"] is False and
            original["automatic_outcome_join"] is False, "original observer scope changed")


def bundle_records(bundle, amendment_sha):
    bundle = Path(bundle).resolve(strict=True)
    amendment = read_pinned(bundle/"amendment.json", amendment_sha)
    original = read_pinned(bundle/"original_protocol.json", amendment["original_protocol_sha256"])
    effective = read_pinned(bundle/"effective_protocol.json", amendment["effective_protocol_sha256"])
    for name, expected in amendment["original_observer_sources"].items():
        require(Path(name).name == name and name.endswith(".py"), "invalid frozen source name")
        require(digest(bundle/"frozen_sources"/name) == expected, "frozen observer source changed")
    require(original["sources"] == amendment["original_observer_sources"], "source population changed")
    for name, expected in amendment["bridge_sources"].items():
        require(Path(name).name == name and name.endswith(".py"), "invalid bridge source name")
        require(digest(Path(__file__).resolve().parent/name) == expected, "bridge implementation changed")
    require(digest(bundle/"DGX_OUTCOME_AMENDMENT.md") == amendment["document_sha256"], "amendment document changed")
    return amendment, original, effective


def checked_module(name, original):
    spec = importlib.util.find_spec(name)
    require(spec is not None and spec.origin is not None, "missing frozen implementation")
    require(digest(spec.origin) == original["sources"][name+".py"], "loaded observer implementation changed")
    module = importlib.import_module(name)
    require(digest(module.__file__) == original["sources"][name+".py"], "loaded observer source mismatch")
    return module


def authenticate(args):
    from replay_recorded_window import load_context
    amendment, original, effective = bundle_records(args.bundle, args.amendment_sha256)
    actual = runtime_record()
    validate_runtime_only(original, effective, amendment, actual)
    manifest = args.root/"recorded_manifest.json"
    reader, request, terminal, audit, _, _, _, barrier = load_context(
        args.root, manifest, amendment["recorded_manifest_sha256"])
    replay = read_pinned(args.bundle/"completed_replay.json", amendment["completed_replay_sha256"])
    require(replay["status"] == "PASS" and replay["complete_recorded_graph_rehashed"] is True and
            replay["future_outcome_access"] is False and
            replay["manifest_sha256"] == amendment["recorded_manifest_sha256"] and
            replay["assembly"] == audit["result"]["assembly"], "completed replay binding differs")
    require(terminal["identity"] == original["identity"] == asdict(request.identity), "observer identity differs")
    require(terminal["decision"]["bracket"] == amendment["issued_bracket"], "issued bracket changed")
    candidate, _ = reader.read_json("results/larger_transformer_third_pass/candidate.json",
                                   original["identity"]["candidate_sha256"])
    require(candidate["config"] == original["config"] and candidate["anchor"] == original["anchor"] and
            candidate["predicted_offset"] == original["predicted_offset"] and
            candidate["anchor_sha256"] == original["checkpoint_sha256"], "observer checkpoint/config differs")
    for field, value in (("horizon", request.identity.horizon), ("target", request.target),
                         ("persistence", request.persistence), ("evaluation_count", len(request.evaluation_examples)),
                         ("training_examples", request.training_examples),
                         ("evaluation_examples", request.evaluation_examples)):
        require(original[field] == value, "observer event or population differs")
    original_module = checked_module("continue_candidate_sealed_outcome", original)
    logger = checked_module("registered_outcome_continuation", original)
    checked_module("verified_artifact_io", original)
    checked_module("anchor_response_evidence", original)
    # Check the neural code without importing it or constructing the candidate.
    spec = importlib.util.find_spec("transformer_hvp_grokking")
    require(spec is not None and digest(spec.origin) == original["sources"]["transformer_hvp_grokking.py"],
            "neural implementation differs")
    logger.unopened(reader, original_module.RUN, original_module.OUTCOME)
    for name, expected in reader.recorded.items():
        reader.check_blob(name, expected)
    barrier()
    gate = {"record_authenticated": True, "outcome_join_authorized": True,
            "future_outcome_access": False, "decision": terminal["decision"]["decision"],
            "bracket": terminal["decision"]["bracket"],
            "disposition_sha256": amendment["original_terminal_sha256"],
            "hardware_amendment_sha256": args.amendment_sha256,
            "authorization_basis": "User-approved, publicly sealed DGX runtime-only amendment",
            "original_terminal_record_changed": False}
    # Bind the terminal digest independently of its inner fields.
    manifest_record = json.loads(manifest.read_text())
    require(manifest_record["original_terminal_sha256"] == gate["disposition_sha256"],
            "terminal digest differs from amendment")
    return reader, original_module, logger, amendment, effective, gate, barrier


def run(args):
    reader, original_module, logger, amendment, effective, gate, barrier = authenticate(args)
    if args.mode == "check":
        return {"status": "authenticated_for_one_dgx_observation", "amendment_sha256": args.amendment_sha256,
                "model_constructed": False, "future_outcome_access": False, "bracket": gate["bracket"]}
    require(args.delegation_sha256 is not None, "source-host delegation required before execution")
    delegation = read_pinned(args.bundle/"source_host_delegation.json", args.delegation_sha256)
    require(delegation["schema"] == "dgx_outcome_source_host_delegation_v1" and
            delegation["amendment_sha256"] == args.amendment_sha256 and
            delegation["original_protocol_sha256"] == amendment["original_protocol_sha256"] and
            delegation["source_host_original_run_reserved"] is True and
            delegation["automatic_retry_or_resume"] is False, "foreign or incomplete delegation")
    require(original_module.memory_available_mib() >= effective["minimum_free_mib"], "memory admission failed")
    observed = dict(reader.observed)

    def recheck():
        bundle_records(args.bundle, args.amendment_sha256)
        validate_runtime_only(read_pinned(args.bundle/"original_protocol.json", amendment["original_protocol_sha256"]),
                              effective, amendment, runtime_record())
        require(digest(args.bundle/"source_host_delegation.json") == args.delegation_sha256, "delegation changed")
        for name, expected in observed.items():
            reader.check_blob(name, expected)
        barrier()
        require(not reader.resolve(original_module.OUTCOME).exists(), "outcome appeared before publication")

    def prepare():
        engine = original_module.NeuralObserver(reader, effective)
        require(engine.runtime["device"] == "cpu" and engine.runtime["threads"] == effective["config"]["threads"] and
                engine.runtime["interop_threads"] == effective["torch_interop_threads"] and
                engine.runtime["mha_fastpath"] == effective["torch_mha_fastpath"] and
                engine.runtime["deterministic_algorithms"] == effective["deterministic_algorithms"],
                "registered execution settings changed")
        checked_module("transformer_hvp_grokking", read_pinned(
            args.bundle/"original_protocol.json", amendment["original_protocol_sha256"]))
        engine.runtime["hardware_amendment_sha256"] = args.amendment_sha256
        engine.runtime["torch_build"] = str(engine.torch.__version__)
        return engine

    result = logger.execute_once(reader, gate=gate, protocol=effective,
        protocol_sha256=amendment["effective_protocol_sha256"],
        run_folder=original_module.RUN, outcome_path=original_module.OUTCOME,
        prepare=prepare, recheck=recheck,
        progress=lambda j,n: print(json.dumps({"completed_observations": j+1,
                                             "outcome_values_not_revealed": True}), flush=True))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("capture-runtime", "check", "execute"), required=True)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--amendment-sha256")
    parser.add_argument("--delegation-sha256")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.mode == "capture-runtime":
        result = runtime_record()
    else:
        require(args.root is not None and args.bundle is not None and args.amendment_sha256 is not None,
                "root, bundle and external amendment digest required")
        result = run(args)
    if args.report:
        from registered_outcome_continuation import publish_json
        publish_json(args.report, result)
    print(json.dumps(result, indent=2))
