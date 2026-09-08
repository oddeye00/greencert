"""Public-seal-only CPU ARM entry point for one bounded final-scale ladder.

Enter through the externally pinned source bootstrap with Python -I -B.
There is no fixture, resume, alternative-seed or arbitrary-phase-directory
option. A controller reserves one root; each rung/phase has one fixed path.
"""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import sys
import time

import torch

from final_scale_artifacts_v1 import PhaseReader, PhaseWriter
from final_scale_construction_v1 import construct_phase, disposition_phase, phase_bindings, phase_input
from final_scale_controller_v1 import run_ladder
from final_scale_engine_v1 import Engine, production_config
from final_scale_observation_v1 import audit_observation, observation_input, observe_phase
from final_scale_runtime_v1 import (MIB, ResourceLimit, install_address_space_limit,
    linux_memory_snapshot, phase_envelope, run_once)
from final_scale_seal_v1 import (Budget, ENVIRONMENT, GROUPS, admit_protocol, digest, read_pinned,
    require, runtime_snapshot, same, sha, verify_publication)
from final_scale_selection_v1 import audit_phase, registered_policy, run_phase
from prospective_ledger_v1 import encode, is_hash, publish_new, sync_directory


SCHEMA = "final_scale_public_execution_v1"


def boot_identity():
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def phase_runtime(protocol, runtime, phase):
    group = GROUPS[phase]
    execution = protocol["scientific_settings"]["execution"]
    threads = execution["kernel_threads"] if group == "construction" else execution["training_threads"]
    return {"identity": runtime, "phase_group": group, "threads": threads,
        "interop_threads": execution["interop_threads"], "deterministic_algorithms": True,
        "mha_fastpath": False, "device": "cpu", "worker_environment": ENVIRONMENT}


def configure_worker(record):
    require(all(os.environ.get(key) == value for key, value in ENVIRONMENT.items()), "worker thread environment differs")
    torch.set_num_threads(record["threads"])
    torch.set_num_interop_threads(record["interop_threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.set_default_device("cpu")
    require(torch.get_num_threads() == record["threads"] and
            torch.get_num_interop_threads() == record["interop_threads"] and
            str(torch.get_default_dtype()) == "torch.float32", "registered Torch initialization/runtime differs")


def selection_bindings(protocol_pin, protocol, runtime, policy):
    return {"protocol_sha256": protocol_pin, "source_manifest_sha256": protocol["source_manifest_sha256"],
        "runtime_manifest_sha256": sha(runtime), "phase_input_sha256": sha(policy)}


def case_arguments(run_root, rung, prior, policy, runtimes, bindings):
    base = Path(run_root) / f"rung_{rung}"
    return {"selection_root": base / "selection", "selection_manifest": prior["selection"]["manifest_sha256"],
        "construction_root": base / "construction", "construction_manifest": prior["construction"]["manifest_sha256"],
        "disposition_root": base / "disposition", "disposition_manifest": prior["disposition"]["manifest_sha256"],
        "policy": policy, "selection_runtime": runtimes["selection"], "construction_runtime": runtimes["construction"],
        "disposition_runtime": runtimes["disposition"], "selection_bindings": bindings}


def expected_phase_binding(run_root, rung, phase, prior, policy, runtimes, bindings):
    if phase == "selection":
        return bindings
    if phase in ("construction", "disposition"):
        inputs = phase_input(phase, policy, prior["selection"]["manifest_sha256"],
            prior["construction"]["manifest_sha256"] if phase == "disposition" else None)
    else:
        inputs = observation_input(case_arguments(run_root, rung, prior, policy, runtimes, bindings))
        if phase == "audit":
            inputs = {"schema": SCHEMA, "observation_input": inputs,
                      "observation_manifest_sha256": prior["observation"]["manifest_sha256"]}
    return phase_bindings(bindings, runtimes[phase], inputs)


def close_selection_arrays(audit):
    for key in ("parameter", "unscaled_velocity", "reference"):
        if audit[key] is not None:
            audit[key]._mmap.close()
    for pair in audit["populations"].values():
        for value in pair:
            value._mmap.close()


def execute_scientific_phase(root, *, phase, rung, prior, protocol, protocol_pin, runtime, guard):
    """Production adapter only: no mocked gate or fixture parameter is accepted."""
    settings = protocol["scientific_settings"]
    policy = registered_policy(settings, rung)
    runtimes = {name: phase_runtime(protocol, runtime, name) for name in GROUPS}
    bindings = selection_bindings(protocol_pin, protocol, runtimes["selection"], policy)
    base, target = Path(root) / f"rung_{rung}", Path(root) / f"rung_{rung}" / phase
    cap = protocol["resources"]["maximum_complete_phase_artifact_mib"]*MIB
    if phase == "selection":
        def factory(record):
            return Engine(production_config(settings, rung), expected_parameters=settings["rungs"][rung]["parameters"],
                expected_counts={role: settings["data"][role+"_count"] for role in ("training", "trigger", "certification")},
                maximum_updates=settings["selection"]["maximum_updates"], guard=guard, record=record)
        result = run_phase(target, policy=policy, runtime_record=runtimes[phase], bindings=bindings,
            engine_factory=factory, guard=guard, maximum_bytes=cap)
        audited = audit_phase(target, expected_manifest_sha256=result["manifest_sha256"], policy=policy,
            runtime_record=runtimes[phase], bindings=bindings, guard=guard)
        close_selection_arrays(audited)
        return result["manifest_sha256"]
    common = {"selection_root": base / "selection", "selection_manifest": prior["selection"]["manifest_sha256"],
        "policy": policy, "selection_runtime": runtimes["selection"], "selection_bindings": bindings, "guard": guard}
    if phase == "construction":
        result = construct_phase(target, **common, runtime_record=runtimes[phase], maximum_bytes=cap)
    elif phase == "disposition":
        result = disposition_phase(target, **common, construction_root=base / "construction",
            construction_manifest=prior["construction"]["manifest_sha256"], construction_runtime=runtimes["construction"],
            runtime_record=runtimes[phase], maximum_bytes=cap)
    else:
        case = case_arguments(root, rung, prior, policy, runtimes, bindings)
        if phase == "observation":
            result = observe_phase(target, case=case, runtime_record=runtimes[phase], guard=guard, maximum_bytes=cap)
        else:
            audit_bindings = expected_phase_binding(root, rung, phase, prior, policy, runtimes, bindings)
            writer = PhaseWriter(target, role="audit", bindings=audit_bindings, guard=guard)
            writer.record("runtime.json", runtimes[phase])
            report = audit_observation(base / "observation", observation_manifest=prior["observation"]["manifest_sha256"],
                case=case, observation_runtime=runtimes["observation"], guard=guard)
            ref = writer.record("report.json", report)
            result = writer.seal({"report_sha256": ref["sha256"]}, maximum_bytes=cap)
    return result["manifest_sha256"]


def authenticated_phase_summary(root, *, phase, rung, prior, pin, protocol, protocol_pin, runtime, guard):
    policy = registered_policy(protocol["scientific_settings"], rung)
    runtimes = {name: phase_runtime(protocol, runtime, name) for name in GROUPS}
    bindings = selection_bindings(protocol_pin, protocol, runtimes["selection"], policy)
    expected = expected_phase_binding(root, rung, phase, prior, policy, runtimes, bindings)
    role = "audit" if phase in ("disposition", "audit") else phase
    reader = PhaseReader(Path(root) / f"rung_{rung}" / phase, expected_manifest_sha256=pin,
                         expected_role=role, expected_bindings=expected, guard=guard)
    if phase == "selection":
        selected = reader.record("selection.json")["result"]
        require(reader.manifest["payload"]["selection_record_sha256"] == reader.files["selection.json"]["sha256"],
                "selection terminal record differs")
        return {"status": selected["status"], "manifest_sha256": pin, "selection": selected}
    if phase == "construction":
        require(reader.manifest["payload"]["event_certificate_issued"] is False and
                reader.manifest["payload"]["future_observation_authorized"] is False, "construction cannot issue or observe")
        return {"status": reader.manifest["payload"]["status"], "manifest_sha256": pin}
    if phase == "disposition":
        result = reader.record("disposition.json")
        require(reader.manifest["payload"]["disposition_sha256"] == reader.files["disposition.json"]["sha256"],
                "disposition terminal record differs")
        return {"status": result["status"], "manifest_sha256": pin, "certificate_issued": result["event_certificate_issued"],
                "bracket": result["bracket"]}
    if phase == "observation":
        summary = reader.record("observation_summary.json")
        require(reader.manifest["payload"]["summary_sha256"] == reader.files["observation_summary.json"]["sha256"],
                "observer terminal record differs")
        return {"status": summary["status"], "manifest_sha256": pin, "summary": summary}
    require(set(reader.files) == {"method.json", "runtime.json", "report.json"}, "output audit file population differs")
    same(reader.record("runtime.json"), runtimes[phase], "output audit runtime differs")
    require(reader.manifest["payload"]["report_sha256"] == reader.files["report.json"]["sha256"], "output audit report differs")
    report = reader.record("report.json")
    return {"status": report["status"], "manifest_sha256": pin, "summary": report["summary"]}


def worker(args, protocol, runtime):
    request = read_pinned(args.request, args.request_sha256)
    root = Path(args.run_root).resolve(strict=True)
    require(set(request) == {"schema", "protocol_sha256", "source_manifest_sha256", "controller_started_sha256",
        "boot_id", "rung", "phase", "prior", "deadlines"} and request["schema"] == SCHEMA and
        request["protocol_sha256"] == args.protocol_sha256 and
        request["source_manifest_sha256"] == protocol["source_manifest_sha256"], "worker request binding differs")
    started = read_pinned(root / "started.json", request["controller_started_sha256"])
    require(started["protocol_sha256"] == args.protocol_sha256 and request["boot_id"] == started["boot_id"] == boot_identity(),
            "controller identity or system boot differs")
    rung, phase = request["rung"], request["phase"]
    require(type(rung) is int and 0 <= rung < len(protocol["scientific_settings"]["rungs"]) and phase in GROUPS,
            "unregistered worker phase")
    expected_prior = list(GROUPS)[:list(GROUPS).index(phase)]
    require(set(request["prior"]) == set(expected_prior), "worker predecessor phases incomplete")
    require(Path(args.request).resolve(strict=True) == root / f"rung_{rung}" / "requests" / (phase+".json"),
            "worker request path is not canonical")
    deadlines = request["deadlines"]
    execution = protocol["scientific_settings"]["execution"]
    require(deadlines["overall_started"] == started["monotonic_started"] and
        deadlines["group_started"] >= deadlines["overall_started"] and
        deadlines["overall_deadline"] == started["monotonic_started"]+execution["overall_seconds"] and
        deadlines["group_deadline"] == deadlines["group_started"]+execution[GROUPS[phase]+"_seconds_per_rung"] and
        deadlines["deadline"] == min(deadlines["overall_deadline"], deadlines["group_deadline"]), "worker deadline differs")
    limits = install_address_space_limit(protocol["resources"]["worker_address_space_mib"])
    phase_record = phase_runtime(protocol, runtime, phase)
    configure_worker(phase_record)
    envelope = phase_envelope(protocol["scientific_settings"], GROUPS[phase])
    last_sample, last_notice = -1.0, -1.0
    def guard():
        nonlocal last_sample, last_notice
        now = time.monotonic()
        if now >= deadlines["deadline"]:
            raise ResourceLimit("worker cumulative deadline reached")
        if now-last_sample >= .5:
            available, rss = linux_memory_snapshot(os.getpid())
            envelope.check(elapsed=now-deadlines["group_started"], available_memory=available,
                           free_disk=shutil.disk_usage(root).free, rss=rss)
            last_sample = now
        if now-last_notice >= 60:
            print(json.dumps({"phase": phase, "rung": rung, "remaining_seconds": deadlines["deadline"]-now}), flush=True)
            last_notice = now
    guard()
    output = root / f"rung_{rung}" / "worker_results" / (phase+".json")
    require(not output.exists() and not (root / f"rung_{rung}" / phase).exists(), "worker phase already reserved or completed")
    manifest = execute_scientific_phase(root, phase=phase, rung=rung, prior=request["prior"], protocol=protocol,
                                       protocol_pin=args.protocol_sha256, runtime=runtime, guard=guard)
    guard()
    publish_new(output, {"request_sha256": args.request_sha256, "phase": phase, "rung": rung,
        "manifest_sha256": manifest, "installed_worker_limit": limits})
    print(json.dumps({"phase": phase, "rung": rung, "status": "phase_artifact_sealed"}), flush=True)


def controller(args, protocol, runtime):
    publication = read_pinned(args.publication, args.publication_sha256)
    public_check = verify_publication(publication, args.protocol_sha256)
    root = Path(args.run_root)
    require(root.is_absolute() and root.parent.is_dir() and not root.parent.is_symlink(), "absolute existing run parent required")
    sync_directory(root.parent)
    root.mkdir(exist_ok=False)
    sync_directory(root.parent)
    root = root.resolve(strict=True)
    budget = Budget(protocol["scientific_settings"])
    started = {"schema": SCHEMA, "protocol_sha256": args.protocol_sha256, "source_manifest_sha256": protocol["source_manifest_sha256"],
        "runtime_sha256": protocol["runtime_sha256"], "publication_receipt_sha256": args.publication_sha256,
        "public_check": public_check, "boot_id": boot_identity(), "monotonic_started": budget.started,
        "automatic_retry_or_resume": False}
    started_pin = publish_new(root / "started.json", started)
    (root / "records").mkdir(exist_ok=False)
    sync_directory(root)
    sequence, head = 0, None
    def record(kind, payload):
        nonlocal sequence, head
        item = {"sequence": sequence, "previous_sha256": head, "kind": kind, "payload": payload}
        head = publish_new(root / "records" / f"record_{sequence:06d}.json", item)
        sequence += 1
        print(json.dumps({"record": sequence-1, "kind": kind, "rung": payload.get("rung"), "phase": payload.get("phase")}), flush=True)
    def execute(rung, phase, prior):
        base = root / f"rung_{rung}"
        if phase == "selection":
            base.mkdir(exist_ok=False)
            for name in ("requests", "supervision", "worker_results"):
                (base / name).mkdir(exist_ok=False)
            sync_directory(base)
            sync_directory(root)
        deadlines = budget.deadlines(rung, phase)
        request = {"schema": SCHEMA, "protocol_sha256": args.protocol_sha256,
            "source_manifest_sha256": protocol["source_manifest_sha256"], "controller_started_sha256": started_pin,
            "boot_id": started["boot_id"], "rung": rung, "phase": phase, "prior": prior, "deadlines": deadlines}
        request_path = base / "requests" / (phase+".json")
        request_pin = publish_new(request_path, request)
        command = [sys.executable, "-I", "-B", str(Path(args.source_root) / "pinned_source_bundle_v1.py"),
            "--source-root", str(args.source_root), "--manifest", str(args.source_manifest),
            "--manifest-sha256", protocol["source_manifest_sha256"], "--", "worker",
            "--protocol", str(args.protocol), "--protocol-sha256", args.protocol_sha256, "--runtime", str(args.runtime),
            "--source-root", str(args.source_root), "--source-manifest", str(args.source_manifest),
            "--run-root", str(root), "--request", str(request_path), "--request-sha256", request_pin]
        envelope = replace(phase_envelope(protocol["scientific_settings"], GROUPS[phase]), seconds=deadlines["remaining_seconds"])
        def guard():
            budget.guard()
            if time.monotonic() >= deadlines["deadline"]:
                raise ResourceLimit("controller cumulative group deadline reached")
        require(digest(Path(args.source_root) / "pinned_source_bundle_v1.py") == protocol["bootstrap_sha256"] and
                digest(args.source_manifest) == protocol["source_manifest_sha256"], "child bootstrap or manifest changed")
        outcome = run_once(base / "supervision" / phase, command, working_directory=root, envelope=envelope,
            bindings={"protocol_sha256": args.protocol_sha256, "source_manifest_sha256": protocol["source_manifest_sha256"],
                      "phase_input_sha256": request_pin}, environment={**os.environ, **ENVIRONMENT}, guard=guard)
        require(outcome["status"] == "worker_completed" and outcome["returncode"] == 0, "worker execution failed")
        output_path = base / "worker_results" / (phase+".json")
        output = read_pinned(output_path, digest(output_path))
        require(output["request_sha256"] == request_pin and output["rung"] == rung and output["phase"] == phase and
                is_hash(output["manifest_sha256"]), "worker response binding differs")
        return authenticated_phase_summary(root, phase=phase, rung=rung, prior=prior, pin=output["manifest_sha256"],
            protocol=protocol, protocol_pin=args.protocol_sha256, runtime=runtime, guard=guard)
    result = run_ladder(protocol["scientific_settings"], execute=execute, record=record, guard=budget.guard)
    publish_new(root / "terminal.json", {"result": result, "record_count": sequence, "record_head_sha256": head})
    return 0 if result["status"] == "ladder_completed" else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "run", "worker"))
    for name in ("protocol", "runtime", "source-root", "source-manifest"):
        parser.add_argument("--"+name, required=True, type=Path)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--publication", type=Path)
    parser.add_argument("--publication-sha256")
    parser.add_argument("--request", type=Path)
    parser.add_argument("--request-sha256")
    args = parser.parse_args()
    for name in ("protocol", "runtime", "source_root", "source_manifest"):
        setattr(args, name, getattr(args, name).resolve(strict=True))
    protocol, runtime = admit_protocol(protocol_path=args.protocol, protocol_sha256=args.protocol_sha256,
        runtime_path=args.runtime, source_root=args.source_root, source_manifest_path=args.source_manifest)
    if args.mode == "check":
        print(json.dumps({"status": "public_source_protocol_runtime_admitted", "model_instantiated": False,
                          "future_outcome_accessed": False}), flush=True)
        return 0
    require(args.run_root is not None, "run root required")
    if args.mode == "run":
        require(args.publication is not None and is_hash(args.publication_sha256), "public publication receipt required")
        return controller(args, protocol, runtime)
    require(args.request is not None and is_hash(args.request_sha256), "pinned worker request required")
    worker(args, protocol, runtime)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
