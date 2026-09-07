"""Explain, without executing it, the retained v1 observation-context loader.

This is a post-outcome compatibility audit. It does not amend the frozen
observer, turn v1 bytes into v3 bytes, or reconstruct historical imports.
"""
import sys
if __name__ == "__main__" and not sys.flags.isolated:
    raise SystemExit("Use python -I -B for this source-only audit")

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile

LEGACY_SHA = "c799935f80b06e6bfd86e71031c0b4b8bdf74264d570aea83319ef1e39c0a2de"
PORTABLE_SHA = "661e34c23358b043ee59b966613ddefe7a7d75ff3c40d4f810a5ce24dae02389"
CHECKER_SHA = "01348e7c130a218fef467e79d4fe54899ca1f4c33aa727a54e483bee3e897129"


def require(value,message):
    if not value:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def archive_source(path,expected):
    require(sha(path.read_bytes()) == expected,"source archive checksum mismatch")
    with zipfile.ZipFile(path) as archive:
        return archive.read("scripts/replay_recorded_window.py")


def audit(args):
    checker_path = Path(__file__).with_name("audit_dgx_observer_source_chain.py")
    require(sha(checker_path.read_bytes()) == CHECKER_SHA,"post-outcome checker changed")
    spec = importlib.util.spec_from_file_location("source_checker",checker_path)
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)  # Authenticated new checker, not observer code.
    old = archive_source(args.legacy_archive,LEGACY_SHA)
    new = archive_source(args.portable_archive,PORTABLE_SHA)
    old_tree, new_tree = ast.parse(old),ast.parse(new)
    old_nodes = [ast.dump(n,include_attributes=False) for n in old_tree.body]
    new_nodes = [ast.dump(n,include_attributes=False) for n in new_tree.body]
    require(len(old_nodes) == len(new_nodes),"loader population changed")
    differences = [j for j,(a,b) in enumerate(zip(old_nodes,new_nodes)) if a != b]
    require(len(differences) == 1,"more than the known import differs")
    a,b = old_tree.body[differences[0]],new_tree.body[differences[0]]
    require(isinstance(a,ast.ImportFrom) and isinstance(b,ast.ImportFrom) and a.module == "reference_margin_evidence" and
            b.module == "portable_reference_margin_evidence" and a.level == b.level == 0 and
            [ast.dump(n) for n in a.names] == [ast.dump(n) for n in b.names],"unrecognized loader import change")
    changed_symbols = [n.asname or n.name for n in a.names]
    context = next(n for n in old_tree.body if isinstance(n,ast.FunctionDef) and n.name == "load_context")
    used_names = {n.id for n in ast.walk(context) if isinstance(n,ast.Name)}
    require(not (set(changed_symbols) & used_names),"changed margin function is referenced by load_context")
    amendment = checker.pinned_json(args.bundle/"amendment.json",checker.AMENDMENT_SHA256)
    original = checker.pinned_json(args.bundle/"original_protocol.json",amendment["original_protocol_sha256"])
    replay = checker.pinned_json(args.bundle/"completed_replay.json",amendment["completed_replay_sha256"])
    require(replay["status"] == "PASS" and replay["identity"] == original["identity"] and
            replay["manifest_sha256"] == amendment["recorded_manifest_sha256"],"replay lineage differs")
    require(replay["sources"]["replay_recorded_window.py"] == sha(new),"saved replay is not the portable loader")
    groups = {"original_observer":amendment["original_observer_sources"],
        "bridge":amendment["bridge_sources"],"completed_replay":dict(replay["sources"])}
    # This explicit, reported substitution is for the compatibility audit only.
    # It is not written into the original replay, amendment, or source chain.
    groups["completed_replay"]["replay_recorded_window.py"] = sha(old)
    pins,origins = checker.merge_pins(groups)
    resolved = checker.inspect_sources(pins,origins,args.bridge_dir,args.source_dir)
    public_sources = {name:{"sha256":r["sha256"],"search_index":r["search_index"]}
                      for name,r in resolved["resolved_sources"].items()}
    return {"schema":"post_outcome_legacy_loader_compatibility_v1", "status":"KNOWN_VARIANT_AUTHENTICATED",
        "strict_completed_replay_source_equality":False,
        "legacy_archive_sha256":LEGACY_SHA,"portable_archive_sha256":PORTABLE_SHA,
        "retained_loader_sha256":sha(old),"completed_replay_loader_sha256":sha(new),
        "single_difference":"reference-margin import uses legacy module instead of portable module",
        "all_function_and_class_asts_equal":True,"load_context_references_changed_margin_symbols":False,
        "pinned_sources_checked_with_explicit_legacy_loader":resolved["pinned_source_count"],
        "bridge_static_closure_count":resolved["bridge_static_closure_count"],
        "bridge_static_internal_dependency_gaps":resolved["bridge_static_internal_dependency_gaps"],
        "other_pinned_source_dependency_gaps":resolved["other_pinned_source_dependency_gaps"],
        "resolved_source_hashes":public_sources,"search_directories":"entry directory; original evidence scripts; portable toolkit scripts; frozen observer scripts",
        "original_payloads_modified":False,"recorded_code_imported_or_executed":False,"observer_reexecuted":False,
        "historical_preexecution_guard_repaired":False,"historical_imports_attested":False,
        "audit_source_sha256":sha(Path(__file__).read_bytes()),"checker_sha256":CHECKER_SHA,
        "scope":"Current source compatibility: all other pinned source bytes agree; the retained loader's sole AST difference changes margin routines not referenced by load_context. This does not prove historical imports or validate the old loader for full portable numerical replay."}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle",type=Path,required=True)
    p.add_argument("--bridge-dir",type=Path,required=True)
    p.add_argument("--source-dir",type=Path,action="append",required=True)
    p.add_argument("--legacy-archive",type=Path,required=True)
    p.add_argument("--portable-archive",type=Path,required=True)
    p.add_argument("--report",type=Path,required=True)
    a = p.parse_args()
    require(not a.report.exists(),"fresh report required")
    result = audit(a)
    with a.report.open("x",encoding="utf-8",newline="\n") as stream:
        json.dump(result,stream,indent=2,sort_keys=True)
        stream.write("\n")
    print(json.dumps({k:v for k,v in result.items() if k != "resolved_source_hashes"},indent=2))
