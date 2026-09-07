"""Post-outcome filesystem source audit; never import the recorded implementation.

Run with ``python -I -B scripts/audit_dgx_observer_source_chain.py``. Supply
--bridge-dir DGXAM/frozen (the executed script's directory), --bundle containing
amendment.json, and --source-dir in the original PYTHONPATH order:
evidence/scripts, portable-tools-v3/scripts, DGXAM/frozen/frozen_sources.
The entry directory is always searched first; this process's sys.path is NOT
used to resolve evidence. Local mirrors audit those mirrors, not remote files.

All manifest source names must resolve to matching .py bytes. AST traversal
checks ordinary imports (including function bodies) and recognized literal
dynamic imports, without evaluating code. Non-flat packages/native loaders
are refused in the supplied directories. Interpreter, installed dependencies,
import hooks, cached bytecode, and historical sys.modules are outside scope.
Current hash agreement is NOT retroactive proof of pre-execution checks, the
historical search path, execution timing, or one-shot execution.

Only an explicitly supplied, new --report is written. --self-test instead
creates disposable synthetic files and never accesses the outcome or model.
"""
import sys

# Do this before importing even stdlib helpers: PYTHONPATH/script-directory
# shadows must not become executable dependencies of the checker itself.
if __name__ == "__main__" and not sys.flags.isolated:
    raise SystemExit("Use an isolated interpreter: python -I -B <this script> ...")

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile


AMENDMENT_SHA256 = "d8f56172db878c1a0986c6cfcaf384166b0a33a37284497e53514ec7a9d3e97a"
ENTRYPOINT = "dgx_outcome_bridge.py"
EXTERNAL_PACKAGES = frozenset({"numpy", "torch", "flint"})
SOURCE_NAME = re.compile(r"[A-Za-z_][A-Za-z_0-9]*\.py\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class SourceChainError(ValueError):
    """A pinned source, resolution rule, or internal dependency did not agree."""


def require(condition, message):
    if not condition:
        raise SourceChainError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def unique_object(pairs):
    value = {}
    for key, item in pairs:
        require(key not in value, "duplicate JSON key: " + key)
        value[key] = item
    return value


def finite_float(text):
    value = float(text)
    require(math.isfinite(value), "nonfinite JSON number")
    return value


def invalid_constant(text):
    raise SourceChainError("nonfinite JSON constant: " + text)


def pinned_json(path, expected):
    require(isinstance(expected, str) and SHA256.fullmatch(expected), "invalid SHA256 pin")
    raw = path.read_bytes()
    require(digest(raw) == expected, "pinned record changed: " + str(path))
    value = json.loads(raw, object_pairs_hook=unique_object,
                       parse_float=finite_float, parse_constant=invalid_constant)
    require(type(value) is dict, "record must be a JSON object: " + str(path))
    return value


def merge_pins(groups):
    pins, origins = {}, {}
    for group, values in groups.items():
        require(type(values) is dict and values, "empty/invalid source map: " + group)
        for name, sha in values.items():
            require(isinstance(name, str) and SOURCE_NAME.fullmatch(name),
                    "non-flat or unsafe source name: " + repr(name))
            require(isinstance(sha, str) and SHA256.fullmatch(sha), "invalid source pin: " + name)
            require(name not in pins or pins[name] == sha, "conflicting source pins: " + name)
            pins[name] = sha
            origins.setdefault(name, []).append(group)
    return pins, origins


def directory(path):
    resolved = Path(path).resolve(strict=True)
    require(resolved.is_dir(), "not a source directory: " + str(path))
    return resolved


def native_candidates(folder, module):
    # Recognize both DGX/Linux and Windows suffixes, regardless of audit host.
    return sorted(p for p in folder.iterdir() if
                  (p.name == module + ".so" or p.name == module + ".pyd" or
                   (p.name.startswith(module + ".") and p.suffix in (".so", ".pyd"))))


def resolve_source(module, directories):
    """Ordinary flat-file lookup only; never call find_spec or an import hook."""
    require(module.isidentifier(), "invalid module basename: " + repr(module))
    namespaces = []
    for index, folder in enumerate(directories):
        package = folder / module
        if package.exists():
            require(package.is_dir(), "module/package path is not a directory: " + str(package))
            require(package.resolve(strict=True).is_relative_to(folder),
                    "package path escapes source directory: " + str(package))
            if ((package / "__init__.py").exists() or (package / "__init__.pyc").exists()
                    or native_candidates(package, "__init__")):
                raise SourceChainError("unsupported package shadows flat source: " + str(package))
            namespaces.append(str(package))
        require(not native_candidates(folder, module),
                "native module can shadow flat source: " + str(folder / module))
        source = folder / (module + ".py")
        if source.exists() or source.is_symlink():
            resolved = source.resolve(strict=True)
            require(resolved.is_relative_to(folder) and resolved.is_file(),
                    "source is not a regular file within its directory: " + str(source))
            return {"path": str(source), "resolved_path": str(resolved), "search_index": index}
        require(not (folder / (module + ".pyc")).exists(),
                "sourceless bytecode shadows later source: " + str(folder / (module + ".pyc")))
    require(not namespaces, "unsupported namespace-only module: " + module)
    return None


def import_references(raw, name):
    """Return static names plus explicitly unproved dynamic-loader sites."""
    tree = ast.parse(raw, filename=name)
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                aliases[item.asname or item.name.split(".")[0]] = item.name if item.asname else item.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            for item in node.names:
                aliases[item.asname or item.name] = (node.module or "") + "." + item.name

    references, dynamic = [], []
    loaders = {"__import__", "importlib.import_module", "importlib.util.find_spec", "checked_module"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            references.extend({"module": item.name, "line": node.lineno, "kind": "import"} for item in node.names)
        elif isinstance(node, ast.ImportFrom):
            references.append({"module": "." * node.level + (node.module or ""),
                               "line": node.lineno, "kind": "from"})
        elif isinstance(node, ast.Call):
            spelling = ast.unparse(node.func)
            head, dot, tail = spelling.partition(".")
            spelling = aliases.get(head, head) + (dot + tail if dot else "")
            if spelling not in loaders:
                continue
            argument = node.args[0] if node.args else next(
                (k.value for k in node.keywords if k.arg == "name"), None)
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                references.append({"module": argument.value, "line": node.lineno, "kind": spelling})
            else:
                dynamic.append({"source": name, "line": node.lineno, "loader": spelling,
                                "limitation": "nonliteral loader argument; no execution/dataflow proof"})
    return references, dynamic


def inspect_sources(pins, origins, bridge_dir, source_dirs):
    require(source_dirs, "at least one ordered --source-dir is required")
    directories = [directory(bridge_dir)] + [directory(p) for p in source_dirs]
    records, raw_sources = {}, {}
    # Hash ALL maps, including archived tests/legacy tools, before any AST walk.
    for name, expected in sorted(pins.items()):
        chosen = resolve_source(name[:-3], directories)
        require(chosen is not None, "missing pinned source: " + name)
        require(name != ENTRYPOINT or chosen["search_index"] == 0,
                "entrypoint must exist in --bridge-dir")
        raw = Path(chosen["resolved_path"]).read_bytes()
        actual = digest(raw)
        require(actual == expected, "resolved source checksum mismatch: " + name + " at " + chosen["path"]
                + " (expected " + expected + ", actual " + actual + ")")
        raw_sources[name] = raw
        records[name] = {**chosen, "sha256": actual, "pin_groups": origins[name]}

    graph, gaps, external, dynamic = {}, [], set(), []
    for name, raw in sorted(raw_sources.items()):
        graph[name] = set()
        references, calls = import_references(raw, name)
        dynamic.extend(calls)
        for ref in references:
            module = ref["module"]
            head = module.split(".")[0]
            gap = {"source": name, **ref}
            if not head or not head.isidentifier():
                gaps.append({**gap, "reason": "relative/non-flat import unsupported"})
            elif head + ".py" in pins:
                if "." in module:
                    gaps.append({**gap, "reason": "dotted import cannot resolve through a pinned flat module"})
                else:
                    graph[name].add(head + ".py")
            elif head in sys.builtin_module_names:
                external.add(module)
            else:
                try:
                    chosen = resolve_source(head, directories)
                except SourceChainError as error:
                    gaps.append({**gap, "reason": str(error)})
                    continue
                if chosen:
                    gaps.append({**gap, "reason": "internal source has no trusted pin", "path": chosen["path"]})
                elif head in sys.stdlib_module_names or head in EXTERNAL_PACKAGES:
                    external.add(module)
                else:
                    gaps.append({**gap, "reason": "missing internal or undeclared external dependency"})

    require(ENTRYPOINT in graph, "entrypoint has no source pin")
    pending, closure = [ENTRYPOINT], set()
    while pending:
        name = pending.pop()
        if name not in closure:
            closure.add(name)
            pending.extend(sorted(graph[name] - closure))
    entry_gaps = [gap for gap in gaps if gap["source"] in closure]
    require(not entry_gaps, "entrypoint internal dependency gap: " + json.dumps(entry_gaps, sort_keys=True))

    # Detect ordinary changes during inspection, including a new earlier shadow.
    for name, record in records.items():
        chosen = resolve_source(name[:-3], directories)
        require(chosen == {k: record[k] for k in ("path", "resolved_path", "search_index")},
                "source resolution changed during audit: " + name)
        require(digest(Path(record["resolved_path"]).read_bytes()) == pins[name],
                "source bytes changed during audit: " + name)
    return {
        "search_directories_entry_first": [str(p) for p in directories],
        "resolved_sources": records, "pinned_source_count": len(records),
        "bridge_static_closure": sorted(closure), "bridge_static_closure_count": len(closure),
        "bridge_static_internal_dependency_gaps": [],
        "other_pinned_source_dependency_gaps": gaps,
        "all_manifest_sources_static_dependencies_pinned": not gaps,
        "external_imports_not_authenticated": sorted(external),
        "nonliteral_dynamic_loader_sites": dynamic,
    }


def audit(bundle, bridge_dir, source_dirs, *, expected_amendment=AMENDMENT_SHA256):
    bundle = directory(bundle)
    amendment = pinned_json(bundle / "amendment.json", expected_amendment)
    require(amendment["schema"] == "dgx_hardware_only_outcome_amendment_v1", "wrong amendment schema")
    original = pinned_json(bundle / "original_protocol.json", amendment["original_protocol_sha256"])
    replay = pinned_json(bundle / "completed_replay.json", amendment["completed_replay_sha256"])
    require(original["sources"] == amendment["original_observer_sources"], "original observer source population differs")
    require(replay["status"] == "PASS" and replay["manifest_sha256"] == amendment["recorded_manifest_sha256"]
            and replay["identity"] == original["identity"], "completed replay lineage differs")
    groups = {"replay": replay["sources"], "original_observer": amendment["original_observer_sources"],
              "bridge": amendment["bridge_sources"]}
    pins, origins = merge_pins(groups)
    require(ENTRYPOINT in groups["bridge"], "bridge entrypoint pin missing")
    result = inspect_sources(pins, origins, bridge_dir, source_dirs)
    for filename, expected in (("amendment.json", expected_amendment),
                               ("original_protocol.json", amendment["original_protocol_sha256"]),
                               ("completed_replay.json", amendment["completed_replay_sha256"])):
        pinned_json(bundle / filename, expected)
    historically_unchecked = sorted(set(groups["replay"]) - set(groups["original_observer"]) - set(groups["bridge"]))
    return {
        "schema": "post_outcome_dgx_observer_source_chain_audit_v1", "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "amendment_sha256": expected_amendment,
        "completed_replay_sha256": amendment["completed_replay_sha256"],
        "original_protocol_sha256": amendment["original_protocol_sha256"],
        "auditor_sha256": digest(Path(__file__).read_bytes()),
        "audit_python": sys.version.split()[0], "audit_platform": sys.platform,
        "current_code_hash_agreement": True, "recorded_code_imported_or_executed": False,
        "observer_reexecuted": False, "historical_execution_path_independently_authenticated": False,
        "historical_preexecution_source_authentication_reestablished": False,
        "current_code_hash_agreement_is_retroactive_time_proof": False,
        "historical_loader_control_limitation": {
            "replay_sources_without_original_observer_or_bridge_checks": historically_unchecked,
            "explanation": "The frozen bridge imported its replay loader before authentication and did not compare the complete replay sources map. This post-outcome check cannot repair that historical pre-execution control gap.",
        },
        "scope": "Current plain-.py resolution under caller-supplied ordered directories; all pinned files hashed, bridge AST closure checked. PASS is not proof of historical imports, execution, timestamps, publication, or a numerical trajectory.",
        "limitations": [
            "The caller supplies the execution-directory mapping; local mirrors do not attest remote state.",
            "AST closure includes imports in function bodies and literal recognized loader calls, not a proved runtime call graph.",
            "Nonliteral imports, import hooks, sys.modules, cached bytecode, installed packages and interpreter provenance are not authenticated.",
            "Legacy/test source dependencies outside the bridge closure are disclosed separately, not silently treated as authenticated.",
            "Final rehash is a consistency check, not an atomic filesystem snapshot or historical time proof.",
        ],
        **result,
    }


def self_test():
    """Synthetic files only, with executable sentinels that must never run."""
    refusals, accepted = [], []
    with tempfile.TemporaryDirectory(prefix="dgx-source-chain-test-") as temporary:
        base = Path(temporary)

        def fixture(label, *, dependency="shared", extra_legacy=False):
            folder = base / label
            bundle, bridge, first, second, frozen = [folder / p for p in ("bundle", "bridge", "first", "second", "frozen")]
            for path in (bundle, bridge, first, second, frozen):
                path.mkdir(parents=True)
            contents = {
                ENTRYPOINT: b"import loader\nimport observer\nraise AssertionError('MUST NOT EXECUTE')\n",
                "loader.py": ("import " + dependency + "\n").encode(),
                "shared.py": b"VALUE = 1\n", "observer.py": b"import shared\nraise AssertionError('NO OBSERVER')\n",
            }
            places = {ENTRYPOINT: bridge, "loader.py": first, "shared.py": second, "observer.py": frozen}
            if extra_legacy:
                contents["legacy.py"] = b"import missing_legacy_dependency\n"
                places["legacy.py"] = frozen
            for name, raw in contents.items():
                (places[name] / name).write_bytes(raw)
            # Later mismatches must never override the earlier selected source.
            (second / "loader.py").write_bytes(b"WRONG_LATER = True\n")
            (first / ENTRYPOINT).write_bytes(b"WRONG_PYTHONPATH_ENTRY = True\n")
            hashes = {name: digest(raw) for name, raw in contents.items()}
            original_sources = {name: sha for name, sha in hashes.items() if name not in (ENTRYPOINT, "loader.py")}

            def save(name, value):
                raw = (json.dumps(value, sort_keys=True) + "\n").encode()
                (bundle / name).write_bytes(raw)
                return digest(raw)

            original_sha = save("original_protocol.json", {"identity": {"fixture": True}, "sources": original_sources})
            replay_sha = save("completed_replay.json", {"status": "PASS", "identity": {"fixture": True},
                "manifest_sha256": "a" * 64, "sources": {n: hashes[n] for n in ("loader.py", "shared.py")}})
            amendment_sha = save("amendment.json", {"schema": "dgx_hardware_only_outcome_amendment_v1",
                "original_protocol_sha256": original_sha, "completed_replay_sha256": replay_sha,
                "recorded_manifest_sha256": "a" * 64, "original_observer_sources": original_sources,
                "bridge_sources": {ENTRYPOINT: hashes[ENTRYPOINT]}})
            return bundle, bridge, [first, second, frozen], amendment_sha

        def check(args):
            bundle, bridge, dirs, sha = args
            return audit(bundle, bridge, dirs, expected_amendment=sha)

        def refuse(label, operation):
            try:
                operation()
            except (SourceChainError, OSError, SyntaxError):
                refusals.append(label)
            else:
                raise AssertionError("invalid fixture accepted: " + label)

        good = fixture("good")
        result = check(good)
        require(result["pinned_source_count"] == 4 and result["bridge_static_closure_count"] == 4,
                "synthetic closure differs")
        require(result["resolved_sources"][ENTRYPOINT]["search_index"] == 0 and
                result["resolved_sources"]["loader.py"]["search_index"] == 1, "source order differs")
        require(result["current_code_hash_agreement_is_retroactive_time_proof"] is False, "inflated scope")
        accepted.extend(["entry_directory_first", "first_pythonpath_match", "no_import_sentinels", "historical_scope_false"])
        legacy = check(fixture("legacy", extra_legacy=True))
        require(len(legacy["other_pinned_source_dependency_gaps"]) == 1 and
                not legacy["all_manifest_sources_static_dependencies_pinned"], "legacy gap hidden")
        accepted.append("non_entrypoint_gap_disclosed")

        for label in ("changed_source", "earlier_shadow", "missing_source", "package_shadow", "bytecode_shadow", "native_shadow",
                      "changed_replay", "changed_amendment", "changed_nonclosure_source"):
            args = fixture(label, extra_legacy=label == "changed_nonclosure_source")
            bundle, bridge, dirs, sha = args
            if label == "changed_source":
                (dirs[0] / "loader.py").write_bytes(b"CHANGED = True\n")
            elif label == "earlier_shadow":
                (bridge / "shared.py").write_bytes(b"SHADOW = True\n")
            elif label == "missing_source":
                (dirs[2] / "observer.py").unlink()
            elif label == "package_shadow":
                (dirs[0] / "loader").mkdir()
                (dirs[0] / "loader/__init__.py").write_bytes(b"raise AssertionError('NO IMPORT')\n")
            elif label == "bytecode_shadow":
                (bridge / "shared.pyc").write_bytes(b"not executable bytecode")
            elif label == "native_shadow":
                (bridge / "shared.cpython-312-aarch64-linux-gnu.so").write_bytes(b"not executable native code")
            elif label == "changed_nonclosure_source":
                (dirs[2] / "legacy.py").write_bytes(b"CHANGED = True\n")
            else:
                (bundle / ("completed_replay.json" if label == "changed_replay" else "amendment.json")).write_bytes(b"{}")
            refuse(label, lambda args=args: check(args))
        missing = fixture("missing_internal", dependency="absent_internal")
        refuse("missing_internal_dependency", lambda: check(missing))
        unpinned = fixture("unpinned_internal", dependency="extra_internal")
        (unpinned[2][0] / "extra_internal.py").write_bytes(b"VALUE = 2\n")
        refuse("unpinned_internal_dependency", lambda: check(unpinned))
        literal_dynamic = fixture("literal_dynamic", dependency="shared\nimport importlib as il\nil.import_module('absent_dynamic_internal')")
        refuse("literal_dynamic_internal_dependency", lambda: check(literal_dynamic))
        reordered = fixture("reordered")
        refuse("pythonpath_reordering", lambda: check((reordered[0], reordered[1], list(reversed(reordered[2])), reordered[3])))
        refuse("conflicting_pins", lambda: merge_pins({"one": {"x.py": "a" * 64}, "two": {"x.py": "b" * 64}}))
        refuse("unsafe_basename", lambda: merge_pins({"one": {"../x.py": "a" * 64}}))
        duplicate = base / "duplicate.json"
        duplicate.write_bytes(b'{"sources": {}, "sources": {}}')
        refuse("duplicate_json_key", lambda: pinned_json(duplicate, digest(duplicate.read_bytes())))
        report = base / "new-report.json"
        write_report(report, '{"original": true}\n')
        refuse("existing_report", lambda: write_report(report, '{"replacement": true}\n'))
        require(report.read_bytes() == b'{"original": true}\n', "existing report was replaced")
        accepted.append("new_report_only")
    return {"status": "PASS", "schema": "dgx_source_chain_synthetic_self_test_v1",
            "accepted_checks": accepted, "refusals": refusals, "refusal_count": len(refusals),
            "recorded_code_imported_or_executed": False, "observer_reexecuted": False,
            "original_artifacts_modified": False}


def write_report(path, payload):
    # This is new audit metadata, not a claim of crash-durable publication.
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--bridge-dir", type=Path)
    parser.add_argument("--source-dir", action="append", type=Path, default=[])
    parser.add_argument("--report", type=Path, help="optional new report path; existing files are never replaced")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        if args.bundle or args.bridge_dir or args.source_dir or args.report:
            parser.error("--self-test cannot be combined with evidence/report arguments")
    elif not args.bundle or not args.bridge_dir or not args.source_dir:
        parser.error("--bundle, --bridge-dir and ordered --source-dir arguments are required")
    try:
        if args.report:
            require(not args.report.exists() and not args.report.is_symlink(), "report already exists")
        result = self_test() if args.self_test else audit(args.bundle, args.bridge_dir, args.source_dir)
        payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.report:
            write_report(args.report, payload)
        print(payload, end="")
    except (ValueError, OSError, SyntaxError, KeyError, TypeError) as error:
        print(json.dumps({"status": "REFUSED", "reason": str(error),
                          "observer_reexecuted": False}), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
