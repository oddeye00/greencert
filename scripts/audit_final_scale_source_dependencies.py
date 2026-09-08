"""Read-only static inventory for the new final-scale pipeline.

This records local import closure and old-case literals without executing
any inspected module. It is not a dynamic call-graph or runtime attestation.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import sys


ROOTS = (
    "final_scale_protocol_v1", "final_scale_response_v1", "final_scale_neural_v1", "final_scale_green_v1",
    "pinned_source_bundle_v1", "prospective_ledger_v1",
    "streaming_variational_centerline", "transformer_hvp_grokking",
    "transformer_optimizer_probe", "arb_deep_envelope",
    "arb_deep_point_gram_bound_strategy", "arb_psd_moment_bounds",
    "arb_native_row_bounds", "arb_reverse_transformer",
    "arb_factored_regularizer_hvp", "arb_deep_transformer_forward",
    "checkpointed_green_products", "green_local_residual_bounds",
    "outward_inexact_anytime_gram", "window_event_assembly",
)
EXTERNAL = {"numpy", "torch", "flint", "scipy", "mpmath", "sklearn", "matplotlib", "pandas"}
FORBIDDEN_DRIVERS = {"build_larger_candidate_reference", "dgx_outcome_bridge",
    "continue_candidate_sealed_outcome", "continue_runtime_anchor_sealed_outcome",
    "run_larger_transformer_third_pass", "dgx_green_power2", "dgx_native_row_pilot"}
OLD_CASE = re.compile(r"\b(?:451008|902016|3925|94003)\b|larger_transformer_third_pass|construction_03925")


def inventory(directory, roots):
    directory = Path(directory).resolve(strict=True)
    todo, sources, edges, external, findings = list(roots), {}, {}, set(), []
    while todo:
        name = todo.pop()
        if name in sources:
            continue
        if not name.isidentifier() or not name.isascii():
            raise ValueError("only flat module names supported")
        path = directory / (name + ".py")
        if path.is_symlink() or not path.is_file():
            raise ValueError("missing or indirect local source: " + name)
        raw = path.read_bytes()
        tree = ast.parse(raw, filename=path.name)
        sources[name] = {"file": path.name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        edges[name] = []
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(v.name.split(".")[0] for v in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level or node.module is None:
                    raise ValueError("relative import in flat source: " + name)
                imported.add(node.module.split(".")[0])
        for dependency in sorted(imported):
            if (directory / (dependency + ".py")).is_file():
                if dependency in sys.stdlib_module_names or dependency in EXTERNAL:
                    raise ValueError("local source shadows a trusted dependency")
                edges[name].append(dependency)
                todo.append(dependency)
            elif dependency in sys.stdlib_module_names:
                pass
            elif dependency in EXTERNAL:
                external.add(dependency)
            else:
                raise ValueError("unresolved/unregistered import: " + name + " -> " + dependency)
        for index, line in enumerate(raw.decode("utf-8").splitlines(), 1):
            if OLD_CASE.search(line):
                findings.append({"file": path.name, "line": index, "text": line.strip()})
    forbidden = sorted(set(sources) & FORBIDDEN_DRIVERS)
    if forbidden:
        raise ValueError("old-case driver enters new core closure: " + ", ".join(forbidden))
    return {"status": "static_source_inventory_completed", "roots": list(roots),
            "modules": len(sources), "sources": sources, "imports": edges,
            "external_top_level": sorted(external), "old_case_literal_findings": findings,
            "old_case_driver_imports": forbidden, "inspected_code_executed": False,
            "future_outcome_access": False, "dynamic_dependency_closure_proved": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--include-neural-tests", action="store_true")
    parser.add_argument("--include-green-tests", action="store_true")
    parser.add_argument("--include-pipeline-tests", action="store_true")
    args = parser.parse_args()
    roots = ROOTS + (("test_final_scale_neural_v1",) if args.include_neural_tests else ())
    roots += (("test_final_scale_green_v1",) if args.include_green_tests else ())
    roots += (("test_final_scale_pipeline_v1",) if args.include_pipeline_tests else ())
    result = inventory(Path(__file__).parent, roots)
    with args.report.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: value for key, value in result.items() if key not in ("sources", "imports")}, indent=2))


if __name__ == "__main__":
    main()
