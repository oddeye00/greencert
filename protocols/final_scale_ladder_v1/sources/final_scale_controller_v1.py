"""Bounded, outcome-independent phase state machine for the registered ladder.

The executable adapter authenticates each worker's phase artifacts. This
state machine does not treat an exit code as a scientific result, does not
resume a phase, and never picks a replacement seed or configuration.
"""
from final_scale_protocol_v1 import validate_settings
from prospective_ledger_v1 import is_hash


PHASES = ("selection", "construction", "disposition", "observation", "audit")


def require(value, message):
    if not value:
        raise ValueError(message)


def run_ladder(settings, *, execute, record, guard):
    validate_settings(settings)
    rungs, largest = [], None
    for rung in settings["rungs"]:
        current = {"index": rung["index"], "parameters": rung["parameters"], "seed": rung["seed"], "phases": {}}
        rungs.append(current)
        phase = None
        try:
            for phase in PHASES:
                guard()
                record("phase_intent", {"rung": rung["index"], "phase": phase})
                result = execute(rung["index"], phase, dict(current["phases"]))
                require(type(result) is dict and is_hash(result.get("manifest_sha256")), "phase lacks authenticated manifest")
                if phase == "selection":
                    require(result["status"] in ("candidate_selected", "no_candidate"), "invalid selection terminal")
                elif phase == "construction":
                    require(result["status"] == "complete_numeric_evidence_constructed", "incomplete construction")
                elif phase == "disposition":
                    require(result["status"] in ("certificate_issued", "scientific_abstention"), "invalid numerical disposition")
                    require(type(result["certificate_issued"]) is bool and
                            result["certificate_issued"] == (result["status"] == "certificate_issued"), "inconsistent issuance status")
                elif phase == "observation":
                    require(result["status"] == "one_shot_float64_observation_complete", "incomplete observer")
                else:
                    require(result["status"] == "complete_observation_and_point_outputs_replayed", "incomplete output audit")
                    require(result["summary"]["certificate_issued"] is current["phases"]["disposition"]["certificate_issued"],
                            "audited issuance differs from prospective disposition")
                current["phases"][phase] = result
                record("phase_completed", {"rung": rung["index"], "phase": phase, "result": result})
                if phase == "selection" and result["status"] == "no_candidate":
                    current["status"] = "no_candidate"
                    break
                if phase == "audit":
                    summary = result["summary"]
                    require(summary.get("all_count_bounds_contained") is not False, "observed count enclosure mismatch")
                    if summary["certificate_issued"]:
                        require(type(summary["issued_bracket_covered"]) is bool, "issued case has no coverage audit")
                        if not summary["issued_bracket_covered"]:
                            current["status"] = "issued_bracket_uncovered"
                            terminal = {"status": "ladder_stopped", "reason": "uncovered_issued_bracket", "rungs": rungs,
                                        "largest_issued_and_covered_parameters": largest}
                            record("terminal", terminal)
                            return terminal
                        largest = rung["parameters"]
                    current["status"] = "issued_and_covered" if summary["certificate_issued"] else "scientific_abstention"
        except Exception as error:
            current["status"] = "resource_limit_or_execution_failure"
            terminal = {"status": "ladder_stopped", "reason": current["status"], "failed_phase": phase,
                "error_type": type(error).__name__, "rungs": rungs, "largest_issued_and_covered_parameters": largest}
            record("terminal", terminal)
            return terminal
    terminal = {"status": "ladder_completed", "rungs": rungs, "largest_issued_and_covered_parameters": largest}
    record("terminal", terminal)
    return terminal
