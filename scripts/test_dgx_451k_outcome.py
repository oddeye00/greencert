"""Independent audit regression using the completed, public observation.

Mutations affect temporary fixtures only; this script never runs an optimizer.
"""
import hashlib
import itertools
import json
from pathlib import Path
import tempfile

from audit_dgx_451k_outcome import audit, first_streak, parse
from package_dgx_451k_outcome import extract, AMENDMENT, REVEAL, SEAL_COMMIT, SOURCE

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_SHA = "92e378a569b6837595ac6f520de6eb6cf004e5c14f52ee5b94be1e0d7e24090c"


def run():
    enumerated = 0
    for h in range(9):
        for counts in itertools.product((0,1), repeat=h):
            for persistence in range(1,6):
                expected = next((j for j in range(h-persistence+1) if all(counts[j:j+persistence])), None)
                assert first_streak(counts,1,persistence) == expected
                enumerated += 1
    refused = []
    with tempfile.TemporaryDirectory(prefix="greencert-outcome-test-") as temporary:
        destination = Path(temporary)/"fixture"
        archive = ROOT/"artifacts/greencert_451k_completed_outcome_20260907_v1.zip"
        extract(archive,ARCHIVE_SHA,destination)
        root, bundle = destination/"root", destination/"amendment"
        source, ledger = root/SOURCE, root/SOURCE/"registered_outcome_run"
        original = {p:p.read_bytes() for p in destination.rglob("*") if p.is_file()}
        baseline = audit(root,bundle,REVEAL,AMENDMENT,SEAL_COMMIT)
        assert baseline["bracket_covered"] and baseline["observed_first_persistent_offset"] == 44
        assert baseline["logits_independently_reconstructed"] == 64090 and baseline["argmax_ties"] == 0
        assert baseline["all_observed_counts_inside_certified_bounds"] and baseline["observed_rows"] == 65
        for label,call in (
            ("duplicate_json_key",lambda:parse('{"a":1,"a":2}')),
            ("nonfinite_json_constant",lambda:parse('{"a":NaN}')),
            ("boolean_count",lambda:first_streak([True],1,1)),
            ("boolean_persistence",lambda:first_streak([1],1,True)),
            ("invalid_commit_reference",lambda:audit(root,bundle,REVEAL,AMENDMENT,"unknown")),
        ):
            try:
                call()
            except ValueError:
                refused.append(label)
            else:
                raise AssertionError(label)
        for label in ("existing_extraction", "bad_archive_hash"):
            try:
                extract(archive, ARCHIVE_SHA if label == "existing_extraction" else "0"*64,
                        destination if label == "existing_extraction" else Path(temporary)/"not-written")
            except ValueError:
                refused.append(label)
            else:
                raise AssertionError(label)
        def write(path, value):
            raw = (json.dumps(value,indent=2,sort_keys=True)+"\n").encode()
            path.write_bytes(raw)
            return hashlib.sha256(raw).hexdigest()
        def mutate_json(path, mutator):
            value = json.loads(path.read_bytes())
            mutator(value)
            return write(path,value)
        cases = [
            ("row_checksum", "row", lambda r:r["observation"].update(correct_count=0),False),
            ("logit_derived_count", "row", lambda r:r["observation"].update(correct_count=0),True),
            ("logit_derived_prediction", "row", lambda r:r["observation"]["predictions"].__setitem__(0,100),True),
            ("reported_ties", "row", lambda r:r["observation"].update(argmax_ties=1),True),
            ("wrong_evaluation_population", "row", lambda r:r["observation"].update(evaluation_count=57),True),
            ("wrong_initial_state", "row", lambda r:r["observation"].update(parameter_sha256="0"*64),True),
            ("wrong_absolute_update", "row", lambda r:r.update(absolute_update=3926),True),
            ("boolean_row_index", "row", lambda r:r.update(step=False),True),
            ("early_initial_row", "row", lambda r:r.update(utc="2000-01-01T00:00:00+00:00"),True),
            ("reveal_count_mismatch", "reveal", lambda r:r["counts"].__setitem__(0,0),True),
            ("wrong_event", "reveal", lambda r:r.update(observed_first_persistent_offset=43),True),
            ("wrong_anchor", "reveal", lambda r:r.update(anchor=0),True),
            ("inflated_numerical_scope", "reveal", lambda r:r.update(exact_real_continuation_verified=True),True),
            ("early_reveal", "reveal", lambda r:r.update(utc="2000-01-01T00:00:00+00:00"),True),
            ("missing_row_population", "reveal", lambda r:r["row_sha256"].pop("row_064.json"),True),
            ("wrong_intent_identity", "intent", lambda r:r.update(step=2),True),
            ("early_intent", "intent", lambda r:r.update(utc="2000-01-01T00:00:00+00:00"),True),
            ("early_engine", "engine", lambda r:r.update(utc="2000-01-01T00:00:00+00:00"),True),
            ("wrong_engine_runtime", "engine", lambda r:r["runtime"].update(device="cuda"),True),
            ("coherent_runtime_threads", "runtime", lambda r:r["runtime"].update(threads=8),True),
            ("coherent_runtime_build", "runtime", lambda r:r["runtime"].update(torch_build="foreign"),True),
            ("coherent_runtime_determinism", "runtime", lambda r:r["runtime"].update(deterministic_algorithms=False),True),
        ]
        for label,kind,mutator,rebind in cases:
            path = {"row":ledger/"row_000.json", "reveal":source/"reveal.json",
                    "intent":ledger/"intent_001.json", "engine":ledger/"engine_ready.json",
                    "runtime":ledger/"engine_ready.json"}[kind]
            changed_sha = mutate_json(path,mutator)
            expected_reveal = REVEAL
            if rebind and kind == "row":
                expected_reveal = mutate_json(source/"reveal.json",lambda r:r["row_sha256"].update({"row_000.json":changed_sha}))
            elif rebind and kind == "reveal":
                expected_reveal = changed_sha
            elif kind == "runtime":
                expected_reveal = mutate_json(source/"reveal.json",mutator)
            try:
                audit(root,bundle,expected_reveal,AMENDMENT,SEAL_COMMIT)
            except ValueError:
                refused.append(label)
            else:
                raise AssertionError("semantic corruption accepted: "+label)
            finally:
                path.write_bytes(original[path])
                (source/"reveal.json").write_bytes(original[source/"reveal.json"])
        # A coherent wrong or absent synthetic outcome is a valid audit result,
        # not grounds to force coverage or to suppress the observation.
        labels = [r["label"] for r in json.loads((bundle/"original_protocol.json").read_bytes())["evaluation_examples"]]
        uncovered = []
        for label,steps,event in (("late_finite",[44],45),("no_crossing",range(44,65),None)):
            reveal = json.loads(original[source/"reveal.json"])
            touched = []
            for step in steps:
                path = ledger/f"row_{step:03d}.json"
                row = json.loads(original[path])
                obs = row["observation"]
                index = next(i for i,pred in enumerate(obs["predictions"]) if pred == labels[i])
                wrong = (labels[index]+1) % 17
                obs["logits_float64_hex"][index] = [float(1 if i == wrong else 0).hex() for i in range(17)]
                obs["predictions"][index] = wrong
                obs["correct_count"] -= 1
                reveal["counts"][step] -= 1
                reveal["row_sha256"][path.name] = write(path,row)
                touched.append(path)
            reveal.update(observed_first_persistent_offset=event,observed_float64_crossing_in_bracket=False,
                          prediction_error_updates=None if event is None else 44-event)
            synthetic_sha = write(source/"reveal.json",reveal)
            result = audit(root,bundle,synthetic_sha,AMENDMENT,SEAL_COMMIT)
            assert result["status"] == "PASS" and result["observed_first_persistent_offset"] == event
            assert result["bracket_covered"] is False and result["all_observed_counts_inside_certified_bounds"] is False
            uncovered.append(label)
            for path in touched+[source/"reveal.json"]:
                path.write_bytes(original[path])
        extra = ledger/"unexpected.json"
        extra.write_text("{}")
        try:
            audit(root,bundle,REVEAL,AMENDMENT,SEAL_COMMIT)
        except ValueError:
            refused.append("unexpected_ledger_record")
        else:
            raise AssertionError("unexpected ledger file accepted")
        # The fixture is disposable; no original record is modified.
    return {"status":"PASS", "event_paths_enumerated":enumerated, "refusals":refused,
            "coherent_uncovered_synthetic_outcomes_retained":uncovered,
            "completed_observation_logits_checked":64090, "neural_optimizer_reexecuted":False,
            "original_recorded_payloads_modified":False}


if __name__ == "__main__":
    print(json.dumps(run(),indent=2,sort_keys=True))
