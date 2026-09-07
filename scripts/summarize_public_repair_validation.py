"""Authenticate the clean-package Win/ARM replay results and their scope."""
import argparse
import hashlib
import json
from pathlib import Path

from read_public_repair_archive import ROOT, ARCHIVE_SHA, MANIFEST_SHA, recorded_brackets

EVIDENCE = {
    "results/public_repair_ledger_windows_20260907.replay.json": "8fd30f5ec0859b5431facd7f32e995c784a1e437aa2921df9e5b96274b57f08b",
    "results/public_repair_ledger_windows_20260907.isolation.json": "68533581bda3f7db8e78d8a32a2b7366ee0096658dc00f5f554c40dac69e3866",
    "results/public_repair_outputs_windows_20260907.replay.json": "893d6a58fb571ca1d6f76a6bf139f52edab6693067702494602efdd5011a0526",
    "results/public_repair_outputs_windows_20260907.isolation.json": "d8bd0df2e155daff04f8be8638d0f75ef4a22f7288b3ec04d7b5db48c9e63323",
    "results/public_repair_dgx_20260907/ledger.json": "f6cdbb17d50849743425bfc32e78f067588f98cf8dd169db41bac674da81dcbc",
    "results/public_repair_dgx_20260907/ledger.isolation.json": "88da457f7f8969d3753ccc334263d8417bfe0b45b9267fed41c2961d56f58bc7",
    "results/public_repair_dgx_20260907/outputs.json": "5f4b5dc3a2f994f42ab94c32a1274874e701fb09510412d1dc1792ec4c69d874",
    "results/public_repair_dgx_20260907/outputs.isolation.json": "e090164e14b73a9154534938c1e7ed1c2ab6fea6850237d80879425f6759259e",
    "results/public_repair_dgx_20260907/neural.json": "96dc7af9a89400780b30a5109413f9f15b42e0dd28352c5598318d5f6b3816bb",
    "results/public_repair_dgx_20260907/neural.isolation.json": "0b92eeb8f3637160ae9ab3cc240f7150ab64a9345ef5001d15adca21a9bf8035",
}


def check():
    loaded = {}
    for name, expected in EVIDENCE.items():
        raw = (ROOT/name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("public replay evidence changed: "+name)
        value = json.loads(raw)
        if value["status"] != "PASS" or value["manifest_sha256"] != MANIFEST_SHA or value["new_prospective_events"] != 0:
            raise ValueError("wrong public replay identity")
        loaded[name] = value
    rows = list(loaded.values())
    for index in range(0, len(rows), 2):
        report, isolation = rows[index:index+2]
        expected_jobs, expected_claims = (3, 4) if report["mode"] == "neural" else (63, 79)
        if (report["selected_jobs"], report["retained_claims"], report["partial_smoke_test"]) != (expected_jobs, expected_claims, False):
            raise ValueError("selected/full replay scope differs")
        if isolation["report_sha256"] != list(EVIDENCE.values())[index] or isolation["unexpected_private_accesses"] != 0 or not (
                isolation["private_roots_denied"] == isolation["guard_refusals_tested"] >= 1):
            raise ValueError("private-read isolation did not pass")
        if report["neural_derivatives_recomputed"] != (report["mode"] == "neural") or report["output_margins_recomputed"] != (report["mode"] != "ledger"):
            raise ValueError("numerical replay mode mislabeled")
        if report["mode"] != "neural" and (report["scalar_transitions"], report["output_margins"]) != (7551, 1485052):
            raise ValueError("fixed transition/margin population differs")
    if rows[0]["rows"] != rows[4]["rows"] or rows[2]["rows"] != rows[6]["rows"]:
        raise ValueError("Win/ARM per-job results differ")
    if [r["ordinal"] for r in rows[8]["rows"]] != [0, 40, 47] or [r["study"] for r in rows[8]["rows"]] != ["wdbc", "digits", "modular_mse"]:
        raise ValueError("full neural representative population changed")
    public = recorded_brackets()
    for report in (rows[0], rows[2], rows[4], rows[6]):
        observed = {c["key"]: c["replayed_bracket"] for row in report["rows"] for c in row["events"] if c["retained"]}
        if observed != public:
            raise ValueError("public recorded brackets differ from independent replay")
    return {"status": "PASS", "archive_sha256": ARCHIVE_SHA, "manifest_sha256": MANIFEST_SHA,
            "fixed_jobs": 63, "historical_brackets": 79, "new_prospective_events": 0,
            "win_arm_ledger_job_results_equal": True, "win_arm_output_job_results_equal": True,
            "scalar_transitions_per_full_replay": 7551, "output_margins_per_full_replay": 1485052,
            "full_output_replay_platforms": ["Windows/x86_64", "Linux/aarch64"],
            "neural_recomputation": {"platform": "Linux/aarch64", "fixed_job_ordinals": [0, 40, 47],
                "full_windows": 3, "retained_brackets": 4, "transitions": 126, "output_margins": 39149},
            "private_read_guard_passed": True, "evidence": EVIDENCE,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "scope": "Clean-package replay of the completed repair. Full 63-job derivative recomputation belongs to the original repair record; this portability check repeats full neural kernels for three fixed representative jobs."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = check()
    if args.report:
        with args.report.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")
    print(json.dumps(result, indent=2))
