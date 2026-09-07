"""Replay the corrected continuation package in a fresh, retained directory.

The default recomputes all 63 neural windows. A ledger or output-only replay
is explicitly labeled and is not a fresh derivative audit. No old evidence
file is rewritten, and a failed attempt is retained for inspection.
"""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from read_public_repair_archive import ROOT, MANIFEST_SHA, extract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("ledger", "outputs", "neural"), default="neural")
    parser.add_argument("--job", type=int, action="append")
    parser.add_argument("--steps", type=int)
    args = parser.parse_args()
    if args.steps is not None and (args.mode != "neural" or args.steps < 1):
        parser.error("--steps is a positive, explicitly partial neural smoke test")
    if args.job is not None and (len(args.job) != len(set(args.job)) or any(j < 0 or j >= 63 for j in args.job)):
        parser.error("--job ordinals must be distinct members of 0..62")
    out = ROOT/"output"
    out.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run = Path(tempfile.mkdtemp(prefix="corrected-continuation-"+stamp+"-", dir=out))
    target = run/"artifact"
    extract(target)
    command = [sys.executable, str(target/"scripts/replay_public_repair.py"),
               "--manifest-sha256", MANIFEST_SHA, "--mode", args.mode,
               "--report", str(run/"replay.json")]
    for job in args.job or []:
        command.extend(("--job", str(job)))
    if args.steps is not None:
        command.extend(("--steps", str(args.steps)))
    environment = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                   "MKL_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": ""}
    print("Retained replay directory: "+run.relative_to(ROOT).as_posix(), flush=True)
    subprocess.run(command, cwd=target, env=environment, check=True)


if __name__ == "__main__":
    main()
