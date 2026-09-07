"""Ensure an early native failure cannot be masked by a later CI command."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys


PREFIX = "$ErrorActionPreference = 'Stop'\n$PSNativeCommandUseErrorActionPreference = $true\n"


def run(pwsh):
    workflow = Path(__file__).resolve().parents[1]/".github/workflows/reproducibility.yml"
    lines = workflow.read_text(encoding="utf-8").splitlines()
    blocks = []
    for i, line in enumerate(lines):
        if line.strip() == "run: |":
            prefix = "\n".join(v.strip() for v in lines[i+1:i+3])+"\n"
            if prefix != PREFIX:
                raise AssertionError("multicommand workflow block lacks native fail-fast guard")
            blocks.append(i+1)
    if not blocks or "shell: pwsh" not in workflow.read_text(encoding="utf-8"):
        raise AssertionError("explicit PowerShell workflow shell/blocks missing")

    executable = "'" + sys.executable.replace("'", "''") + "'"
    marker = "GREENCERT_SECOND_COMMAND_EXECUTED"
    for code in (0, 7):
        script = PREFIX + f'& {executable} -c "raise SystemExit({code})"\nWrite-Output "{marker}"'
        result = subprocess.run([pwsh, "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", script],
                                capture_output=True, text=True, timeout=30)
        if code == 0:
            assert result.returncode == 0 and marker in result.stdout
        else:
            assert result.returncode != 0 and marker not in result.stdout
    return {"status": "PASS", "guarded_workflow_blocks": len(blocks),
            "successful_native_command_continues": True,
            "failed_native_command_stops_before_next_command": True,
            "numerical_or_empirical_result": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pwsh", default=shutil.which("pwsh"))
    args = parser.parse_args()
    if not args.pwsh:
        parser.error("PowerShell 7 executable required")
    print(json.dumps(run(args.pwsh), indent=2))
