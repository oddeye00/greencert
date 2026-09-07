"""Scan a public manifest and nested ZIP text without printing matched values."""
import argparse
import ast
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import zipfile

TEXT = {".py", ".json", ".md", ".txt", ".in", ".log", ".csv", ".yml", ".yaml", ".tex", ".bib", ".cff", ".sty"}
LOCAL = (re.compile(r'''[A-Za-z]:[\\/]Users[\\/][^<\\/\s"'`;,:()\[\]{}]+''', re.I),
         re.compile("/"+r"home/[^/<\s]+/"), re.compile("/"+r"Users/[^/<\s]+/"))
SECRET = (re.compile(r"ghp_[A-Za-z0-9]{20,}"), re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
          re.compile(r"AKIA[0-9A-Z]{16}"), re.compile("-----"+"BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"))


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from strings(key)
            yield from strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)


def audit(root):
    root = root.resolve()
    manifest = json.loads((root/"PUBLIC_MANIFEST_SHA256.json").read_text(encoding="utf-8"))
    findings = {"machine_paths": [], "credential_shapes": [], "blocked": []}
    inspected = 0
    def inspect(name, raw, depth=0):
        nonlocal inspected
        inspected += 1
        suffix = Path(name).suffix.lower()
        if suffix in {".ckpt", ".pem", ".pt", ".pth"} or len(raw) >= 95*1024*1024:
            findings["blocked"].append(name)
        if suffix == ".zip":
            if depth >= 3:
                raise ValueError("unexpected archive nesting")
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                members = archive.infolist()
                if len({m.filename for m in members}) != len(members):
                    raise ValueError("duplicate archive names")
                for member in members:
                    if member.is_dir():
                        continue
                    p = PurePosixPath(member.filename)
                    if p.is_absolute() or ".." in p.parts or ":" in member.filename or "\\" in member.filename:
                        raise ValueError("unsafe archive member")
                    inspect(name+"!"+member.filename, archive.read(member), depth+1)
        if suffix not in TEXT and Path(name).name not in {"Dockerfile", "Makefile", "LICENSE", "LICENSE-PAPER"}:
            return
        text = raw.decode("utf-8", errors="replace")
        values = [text]
        if suffix == ".json":
            values.extend(strings(json.loads(text)))
        elif suffix == ".py":
            values.extend(n.value for n in ast.walk(ast.parse(text))
                          if isinstance(n, ast.Constant) and isinstance(n.value, str))
        for label, patterns in (("machine_paths", LOCAL), ("credential_shapes", SECRET)):
            if any(pattern.search(value) for pattern in patterns for value in values):
                findings[label].append(name)
    for name, expected in manifest["files"].items():
        path = (root/name).resolve(strict=True)
        path.relative_to(root)
        raw = path.read_bytes()
        if len(raw) != expected["bytes"] or hashlib.sha256(raw).hexdigest().upper() != expected["sha256"]:
            raise ValueError("snapshot changed after manifest construction")
        inspect(name, raw)
    report = {"status": "PASS" if not any(findings.values()) else "publication_intake_required",
              "inspected_files_and_members": inspected, **findings,
              "scope": "Static decoded-text publication scan, not a numerical proof."}
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.root)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["status"] == "PASS" else 1)
