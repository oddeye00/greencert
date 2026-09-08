"""Pre-import authentication for new flat Python source bundles.

Launch with ``python -I -B`` and an externally pinned manifest digest. All
registered source bytes are authenticated before any registered code runs;
imports then compile those same in-memory bytes, not a second disk read.
The bootstrap, interpreter and installed dependencies remain trusted. This
is a cooperative reproducibility control, not a sandbox or attestation.
"""
import argparse
import ast
import hashlib
import importlib.abc
import importlib.util
import json
import keyword
from pathlib import Path
import re
import runpy
import sys


SCHEMA = "pinned_python_bundle_v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unique_object(pairs):
    value = {}
    for key, item in pairs:
        require(key not in value, "duplicate JSON key")
        value[key] = item
    return value


def invalid_constant(value):
    raise ValueError("nonfinite JSON constant")


def valid_hash(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def valid_name(value):
    return (type(value) is str and value.isascii() and value.isidentifier()
            and not keyword.iskeyword(value) and not value.startswith("__"))


def authenticate(source_root, manifest_path, expected_sha256):
    """Read/authenticate every payload; do not import or execute bundle code."""
    require(valid_hash(expected_sha256), "invalid external manifest digest")
    path = Path(manifest_path)
    require(not path.is_symlink() and path.is_file(), "manifest must be a regular file")
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected_sha256, "manifest digest differs")
    manifest = json.loads(raw, object_pairs_hook=unique_object, parse_constant=invalid_constant)
    require(type(manifest) is dict and set(manifest) ==
            {"schema", "entry_module", "sources", "external_top_level"}, "manifest fields differ")
    require(manifest["schema"] == SCHEMA, "unknown source manifest schema")
    sources, external = manifest["sources"], manifest["external_top_level"]
    require(type(sources) is dict and sources, "empty or invalid source population")
    require(type(external) is list and all(valid_name(n) for n in external)
            and len(set(external)) == len(external), "invalid external module names")
    names = {}
    for filename, row in sources.items():
        require(type(filename) is str and filename.endswith(".py") and
                valid_name(filename[:-3]), "only flat Python module names are supported")
        name = filename[:-3]
        require(name not in sys.stdlib_module_names and name not in external,
                "bundle shadows a trusted dependency")
        require(type(row) is dict and set(row) == {"sha256", "bytes"}
                and valid_hash(row["sha256"]) and type(row["bytes"]) is int
                and row["bytes"] >= 0, "invalid source identity")
        names[name] = row
    require(valid_name(manifest["entry_module"]) and manifest["entry_module"] in names,
            "entry point is not registered")
    require(not set(names).intersection(sys.modules), "bundle module was imported before authentication")
    root = Path(source_root)
    require(not root.is_symlink() and root.is_dir(), "source root must be a real directory")
    root = root.resolve(strict=True)
    children = list(root.iterdir())
    require({p.name for p in children} == set(sources), "unregistered or missing bundle files")
    payloads = {}
    for name, row in names.items():
        path = root / (name + ".py")
        require(not path.is_symlink() and path.is_file(), "source must be a regular file")
        data = path.read_bytes()
        require(len(data) == row["bytes"] and hashlib.sha256(data).hexdigest() == row["sha256"],
                "source payload differs: " + name)
        payloads[name] = data
    allowed = set(names) | set(external) | set(sys.stdlib_module_names)
    for name, data in payloads.items():
        tree = ast.parse(data, filename=str(root / (name + ".py")))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported = [item.name.split(".")[0] for item in node.names]
            elif isinstance(node, ast.ImportFrom):
                require(node.level == 0 and node.module is not None, "relative import in flat bundle")
                imported = [node.module.split(".")[0]]
            else:
                continue
            require(set(imported) <= allowed, "undeclared direct import in " + name)
    return manifest, root, payloads


class PinnedFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Serve registered modules from authenticated bytes, independent of PATH."""

    def __init__(self, root, payloads):
        self.root = root
        self.payloads = dict(payloads)

    def find_spec(self, fullname, path=None, target=None):
        if fullname in self.payloads:
            return importlib.util.spec_from_file_location(fullname, self.root / (fullname + ".py"), loader=self)
        return None

    def create_module(self, spec):
        return None

    def get_code(self, fullname):
        return compile(self.payloads[fullname], str(self.root / (fullname + ".py")), "exec", dont_inherit=True)

    def exec_module(self, module):
        exec(self.get_code(module.__name__), module.__dict__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("entry_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    require(sys.flags.isolated == 1 and sys.dont_write_bytecode,
            "bootstrap requires python -I -B")
    manifest, root, payloads = authenticate(args.source_root, args.manifest, args.manifest_sha256)
    if args.check_only:
        print(json.dumps({"status": "authenticated_without_execution", "modules": len(payloads),
                          "manifest_sha256": args.manifest_sha256, "bundle_code_executed": False}))
        return
    finder = PinnedFinder(root, payloads)
    entry_args = args.entry_args[1:] if args.entry_args[:1] == ["--"] else args.entry_args
    sys.argv = [str(root / (manifest["entry_module"] + ".py")), *entry_args]
    sys.meta_path.insert(0, finder)
    runpy.run_module(manifest["entry_module"], run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
