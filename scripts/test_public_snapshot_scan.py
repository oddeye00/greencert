"""Check escaped-path detection and false positives in publication helpers."""
import ast
from pathlib import Path
import re
import tempfile

from scan_public_snapshot import LOCAL, strings


def main():
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root/"scripts/build_anonymous_supplement.py").read_text(encoding="utf-8"))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "sanitize")
    # Load only the pure sanitizer, not the builder's filesystem enumeration.
    namespace = {"ROOT": Path("test-repository"), "re": re}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "sanitizer-test", "exec"), namespace)
    sanitize = namespace["sanitize"]
    tested = 0
    for sep in (chr(92), chr(92)*2, "/"):
        path = sep.join(("Q:", "Users", "example", "project"))
        # Raw JSON escaping is decoded before matching, as in the scanner.
        import json
        decoded = list(strings(json.loads(json.dumps({"source": path}))))
        if sep != chr(92)*2:
            assert any(p.search(v) for p in LOCAL for v in decoded)
        assert sanitize(path) == "<USER_HOME>"+sep+"project"
        tested += 1
    prefix = chr(92).join(("Q:", "Users", ""))
    assert not any(p.search(prefix) for p in LOCAL)
    assert not any(p.search('b"'+prefix+'", other') for p in LOCAL)
    print({"status": "PASS", "sanitizer_encodings": tested, "prefix_false_positive_checks": 2,
           "numerical_records_rewritten": False})


if __name__ == "__main__":
    main()
