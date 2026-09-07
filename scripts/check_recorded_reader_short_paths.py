"""Reproduce the Windows CI short-TEMP fixture failure without changing a reader."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main(legacy,corrected):
    if sys.platform!="win32":
        return {"status":"not_applicable","reason":"Windows 8.3 path regression"}
    with tempfile.TemporaryDirectory(prefix="greencert_short_temporary_root_") as temporary:
        root=Path(temporary).resolve()
        buffer=ctypes.create_unicode_buffer(32768)
        length=ctypes.windll.kernel32.GetShortPathNameW(str(root),buffer,len(buffer))
        if not length or length>=len(buffer) or buffer.value.casefold()==str(root).casefold():
            return {"status":"not_exercised","reason":"8.3 alias unavailable on this volume"}
        env={**os.environ,"TEMP":buffer.value,"TMP":buffer.value}
        old=subprocess.run([sys.executable,str(legacy.resolve())],capture_output=True,text=True,env=env)
        new=subprocess.run([sys.executable,str(corrected.resolve())],capture_output=True,text=True,env=env)
        if old.returncode==0 or "is not in the subpath" not in old.stderr:
            raise AssertionError("legacy short-path failure not reproduced")
        if new.returncode!=0:
            raise AssertionError("corrected fixture failed: "+new.stderr)
        report=json.loads(new.stdout)
        if report["status"]!="PASS" or report["refusals"]!=31:
            raise AssertionError("corrected fixture did not exercise its full refusal set")
        return {"status":"PASS","legacy_short_path_failure_reproduced":True,
                "corrected_fixture_refusals":31,"reader_implementation_changed":False}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy",type=Path,required=True)
    parser.add_argument("--corrected",type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(main(args.legacy,args.corrected),indent=2))
