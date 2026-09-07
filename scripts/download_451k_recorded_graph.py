"""Download the public recorded graph without credentials; authenticate bytes.

Existing complete files are accepted only after hash checks. Failed partial
downloads are retained and never mistaken for a complete file. No code in
the graph is executed, and no optimizer or certificate is run.
"""
import argparse
import hashlib
import json
from pathlib import Path
import os
import tempfile
from urllib.parse import urlsplit
from urllib.request import build_opener, HTTPRedirectHandler, Request

BASE = "https://github.com/oddeye00/greencert/releases/download/evidence-451k-20260907/"
TRANSPORT_SHA = "b8490b76a42338d7e8bc1c08744e05f537c7a912bbc14c7842bc567f697ae25e"


class Redirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        u = urlsplit(newurl)
        if u.scheme != "https" or u.hostname not in {"github.com","release-assets.githubusercontent.com","objects.githubusercontent.com"}:
            raise ValueError("unexpected download redirect")
        return super().redirect_request(req,fp,code,msg,headers,newurl)


def file_sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def fetch(folder, name, expected, size=None):
    if not name or Path(name).name != name or any(c in name for c in ("/","\\",":")):
        raise ValueError("invalid download filename")
    target = folder/name
    if target.exists():
        if target.is_symlink() or (size is not None and target.stat().st_size != size) or file_sha(target) != expected:
            raise ValueError("existing asset differs: "+name)
        print("Authenticated existing "+name,flush=True)
        return
    fd, temporary = tempfile.mkstemp(prefix=name+".",suffix=".partial",dir=folder)
    downloaded, digest, next_progress = 0, hashlib.sha256(), 256*1024*1024
    with os.fdopen(fd,"wb") as stream:
        request = Request(BASE+name,headers={"User-Agent":"greencert-public-evidence-downloader/1"})
        with build_opener(Redirects()).open(request,timeout=60) as response:
            if response.status != 200:
                raise ValueError("download did not return HTTP200")
            while block := response.read(1024*1024):
                downloaded += len(block)
                if downloaded > (size if size is not None else 16*1024*1024):
                    raise ValueError("download exceeds registered size")
                stream.write(block)
                digest.update(block)
                if downloaded >= next_progress:
                    print(f"Downloading {name}: {downloaded} bytes",flush=True)
                    next_progress += 256*1024*1024
        stream.flush()
        os.fsync(stream.fileno())
    if digest.hexdigest() != expected or (size is not None and downloaded != size):
        raise ValueError("download checksum/size differs: "+name)
    # Atomic no-replacement installation. Partial filename is deliberately kept
    # as a hardlink, so interrupted-download evidence is never auto-deleted.
    os.link(temporary,target)
    print("Downloaded and authenticated "+name,flush=True)


def run(folder):
    folder.mkdir(parents=True,exist_ok=True)
    fetch(folder,"transport.json",TRANSPORT_SHA)
    descriptor = json.loads((folder/"transport.json").read_bytes())
    if descriptor["schema"] != "recorded_graph_transport_v1" or len(descriptor["files"]) != 13:
        raise ValueError("unexpected pinned transport")
    for name,row in sorted(descriptor["files"].items()):
        fetch(folder,name,row["sha256"],row["bytes"])
    print(json.dumps({"status":"complete_public_transport_authenticated","assets":14,
        "transport_sha256":TRANSPORT_SHA,"optimizer_executed":False},indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--destination",type=Path,required=True)
    run(p.parse_args().destination)
