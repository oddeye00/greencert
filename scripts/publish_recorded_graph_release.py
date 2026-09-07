"""Upload only the approved, unchanged full451k transport (stdlib only).

Run from this repository after review; TARGET must be a full GitHub commit SHA.
Each invocation requires a DIFFERENT, nonexistent report directory:

  python scripts/publish_recorded_graph_release.py --target-commit TARGET \
      --report-dir output/evidence-upload-01
  python scripts/publish_recorded_graph_release.py --target-commit TARGET \
      --resume RELEASE_ID --report-dir output/evidence-upload-02
  python scripts/publish_recorded_graph_release.py --target-commit TARGET \
      --resume RELEASE_ID --publish --report-dir output/evidence-publish-01

The first command creates a new DRAFT and uploads. Resume uploads only missing
assets, reading metadata (never downloading assets) to skip exact matches.
Publish is a separate invocation: it uploads NOTHING and refuses unless all 14
assets already match. It never marks this evidence release as latest. Existing
unrelated releases, tags, assets, source files, and the paper are never edited.

Every invocation hashes all local assets before even creating its report folder
or obtaining a credential. Keep the transport unchanged and run only one writer
for this release at a time: GitHub offers no atomic verify-and-publish operation.
Open handles, stat checks, and upload-time hashing detect ordinary local changes;
they are not a filesystem snapshot or a lock against other GitHub writers.

No request is retried, including timeouts, 502/starter assets, and lost responses.
Inspect the preserved intent/receipt files and GitHub metadata after a failure;
use an explicit resume only after resolving the uncertainty. Conflicting assets
are NEVER removed or replaced. Published releases are refused even on resume.
Reports are exclusive, numbered, fsynced JSON files; no raw HTTP/credential data
or exception text is logged. Interrupted runs retain all prior records.

API contracts (version 2026-03-10, also selectable: 2022-11-28):
https://docs.github.com/en/rest/releases/releases
https://docs.github.com/en/rest/releases/assets
"""

import argparse
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import ssl
import stat
import subprocess
import sys
import time
from urllib.parse import urlencode, urlsplit


REPO = "oddeye00/greencert"
ROOT = Path(__file__).resolve().parents[1]
TRANSPORT = ROOT / "output/full_recorded_graph_transport_20260907_v1"
TRANSPORT_SHA = "b8490b76a42338d7e8bc1c08744e05f537c7a912bbc14c7842bc567f697ae25e"
DEFAULT_TAG = "evidence-451k-20260907"
API = "https://api.github.com/repos/" + REPO
TITLE = "Full451k unchanged recorded evidence graph"
ASSET_NAMES = {"recorded_manifest.json", "replay_sources.zip", "transport.json"} | {
    f"recorded_objects_{i:03d}.zip" for i in range(11)
}
TOTAL_BYTES = 11_193_287_292
CHUNK = 1024 * 1024
MAX_JSON = 16 * 1024 * 1024


class Refusal(Exception):
    """Only locally constructed, non-secret diagnostic messages belong here."""


def require(condition, message):
    if not condition:
        raise Refusal(message)


def positive_id(value):
    require(type(value) is int and value > 0, "Invalid GitHub numeric ID.")
    return value


def stamp(stream):
    info = os.fstat(stream.fileno())
    require(stat.S_ISREG(info.st_mode), "Transport asset is not a regular file.")
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


@dataclass
class Asset:
    name: str
    size: int
    sha256: str
    stream: object
    original_stat: tuple

    def unchanged(self):
        require(stamp(self.stream) == self.original_stat,
                "Local asset changed: " + self.name)

    def description(self):
        return {"name": self.name, "bytes": self.size, "sha256": self.sha256}


def authenticate(folder, stack):
    """Authenticate the pinned descriptor, then every byte of exactly 14 files."""
    folder = folder.resolve(strict=True)

    def open_asset(name):
        path = folder / name
        require(not path.is_symlink() and path.resolve(strict=True).parent == folder,
                "Transport asset must be a direct, non-symlink file.")
        stream = stack.enter_context(path.open("rb"))
        initial = stamp(stream)
        return stream, initial

    descriptor_stream, initial = open_asset("transport.json")
    raw = descriptor_stream.read(MAX_JSON + 1)
    require(len(raw) <= MAX_JSON and stamp(descriptor_stream) == initial,
            "Transport descriptor changed or exceeds the size limit.")
    require(hashlib.sha256(raw).hexdigest() == TRANSPORT_SHA,
            "Transport descriptor SHA256 differs from the approved graph.")
    descriptor = json.loads(raw)
    require(descriptor.get("schema") == "recorded_graph_transport_v1",
            "Wrong transport schema.")
    files = descriptor.get("files")
    require(isinstance(files, dict) and set(files) == ASSET_NAMES - {"transport.json"},
            "Transport must describe exactly the approved 13 other assets.")
    assets = [Asset("transport.json", len(raw), TRANSPORT_SHA,
                    descriptor_stream, initial)]
    print("Authenticated transport.json (1/14)", flush=True)
    for name, row in sorted(files.items()):
        require(isinstance(row, dict) and set(row) == {"bytes", "sha256"}
                and type(row["bytes"]) is int and 0 < row["bytes"] < 2**31
                and isinstance(row["sha256"], str)
                and re.fullmatch(r"[0-9a-f]{64}", row["sha256"]),
                "Invalid asset descriptor.")
        stream, initial = open_asset(name)
        asset = Asset(name, row["bytes"], row["sha256"], stream, initial)
        require(initial[2] == asset.size, "Local asset size differs: " + name)
        digest, size = hashlib.sha256(), 0
        while block := stream.read(CHUNK):
            digest.update(block)
            size += len(block)
        asset.unchanged()
        require(size == asset.size and digest.hexdigest() == asset.sha256,
                "Local asset SHA256/size differs: " + name)
        assets.append(asset)
        print(f"Authenticated {name} ({len(assets)}/14)", flush=True)
    require(sum(a.size for a in assets) == TOTAL_BYTES, "Unexpected transport total.")
    check_unchanged(assets)
    return assets


def check_unchanged(assets):
    for asset in assets:
        asset.unchanged()


class Journal:
    """Create-only directory and records, never overwrite even after a crash."""

    def __init__(self, directory):
        self.directory = directory
        directory.mkdir(mode=0o700, parents=False, exist_ok=False)
        self.sequence = 0

    def record(self, kind, **fields):
        self.sequence += 1
        path = self.directory / f"{self.sequence:04d}_{kind}.json"
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump({"event": kind, "utc": datetime.now(timezone.utc).isoformat(),
                       **fields}, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())


def credential():
    """Capture privately; never approve/store credentials or expose helper output."""
    env = {key: value for key, value in os.environ.items()
           if not key.upper().startswith(("GIT_TRACE", "GCM_TRACE"))}
    env.update(GIT_TERMINAL_PROMPT="0", GCM_INTERACTIVE="Never", GCM_GUI_PROMPT="false")
    try:
        result = subprocess.run(
            ["git", "-C", str(ROOT), "-c", "credential.interactive=false",
             "-c", "credential.trace=false", "-c", "credential.traceSecrets=false",
             "credential", "fill"],
            input=f"protocol=https\nhost=github.com\npath={REPO}.git\n\n".encode(),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, timeout=30,
            check=False,
        )
        require(result.returncode == 0, "Git credential lookup failed (output withheld).")
        passwords = [line[len(b"password="):] for line in result.stdout.splitlines()
                     if line.startswith(b"password=")]
        require(len(passwords) == 1 and 0 < len(passwords[0]) <= 8192
                and all(33 <= char <= 126 for char in passwords[0]),
                "Git did not return a usable password/token (output withheld).")
        return passwords[0].decode("ascii")
    except Refusal:
        raise
    except Exception:
        raise Refusal("Git credential lookup failed (details withheld).") from None


class GitHub:
    def __init__(self, token, version, timeout, upload_timeout):
        self._token = token
        self.version, self.timeout = version, timeout
        self.upload_timeout = upload_timeout

    def request(self, method, url, *, data=None, asset=None, missing_ok=False):
        """Direct TLS only; no proxies, redirects, retries, or raw-response logging."""
        parsed = urlsplit(url)
        require(parsed.scheme == "https" and parsed.netloc in
                {"api.github.com", "uploads.github.com"} and not parsed.fragment
                and (parsed.path == f"/repos/{REPO}"
                     or parsed.path.startswith(f"/repos/{REPO}/")),
                "Refusing an unapproved authenticated destination.")
        require(method in {"GET", "POST", "PATCH"}, "Unsupported HTTP method.")
        require(not missing_ok or method == "GET", "Only reads may accept 404.")
        require((asset is not None) == (parsed.netloc == "uploads.github.com")
                and (asset is None or (method == "POST" and data is None)),
                "Invalid upload destination or request.")
        body = json.dumps(data, allow_nan=False).encode() if data is not None else b""
        headers = {"Authorization": "Bearer " + self._token,
                   "Accept": "application/vnd.github+json",
                   "X-GitHub-Api-Version": self.version,
                   "User-Agent": "greencert-recorded-graph-uploader/1",
                   "Content-Length": str(asset.size if asset else len(body)),
                   "Content-Type": "application/octet-stream" if asset else "application/json"}
        connection = http.client.HTTPSConnection(parsed.netloc, timeout=self.timeout,
                                                  context=ssl.create_default_context())
        connection.set_debuglevel(0)
        deadline = time.monotonic() + (self.upload_timeout if asset else self.timeout)
        writing = method != "GET"
        suffix = (" Write outcome may be ambiguous; STOP, inspect metadata, and do not "
                  "blindly rerun. No retry or cleanup was attempted." if writing else
                  " No retry was attempted.")

        def budget():
            remaining = deadline - time.monotonic()
            require(remaining > 0, "Request deadline exceeded.")
            connection.timeout = min(self.timeout, remaining)
            if connection.sock is not None:
                connection.sock.settimeout(connection.timeout)

        try:
            if asset:
                asset.unchanged()
                asset.stream.seek(0)
            budget()
            connection.connect()
            budget()
            target = parsed.path + ("?" + parsed.query if parsed.query else "")
            connection.putrequest(method, target)
            for name, value in headers.items():
                connection.putheader(name, value)
            connection.endheaders()
            if asset:
                sent, next_progress, digest = 0, 10, hashlib.sha256()
                while sent < asset.size:
                    budget()
                    block = asset.stream.read(min(CHUNK, asset.size - sent))
                    require(bool(block), "Local upload stream ended early.")
                    connection.send(block)
                    digest.update(block)
                    sent += len(block)
                    percent = sent * 100 // asset.size
                    if percent >= next_progress:
                        print(f"Upload {asset.name}: {percent}% ({sent}/{asset.size} bytes)",
                              flush=True)
                        next_progress = (percent // 10 + 1) * 10
                require(not asset.stream.read(1) and digest.hexdigest() == asset.sha256,
                        "Upload stream changed after preflight authentication.")
                asset.unchanged()
            elif body:
                budget()
                connection.send(body)
            budget()
            response = connection.getresponse()
            require(not 300 <= response.status < 400,
                    "Authenticated redirect rejected (destination withheld).")
            if response.status == 404 and missing_ok:
                return None
            expected = 201 if method == "POST" else 200
            require(response.status == expected, f"GitHub returned HTTP {response.status}.")
            raw = bytearray()
            while True:
                budget()
                block = response.read1(64 * 1024)
                if not block:
                    break
                raw.extend(block)
                require(len(raw) <= MAX_JSON, "GitHub JSON response exceeds limit.")
            return json.loads(raw)
        except Refusal as exc:
            raise Refusal(str(exc) + suffix) from None
        except Exception:
            # Exceptions can contain headers, credentials, URLs, or server body text.
            raise Refusal("GitHub request failed (details withheld)." + suffix) from None
        finally:
            connection.close()

    def get(self, path, *, missing_ok=False):
        return self.request("GET", API + path, missing_ok=missing_ok)

    def listing(self, path):
        rows = []
        # Construct our own pagination URLs; never follow a server-supplied Link.
        for page in range(1, 101):
            batch = self.get(path + f"?per_page=100&page={page}")
            require(isinstance(batch, list) and len(batch) <= 100
                    and all(isinstance(row, dict) for row in batch),
                    "Invalid GitHub list response.")
            rows.extend(batch)
            if len(batch) < 100:
                return rows
        raise Refusal("Pagination limit reached; refusing an incomplete inventory.")


def release_body(target):
    return ("Unchanged full451k saved-evidence graph; 14 exact-byte transport assets.\n\n"
            "This evidence-only release does not change the paper or issue a new "
            "certificate, recompute neural derivatives, or reveal a future outcome.\n\n"
            f"Transport SHA256: `{TRANSPORT_SHA}`\nTarget commit: `{target}`\n\n"
            f"<!-- greencert-recorded-graph-release-v1:{TRANSPORT_SHA}:{target} -->")


def validate_release(row, args, *, draft=True):
    require(isinstance(row, dict), "Invalid release response.")
    release_id = positive_id(row.get("id"))
    require(row.get("tag_name") == args.tag and row.get("target_commitish") == args.target_commit
            and row.get("name") == TITLE and row.get("body") == release_body(args.target_commit)
            and row.get("draft") is draft and row.get("prerelease") is False,
            "Release identity/state differs; existing releases will not be altered.")
    if draft:
        require(row.get("published_at") is None and row.get("immutable") is not True,
                "Release is not an editable, never-published draft.")
    expected = f"https://uploads.github.com/repos/{REPO}/releases/{release_id}/assets"
    require(row.get("upload_url") in {expected, expected + "{?name,label}"},
            "Unexpected release upload URL.")
    return release_id, expected


def check_tag(client, tag, target, *, must_be_new=False):
    ref = client.get("/git/ref/tags/" + tag, missing_ok=True)
    if ref is None:
        return
    require(not must_be_new, "Tag already exists; a NEW dedicated evidence tag is required.")
    require(isinstance(ref, dict) and ref.get("ref") == "refs/tags/" + tag,
            "Unexpected tag reference.")
    obj = ref.get("object")
    for _ in range(10):
        require(isinstance(obj, dict) and isinstance(obj.get("sha"), str)
                and re.fullmatch(r"[0-9a-f]{40}", obj["sha"]), "Invalid tag object.")
        if obj.get("type") == "commit":
            require(obj["sha"] == target, "Existing tag points at a different commit.")
            return
        require(obj.get("type") == "tag", "Tag does not resolve to a commit.")
        obj = client.get("/git/tags/" + obj["sha"]).get("object")
    raise Refusal("Tag nesting limit exceeded.")


def verified_asset(row, asset):
    require(isinstance(row, dict), "Invalid asset response.")
    asset_id = positive_id(row.get("id"))
    require(row.get("name") == asset.name and type(row.get("size")) is int
            and row["size"] == asset.size and row.get("state") == "uploaded"
            and row.get("digest") == "sha256:" + asset.sha256,
            "Asset conflict or unverified upload: " + asset.name
            + ". Nothing will be deleted or replaced; inspect before resuming.")
    # Return a whitelist built from trusted expectations, never a raw API object.
    return {**asset.description(), "asset_id": asset_id,
            "digest": "sha256:" + asset.sha256, "state": "uploaded"}


def inventory(client, release_id, assets, *, complete=False):
    expected = {asset.name: asset for asset in assets}
    found, ids = {}, set()
    for row in client.listing(f"/releases/{release_id}/assets"):
        name = row.get("name")
        require(isinstance(name, str) and name in expected and name not in found,
                "Unexpected or duplicate release asset; refusing to modify this release.")
        receipt = verified_asset(row, expected[name])
        require(receipt["asset_id"] not in ids, "Duplicate remote asset ID.")
        ids.add(receipt["asset_id"])
        found[name] = receipt
    require(not complete or len(found) == 14,
            "Publish requires all 14 assets already uploaded and exactly verified.")
    return found


def run(args, assets, journal, client):
    require(not args.publish or args.resume is not None,
            "Publishing requires a separate --resume run.")
    repo = client.get("")
    require(repo.get("full_name", "").lower() == REPO
            and repo.get("permissions", {}).get("push") is True,
            "Repository identity/push access unverified; draft inventory may be incomplete.")
    commit = client.get("/git/commits/" + args.target_commit)
    require(commit.get("sha") == args.target_commit, "Target commit not verified on GitHub.")
    matching = [r for r in client.listing("/releases") if r.get("tag_name") == args.tag]
    if args.resume is None:
        require(not matching, "Release tag already in use; inspect and explicitly resume by ID.")
        check_tag(client, args.tag, args.target_commit, must_be_new=True)
        check_unchanged(assets)
        journal.record("create_draft_intent", tag=args.tag, target_commit=args.target_commit)
        row = client.request("POST", API + "/releases", data={
            "tag_name": args.tag, "target_commitish": args.target_commit, "name": TITLE,
            "body": release_body(args.target_commit), "draft": True, "prerelease": False,
            "generate_release_notes": False, "make_latest": "false",
        })
    else:
        require(len(matching) == 1 and matching[0].get("id") == args.resume,
                "Resume ID does not uniquely match the dedicated release tag.")
        row = client.get(f"/releases/{args.resume}")
    release_id, upload_url = validate_release(row, args)
    require(args.resume is None or release_id == args.resume, "Resume release ID mismatch.")
    journal.record("draft_verified", release_id=release_id, tag=args.tag,
                   target_commit=args.target_commit, transport_sha256=TRANSPORT_SHA)
    print(f"Verified draft release {release_id}", flush=True)
    check_tag(client, args.tag, args.target_commit)
    found = inventory(client, release_id, assets, complete=args.publish)
    for asset in assets:
        if asset.name in found:
            journal.record("asset_receipt", release_id=release_id,
                           action="reused_exact_metadata", **found[asset.name])
            print("Reused exact asset: " + asset.name, flush=True)
            continue
        require(not args.publish, "Publish mode cannot upload assets.")
        current = client.get(f"/releases/{release_id}")
        current_id, current_url = validate_release(current, args)
        require((current_id, current_url) == (release_id, upload_url), "Release changed.")
        check_unchanged(assets)
        journal.record("upload_intent", release_id=release_id, **asset.description())
        result = client.request("POST", upload_url + "?" + urlencode({"name": asset.name}),
                                asset=asset)
        receipt = verified_asset(result, asset)
        # A second, metadata-only read must agree with the upload response.
        reread = client.get(f"/releases/assets/{receipt['asset_id']}")
        require(verified_asset(reread, asset) == receipt, "Uploaded asset metadata changed.")
        journal.record("asset_receipt", release_id=release_id, action="uploaded", **receipt)
    final_assets = inventory(client, release_id, assets, complete=True)
    current_id, _ = validate_release(client.get(f"/releases/{release_id}"), args)
    require(current_id == release_id, "Release ID changed.")
    check_tag(client, args.tag, args.target_commit)
    check_unchanged(assets)
    journal.record("all_assets_verified", release_id=release_id,
                   assets=list(final_assets.values()), count=14)
    if args.publish:
        # The only PATCH in the program; never rename, replace assets, or change latest.
        journal.record("publish_intent", release_id=release_id, verified_assets=14)
        published = client.request("PATCH", API + f"/releases/{release_id}",
                                   data={"draft": False, "make_latest": "false"})
        published_id, _ = validate_release(published, args, draft=False)
        require(published_id == release_id, "Published release ID mismatch.")
        checked_id, _ = validate_release(client.get(f"/releases/{release_id}"), args, draft=False)
        require(checked_id == release_id and inventory(client, release_id, assets, complete=True)
                == final_assets, "Post-publication verification differs; inspect manually.")
    status = "published_verified" if args.publish else "draft_complete_not_published"
    journal.record("report", status=status, repository=REPO, release_id=release_id,
                   tag=args.tag, target_commit=args.target_commit,
                   transport_sha256=TRANSPORT_SHA, asset_count=14, total_bytes=TOTAL_BYTES)
    print(f"{status}: release {release_id}, 14 verified assets", flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target-commit", required=True, help="Full 40-character GitHub commit SHA")
    parser.add_argument("--tag", default=DEFAULT_TAG, help="Dedicated evidence-451k-YYYYMMDD[-suffix] tag")
    parser.add_argument("--transport-dir", type=Path, default=TRANSPORT)
    parser.add_argument("--report-dir", type=Path, required=True,
                        help="New folder; its parent must exist. Never reuse a previous run folder.")
    parser.add_argument("--resume", type=int, metavar="RELEASE_ID", help="Explicit existing draft ID")
    parser.add_argument("--publish", action="store_true", help="With --resume: verify and publish ONLY")
    parser.add_argument("--api-version", choices=["2026-03-10", "2022-11-28"], default="2026-03-10")
    parser.add_argument("--timeout", type=int, default=60, help="Socket/metadata timeout seconds (1..600)")
    parser.add_argument("--upload-timeout", type=int, default=3600,
                        help="Overall deadline per upload in seconds (1..86400)")
    args = parser.parse_args(argv)
    require(re.fullmatch(r"[0-9a-f]{40}", args.target_commit), "A full lowercase commit SHA is required.")
    require(re.fullmatch(r"evidence-451k-[0-9]{8}(?:-[A-Za-z0-9]+)*", args.tag)
            and len(args.tag) <= 100, "Only dedicated evidence-451k date tags are allowed.")
    require(args.resume is None or args.resume > 0, "Resume requires a positive release ID.")
    require(not args.publish or args.resume is not None, "Publishing requires a separate --resume run.")
    require(1 <= args.timeout <= 600 and 1 <= args.upload_timeout <= 86400, "Invalid timeout.")
    args.report_dir = args.report_dir.absolute()
    require(not args.report_dir.exists() and not args.report_dir.is_symlink()
            and args.report_dir.parent.is_dir(), "Report folder must be new with an existing parent.")
    require(not args.report_dir.resolve().is_relative_to(args.transport_dir.resolve()),
            "Reports must be outside the unchanged transport directory.")
    return args


def main(argv=None):
    journal, client = None, None
    try:
        args = parse_args(argv)
        with ExitStack() as stack:
            assets = authenticate(args.transport_dir, stack)
            journal = Journal(args.report_dir)
            journal.record("plan", repository=REPO, tag=args.tag, target_commit=args.target_commit,
                           transport_sha256=TRANSPORT_SHA, assets=[a.description() for a in assets],
                           resume_release_id=args.resume, publish=args.publish, api_version=args.api_version)
            client = GitHub(credential(), args.api_version, args.timeout, args.upload_timeout)
            run(args, assets, journal, client)
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        message = str(exc) if isinstance(exc, Refusal) else "Run failed or interrupted (details withheld)."
        if journal is not None:
            message += " No automatic retry/rollback. Inspect preserved intents and remote metadata before resuming."
            try:
                journal.record("report", status="failed_or_uncertain", message=message)
            except Exception:
                pass  # Never mask the failure or expose raw exception/helper output.
        print(message, file=sys.stderr, flush=True)
        return 1
    finally:
        if client is not None:
            client._token = None


if __name__ == "__main__":
    sys.exit(main())
