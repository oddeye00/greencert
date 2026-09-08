"""Linux worker limits and a one-shot, PID-bound subprocess supervisor.

The production launcher must bind these limits in its public protocol seal.
This module is not an experiment launcher and never selects a neural seed.
No worker is resumed after timeout/failure. Only a process created by this
supervisor is signalled, through its Linux pidfd rather than a reusable PID.
"""
from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

from prospective_ledger_v1 import encode, is_hash, publish_new, sync_directory


MIB = 1024**2


def hard_resource_policy():
    """Additional pre-seal resource choices, separate from scientific settings."""
    return {"worker_address_space_mib": 49152, "minimum_free_memory_mib": 16384,
            "minimum_free_disk_mib": 262144, "maximum_worker_rss_mib": 49152,
            "maximum_complete_phase_artifact_mib": 524288,
            "supervisor_poll_seconds": 1.0, "termination_grace_seconds": 5.0,
            "worker_subprocesses_permitted": False, "automatic_retry_or_resume": False}


class ResourceLimit(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise ValueError(message)


def positive(value, name):
    require(type(value) in (int, float) and math.isfinite(value) and value > 0, "invalid " + name)
    return value


def install_address_space_limit(mib):
    """Irreversibly lower this worker's RLIMIT_AS; call only in a new worker."""
    require(os.name == "posix" and Path("/proc/self/status").exists(), "Linux worker required")
    require(type(mib) is int and mib > 0, "positive integer memory limit required")
    import resource
    cap = mib * MIB
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    effective = min(cap, hard) if hard != resource.RLIM_INFINITY else cap
    if soft != resource.RLIM_INFINITY:
        effective = min(effective, soft)
    resource.setrlimit(resource.RLIMIT_AS, (effective, effective))
    return {"address_space_limit_bytes": effective, "inherited_limit_not_raised": True}


def linux_memory_snapshot(pid):
    require(type(pid) is int and pid > 0, "invalid worker PID")
    fields = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        fields[key] = value.split()
    require("MemAvailable" in fields and fields["MemAvailable"][1:] == ["kB"], "missing Linux memory availability")
    available = int(fields["MemAvailable"][0]) * 1024
    statm = Path(f"/proc/{pid}/statm").read_text().split()
    require(len(statm) >= 2, "missing worker resident size")
    return available, int(statm[1]) * os.sysconf("SC_PAGE_SIZE")


@dataclass(frozen=True)
class Envelope:
    seconds: float
    minimum_free_memory_bytes: int
    minimum_free_disk_bytes: int
    maximum_rss_bytes: int
    poll_seconds: float = 1.0
    termination_grace_seconds: float = 5.0

    def validate(self):
        positive(self.seconds, "time limit")
        positive(self.poll_seconds, "poll interval")
        positive(self.termination_grace_seconds, "termination grace")
        require(self.poll_seconds <= 5 and self.termination_grace_seconds <= 30, "unbounded monitoring interval")
        for value in (self.minimum_free_memory_bytes, self.minimum_free_disk_bytes, self.maximum_rss_bytes):
            require(type(value) is int and value > 0, "positive integer resource bounds required")

    def check(self, *, elapsed, available_memory, free_disk, rss):
        self.validate()
        require(type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed >= 0, "invalid elapsed time")
        require(all(type(v) is int and v >= 0 for v in (available_memory, free_disk, rss)), "invalid resource sample")
        if elapsed >= self.seconds:
            raise ResourceLimit("registered phase wall-time cap")
        if available_memory < self.minimum_free_memory_bytes:
            raise ResourceLimit("registered free-memory reserve")
        if free_disk < self.minimum_free_disk_bytes:
            raise ResourceLimit("registered free-disk reserve")
        if rss > self.maximum_rss_bytes:
            raise ResourceLimit("registered worker resident-memory cap")


def phase_envelope(settings, phase):
    require(phase in ("selection", "construction", "observation_audit"), "unregistered phase")
    policy = hard_resource_policy()
    return Envelope(seconds=settings["execution"][phase + "_seconds_per_rung"],
        minimum_free_memory_bytes=policy["minimum_free_memory_mib"]*MIB,
        minimum_free_disk_bytes=policy["minimum_free_disk_mib"]*MIB,
        maximum_rss_bytes=policy["maximum_worker_rss_mib"]*MIB,
        poll_seconds=policy["supervisor_poll_seconds"], termination_grace_seconds=policy["termination_grace_seconds"])


def terminate_owned(worker, pidfd, grace):
    """Signal only the kernel-pinned process created by this supervisor."""
    if worker.poll() is not None:
        return
    try:
        signal.pidfd_send_signal(pidfd, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        worker.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        try:
            signal.pidfd_send_signal(pidfd, signal.SIGKILL)
        except ProcessLookupError:
            pass
        worker.wait(timeout=grace)


def run_once(folder, command, *, working_directory, envelope, bindings, environment=None,
             guard, sync=sync_directory):
    """Supervise a single non-forking worker in a fresh reserved directory.

Scientific completion is a separate authenticated artifact; exit zero is
not a certificate. Caller must enforce the declared no-child-process worker
policy. The OS may delay stopping an uninterruptible process; a failed stop
is reported, not silently treated as a completed timeout.
"""
    require(os.name == "posix" and hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal"),
            "Linux pidfd supervision required")
    envelope.validate()
    require(type(command) is list and command and all(type(v) is str and v for v in command), "invalid worker command")
    require(Path(command[0]).is_absolute() and Path(command[0]).is_file(), "absolute worker interpreter required")
    cwd = Path(working_directory)
    require(cwd.is_dir() and not cwd.is_symlink(), "invalid worker directory")
    folder = Path(folder)
    require(folder.parent.is_dir() and not folder.parent.is_symlink(), "invalid supervisor parent")
    require(type(bindings) is dict and set(bindings) ==
            {"protocol_sha256", "source_manifest_sha256", "phase_input_sha256"} and
            all(is_hash(value) for value in bindings.values()), "authenticated launch bindings required")
    require(callable(guard), "source/overall-resource guard required")
    guard()
    available, _ = linux_memory_snapshot(os.getpid())
    envelope.check(elapsed=0.0, available_memory=available, free_disk=shutil.disk_usage(cwd).free, rss=0)
    sync(folder.parent)
    folder.mkdir(exist_ok=False)
    sync(folder.parent)
    start = time.monotonic()
    publish_new(folder / "started.json", {"command_sha256": hashlib.sha256(encode(command)).hexdigest(),
        "bindings": bindings, "limits": envelope.__dict__, "automatic_retry_or_resume": False,
        "worker_subprocesses_permitted": False}, sync=sync)
    worker, pidfd, reason = None, None, None
    try:
        with (folder / "worker.log").open("xb") as log:
            worker = subprocess.Popen(command, cwd=cwd, env=environment, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            # This descriptor pins process identity even after PID reuse.
            pidfd = os.pidfd_open(worker.pid)
            publish_new(folder / "worker_started.json", {"pid": worker.pid, "pidfd_identity_pinned": True}, sync=sync)
            while worker.poll() is None:
                guard()
                try:
                    available, rss = linux_memory_snapshot(worker.pid)
                except FileNotFoundError:
                    if worker.poll() is not None:
                        break
                    raise
                envelope.check(elapsed=time.monotonic()-start, available_memory=available,
                    free_disk=shutil.disk_usage(cwd).free, rss=rss)
                time.sleep(envelope.poll_seconds)
            guard()
            if time.monotonic()-start >= envelope.seconds:
                raise ResourceLimit("registered phase wall-time cap")
            log.flush()
            os.fsync(log.fileno())
            sync(folder)
        result = {"status": "worker_completed" if worker.returncode == 0 else "worker_failed",
                  "returncode": worker.returncode, "elapsed_seconds": time.monotonic()-start,
                  "scientific_completion_authenticated": False, "automatic_retry_or_resume": False}
        publish_new(folder / "terminal.json", result, sync=sync)
        return result
    except BaseException as error:
        reason = {"status": "worker_interrupted", "error_type": type(error).__name__,
                  "error": str(error), "elapsed_seconds": time.monotonic()-start,
                  "automatic_retry_or_resume": False}
        try:
            if worker is not None and worker.poll() is None:
                if pidfd is None:
                    # An unreaped child cannot have its PID reused. This
                    # fallback is only for failure to acquire the first pidfd.
                    worker.terminate()
                    try:
                        worker.wait(timeout=envelope.termination_grace_seconds)
                    except subprocess.TimeoutExpired:
                        worker.kill()
                        worker.wait(timeout=envelope.termination_grace_seconds)
                else:
                    terminate_owned(worker, pidfd, envelope.termination_grace_seconds)
        except BaseException as stop_error:
            reason["stop_error"] = {"type": type(stop_error).__name__, "message": str(stop_error)}
        try:
            publish_new(folder / "interrupted.json", reason, sync=sync)
        except BaseException:
            pass  # The original reservation remains; completion must not be inferred.
        raise
    finally:
        if pidfd is not None:
            os.close(pidfd)
