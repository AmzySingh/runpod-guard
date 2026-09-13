from __future__ import annotations

import os
import math
import shlex
import signal
import subprocess
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from .api import RunpodAPI, RunpodAPIError
from .models import Artifact, JobResult, JobSpec
from .state import LeaseStore


class RunpodRunner:
    def __init__(self, api_key: str | None = None, ssh_key: Path | None = None,
                 api: RunpodAPI | None = None, leases: LeaseStore | None = None,
                 log: Callable[[str], None] = print) -> None:
        key = api_key or os.environ.get("RUNPOD_API_KEY", "")
        self.api = api or RunpodAPI(key)
        self.ssh_key = (ssh_key or Path.home() / ".ssh" / "runpod_engram").expanduser()
        self.leases = leases or LeaseStore()
        self.log = log
        self.reap_failures: list[str] = []

    def _log(self, message: str) -> None:
        """Logging must never get between an allocated Pod and its teardown."""
        try:
            self.log(message)
        except Exception:
            pass

    def preflight(self) -> int:
        if not self.ssh_key.is_file():
            raise RuntimeError(f"SSH private key does not exist: {self.ssh_key}")
        self._public_key()
        return len(self.api.list_pods())

    def _public_key(self) -> str:
        path = Path(str(self.ssh_key) + ".pub")
        if not path.is_file():
            raise RuntimeError(f"SSH public key does not exist: {path}")
        value = path.read_text(encoding="utf-8").strip()
        if not value.startswith(("ssh-ed25519 ", "ssh-rsa ", "ecdsa-sha2-")):
            raise RuntimeError(f"SSH public key has an unexpected format: {path}")
        try:
            derived = subprocess.run(
                ["ssh-keygen", "-y", "-f", str(self.ssh_key)], check=True,
                text=True, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, timeout=10,
            ).stdout.strip()
        except (FileNotFoundError, subprocess.CalledProcessError,
                subprocess.TimeoutExpired) as error:
            raise RuntimeError("could not derive the public key from the SSH private key") from error
        if derived.split()[:2] != value.split()[:2]:
            raise RuntimeError(f"SSH public key does not match private key: {path}")
        return value

    def _pod_body(self, spec: JobSpec, name: str) -> dict:
        public_key = self._public_key()
        return {
            "name": name,
            "imageName": spec.image,
            "gpuTypeIds": list(spec.selected_gpus),
            "gpuTypePriority": "custom",
            "cloudType": spec.cloud,
            "gpuCount": 1,
            "containerDiskInGb": spec.container_disk_gb,
            "volumeInGb": 0,
            "ports": ["22/tcp"],
            "supportPublicIp": True,
            "interruptible": False,
            # Supplying it explicitly makes SSH bootstrap independent of whether
            # account-level key injection races or differs across Pod APIs.
            "env": {**spec.env, "PUBLIC_KEY": public_key, "SSH_PUBLIC_KEY": public_key},
        }

    @staticmethod
    def _cost(pod: dict) -> float | None:
        raw = pod.get("adjustedCostPerHr", pod.get("costPerHr"))
        try:
            value = float(raw)
            return value if math.isfinite(value) and value >= 0 else None
        except (TypeError, ValueError):
            return None

    def _wait_for_address(self, pod_id: str, timeout: int = 900) -> tuple[str, int]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            pod = self.api.get_pod(pod_id)
            mappings = pod.get("portMappings") or {}
            ip = pod.get("publicIp")
            port = mappings.get("22") or mappings.get(22)
            status = pod.get("desiredStatus") or pod.get("status")
            if status == "RUNNING" and ip and port:
                return str(ip), int(port)
            time.sleep(8)
        raise TimeoutError("Pod did not publish an SSH address within 15 minutes")

    def _ssh_command(self, ip: str, port: int) -> list[str]:
        return [
            "ssh", "-p", str(port), "-i", str(self.ssh_key),
            "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
            "-o", "ForwardAgent=no", "-o", "IdentitiesOnly=yes",
            "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ServerAliveInterval=20", "-o", "ServerAliveCountMax=3",
            f"root@{ip}", "bash -s",
        ]

    def _ssh(self, ip: str, port: int, script: str, timeout: int) -> int:
        process = subprocess.run(self._ssh_command(ip, port), input=script, text=True,
                                 timeout=timeout)
        return process.returncode

    def _wait_for_ssh(self, ip: str, port: int, timeout: int = 300) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                if self._ssh(ip, port, "true\n", 30) == 0:
                    return
            except subprocess.TimeoutExpired:
                pass
            time.sleep(8)
        raise TimeoutError("Pod published a port but SSH did not become ready")

    @staticmethod
    def _watchdog_script(pod_id: str, seconds: int) -> str:
        # It is intentionally detached from SSH. It survives a dead caller/VPS and
        # uses the Pod-scoped credentials that Runpod injects into every Pod.
        command = ("unset RUNPOD_API_URL RUNPOD_BASE_URL; "
                   "export PATH=/usr/local/bin:/usr/bin:/bin; "
                   f"sleep {seconds}; while true; do "
                   f"timeout 30s runpodctl pod delete {shlex.quote(pod_id)} && exit 0; "
                   "sleep 15; done")
        return (
            "set -eu\n"
            "command -v runpodctl >/dev/null\n"
            "command -v timeout >/dev/null\n"
            f"nohup bash -c {shlex.quote(command)} "
            ">/tmp/runpod-guard-watchdog.log 2>&1 </dev/null &\n"
            "echo $! >/tmp/runpod-guard-watchdog.pid\n"
            "kill -0 $!\n"
        )

    @staticmethod
    def _bootstrap_script() -> str:
        return """set -eu
command -v git
command -v runpodctl
command -v nvidia-smi
nvidia-smi -L
"""

    @staticmethod
    def _job_script(spec: JobSpec) -> str:
        repo = shlex.quote(spec.repo)
        ref = shlex.quote(spec.ref)
        return f"""set -euo pipefail
rm -rf /root/runpod-guard-job
git clone --filter=blob:none {repo} /root/runpod-guard-job
cd /root/runpod-guard-job
git fetch --depth 1 origin {ref}
git checkout --detach FETCH_HEAD
mkdir -p .runpod-guard
exec > >(tee /tmp/runpod-guard-job.log) 2>&1
trap 'cp /tmp/runpod-guard-job.log .runpod-guard/job.log 2>/dev/null || true' EXIT
{spec.setup}
set +e
bash -o pipefail -c {shlex.quote(spec.command)}
job_status=$?
set -e
exit "$job_status"
"""

    def _fetch(self, ip: str, port: int, artifact: Artifact, timeout: int = 120) -> bool:
        local_dir = artifact.local_dir.expanduser().resolve()
        local_dir.mkdir(parents=True, exist_ok=True)
        remote_path = "/root/runpod-guard-job/" + artifact.remote
        validate = f"""set -eu
candidate=$(realpath -e -- {shlex.quote(remote_path)})
case "$candidate" in /root/runpod-guard-job/*) exit 0;; *) exit 1;; esac
"""
        try:
            if self._ssh(ip, port, validate, min(timeout, 30)) != 0:
                return False
        except subprocess.TimeoutExpired:
            return False
        command = [
            "scp", "-P", str(port), "-i", str(self.ssh_key), "-r",
            "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
            "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
            f"root@{ip}:{remote_path}", str(local_dir),
        ]
        try:
            return subprocess.run(command, timeout=timeout).returncode == 0
        except subprocess.TimeoutExpired:
            return False

    def execute(self, spec: JobSpec) -> JobResult:
        started = time.monotonic()
        created = datetime.now(timezone.utc)
        # max_minutes bounds the entire billable lifetime, including provisioning
        # and setup—not merely the user's command.
        expires = created + timedelta(minutes=spec.max_minutes)
        monotonic_deadline = started + spec.max_minutes * 60
        name = f"rpg-{spec.name[:30]}-{uuid.uuid4().hex[:8]}"
        # Record intent before POST. If the create succeeds but its response or the
        # caller disappears, the reaper can resolve the unique name later.
        pending_id = f"pending-{uuid.uuid4().hex}"
        pending_lease = {
            "pod_id": None, "name": name, "created_at": created.isoformat(),
            "expires_at": expires.isoformat(), "max_minutes": spec.max_minutes,
            "api_identity": self.api.identity,
        }
        self.leases.put(pending_id, pending_lease)
        try:
            pod = self.api.create_pod(self._pod_body(spec, name))
            if not isinstance(pod, dict) or not pod.get("id"):
                raise RunpodAPIError("Runpod create returned no Pod ID")
        except BaseException:
            # A network error after Runpod accepted POST is ambiguous. Best effort
            # cleanup now; the pending name lease remains for the scheduled reaper.
            try:
                for candidate in self.api.list_pods():
                    if candidate.get("name") == name and candidate.get("id"):
                        self.api.delete_and_confirm(candidate["id"])
            except RunpodAPIError:
                pass
            raise
        pod_id = pod["id"]
        cost = self._cost(pod)
        terminated = False
        returncode: int | None = None
        timed_out = False
        artifacts_ok = True
        ip: str | None = None
        port: int | None = None
        previous_handlers: dict[int, object] = {}

        def interrupted(signum, _frame):
            raise KeyboardInterrupt(f"received signal {signum}")

        def remaining() -> int:
            seconds = int(monotonic_deadline - time.monotonic())
            if seconds <= 0:
                raise TimeoutError("job exhausted its total Pod-lifetime deadline")
            return seconds

        try:
            self.leases.put(pod_id, {
                "pod_id": pod_id, "name": name, "created_at": created.isoformat(),
                "expires_at": expires.isoformat(), "max_minutes": spec.max_minutes,
                "api_identity": self.api.identity,
            })
            self.leases.remove(pending_id)
            self._log(f"created {pod_id} ({name}); hard deadline {expires.isoformat()}")
            if threading.current_thread() is threading.main_thread():
                for signum in (signal.SIGINT, signal.SIGTERM):
                    previous_handlers[signum] = signal.signal(signum, interrupted)
            if spec.max_cost_per_hour is not None:
                if cost is None:
                    raise RuntimeError("Runpod did not report an hourly cost; refusing bounded-cost job")
                if cost > spec.max_cost_per_hour:
                    raise RuntimeError(
                        f"Pod costs ${cost:.3f}/hr, above ${spec.max_cost_per_hour:.3f}/hr limit"
                    )
            ip, port = self._wait_for_address(pod_id, min(900, remaining()))
            self._wait_for_ssh(ip, port, min(300, remaining()))
            # Five minutes beyond the local deadline is enough for ordinary local
            # teardown, but still bounds spend if this process or host disappears.
            if self._ssh(ip, port, self._watchdog_script(pod_id, remaining() + 300), 60) != 0:
                raise RuntimeError("could not arm the independent on-Pod termination watchdog")
            self._log("independent on-Pod watchdog armed")
            if self._ssh(ip, port, self._bootstrap_script(), min(60, remaining())) != 0:
                raise RuntimeError("Pod is missing git, runpodctl, or a visible NVIDIA GPU")
            try:
                reserve = min(120, max(30, spec.max_minutes * 6))
                job_budget = remaining() - reserve
                if job_budget <= 0:
                    raise TimeoutError("no execution budget remains after provisioning")
                returncode = self._ssh(ip, port, self._job_script(spec), job_budget)
            except (subprocess.TimeoutExpired, TimeoutError):
                timed_out = True
                self._log("job reached its local deadline")
            finally:
                if ip is not None and port is not None:
                    for artifact in spec.artifacts:
                        try:
                            fetch_budget = min(120, remaining())
                        except TimeoutError:
                            fetch_budget = 0
                        fetched = fetch_budget > 0 and self._fetch(
                            ip, port, artifact, fetch_budget
                        )
                        if artifact.required and not fetched:
                            artifacts_ok = False
        finally:
            # Once teardown begins, a second Ctrl-C/SIGTERM must not interrupt it.
            if previous_handlers and threading.current_thread() is threading.main_thread():
                for signum in (signal.SIGINT, signal.SIGTERM):
                    signal.signal(signum, signal.SIG_IGN)
            try:
                self._log(f"terminating {pod_id}")
                try:
                    terminated = self.api.delete_and_confirm(pod_id)
                except Exception as error:
                    self._log(f"delete failed for {pod_id}: {error}")
                    terminated = False
                if terminated:
                    try:
                        self.leases.remove(pod_id)
                        self.leases.remove(pending_id)
                    except Exception as error:
                        self._log(f"Pod is gone but lease cleanup failed for {pod_id}: {error}")
                    self._log(f"confirmed {pod_id} is gone")
                else:
                    self._log(f"CRITICAL: could not confirm {pod_id} is gone; run `runpod-guard reap`")
            finally:
                for signum, handler in previous_handlers.items():
                    signal.signal(signum, handler)

        return JobResult(
            pod_id=pod_id, returncode=returncode, timed_out=timed_out,
            artifacts_ok=artifacts_ok, terminated=terminated,
            elapsed_seconds=time.monotonic() - started, cost_per_hour=cost,
        )

    def reap(self, all_managed: bool = False) -> list[str]:
        self.reap_failures = []
        first_pods = {pod["id"]: pod for pod in self.api.list_pods()}
        expired = [lease for lease in self.leases.expired()
                   if lease.get("api_identity") == self.api.identity]
        self.reap_failures.extend(self.leases.errors)
        time.sleep(2)
        pods = {pod["id"]: pod for pod in self.api.list_pods()}
        candidates = {lease.get("pod_id") for lease in expired if lease.get("pod_id")}
        expired_names = {lease.get("name") for lease in expired if lease.get("name")}
        candidates.update(
            pod_id for pod_id, pod in pods.items()
            if pod_id and pod.get("name") in expired_names
        )
        if all_managed:
            candidates.update(
                pod_id for pod_id, pod in pods.items()
                if pod_id and str(pod.get("name", "")).startswith("rpg-")
            )
        removed = []

        def remove_lease(lease_id: str) -> None:
            try:
                self.leases.remove(lease_id)
            except Exception as error:
                self.reap_failures.append(f"could not remove lease {lease_id}: {error}")

        for pod_id in sorted(candidates - {None}):
            if pod_id not in pods:
                if pod_id not in first_pods:
                    remove_lease(pod_id)
                continue
            delete_errored = False
            try:
                confirmed = self.api.delete_and_confirm(pod_id)
            except Exception as error:
                confirmed = False
                delete_errored = True
                self.reap_failures.append(f"delete failed for {pod_id}: {error}")
            if confirmed:
                remove_lease(pod_id)
                removed.append(pod_id)
            elif not delete_errored:
                self.reap_failures.append(f"could not confirm deletion of {pod_id}")
        # Clear expired pending intents once no current Pod has their unique name.
        first_names = {pod.get("name") for pod in first_pods.values()}
        live_names = {pod.get("name") for pod in pods.values()}
        for lease in expired:
            if (not lease.get("pod_id") and lease.get("name") not in live_names and
                    lease.get("name") not in first_names):
                try:
                    Path(lease["_path"]).unlink(missing_ok=True)
                except Exception as error:
                    self.reap_failures.append(
                        f"could not remove pending lease {lease.get('_path')}: {error}"
                    )
        return removed
