from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import hashlib
import math
import os
import re
import shlex
import signal
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Callable

from .api import RunpodAPI, RunpodAPIError
from .models import Artifact, JobResult, JobSpec
from .state import LeaseStore


class RetainedPodUnavailable(RunpodAPIError):
    """A retained Pod stayed stopped after every retryable start attempt."""


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

    def _pod_body(self, spec: JobSpec, name: str, public_key: str | None = None) -> dict:
        public_key = public_key or self._public_key()
        body = {
            "name": name,
            "imageName": spec.image,
            "gpuTypeIds": list(spec.selected_gpus),
            "gpuTypePriority": "custom",
            "cloudType": spec.cloud,
            "gpuCount": 1,
            "containerDiskInGb": spec.container_disk_gb,
            "volumeInGb": spec.workspace_gb if spec.retest_window_minutes else 0,
            "ports": ["22/tcp"],
            "supportPublicIp": True,
            "interruptible": False,
            # Supplying it explicitly makes SSH bootstrap independent of whether
            # account-level key injection races or differs across Pod APIs.
            "env": {**spec.env, "PUBLIC_KEY": public_key, "SSH_PUBLIC_KEY": public_key},
        }
        if spec.retest_window_minutes:
            body["volumeMountPath"] = "/workspace"
        return body

    @staticmethod
    def _cost(pod: dict) -> float | None:
        raw = pod.get("adjustedCostPerHr", pod.get("costPerHr"))
        try:
            value = float(raw)
            return value if math.isfinite(value) and value >= 0 else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _require_bounded_cost(cost: float | None, limit: float | None) -> None:
        if limit is None:
            return
        if cost is None:
            raise RuntimeError("Runpod did not report an hourly cost; refusing bounded-cost job")
        if cost > limit:
            raise RuntimeError(f"Pod costs ${cost:.3f}/hr, above ${limit:.3f}/hr limit")

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
command -v tar
command -v runpodctl
command -v nvidia-smi
nvidia-smi -L
"""

    @staticmethod
    def _job_script(spec: JobSpec, persistent_workspace: bool = False) -> str:
        job_root = "/workspace/runpod-guard-job" if persistent_workspace else "/root/runpod-guard-job"
        cache_root = "/workspace/runpod-guard-cache" if persistent_workspace else "/root/runpod-guard-cache"
        if spec.source_dir is not None:
            checkout = f"""rm -rf {job_root}
mkdir -p {job_root}
tar -xf /tmp/runpod-guard-source.tar -C {job_root}
rm -f /tmp/runpod-guard-source.tar
"""
        else:
            repo = shlex.quote(spec.repo or "")
            ref = shlex.quote(spec.ref)
            checkout = f"""rm -rf {job_root}
mkdir -p {job_root}
git -C {job_root} init -q
git -C {job_root} remote add origin {repo}
cd {job_root}
git -c credential.helper= fetch --depth 1 --no-tags origin {ref}
git checkout --detach --force FETCH_HEAD
"""
        return f"""set -euo pipefail
{checkout}
cd {job_root}
mkdir -p .runpod-guard
mkdir -p {cache_root}
export RUNPOD_GUARD_CACHE={cache_root}
exec > >(tee /tmp/runpod-guard-job.log) 2>&1
trap 'cp /tmp/runpod-guard-job.log .runpod-guard/job.log 2>/dev/null || true' EXIT
{spec.setup}
set +e
bash -o pipefail -c {shlex.quote(spec.command)}
job_status=$?
set -e
exit "$job_status"
"""

    @staticmethod
    def _source_archive(spec: JobSpec) -> Path:
        if spec.source_dir is None:
            raise ValueError("source_dir is required")
        source = spec.source_dir.expanduser().resolve()
        handle = tempfile.NamedTemporaryFile(prefix="runpod-guard-", suffix=".tar", delete=False)
        archive = Path(handle.name)
        handle.close()
        try:
            subprocess.run(
                ["git", "-C", str(source), "archive", "--format=tar",
                 f"--output={archive}", spec.ref],
                check=True, stdin=subprocess.DEVNULL, timeout=120,
            )
            archive.chmod(0o600)
            return archive
        except BaseException:
            archive.unlink(missing_ok=True)
            raise

    def _upload_source(self, ip: str, port: int, archive: Path, timeout: int) -> bool:
        command = [
            "scp", "-P", str(port), "-i", str(self.ssh_key),
            "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
            "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
            str(archive), f"root@{ip}:/tmp/runpod-guard-source.tar",
        ]
        try:
            return subprocess.run(command, timeout=timeout).returncode == 0
        except subprocess.TimeoutExpired:
            return False

    def _fetch(self, ip: str, port: int, artifact: Artifact, timeout: int = 120,
               persistent_workspace: bool = False) -> bool:
        local_dir = artifact.local_dir.expanduser().resolve()
        local_dir.mkdir(parents=True, exist_ok=True)
        job_root = "/workspace/runpod-guard-job" if persistent_workspace else "/root/runpod-guard-job"
        remote_path = job_root + "/" + artifact.remote
        validate = f"""set -eu
candidate=$(realpath -e -- {shlex.quote(remote_path)})
case "$candidate" in {job_root}/*) exit 0;; *) exit 1;; esac
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

    def _retained_lease(self, pod_id: str) -> dict:
        now = datetime.now(timezone.utc)
        for lease in self.leases.all():
            if lease.get("pod_id") != pod_id or lease.get("api_identity") != self.api.identity:
                continue
            try:
                expires = datetime.fromisoformat(lease["expires_at"])
            except (KeyError, TypeError, ValueError):
                break
            if (expires.tzinfo is not None and lease.get("state") == "paused-for-retest" and
                    expires > now):
                return lease
            break
        raise RuntimeError(f"{pod_id} is not an unexpired Pod retained by this API identity")

    def extend_retest(self, pod_id: str, minutes: int) -> str:
        """Renew a stopped, guard-owned Pod lease without starting its GPU."""
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", pod_id):
            raise ValueError("pod_id has an unexpected format")
        if not 1 <= minutes <= 24 * 60:
            raise ValueError("minutes must be between 1 and 1440")
        with self.leases.claim(pod_id):
            lease = next((row for row in self.leases.all()
                          if row.get("pod_id") == pod_id), None)
            if (lease is None or lease.get("api_identity") != self.api.identity or
                    lease.get("state") != "paused-for-retest"):
                raise RuntimeError(
                    f"{pod_id} is not a stopped Pod retained by this API identity"
                )
            pod = self.api.get_pod(pod_id)
            if pod.get("name") != lease.get("name"):
                raise RuntimeError("retained Pod name no longer matches its local lease")
            status = pod.get("desiredStatus") or pod.get("status")
            if status not in {"EXITED", "STOPPED"}:
                raise RuntimeError("retained Pod is not stopped; refusing to extend its lease")
            expires = datetime.now(timezone.utc) + timedelta(minutes=minutes)
            renewed = {key: value for key, value in lease.items() if key != "_path"}
            renewed["expires_at"] = expires.isoformat()
            self.leases.put(pod_id, renewed)
            return expires.isoformat()

    def execute(self, spec: JobSpec) -> JobResult:
        if spec.reuse_pod_id:
            started = time.monotonic()
            with self.leases.claim(spec.reuse_pod_id):
                try:
                    return self._execute(spec)
                except RetainedPodUnavailable as error:
                    if not spec.fallback_fresh_on_reuse_unavailable:
                        raise
                    leases = [lease for lease in self.leases.all()
                              if lease.get("pod_id") == spec.reuse_pod_id]
                    if len(leases) != 1 or leases[0].get("state") != "paused-for-retest":
                        raise RuntimeError(
                            "retained Pod was unavailable but was not confirmed stopped; "
                            "refusing a fresh fallback"
                        ) from error
            elapsed_minutes = math.ceil((time.monotonic() - started) / 60)
            fallback_minutes = spec.max_minutes - elapsed_minutes
            if fallback_minutes < 1:
                raise TimeoutError("no job budget remains for a fresh fallback")
            self._log("retained Pod remained unavailable; allocating a fresh fallback")
            return self._execute(replace(
                spec, reuse_pod_id=None, max_minutes=fallback_minutes
            ))
        return self._execute(spec)

    def _start_retained_pod(self, pod_id: str, spec: JobSpec,
                            remaining: Callable[[], int]) -> None:
        for attempt in range(1, spec.reuse_start_attempts + 1):
            # A previous delay may have consumed the rest of the job budget.
            remaining()
            try:
                self.api.start_pod(pod_id)
                return
            except RunpodAPIError as error:
                # POST can succeed at Runpod even when its response is lost. Reconcile
                # provider state before deciding whether to retry or fail.
                try:
                    pod = self.api.get_pod(pod_id)
                except RunpodAPIError as status_error:
                    if not status_error.retryable:
                        raise status_error from error
                else:
                    status = pod.get("desiredStatus") or pod.get("status")
                    if status == "RUNNING":
                        self._log("retained Pod start was confirmed after a failed response")
                        return
                if not error.retryable:
                    raise
                if attempt == spec.reuse_start_attempts:
                    raise RetainedPodUnavailable(str(error), error.status_code) from error
                delay = min(spec.reuse_start_delay_seconds, remaining())
                self._log(
                    f"retained Pod start attempt {attempt}/{spec.reuse_start_attempts} failed; "
                    f"retrying in {delay} seconds"
                )
                time.sleep(delay)

    def _execute(self, spec: JobSpec) -> JobResult:
        started = time.monotonic()
        created = datetime.now(timezone.utc)
        public_key = self._public_key()
        pod_configuration = {
            **spec.pod_configuration,
            "ssh_public_key_sha256": hashlib.sha256(public_key.encode()).hexdigest(),
        }
        # max_minutes bounds provisioning, setup, and execution. An explicit retest
        # window is recorded only after the Pod has completed and is being stopped.
        expires = created + timedelta(minutes=spec.max_minutes)
        monotonic_deadline = started + spec.max_minutes * 60
        pending_id: str | None = None
        retained_lease: dict | None = None
        if spec.reuse_pod_id:
            retained_lease = self._retained_lease(spec.reuse_pod_id)
            retained_configuration = retained_lease.get("pod_configuration")
            legacy_configuration = {
                key: value for key, value in pod_configuration.items()
                if key != "ssh_public_key_sha256"
            }
            legacy_ssh_binding = retained_configuration == legacy_configuration
            if retained_configuration != pod_configuration and not legacy_ssh_binding:
                raise RuntimeError("retained Pod configuration differs from the requested job")
            pod = self.api.get_pod(spec.reuse_pod_id)
            name = str(retained_lease["name"])
            if pod.get("name") != name:
                raise RuntimeError("retained Pod name no longer matches its local lease")
            status = pod.get("desiredStatus") or pod.get("status")
            if status not in {"EXITED", "STOPPED"}:
                if not self.api.stop_and_confirm(spec.reuse_pod_id):
                    if self.api.delete_and_confirm(spec.reuse_pod_id):
                        self.leases.remove(spec.reuse_pod_id)
                    raise RuntimeError("retained Pod could not be confirmed stopped before reuse")
            if legacy_ssh_binding:
                environment = pod.get("env")
                if (not isinstance(environment, dict) or
                        environment.get("PUBLIC_KEY") != public_key or
                        environment.get("SSH_PUBLIC_KEY") != public_key):
                    raise RuntimeError(
                        "legacy retained Pod does not match the current SSH public key"
                    )
                retained_lease = {
                    **{key: value for key, value in retained_lease.items() if key != "_path"},
                    "pod_configuration": pod_configuration,
                }
                self.leases.put(spec.reuse_pod_id, retained_lease)
        else:
            name = f"rpg-{spec.name[:30]}-{uuid.uuid4().hex[:8]}"
            # Record intent before POST. If creation succeeds but its response or the
            # caller disappears, the reaper can resolve the unique name later.
            pending_id = f"pending-{uuid.uuid4().hex}"
            self.leases.put(pending_id, {
                "pod_id": None, "name": name, "created_at": created.isoformat(),
                "expires_at": expires.isoformat(), "max_minutes": spec.max_minutes,
                "api_identity": self.api.identity, "state": "creating",
            })
            try:
                pod = self.api.create_pod(self._pod_body(spec, name, public_key))
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
        paused = False
        retest_expires_at: str | None = None
        returncode: int | None = None
        timed_out = False
        artifacts_ok = True
        job_started = False
        job_retestable = False
        persistent_workspace = bool(spec.retest_window_minutes or spec.reuse_pod_id)
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
            if threading.current_thread() is threading.main_thread():
                for signum in (signal.SIGINT, signal.SIGTERM):
                    previous_handlers[signum] = signal.signal(signum, interrupted)
            self.leases.put(pod_id, {
                "pod_id": pod_id, "name": name, "created_at": created.isoformat(),
                "expires_at": expires.isoformat(), "max_minutes": spec.max_minutes,
                "api_identity": self.api.identity, "state": "running",
                "pod_configuration": pod_configuration,
            })
            if pending_id:
                self.leases.remove(pending_id)
                self._log(f"created {pod_id} ({name}); hard deadline {expires.isoformat()}")
            else:
                self._start_retained_pod(pod_id, spec, remaining)
                self._log(f"restarting retained Pod {pod_id}; hard deadline {expires.isoformat()}")
            if retained_lease is None:
                self._require_bounded_cost(cost, spec.max_cost_per_hour)
            ip, port = self._wait_for_address(pod_id, min(900, remaining()))
            if retained_lease is not None:
                # A stopped Pod may report only storage cost. Check refreshed running cost.
                cost = self._cost(self.api.get_pod(pod_id))
                self._require_bounded_cost(cost, spec.max_cost_per_hour)
            self._wait_for_ssh(ip, port, min(300, remaining()))
            # Five minutes beyond the local deadline is enough for ordinary local
            # teardown, but still bounds spend if this process or host disappears.
            watchdog_seconds = remaining() + 300
            if self._ssh(ip, port, self._watchdog_script(pod_id, watchdog_seconds), 60) != 0:
                raise RuntimeError("could not arm the independent on-Pod termination watchdog")
            self._log("independent on-Pod watchdog armed")
            if self._ssh(ip, port, self._bootstrap_script(), min(60, remaining())) != 0:
                raise RuntimeError("Pod is missing git, runpodctl, or a visible NVIDIA GPU")
            if spec.source_dir is not None:
                archive = self._source_archive(spec)
                try:
                    if not self._upload_source(ip, port, archive, min(120, remaining())):
                        raise RuntimeError("could not upload the local source archive")
                finally:
                    archive.unlink(missing_ok=True)
            try:
                reserve = min(120, max(30, spec.max_minutes * 6))
                job_budget = remaining() - reserve
                if job_budget <= 0:
                    raise TimeoutError("no execution budget remains after provisioning")
                job_started = True
                returncode = self._ssh(
                    ip, port, self._job_script(spec, persistent_workspace), job_budget
                )
                job_retestable = True
            except (subprocess.TimeoutExpired, TimeoutError):
                timed_out = True
                job_retestable = job_started
                self._log("job reached its local deadline")
            finally:
                if ip is not None and port is not None:
                    for artifact in spec.artifacts:
                        try:
                            fetch_budget = min(120, remaining())
                        except TimeoutError:
                            fetch_budget = 0
                        fetched = fetch_budget > 0 and self._fetch(
                            ip, port, artifact, fetch_budget, persistent_workspace
                        )
                        if artifact.required and not fetched:
                            artifacts_ok = False
        finally:
            # Once teardown begins, a second Ctrl-C/SIGTERM must not interrupt it.
            if previous_handlers and threading.current_thread() is threading.main_thread():
                for signum in (signal.SIGINT, signal.SIGTERM):
                    signal.signal(signum, signal.SIG_IGN)
            try:
                if job_retestable and spec.retest_window_minutes:
                    retest_expires = datetime.now(timezone.utc) + timedelta(
                        minutes=spec.retest_window_minutes
                    )
                    try:
                        self._log(f"pausing {pod_id} for a bounded retest window")
                        if self.api.stop_and_confirm(pod_id):
                            self.leases.put(pod_id, {
                                "pod_id": pod_id, "name": name,
                                "created_at": created.isoformat(),
                                "expires_at": retest_expires.isoformat(),
                                "max_minutes": spec.max_minutes,
                                "api_identity": self.api.identity,
                                "state": "paused-for-retest",
                                "pod_configuration": pod_configuration,
                            })
                            paused = True
                    except Exception as error:
                        self._log(f"pause failed for {pod_id}: {error}")
                        paused = False
                    if paused:
                        retest_expires_at = retest_expires.isoformat()
                        self._log(
                            f"confirmed {pod_id} is stopped; reaper deadline {retest_expires_at}"
                        )
                elif retained_lease is not None and not job_started:
                    # Reacquiring a GPU can fail normally. Preserve the existing
                    # workspace/window if the Pod can still be proven stopped.
                    original_expires = datetime.fromisoformat(retained_lease["expires_at"])
                    try:
                        if (original_expires > datetime.now(timezone.utc) and
                                self.api.stop_and_confirm(pod_id)):
                            original = {
                                key: value for key, value in retained_lease.items()
                                if key != "_path"
                            }
                            self.leases.put(pod_id, original)
                            paused = True
                            retest_expires_at = original_expires.isoformat()
                            self._log(
                                f"restart failed; {pod_id} remains stopped until "
                                f"{retest_expires_at}"
                            )
                    except Exception as error:
                        self._log(f"could not restore stopped Pod {pod_id}: {error}")
                        paused = False
                if not paused:
                    self._log(f"terminating {pod_id}")
                    try:
                        terminated = self.api.delete_and_confirm(pod_id)
                    except Exception as error:
                        self._log(f"delete failed for {pod_id}: {error}")
                        terminated = False
                if terminated:
                    try:
                        self.leases.remove(pod_id)
                        if pending_id:
                            self.leases.remove(pending_id)
                    except Exception as error:
                        self._log(f"Pod is gone but lease cleanup failed for {pod_id}: {error}")
                    self._log(f"confirmed {pod_id} is gone")
                elif not paused:
                    self._log(f"CRITICAL: could not confirm {pod_id} is gone; run `runpod-guard reap`")
            finally:
                for signum, handler in previous_handlers.items():
                    signal.signal(signum, handler)

        return JobResult(
            pod_id=pod_id, returncode=returncode, timed_out=timed_out,
            artifacts_ok=artifacts_ok, terminated=terminated,
            elapsed_seconds=time.monotonic() - started, cost_per_hour=cost,
            paused=paused, retest_expires_at=retest_expires_at,
            requested=spec.receipt,
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

        def delete_candidate(pod_id: str) -> None:
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

        for pod_id in sorted(candidates - {None}):
            if pod_id not in pods:
                if pod_id not in first_pods:
                    remove_lease(pod_id)
                continue
            if all_managed:
                delete_candidate(pod_id)
                continue
            claim = getattr(self.leases, "claim", None)
            try:
                with claim(pod_id) if claim else nullcontext():
                    # A reuse caller may have renewed a lease after the first
                    # expiry snapshot but before this lock was acquired.
                    current_expired = self.leases.expired()
                    pod_name = pods[pod_id].get("name")
                    still_expired = any(
                        lease.get("api_identity") == self.api.identity and
                        (lease.get("pod_id") == pod_id or lease.get("name") == pod_name)
                        for lease in current_expired
                    )
                    if still_expired:
                        delete_candidate(pod_id)
            except RuntimeError as error:
                if "already claimed by another retest" not in str(error):
                    self.reap_failures.append(f"could not claim {pod_id}: {error}")
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
