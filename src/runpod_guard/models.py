from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
from typing import Any
from urllib.parse import urlsplit


GPU_PROFILES: dict[str, tuple[str, ...]] = {
    # Ordered fallbacks let the scheduler choose an available card in the class.
    "small": ("NVIDIA L4", "NVIDIA RTX A5000", "NVIDIA GeForce RTX 3090"),
    "fast": ("NVIDIA GeForce RTX 4090", "NVIDIA GeForce RTX 5090"),
    "large": ("NVIDIA RTX A6000", "NVIDIA A40", "NVIDIA L40S"),
    "xlarge": ("NVIDIA A100 80GB PCIe", "NVIDIA A100-SXM4-80GB"),
}


def _safe_remote_path(value: str) -> str:
    path = PurePosixPath(value)
    # SCP's remote operand is interpreted by a remote shell. A narrow portable
    # alphabet avoids turning an artifact name into command execution.
    if (not value or path.is_absolute() or ".." in path.parts or
            not re.fullmatch(r"[A-Za-z0-9._/-]+", value)):
        raise ValueError(f"remote path must stay inside the checkout: {value!r}")
    return value


@dataclass(frozen=True)
class Artifact:
    remote: str
    local_dir: Path = Path("runpod-output")
    required: bool = True

    def __post_init__(self) -> None:
        _safe_remote_path(self.remote)


@dataclass(frozen=True)
class JobSpec:
    repo: str | None
    ref: str
    command: str
    setup: str = ""
    source_dir: Path | None = None
    profile: str = "small"
    gpu_types: tuple[str, ...] = ()
    cloud: str = "SECURE"
    max_minutes: int = 60
    max_cost_per_hour: float | None = 1.0
    image: str = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
    container_disk_gb: int = 40
    workspace_gb: int = 20
    retest_window_minutes: int = 0
    reuse_pod_id: str | None = None
    reuse_start_attempts: int = 4
    reuse_start_delay_seconds: int = 20
    fallback_fresh_on_reuse_unavailable: bool = False
    artifacts: tuple[Artifact, ...] = ()
    name: str = "job"
    env: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.ref or not self.command:
            raise ValueError("ref and command are required")
        if bool(self.repo) == bool(self.source_dir):
            raise ValueError("exactly one of repo or source_dir is required")
        if self.repo:
            parsed_repo = urlsplit(self.repo)
            if (parsed_repo.scheme != "https" or not parsed_repo.hostname or
                    parsed_repo.username or parsed_repo.password or parsed_repo.query or
                    parsed_repo.fragment):
                raise ValueError("repo must be a public HTTPS URL without embedded credentials")
        if self.source_dir is not None:
            source = self.source_dir.expanduser()
            if not source.is_dir() or not (source / ".git").exists():
                raise ValueError("source_dir must be a local Git working tree")
        if (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", self.ref) or
                ".." in self.ref or "@{" in self.ref):
            raise ValueError("ref contains syntax Git could interpret unsafely")
        if self.profile not in GPU_PROFILES:
            raise ValueError(f"unknown profile {self.profile!r}; choose {', '.join(GPU_PROFILES)}")
        if self.cloud not in {"COMMUNITY", "SECURE"}:
            raise ValueError("cloud must be COMMUNITY or SECURE")
        if not 1 <= self.max_minutes <= 24 * 60:
            raise ValueError("max_minutes must be between 1 and 1440")
        if not 10 <= self.container_disk_gb <= 1000:
            raise ValueError("container_disk_gb must be between 10 and 1000")
        if not 1 <= self.workspace_gb <= 1000:
            raise ValueError("workspace_gb must be between 1 and 1000")
        if not 0 <= self.retest_window_minutes <= 24 * 60:
            raise ValueError("retest_window_minutes must be between 0 and 1440")
        if (self.reuse_pod_id is not None and
                not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", self.reuse_pod_id)):
            raise ValueError("reuse_pod_id has an unexpected format")
        if not 1 <= self.reuse_start_attempts <= 10:
            raise ValueError("reuse_start_attempts must be between 1 and 10")
        if not 1 <= self.reuse_start_delay_seconds <= 300:
            raise ValueError("reuse_start_delay_seconds must be between 1 and 300")
        if (self.max_cost_per_hour is not None and
                (not math.isfinite(self.max_cost_per_hour) or self.max_cost_per_hour <= 0)):
            raise ValueError("max_cost_per_hour must be positive")
        forbidden = {"PUBLIC_KEY", "SSH_PUBLIC_KEY", "RUNPOD_API_KEY", "RUNPOD_POD_ID",
                     "RUNPOD_API_URL", "RUNPOD_BASE_URL"}.intersection(self.env)
        if forbidden:
            raise ValueError(f"refusing to pass privileged environment variables: {sorted(forbidden)}")

    @property
    def selected_gpus(self) -> tuple[str, ...]:
        return self.gpu_types or GPU_PROFILES[self.profile]

    @property
    def pod_configuration(self) -> dict[str, Any]:
        """Immutable provisioned fields that a retained Pod cannot change on restart."""
        environment = json.dumps(self.env, sort_keys=True, separators=(",", ":")).encode()
        return {
            "gpu_types": list(self.selected_gpus),
            "cloud": self.cloud,
            "image": self.image,
            "container_disk_gb": self.container_disk_gb,
            "workspace_gb": self.workspace_gb,
            # Detect a changed environment without retaining its values in local state.
            "environment_sha256": hashlib.sha256(environment).hexdigest(),
        }

    @property
    def receipt(self) -> dict[str, Any]:
        """Non-secret requested configuration suitable for a result receipt."""
        configuration = {
            key: value for key, value in self.pod_configuration.items()
            if key != "environment_sha256"
        }
        return {
            "ref": self.ref,
            "source": "local-archive" if self.source_dir is not None else "public-repository",
            "profile": self.profile,
            **configuration,
            "max_minutes": self.max_minutes,
            "max_cost_per_hour": self.max_cost_per_hour,
            "retest_window_minutes": self.retest_window_minutes,
            "reuse_requested": self.reuse_pod_id is not None,
            "reuse_start_attempts": self.reuse_start_attempts,
            "reuse_start_delay_seconds": self.reuse_start_delay_seconds,
            "fallback_fresh_on_reuse_unavailable": self.fallback_fresh_on_reuse_unavailable,
        }


@dataclass(frozen=True)
class JobResult:
    pod_id: str
    returncode: int | None
    timed_out: bool
    artifacts_ok: bool
    terminated: bool
    elapsed_seconds: float
    cost_per_hour: float | None = None
    paused: bool = False
    retest_expires_at: str | None = None
    requested: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return (self.returncode == 0 and not self.timed_out and self.artifacts_ok and
                (self.terminated or self.paused))

    def to_dict(self) -> dict[str, Any]:
        return {
            "pod_id": self.pod_id,
            "returncode": self.returncode,
            "timed_out": self.timed_out,
            "artifacts_ok": self.artifacts_ok,
            "terminated": self.terminated,
            "paused": self.paused,
            "retest_expires_at": self.retest_expires_at,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "cost_per_hour": self.cost_per_hour,
            "requested": self.requested,
            "ok": self.ok,
        }
