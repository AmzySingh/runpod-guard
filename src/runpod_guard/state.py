from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
from typing import Any


class LeaseStore:
    """One atomic file per Pod survives caller crashes for the external reaper."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or Path.home() / ".local" / "state" / "runpod-guard" / "leases"
        self.errors: list[str] = []

    def put(self, pod_id: str, value: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        destination = self.root / f"{pod_id}.json"
        temporary = self.root / f".{pod_id}.{os.getpid()}.tmp"
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(value, indent=2))
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o600)
        os.replace(temporary, destination)
        directory_fd = os.open(self.root, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

    def remove(self, pod_id: str) -> None:
        try:
            (self.root / f"{pod_id}.json").unlink()
        except FileNotFoundError:
            pass

    @contextmanager
    def claim(self, pod_id: str):
        """Exclusively claim a retained Pod for the duration of one reuse attempt."""
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        path = self.root / f".{pod_id}.lock"
        descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(f"Pod {pod_id} is already claimed by another retest") from error
            yield
        finally:
            os.close(descriptor)

    def all(self) -> list[dict[str, Any]]:
        self.errors = []
        if not self.root.exists():
            return []
        leases = []
        for path in self.root.glob("*.json"):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(value, dict):
                    raise ValueError("lease is not a JSON object")
                value["_path"] = str(path)
                leases.append(value)
            except (OSError, json.JSONDecodeError, ValueError) as error:
                self.errors.append(f"{path}: {error}")
                continue
        return leases

    def expired(self, now: datetime | None = None) -> list[dict[str, Any]]:
        now = now or datetime.now(timezone.utc)
        result = []
        for lease in self.all():
            try:
                expires = datetime.fromisoformat(lease["expires_at"])
                if expires.tzinfo is None:
                    raise ValueError("expiry has no timezone")
            except (KeyError, TypeError, ValueError):
                self.errors.append(f"{lease.get('_path', '<unknown>')}: invalid expires_at")
                continue
            if expires <= now:
                result.append(lease)
        return result
