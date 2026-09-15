from __future__ import annotations

import json
import hashlib
import time
import urllib.error
import urllib.request
from typing import Any, Callable


class RunpodAPIError(RuntimeError):
    """A Runpod REST request failed."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code

    @property
    def retryable(self) -> bool:
        return (self.status_code is None or self.status_code in {408, 429} or
                500 <= self.status_code < 600)


class RunpodPodNotFound(RunpodAPIError):
    """An authoritative Pod lookup reported that the Pod does not exist."""


class RunpodAPI:
    def __init__(self, api_key: str, base_url: str = "https://rest.runpod.io/v1",
                 opener: Callable[..., Any] = urllib.request.urlopen) -> None:
        if not api_key.strip():
            raise ValueError("Runpod API key is empty")
        self.api_key = api_key.strip()
        self.base_url = base_url.rstrip("/")
        self._opener = opener

    @property
    def identity(self) -> str:
        return hashlib.sha256(self.api_key.encode()).hexdigest()[:16]

    def request(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        request = urllib.request.Request(
            f"{self.base_url}{path}", method=method,
            data=json.dumps(body).encode() if body is not None else None,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
                "Content-Type": "application/json",
                # Cloudflare rejects urllib's default signature with error 1010.
                "User-Agent": "runpod-guard/0.1",
            },
        )
        try:
            with self._opener(request, timeout=60) as response:
                raw = response.read().decode()
                return json.loads(raw) if raw.strip() else {}
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")[:800].replace(self.api_key, "[REDACTED]")
            raise RunpodAPIError(
                f"Runpod {method} {path} returned {error.code}: {detail}", error.code
            ) from None
        except urllib.error.URLError as error:
            raise RunpodAPIError(f"Runpod {method} {path} failed: {error.reason}") from error
        except TimeoutError as error:
            raise RunpodAPIError(f"Runpod {method} {path} timed out") from error

    def list_pods(self) -> list[dict[str, Any]]:
        response = self.request("GET", "/pods")
        if isinstance(response, list):
            pods = response
        elif isinstance(response, dict) and isinstance(response.get("data"), list):
            pods = response["data"]
        else:
            raise RunpodAPIError("Runpod GET /pods returned an unexpected response shape")
        if any(not isinstance(pod, dict) or not isinstance(pod.get("id"), str) or
               not pod["id"] for pod in pods):
            raise RunpodAPIError("Runpod GET /pods returned a Pod without a valid ID")
        return pods

    def get_pod(self, pod_id: str) -> dict[str, Any]:
        try:
            return self.request("GET", f"/pods/{pod_id}")
        except RunpodAPIError as error:
            if error.status_code == 404:
                # Keep this typed signal narrower than generic endpoint 404s. An
                # ordered reuse caller may safely retire only a failed GET lookup.
                raise RunpodPodNotFound("Runpod Pod was not found", 404) from None
            raise

    def create_pod(self, body: dict[str, Any]) -> dict[str, Any]:
        return self.request("POST", "/pods", body)

    def start_pod(self, pod_id: str) -> None:
        self.request("POST", f"/pods/{pod_id}/start")

    def stop_pod(self, pod_id: str) -> None:
        self.request("POST", f"/pods/{pod_id}/stop")

    def stop_and_confirm(self, pod_id: str, attempts: int = 8,
                         sleeper: Callable[[float], None] = time.sleep) -> bool:
        """Stop repeatedly and confirm a non-billable compute state."""
        stopped_reads = 0
        for attempt in range(attempts):
            try:
                self.stop_pod(pod_id)
            except (RunpodAPIError, TimeoutError, OSError):
                pass
            sleeper(min(2 + attempt * 2, 10))
            try:
                pod = self.get_pod(pod_id)
                status = pod.get("desiredStatus") or pod.get("status")
                if status in {"EXITED", "STOPPED"}:
                    stopped_reads += 1
                    if stopped_reads >= 2:
                        return True
                else:
                    stopped_reads = 0
            except (RunpodAPIError, TimeoutError, OSError):
                stopped_reads = 0
        return False

    def delete_pod(self, pod_id: str) -> None:
        self.request("DELETE", f"/pods/{pod_id}")

    def delete_and_confirm(self, pod_id: str, attempts: int = 8,
                           sleeper: Callable[[float], None] = time.sleep) -> bool:
        """Delete repeatedly and confirm absence; never trust DELETE alone."""
        absent_reads = 0
        for attempt in range(attempts):
            try:
                self.delete_pod(pod_id)
            except (RunpodAPIError, TimeoutError, OSError):
                pass
            sleeper(min(2 + attempt * 2, 10))
            try:
                if pod_id not in {pod.get("id") for pod in self.list_pods()}:
                    absent_reads += 1
                    if absent_reads >= 2:
                        return True
                else:
                    absent_reads = 0
            except (RunpodAPIError, TimeoutError, OSError):
                absent_reads = 0
        return False
