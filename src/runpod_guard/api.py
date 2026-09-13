from __future__ import annotations

import json
import hashlib
import time
import urllib.error
import urllib.request
from typing import Any, Callable


class RunpodAPIError(RuntimeError):
    """A Runpod REST request failed."""


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
            raise RunpodAPIError(f"Runpod {method} {path} returned {error.code}: {detail}") from None
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
        return self.request("GET", f"/pods/{pod_id}")

    def create_pod(self, body: dict[str, Any]) -> dict[str, Any]:
        return self.request("POST", "/pods", body)

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
