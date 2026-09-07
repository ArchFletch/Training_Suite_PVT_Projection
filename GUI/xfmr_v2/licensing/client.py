"""Small stdlib HTTP client for the desktop floating-license endpoints."""

from __future__ import annotations

import http.client
import json
import urllib.error
import urllib.request
from typing import Any

from .models import (
    LicenseCheckoutResult,
    LicenseHeartbeatResult,
    LicenseIdentity,
    LicenseReleaseResult,
    LicenseStatus,
    normalize_server_url,
)


class LicenseClientError(RuntimeError):
    """Base exception raised for desktop-side license client failures."""


class LicenseConnectionError(LicenseClientError):
    """Raised when the server cannot be reached."""


class LicenseProtocolError(LicenseClientError):
    """Raised when the server response is invalid or indicates an HTTP failure."""


class LicenseHttpClient:
    """HTTP client for ``status``, ``checkout``, ``heartbeat``, and ``release``."""

    def __init__(self, server_url: str, *, timeout_seconds: float = 5.0) -> None:
        self.server_url = normalize_server_url(server_url)
        if not self.server_url:
            raise ValueError("A license server URL is required.")
        if self.server_url.endswith("/api/v1"):
            self.api_base = self.server_url
        else:
            self.api_base = f"{self.server_url}/api/v1"
        self.timeout_seconds = float(timeout_seconds)

    def get_status(self) -> LicenseStatus:
        return LicenseStatus.from_payload(self._request_json("GET", "/status"))

    def checkout(self, identity: LicenseIdentity | dict[str, Any]) -> LicenseCheckoutResult:
        payload = identity.to_checkout_payload() if isinstance(identity, LicenseIdentity) else dict(identity)
        return LicenseCheckoutResult.from_payload(self._request_json("POST", "/checkout", payload=payload))

    def heartbeat(self, *, lease_id: str, machine_id: str) -> LicenseHeartbeatResult:
        return LicenseHeartbeatResult.from_payload(
            self._request_json("POST", "/heartbeat", payload={"lease_id": lease_id, "machine_id": machine_id})
        )

    def release(self, *, lease_id: str, machine_id: str) -> LicenseReleaseResult:
        return LicenseReleaseResult.from_payload(
            self._request_json("POST", "/release", payload={"lease_id": lease_id, "machine_id": machine_id})
        )

    def _request_json(self, method: str, path: str, *, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        body: bytes | None = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        url = f"{self.api_base}{path}"
        try:
            request = urllib.request.Request(url, data=body, headers=headers, method=method)
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            message = _extract_error_message(raw) or f"License server returned HTTP {exc.code}."
            raise LicenseProtocolError(message) from exc
        except urllib.error.URLError as exc:
            reason = getattr(exc, "reason", exc)
            raise LicenseConnectionError(f"Could not reach {self.server_url}: {reason}") from exc
        except (ValueError, http.client.HTTPException) as exc:
            # A malformed URL (no scheme, a space in the host, a non-numeric port)
            # surfaces from urllib as ValueError or http.client.InvalidURL, not as a
            # URLError. Left alone these escaped every `except LicenseClientError`
            # in the GUI and reached the user as a raw traceback.
            raise LicenseConnectionError(
                f"'{self.server_url}' is not a valid license server URL ({exc}). "
                "Use the form http://host:port, for example http://licsrv01:27850."
            ) from exc
        except OSError as exc:
            raise LicenseConnectionError(f"Could not reach {self.server_url}: {exc}") from exc

        if not raw:
            return {}
        try:
            decoded = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LicenseProtocolError("License server returned invalid JSON.") from exc
        if not isinstance(decoded, dict):
            raise LicenseProtocolError("License server returned an unexpected response payload.")
        return decoded


def test_license_connection(server_url: str, *, timeout_seconds: float = 5.0) -> LicenseStatus:
    """Return the user-facing connection summary for one configured server URL."""

    return LicenseHttpClient(server_url, timeout_seconds=timeout_seconds).get_status()


def _extract_error_message(raw: bytes) -> str | None:
    if not raw:
        return None
    try:
        decoded = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return raw.decode("utf-8", errors="ignore").strip() or None
    if isinstance(decoded, dict):
        for key in ("message", "detail", "error"):
            if decoded.get(key):
                return str(decoded[key]).strip()
    return None
