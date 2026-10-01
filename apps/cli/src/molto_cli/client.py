"""Transport and credential validation for the public Molto management API."""

from __future__ import annotations

import ipaddress
import os
from pathlib import Path
from urllib.parse import unquote, urlsplit

import httpx


class CLIError(Exception):
    """An actionable command failure with a stable process exit code."""

    def __init__(self, message: str, exit_code: int = 1, details=None):
        super().__init__(message)
        self.exit_code = exit_code
        self.details = details


def server_url(host: str, port: int) -> str:
    host = host.strip().split(",")[0].strip().strip("[]")
    if host == "::":
        host = "::1"
    elif host in ("", "0.0.0.0"):
        host = "127.0.0.1"
    return f"http://{'[' + host + ']' if ':' in host else host}:{port}"


def validate_origin(value: str) -> str:
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as exc:
        raise CLIError(
            "Invalid --url. Use an HTTP(S) origin such as http://127.0.0.1:8000.", 2
        ) from exc
    if (
        parsed.scheme not in ("http", "https")
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in ("", "/")
        or parsed.query
        or parsed.fragment
        or any(ord(char) <= 32 or ord(char) == 127 for char in value)
        or port == 0
    ):
        raise CLIError(
            "--url must be an HTTP(S) origin without credentials, a path, or a query.",
            2,
        )
    return value.rstrip("/")


def loopback_origin(value: str) -> bool:
    host = urlsplit(value).hostname
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host or "").is_loopback
    except ValueError:
        return False


def validate_path(path: str) -> str:
    path = path.removeprefix("/")
    decoded = unquote(path)
    if (
        not path
        or any(char in decoded for char in "\\%?#")
        or any(ord(char) <= 32 or ord(char) == 127 for char in decoded)
        or any(part in ("", ".", "..") for part in decoded.split("/"))
        or decoded.split("/")[0] == "setup"
    ):
        raise CLIError(
            "Use a management operation path without traversal, query strings, or setup.",
            2,
        )
    return path


def read_key_file(filename: str) -> str:
    try:
        path = Path(filename).expanduser()
        if path.stat().st_mode & 0o077:
            raise CLIError(f"Key file must be private. Run chmod 600 on {path}.", 2)
        with path.open(encoding="utf-8") as stream:
            key = stream.read(4097).strip()
        if not key or len(key) > 4096:
            raise CLIError(
                "Key file must contain one key of at most 4096 characters.", 2
            )
        return key
    except (OSError, UnicodeError) as exc:
        raise CLIError(
            "Cannot read the API key file. Check its path and permissions.", 2
        ) from exc


def resolve_connection(args, *, require_key: bool = True):
    from molto_config.settings import GlobalSettings

    settings = GlobalSettings.load(base_path=getattr(args, "base_path", None))
    local_url = server_url(settings.server.host, settings.server.port)
    url = validate_origin(
        getattr(args, "url", None) or os.environ.get("MOLTO_URL") or local_url
    )
    key_file = getattr(args, "api_key_file", None)
    key = (
        read_key_file(key_file)
        if key_file
        else getattr(args, "api_key", None) or os.environ.get("MOLTO_API_KEY")
    )
    # A local saved main key must never follow an explicit remote destination.
    if not key and url == local_url and loopback_origin(url):
        key = settings.auth.api_key
    if key and (
        not key.isascii() or any(ord(char) <= 32 or ord(char) == 127 for char in key)
    ):
        raise CLIError(
            "The API key must contain printable ASCII characters without spaces.", 2
        )
    if require_key and not key:
        raise CLIError(
            "A main API key is required. Set MOLTO_API_KEY, use --api-key-file, or run molto init locally.",
            3,
        )
    return url, key or "", settings


class ManagementClient:
    """A thin synchronous client. Redirects never receive management credentials."""

    def __init__(self, url: str, api_key: str, timeout: float = 30, *, transport=None):
        self.url = validate_origin(url)
        self.api_key = api_key
        self._http = httpx.Client(
            timeout=timeout,
            follow_redirects=False,
            transport=transport,
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
        )

    @classmethod
    def from_args(cls, args, *, require_key: bool = True):
        url, key, _ = resolve_connection(args, require_key=require_key)
        return cls(url, key, getattr(args, "timeout", 30))

    def close(self):
        self._http.close()

    def request(self, method: str, path: str, *, json=None, params=None):
        return self.public_request(
            method,
            f"/api/management/v1/{validate_path(path)}",
            json=json,
            params=params,
        )

    def public_request(
        self, method: str, path: str, *, json=None, params=None, headers=None
    ):
        try:
            response = self._http.request(
                method, self.url + path, json=json, params=params, headers=headers
            )
        except httpx.TimeoutException as exc:
            raise CLIError(
                f"Timed out contacting {self.url}. Check server status or increase --timeout.",
                4,
            ) from exc
        except httpx.RequestError as exc:
            raise CLIError(
                f"Cannot reach {self.url}. Start the local server with molto start or check --url.",
                4,
            ) from exc
        except (ValueError, TypeError) as exc:
            raise CLIError(
                "The request body must contain valid, finite JSON values.", 2
            ) from exc
        if not response.is_success:
            try:
                detail = response.json().get("detail", "")
            except (ValueError, AttributeError):
                detail = ""
            if isinstance(detail, dict):
                detail = detail.get("message", detail.get("code", ""))
            elif isinstance(detail, list):
                # Pydantic validation also returns input values, potentially credentials.
                detail = "; ".join(
                    f"{'.'.join(str(part) for part in item.get('loc', []))}: {item.get('msg', 'invalid value')}"
                    for item in detail
                    if isinstance(item, dict)
                )
            if not isinstance(detail, str):
                detail = ""
            if self.api_key:
                detail = detail.replace(self.api_key, "[redacted]")

            def redact_inputs(body):
                nonlocal detail
                if isinstance(body, dict):
                    for name, value in body.items():
                        if (
                            isinstance(value, str)
                            and value
                            and any(
                                part in name.lower()
                                for part in ("key", "token", "secret", "password")
                            )
                        ):
                            detail = detail.replace(value, "[redacted]")
                        else:
                            redact_inputs(value)
                elif isinstance(body, list):
                    for value in body:
                        redact_inputs(value)

            redact_inputs(json)
            if response.status_code in (401, 403):
                message = "Management access denied. Use the main API key; inference subkeys cannot administer the server."
                code = 3
            else:
                message = (
                    detail
                    or f"Server returned HTTP {response.status_code} for this operation."
                )
                code = 5 if response.status_code == 409 else 1
            raise CLIError(message, code, {"status": response.status_code})
        if response.status_code == 204:
            return {"success": True}
        try:
            return response.json()
        except ValueError as exc:
            raise CLIError(
                "Server returned an invalid management response. Check that --url points to the public Molto dashboard port.",
                1,
            ) from exc
