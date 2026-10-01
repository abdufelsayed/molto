"""Connection isolation, safe errors, and public management transport."""

import argparse
import json
from types import SimpleNamespace

import httpx
import pytest

from omlx.cli_client import (
    CLIError,
    ManagementClient,
    add_connection_options,
    resolve_connection,
    server_url,
    validate_origin,
    validate_path,
)
from omlx.cli_output import Output
from omlx.settings import GlobalSettings


@pytest.fixture
def local_args(tmp_path, monkeypatch):
    for name in ("OMLX_URL", "OMLX_API_KEY", "OMLX_HOST", "OMLX_PORT"):
        monkeypatch.delenv(name, raising=False)
    settings = GlobalSettings(base_path=tmp_path)
    settings.auth.api_key = "local-main-key"
    settings.save()
    return SimpleNamespace(base_path=str(tmp_path))


def test_remote_destination_does_not_inherit_local_key(local_args):
    local_args.url = "https://remote.example"
    with pytest.raises(CLIError, match="main API key") as error:
        resolve_connection(local_args)
    assert error.value.exit_code == 3


def test_local_credentials_and_explicit_remote_key(local_args):
    assert resolve_connection(local_args)[:2] == (
        "http://127.0.0.1:8000",
        "local-main-key",
    )
    local_args.url = "https://remote.example"
    local_args.api_key = "remote-main-key"
    assert resolve_connection(local_args)[:2] == (
        "https://remote.example",
        "remote-main-key",
    )


def test_remote_environment_url_does_not_leak_key(local_args, monkeypatch):
    monkeypatch.setenv("OMLX_URL", "https://remote.example")
    with pytest.raises(CLIError):
        resolve_connection(local_args)


def test_private_key_file(local_args, tmp_path):
    path = tmp_path / "key"
    path.write_text("remote-key\n")
    path.chmod(0o644)
    local_args.api_key_file = str(path)
    with pytest.raises(CLIError, match="private"):
        resolve_connection(local_args)
    path.chmod(0o600)
    assert resolve_connection(local_args)[1] == "remote-key"


@pytest.mark.parametrize(
    "url",
    [
        "ftp://host",
        "http://secret:secret@host",
        "http://host/foo",
        "http://host?key=secret",
        "http://host#frag",
        "http://host:0",
        "http://host:invalid",
        "http://host\n",
    ],
)
def test_invalid_origins(url):
    with pytest.raises(CLIError):
        validate_origin(url)


def test_ipv6_and_wildcard_urls():
    assert server_url("::1", 8000) == "http://[::1]:8000"
    assert server_url("[::1]", 8000) == "http://[::1]:8000"
    assert server_url("::", 8000) == "http://[::1]:8000"
    assert server_url("0.0.0.0", 8000) == "http://127.0.0.1:8000"


@pytest.mark.parametrize(
    "path",
    [
        "../settings",
        "models/%2e%2e/settings",
        "models/%252e%252e/settings",
        "//other",
        "settings?token=x",
        "setup",
        "setup/anything",
        "models/%5csettings",
        "models/%00",
    ],
)
def test_invalid_paths(path):
    with pytest.raises(CLIError):
        validate_path(path)


def test_management_request_uses_public_gateway_and_preserves_model_id():
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(202, json={"status": "loading"})

    client = ManagementClient(
        "http://localhost:3000", "main-key", transport=httpx.MockTransport(handle)
    )
    try:
        assert client.request("POST", "models/owner%2Fmodel/load", json={}) == {
            "status": "loading"
        }
    finally:
        client.close()
    assert requests[0].url.raw_path == b"/api/management/v1/models/owner%2Fmodel/load"
    assert requests[0].headers["Authorization"] == "Bearer main-key"
    assert "cookie" not in requests[0].headers


@pytest.mark.parametrize(
    "status,exit_code", [(401, 3), (403, 3), (409, 5), (500, 1), (302, 1)]
)
def test_http_errors_are_actionable_and_redirects_not_followed(status, exit_code):
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(
            status,
            json={"detail": "failure main-key"},
            headers={"Location": "https://other.example"},
        )

    client = ManagementClient(
        "http://localhost:3000", "main-key", transport=httpx.MockTransport(handle)
    )
    with pytest.raises(CLIError) as error:
        client.request("GET", "models")
    client.close()
    assert error.value.exit_code == exit_code
    assert "main-key" not in str(error.value)
    assert len(calls) == 1


def test_validation_input_secrets_are_not_printed():
    client = ManagementClient(
        "http://localhost:3000",
        "main-key",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                422,
                json={
                    "detail": [
                        {
                            "loc": ["body", "key"],
                            "msg": "invalid",
                            "input": "new-secret",
                        }
                    ]
                },
            )
        ),
    )
    with pytest.raises(CLIError) as error:
        client.request("POST", "auth/subkeys", json={"key": "new-secret"})
    client.close()
    assert "new-secret" not in str(error.value)


def test_flags_before_and_after_nested_command():
    parser = argparse.ArgumentParser()
    add_connection_options(parser)
    child = parser.add_subparsers(dest="command").add_parser("models")
    add_connection_options(child)
    leaf = child.add_subparsers(dest="action").add_parser("list")
    add_connection_options(leaf)
    args = parser.parse_args(
        ["--url", "https://remote.example", "--json", "models", "list", "--no-color"]
    )
    assert args.url == "https://remote.example"
    assert args.json and args.no_color


def test_json_output_stays_clean_and_errors_use_stderr(capsys):
    output = Output(SimpleNamespace(json=True))
    output.emit({"models": []})
    output.error(CLIError("Cannot connect", 4))
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"models": []}
    assert json.loads(captured.err)["error"]["exit_code"] == 4


def test_noninteractive_confirmation_requires_explicit_yes(monkeypatch):
    import sys

    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    with pytest.raises(CLIError, match="--yes"):
        Output(SimpleNamespace()).confirm("Delete?")
    Output(SimpleNamespace(yes=True)).confirm("Delete?")


def test_terminal_strings_do_not_execute_rich_markup(capsys):
    Output(SimpleNamespace(no_color=True)).emit("[link=https://evil.example]log[/link]")
    assert "[link=" in capsys.readouterr().out


def test_serialization_errors_become_usage_errors():
    client = ManagementClient(
        "http://localhost:3000",
        "main-key",
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={})),
    )
    with pytest.raises(CLIError) as error:
        client.request(
            "POST", "settings", json={"sampling": {"temperature": float("inf")}}
        )
    client.close()
    assert error.value.exit_code == 2


def test_server_error_redacts_submitted_new_secret():
    client = ManagementClient(
        "http://localhost:3000",
        "main-key",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                500, json={"detail": "Unable to save new-secret"}
            )
        ),
    )
    with pytest.raises(CLIError) as error:
        client.request("PATCH", "auth/main-key", json={"key": "new-secret"})
    client.close()
    assert "new-secret" not in str(error.value)


def test_setup_eof_is_clean_cancellation(local_args, monkeypatch):
    import getpass
    import sys

    from omlx import cli

    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

    def eof(prompt):
        raise EOFError()

    monkeypatch.setattr(getpass, "getpass", eof)
    client = SimpleNamespace(
        public_request=lambda *a, **k: {"allowed": True, "setup_required": True},
        close=lambda: None,
    )
    monkeypatch.setattr("omlx.cli_client.ManagementClient", lambda *a, **k: client)
    with pytest.raises(CLIError) as error:
        cli.init_command(local_args)
    assert error.value.exit_code == 130
