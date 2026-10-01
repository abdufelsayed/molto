"""Prove management CLI dispatch without real requests or model mutations."""

import argparse
import json
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from molto_cli import cli_commands as commands
from molto_cli.client import CLIError


@pytest.fixture
def cli(monkeypatch):
    client = Mock()
    output = Mock()
    output.watch.return_value = nullcontext()
    monkeypatch.setattr(commands.ManagementClient, "from_args", lambda args: client)
    monkeypatch.setattr(commands, "Output", lambda args: output)
    parser = argparse.ArgumentParser()
    commands.register_commands(parser.add_subparsers(required=True))

    def invoke(*argv):
        args = parser.parse_args(argv)
        return commands.run(args)

    return SimpleNamespace(client=client, output=output, invoke=invoke, parser=parser)


def test_model_id_with_slashes_stays_one_parameter(cli):
    cli.invoke(
        "models", "load", "org/model", "--json", "--url", "http://localhost:8000"
    )
    cli.client.request.assert_called_once_with("POST", "/models/org%2Fmodel/load")
    cli.client.close.assert_called_once()


def test_nested_server_settings_are_typed(cli):
    cli.invoke(
        "settings",
        "set",
        "scheduler.max_num_seqs=4",
        "auth.allow_unauthenticated_inference=false",
    )
    cli.client.request.assert_called_once_with(
        "PATCH",
        "/server/settings",
        json={
            "scheduler": {"max_num_seqs": 4},
            "auth": {"allow_unauthenticated_inference": False},
        },
    )


@pytest.mark.parametrize(
    "assignment", ["max_num_seqs=4", "scheduler.x=bad", "scheduler..x=2"]
)
def test_invalid_settings_do_not_mutate(cli, assignment):
    with pytest.raises(CLIError):
        cli.invoke("settings", "set", assignment)
    cli.client.request.assert_not_called()
    cli.client.close.assert_called_once()


def test_key_list_hides_all_secrets_even_in_json(cli):
    cli.client.request.return_value = {
        "main_key": "MAIN-SECRET",
        "sub_keys": [{"id": "a", "key": "SUB-SECRET", "name": "CI"}],
    }
    cli.invoke("keys", "list", "--json")
    emitted = cli.output.emit.call_args.args[0]
    assert "SECRET" not in json.dumps(emitted)
    assert emitted["sub_keys"][0]["name"] == "CI"


def test_key_list_reveal_is_deliberate(cli):
    cli.client.request.return_value = {"main_key": "visible"}
    cli.invoke("keys", "list", "--reveal")
    assert cli.output.emit.call_args.args[0]["main_key"] == "visible"


def test_create_reveals_returned_new_key_once(cli):
    cli.client.request.return_value = {"sub_key": {"key": "new-secret", "id": "1"}}
    cli.invoke("keys", "create", "--name", "CI")
    cli.client.request.assert_called_once_with(
        "POST", "/auth/subkeys", json={"name": "CI"}
    )
    assert cli.output.emit.call_args.args[0]["sub_key"]["key"] == "new-secret"


def test_main_rotation_returns_generated_key(cli, monkeypatch):
    monkeypatch.setattr(commands.secrets, "token_hex", lambda n: "generated-key")
    cli.client.request.return_value = {"changed": True}
    cli.invoke("keys", "rotate", "--main", "--yes")
    cli.output.confirm.assert_called_once()
    cli.client.request.assert_called_once_with(
        "PATCH", "/auth/main-key", json={"key": "generated-key"}
    )
    assert cli.output.emit.call_args.args[0]["new_key"] == "generated-key"


def test_remove_uses_fresh_plan_and_confirmation_before_mutation(cli):
    events = []

    def request(method, path, **kwargs):
        events.append((method, path, kwargs))
        return (
            {"plan_token": "fresh", "files": ["weights"]}
            if method == "GET"
            else {"deleted": True}
        )

    cli.client.request.side_effect = request
    cli.output.confirm.side_effect = lambda message: events.append("confirmed")
    cli.invoke("models", "remove", "model", "--drain", "--yes")
    assert events == [
        ("GET", "/workspace/models/model/delete-plan", {}),
        "confirmed",
        (
            "DELETE",
            "/workspace/models/model/delete",
            {"json": {"plan_token": "fresh", "drain": True}},
        ),
    ]
    assert (
        cli.output.emit.call_args_list[0].kwargs["title"]
        == "Files and settings to remove"
    )


def test_noninteractive_confirmation_failure_never_mutates(cli):
    cli.output.confirm.side_effect = CLIError("Use --yes in noninteractive mode")
    with pytest.raises(CLIError):
        cli.invoke("cache", "clear", "ssd")
    cli.client.request.assert_not_called()


def test_diffusion_retry_is_not_invented(cli):
    with pytest.raises(CLIError, match="do not support retry"):
        cli.invoke("jobs", "retry", "abc", "--kind", "diffusion")
    cli.client.request.assert_not_called()


def test_watch_stops_on_failure_and_emits_one_json(cli, monkeypatch):
    cli.client.request.side_effect = [
        {"status": "running"},
        {"status": "failed", "error": "failed conversion"},
    ]
    monkeypatch.setattr(commands.time, "sleep", lambda seconds: None)
    assert cli.invoke("jobs", "watch", "x", "--json") == 1
    cli.output.emit.assert_called_once_with(
        {"status": "failed", "error": "failed conversion"}
    )


def test_watch_is_bounded(cli, monkeypatch):
    cli.client.request.return_value = {"status": "running"}
    times = iter([0, 301])
    monkeypatch.setattr(commands.time, "monotonic", lambda: next(times))
    with pytest.raises(CLIError, match="timed out"):
        cli.invoke("jobs", "watch", "x")
    cli.client.request.assert_called_once()


def test_watch_interrupt_closes_client(cli):
    cli.client.request.side_effect = KeyboardInterrupt
    assert cli.invoke("jobs", "watch", "x") == 130
    cli.client.close.assert_called_once()


@pytest.mark.parametrize(
    "path",
    [
        "https://evil.test/server",
        "//evil.test",
        "/../auth/keys",
        "/server/../keys",
        "/server#fragment",
        "/server\\info",
    ],
)
def test_raw_api_rejects_outside_paths(cli, path):
    with pytest.raises(CLIError):
        cli.invoke("api", "GET", path)
    cli.client.request.assert_not_called()


def test_raw_api_json_file(cli, tmp_path):
    body = tmp_path / "body.json"
    body.write_text('{"scope":"session"}')
    cli.invoke("api", "post", "/monitoring/stats/reset", "--body", f"@{body}")
    cli.client.request.assert_called_once_with(
        "POST", "/monitoring/stats/reset", json={"scope": "session"}
    )


def test_profile_create_contract(cli):
    cli.invoke(
        "models",
        "profiles",
        "create",
        "org/model",
        "fast",
        "--settings",
        '{"temperature":0.2}',
        "--json",
    )
    assert cli.client.request.call_args.args == ("POST", "/models/org%2Fmodel/profiles")
    assert cli.client.request.call_args.kwargs["json"]["settings"] == {
        "temperature": 0.2
    }


def test_diagnostic_contract(cli):
    cli.invoke("diagnostics", "start", "throughput", "--options", '{"model_id":"test"}')
    cli.client.request.assert_called_once_with(
        "POST",
        "/diagnostics/runs",
        json={"kind": "throughput", "options": {"model_id": "test"}},
    )


def test_json_log_output_retains_metadata(cli):
    cli.client.request.return_value = {"logs": "hello\n", "scan_truncated": True}
    cli.invoke("logs", "--level", "ERROR", "--json")
    cli.output.emit.assert_called_once_with({"logs": "hello\n", "scan_truncated": True})


def test_settings_metadata_masks_arbitrary_secret_field(cli):
    cli.client.request.return_value = {
        "sections": {"provider": {"credential": "secret"}},
        "fields": [{"secret": True, "section": "provider", "key": "credential"}],
    }
    cli.invoke("settings", "get", "--json")
    assert (
        cli.output.emit.call_args.args[0]["sections"]["provider"]["credential"]
        == "••••••••"
    )


def test_human_model_list_is_compact(cli):
    cli.client.request.return_value = {
        "models": [
            {
                "id": "model",
                "loaded": True,
                "settings": {"lots": "of data"},
                "model_type": "llm",
                "engine_type": "batched",
            }
        ]
    }
    cli.invoke("models", "list")
    assert cli.output.emit.call_args.args[0] == [
        {"model": "model", "state": "loaded", "type": "llm", "engine": "batched"}
    ]


def test_diagnostics_results_contract(cli):
    cli.invoke("diagnostics", "results", "run-1")
    cli.client.request.assert_called_once_with(
        "GET", "/diagnostics/runs/run-1/results", json=None
    )


def test_monitoring_usage_contract(cli):
    cli.invoke("monitoring", "usage", "--range", "7d", "--model", "model", "--details")
    cli.client.request.assert_called_once_with(
        "GET",
        "/monitoring/usage",
        params={"range": "7d", "model": "model", "include_details": True},
    )


@pytest.mark.parametrize(
    "path", ["/%2e%2e/keys", "/%252e%252e/keys", "/%5cevil", "/%2f%2fevil"]
)
def test_encoded_raw_api_paths_stay_local(cli, path):
    with pytest.raises(CLIError):
        cli.invoke("api", "GET", path)
    cli.client.request.assert_not_called()


@pytest.mark.parametrize(
    "old,new,expected",
    [
        ("a\nb\n", "b\nc\n", "c\n"),
        ("x\nx\n", "x\nx\ny\n", "y\n"),
        ("old\n", "new\n", "new\n"),
        ("same", "same", ""),
        ("", "start", "start"),
    ],
)
def test_follow_logs_overlap(old, new, expected):
    assert commands._log_suffix(old, new) == expected


def test_real_non_tty_confirmation_blocks_mutation(cli, monkeypatch, capsys):
    import sys

    from molto_cli.cli_output import Output

    monkeypatch.setattr(commands, "Output", Output)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    with pytest.raises(CLIError, match="confirmation"):
        cli.invoke("keys", "revoke", "old", "--json")
    cli.client.request.assert_not_called()
    assert capsys.readouterr().out == ""


def test_real_json_key_listing_masks_secrets(cli, monkeypatch, capsys):
    from molto_cli.cli_output import Output

    monkeypatch.setattr(commands, "Output", Output)
    cli.client.request.return_value = {
        "main_key": "main-secret",
        "sub_keys": [{"id": "a", "key": "sub-secret", "name": "CI"}],
    }
    cli.invoke("keys", "list", "--json")
    emitted = json.loads(capsys.readouterr().out)
    assert emitted["main_key"] == "••••••••"
    assert emitted["sub_keys"][0]["key"] == "••••••••"


def test_workspace_jobs_use_actual_shared_operation_types(cli):
    cli.client.request.return_value = {
        "operations": [
            {"id": "a", "kind": "download_hf"},
            {"id": "b", "kind": "update_check"},
            {"id": "c", "kind": "verify"},
        ]
    }
    cli.invoke("jobs", "list", "--kind", "workspace", "--json")
    assert [item["id"] for item in cli.output.emit.call_args.args[0]["operations"]] == [
        "b",
        "c",
    ]


def test_external_job_error_is_terminal(cli):
    cli.client.request.return_value = {
        "status": "error",
        "error": "download interrupted",
    }
    assert cli.invoke("jobs", "watch", "external", "--json") == 1
    cli.client.request.assert_called_once()


def test_human_watch_uses_compact_progress(cli):
    cli.client.request.return_value = {
        "id": "x",
        "status": "succeeded",
        "progress": 100,
        "payload": {"big": "data"},
        "result": {"big": "output"},
    }
    assert cli.invoke("jobs", "watch", "x") == 0
    cli.output.watch.assert_called_once()
    assert cli.output.emit.call_args.args[0] == {
        "id": "x",
        "status": "succeeded",
        "progress": 100,
    }


def test_private_provider_credential_file_required(cli, tmp_path):
    token = tmp_path / "token"
    token.write_text("provider-secret")
    token.chmod(0o644)
    with pytest.raises(CLIError, match="private"):
        cli.invoke("models", "download", "org/model", "--token-file", str(token))
    cli.client.request.assert_not_called()
    token.chmod(0o600)
    cli.invoke("models", "download", "org/model", "--token-file", str(token))
    assert cli.client.request.call_args.kwargs["json"]["token"] == "provider-secret"


def test_malformed_assignment_error_does_not_echo_secret(cli):
    with pytest.raises(CLIError) as error:
        cli.invoke("settings", "set", "bad-field=secret-value")
    assert "secret-value" not in str(error.value)


def test_human_settings_keep_live_application_signal(cli):
    cli.client.request.return_value = {
        "sections": {},
        "changed": ["sampling.temperature"],
        "live_applied": ["sampling.temperature"],
        "restart_required": [],
    }
    cli.invoke("settings", "set", "sampling.temperature=0.2")
    assert cli.output.emit.call_args.args[0]["live_applied"] == ["sampling.temperature"]


def test_default_job_list_combines_operations_and_diffusion(cli):
    cli.client.request.side_effect = [
        {"operations": [{"id": "op", "kind": "verify"}]},
        {"jobs": [{"id": "diff", "kind": "calibration"}]},
    ]
    cli.invoke("jobs", "list")
    rows = cli.output.emit.call_args.args[0]
    assert [(row["id"], row["source"]) for row in rows] == [
        ("op", "workspace"),
        ("diff", "diffusion"),
    ]


def test_explicit_acquisition_list_excludes_workspace(cli):
    cli.client.request.return_value = {
        "operations": [
            {"id": "w", "kind": "verify"},
            {"id": "a", "kind": "download_hf"},
        ]
    }
    cli.invoke("jobs", "list", "--kind", "acquisition", "--json")
    assert [item["id"] for item in cli.output.emit.call_args.args[0]["operations"]] == [
        "a"
    ]


def test_nonfinite_json_is_rejected_before_request(cli):
    with pytest.raises(CLIError, match="Non-finite"):
        cli.invoke("settings", "set", "sampling.temperature=NaN")
    cli.client.request.assert_not_called()


@pytest.mark.parametrize("path", ["server/restart", "/server/restart"])
def test_raw_api_accepts_both_relative_path_spellings(cli, path):
    cli.invoke("api", "POST", path)
    cli.client.request.assert_called_once_with("POST", "/server/restart", json=None)


def test_raw_api_query_parameters_are_separate_and_repeatable(cli):
    cli.invoke(
        "api",
        "GET",
        "stats",
        "--query",
        "model_id=org/model",
        "--query",
        "include_details=true",
        "--query",
        "tag=first",
        "--query",
        "tag=second",
    )
    cli.client.request.assert_called_once_with(
        "GET",
        "/stats",
        json=None,
        params=[
            ("model_id", "org/model"),
            ("include_details", "true"),
            ("tag", "first"),
            ("tag", "second"),
        ],
    )


@pytest.mark.parametrize(
    "query", ["missing_separator", "=empty-name", "bad name=value"]
)
def test_raw_api_invalid_query_never_requests(cli, query):
    with pytest.raises(CLIError, match="NAME=VALUE"):
        cli.invoke("api", "GET", "stats", "--query", query)
    cli.client.request.assert_not_called()


@pytest.mark.parametrize(
    "path",
    [
        "stats?model_id=x",
        "/stats?model_id=x",
        "../server",
        "setup",
        "https://evil.example",
        "//evil.example",
    ],
)
def test_raw_api_client_validation_rejects_queries_and_outside_paths(cli, path):
    with pytest.raises(CLIError):
        cli.invoke("api", "GET", path)
    cli.client.request.assert_not_called()


@pytest.mark.parametrize(
    "body",
    [
        "1e999",
        "-1e999",
        '{"nested":{"items":[1e999]}}',
        '{"nested":NaN}',
        '{"nested":Infinity}',
        '{"nested":-Infinity}',
    ],
)
def test_raw_api_rejects_all_nonfinite_json_numbers_before_request(cli, body):
    with pytest.raises(CLIError, match="Non-finite") as error:
        cli.invoke("api", "POST", "monitoring/cache/probe", f"--body={body}")
    assert error.value.exit_code == 2
    cli.client.request.assert_not_called()


def test_json_file_rejects_nested_overflow(cli, tmp_path):
    body = tmp_path / "body.json"
    body.write_text('{"options":{"temperature":1e999}}')
    with pytest.raises(CLIError, match="Non-finite") as error:
        cli.invoke("diagnostics", "start", "throughput", "--options", f"@{body}")
    assert error.value.exit_code == 2
    cli.client.request.assert_not_called()


@pytest.mark.parametrize("status", [404, 503])
def test_default_jobs_preserve_operations_when_optional_diffusion_unavailable(
    cli, status
):
    cli.client.request.side_effect = [
        {"operations": [{"id": "op", "kind": "verify"}]},
        CLIError("Diffusion provider unavailable", details={"status": status}),
    ]
    assert cli.invoke("jobs", "list", "--json") == 0
    cli.output.emit.assert_called_once()
    result = cli.output.emit.call_args.args[0]
    assert result["operations"][0]["id"] == "op"
    assert result["diffusion_jobs"] == []
    assert result["availability"]["diffusion"] == {
        "available": False,
        "status": status,
        "message": "Diffusion provider unavailable",
    }


def test_optional_diffusion_unavailable_is_visible_in_human_output(cli):
    cli.client.request.side_effect = [
        {"operations": [{"id": "op", "kind": "verify"}]},
        CLIError("Unavailable", details={"status": 503}),
    ]
    assert cli.invoke("jobs", "list") == 0
    assert "unavailable" in cli.output.emit.call_args_list[0].args[0]
    assert cli.output.emit.call_args_list[1].args[0][0]["id"] == "op"


@pytest.mark.parametrize("status", [401, 403, 409, 500])
def test_default_jobs_do_not_hide_other_diffusion_errors(cli, status):
    cli.client.request.side_effect = [
        {"operations": []},
        CLIError("Actual failure", details={"status": status}),
    ]
    with pytest.raises(CLIError, match="Actual failure"):
        cli.invoke("jobs", "list", "--json")
    cli.output.emit.assert_not_called()


def test_explicit_diffusion_jobs_unavailability_remains_an_error(cli):
    cli.client.request.side_effect = CLIError("Unavailable", details={"status": 503})
    with pytest.raises(CLIError):
        cli.invoke("jobs", "list", "--kind", "diffusion")
