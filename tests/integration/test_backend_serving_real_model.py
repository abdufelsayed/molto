# SPDX-License-Identifier: Apache-2.0
"""Opt-in local Gemma proof through a separate, temporary backend process."""

import base64
import io
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest
from PIL import Image
from repo_paths import repository_root

ROOT = repository_root(__file__)

pytestmark = [pytest.mark.slow, pytest.mark.integration]


def test_gemma_backend_text_vision_and_streaming(tmp_path: Path) -> None:
    configured = os.environ.get("MOLTO_TEST_GEMMA_VLM")
    if not configured:
        pytest.skip("Set MOLTO_TEST_GEMMA_VLM to a complete local Gemma VLM checkpoint")
    checkpoint = Path(configured).expanduser().resolve()
    assert (checkpoint / "config.json").is_file()
    root = tmp_path
    models = root / "models"
    models.mkdir()
    (models / "gemma").symlink_to(checkpoint, target_is_directory=True)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = os.environ.copy()
    env.update(
        HOME=str(root),
        MOLTO_BASE_PATH=str(root / "state"),
        MOLTO_API_KEY="probe-only-key",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    logpath = root / "server.log"
    with logpath.open("w") as log:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "molto_cli.cli",
                "serve",
                "--model-dir",
                str(models),
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--memory-guard-gb",
                "12",
            ],
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        base = f"http://127.0.0.1:{port}"

        def request(path, body=None, method=None):
            req = urllib.request.Request(
                base + path,
                data=None if body is None else json.dumps(body).encode(),
                method=method,
                headers={
                    "Authorization": "Bearer probe-only-key",
                    "Content-Type": "application/json",
                },
            )
            with urllib.request.urlopen(req, timeout=180) as response:
                return json.load(response)

        try:
            until = time.monotonic() + 90
            while time.monotonic() < until:
                if proc.poll() is not None:
                    raise RuntimeError(logpath.read_text()[-5000:])
                try:
                    request("/health")
                    break
                except (urllib.error.URLError, OSError):
                    time.sleep(0.25)
            else:
                raise RuntimeError("startup timeout")
            print("inventory", json.dumps(request("/management/v1/models")), flush=True)
            print(
                "load",
                request("/management/v1/models/gemma/load", {}, "POST"),
                flush=True,
            )
            body = {
                "model": "gemma",
                "messages": [
                    {"role": "user", "content": "Reply with exactly the word Hello."}
                ],
                "temperature": 0,
                "max_tokens": 24,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            response = request("/v1/chat/completions", body)
            text = response["choices"][0]["message"]["content"]
            assert text and "hello" in text.lower(), response
            print("text", text, response["usage"], flush=True)
            data = io.BytesIO()
            Image.new("RGB", (128, 128), "red").save(data, format="PNG")
            image = (
                "data:image/png;base64," + base64.b64encode(data.getvalue()).decode()
            )
            body["messages"] = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What color is the square? Answer with one word.",
                        },
                        {"type": "image_url", "image_url": {"url": image}},
                    ],
                }
            ]
            response = request("/v1/chat/completions", body)
            text = response["choices"][0]["message"]["content"]
            assert text and "red" in text.lower(), response
            print("image", text, response["usage"], flush=True)
            body["stream"] = True
            body["stream_options"] = {"include_usage": True}
            req = urllib.request.Request(
                base + "/v1/chat/completions",
                data=json.dumps(body).encode(),
                headers={
                    "Authorization": "Bearer probe-only-key",
                    "Content-Type": "application/json",
                },
            )
            with urllib.request.urlopen(req, timeout=180) as response:
                stream = response.read().decode()
            assert "data: [DONE]" in stream, stream
            chunks = [
                json.loads(line[6:])
                for line in stream.splitlines()
                if line.startswith("data: ") and line[6:] != "[DONE]"
            ]
            streamed = "".join(
                c["choices"][0]["delta"].get("content", "") or ""
                for c in chunks
                if c.get("choices")
            )
            assert "red" in streamed.lower(), stream
            print("sse", streamed, flush=True)
            print(
                "unload",
                request("/management/v1/models/gemma/unload", {}, "POST"),
                flush=True,
            )
            until = time.monotonic() + 30
            while time.monotonic() < until:
                state = request("/management/v1/state")
                if (
                    state["loaded_count"] == 0
                    and all(not row["loaded"] for row in state["models"])
                    and state["current_model_memory"] == 0
                ):
                    break
                time.sleep(0.2)
            assert (
                state["loaded_count"] == 0
                and all(not row["loaded"] for row in state["models"])
                and state["current_model_memory"] == 0
            ), state
            print("final_state", state, flush=True)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
            print("server_stopped", proc.returncode, flush=True)
