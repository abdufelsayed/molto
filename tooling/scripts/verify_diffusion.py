"""Opt-in real image API proof using two existing local checkpoints.

Run with uv run --all-packages and the image extra installed. This script does not
download models or connect to a running server. It uses an isolated engine pool
and ASGI requests, writes PNGs/report.json, and unloads each model after use.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import platform
import tempfile
import time
from importlib.metadata import version
from pathlib import Path

import httpx
import mlx.core as mx
from molto_runtime.engine_core import get_mlx_executor
from molto_runtime.engine_pool import EnginePool
from molto_server.server import create_app
from PIL import Image


async def verify(z_image: Path, flux2: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "platform": platform.platform(),
        "versions": {name: version(name) for name in ("mflux", "mlx", "molto")},
        "models": {"z-image": str(z_image), "flux2": str(flux2)},
        "results": [],
    }
    with tempfile.TemporaryDirectory(prefix="molto-diffusion-models-") as directory:
        roots = Path(directory)
        (roots / "z-image").symlink_to(z_image, target_is_directory=True)
        (roots / "flux2").symlink_to(flux2, target_is_directory=True)
        pool = EnginePool()
        pool.discover_models(str(roots))
        assert set(pool.get_model_ids()) == {"z-image", "flux2"}, pool.get_model_ids()
        app = create_app()
        app.state.server_state.engine_pool = pool
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://127.0.0.1:19387"
        ) as client:

            async def generate(name, route, body):
                await asyncio.get_running_loop().run_in_executor(
                    get_mlx_executor(), mx.reset_peak_memory
                )
                started = time.perf_counter()
                response = await client.post(route, json=body)
                assert response.status_code == 200, response.text[:1000]
                elapsed = time.perf_counter() - started
                png = base64.b64decode(
                    response.json()["data"][0]["b64_json"], validate=True
                )
                target = output / f"{name}.png"
                target.write_bytes(png)
                with Image.open(target) as image:
                    assert image.size == (512, 512), image.size
                    assert image.format == "PNG"
                result = {
                    "name": name,
                    "route": route,
                    "pipeline": body["pipeline"],
                    "seconds_including_load": elapsed,
                    "size": [512, 512],
                    "steps": body["steps"],
                    "seed": body["seed"],
                    "sha256": hashlib.sha256(png).hexdigest(),
                    "peak_active_bytes": await asyncio.get_running_loop().run_in_executor(
                        get_mlx_executor(), mx.get_peak_memory
                    ),
                }
                report["results"].append(result)
                print(json.dumps(result), flush=True)
                return png

            try:
                for model_id in ("z-image", "flux2"):
                    response = await client.get(
                        "/v1/images/capabilities", params={"model": model_id}
                    )
                    assert response.status_code == 200, response.text
                z_body = {
                    "model": "z-image",
                    "pipeline": "z-image-turbo",
                    "size": "512x512",
                    "steps": 9,
                    "seed": 42,
                    "prompt": "A glossy red ceramic teapot on a wooden table, soft window light, studio product photograph",
                }
                z_png = await generate(
                    "z-image-turbo", "/v1/images/generations", z_body
                )
                repeat = await generate(
                    "z-image-turbo-warm", "/v1/images/generations", z_body
                )
                report["z_image_repeat_identical"] = z_png == repeat
                assert z_png == repeat, "Fixed-seed warm output differed"
                assert await pool.request_unload("z-image", reason="verification")
                await generate(
                    "flux2-text",
                    "/v1/images/generations",
                    {
                        "model": "flux2",
                        "pipeline": "flux2-klein-4b",
                        "size": "512x512",
                        "steps": 4,
                        "seed": 17,
                        "prompt": z_body["prompt"],
                    },
                )
                edited = await generate(
                    "flux2-edit",
                    "/v1/images/edits",
                    {
                        "model": "flux2",
                        "pipeline": "flux2-klein-4b/edit",
                        "size": "512x512",
                        "steps": 4,
                        "seed": 17,
                        "prompt": "Change the red teapot to cobalt blue. Preserve the teapot shape, the table and the background.",
                        "image": base64.b64encode(z_png).decode("ascii"),
                    },
                )
                assert edited != z_png, "Editing returned the original bytes"
                assert pool.get_entry("flux2").in_use == 0
                rejected = await client.post(
                    "/v1/images/generations",
                    json={
                        "model": "flux2",
                        "pipeline": "flux2-klein-4b",
                        "prompt": "tree",
                        "negative_prompt": "blur",
                    },
                )
                assert rejected.status_code == 400, rejected.text
                assert "negative_prompt" in rejected.text, rejected.text
                report["unsupported_negative_prompt_status"] = rejected.status_code
            finally:
                for model_id in pool.get_model_ids():
                    await pool.request_unload(model_id, reason="verification cleanup")
                report["all_models_unloaded"] = all(
                    pool.get_entry(model_id).engine is None
                    for model_id in pool.get_model_ids()
                )
                (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
                assert report["all_models_unloaded"], (
                    "Verification cleanup left a model loaded"
                )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--z-image", required=True, type=Path)
    parser.add_argument("--flux2", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    for checkpoint in (args.z_image, args.flux2):
        if not checkpoint.is_dir():
            parser.error(f"Local checkpoint does not exist: {checkpoint}")
    asyncio.run(
        verify(args.z_image.resolve(), args.flux2.resolve(), args.output.resolve())
    )
