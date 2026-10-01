"""Opt-in offline proof using an existing local FLUX.2 Klein 4B checkpoint.

Usage: PYTHONPATH=. python scripts/verify_diffusion_performance.py MODEL REF.png NEW_OUTPUT [--batch4]
Runs real inference through an isolated pool/shared executor and ASGI image APIs.
Uses 256x256, four steps, three alternating trials (override with
OMLX_DIFFUSION_PROOF_TRIALS=1..10). Timings are exploratory; ratios below one
mean slower performance. Batch pixels need not match serial pixels.
Reports survive later failures. Source provenance covers headers/sizes only.
"""

import asyncio
import base64
import hashlib
import io
import json
import os
import platform
import statistics
import sys
import tempfile
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

# Set these before importing any hub, transformer or oMLX modules.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return "uninstalled source checkout"


def memory():
    import mlx.core as mx
    import psutil

    return dict(
        active_bytes=mx.get_active_memory(),
        cache_bytes=mx.get_cache_memory(),
        peak_bytes=mx.get_peak_memory(),
        rss_bytes=psutil.Process().memory_info().rss,
    )


def compare(left, right):
    import numpy as np
    from PIL import Image

    with Image.open(io.BytesIO(left)) as image:
        a = np.asarray(image.convert("RGB"), dtype=np.float64)
    with Image.open(io.BytesIO(right)) as image:
        b = np.asarray(image.convert("RGB"), dtype=np.float64)
    require(a.shape == b.shape, "Compared outputs have different dimensions")
    mse = float(np.mean((a - b) ** 2))
    return dict(
        byte_identical=left == right,
        mae=float(np.mean(np.abs(a - b))),
        max_pixel_delta=float(np.max(np.abs(a - b))),
        psnr_db=None if mse == 0 else float(20 * np.log10(255 / np.sqrt(mse))),
        zero_error=mse == 0,
    )


class Evidence:
    def __init__(self, output, source, trials):
        self.output = output
        self.data = dict(
            status="running",
            offline=True,
            model=str(source),
            platform=platform.platform(),
            versions={name: package_version(name) for name in ("mlx", "mflux", "omlx")},
            size=256,
            steps=4,
            trials=trials,
            runs=[],
            summaries=[],
            api=[],
        )
        self.save()

    def save(self):
        temporary = self.output / ".report.json.tmp"
        temporary.write_text(json.dumps(self.data, indent=2, allow_nan=False) + "\n")
        temporary.replace(self.output / "report.json")

    def record(self, item):
        self.data["runs"].append(item)
        with (self.output / "runs.jsonl").open("a") as stream:
            stream.write(json.dumps(item, allow_nan=False) + "\n")
        self.save()
        print(json.dumps(item, allow_nan=False), flush=True)

    async def run(self, engine, label, prompt, pipeline, refs=(), seeds=(42,)):
        import mlx.core as mx
        from PIL import Image

        self.data["active_run"] = label
        self.save()
        item = dict(
            label=label,
            pipeline=pipeline,
            seeds=list(seeds),
            batch_size=len(seeds),
            cache_before=engine.get_runtime_cache_stats(),
            memory_before=memory(),
        )
        mx.reset_peak_memory()
        start = time.perf_counter()
        try:
            outputs = await engine.generate_images(
                prompt=prompt,
                seeds=seeds,
                width=256,
                height=256,
                steps=4,
                image_paths=refs,
                pipeline=pipeline,
            )
            item.update(
                seconds=time.perf_counter() - start,
                status="complete",
                memory=memory(),
                cache_after=engine.get_runtime_cache_stats(),
                hashes=[],
            )
            require(len(outputs) == len(seeds), "Incomplete native batch")
            for index, output in enumerate(outputs):
                with Image.open(io.BytesIO(output)) as image:
                    require(image.size == (256, 256), "Incorrect output dimensions")
                (self.output / f"{label}-{index}.png").write_bytes(output)
                item["hashes"].append(hashlib.sha256(output).hexdigest())
        except BaseException as exc:
            item.update(
                status="failed",
                seconds=time.perf_counter() - start,
                error=str(exc),
                memory=memory(),
            )
            self.record(item)
            raise
        self.record(item)
        return outputs, item


async def operation(pool, evidence, model_id, pipeline, reference, batch4):
    prompt = (
        "Turn the scene into a watercolor painting."
        if reference
        else "A red ceramic teapot on a white table."
    )
    refs = (str(reference),) if reference else ()
    label = "edit" if reference else "txt2img"
    summary = dict(
        operation=label,
        pipeline=pipeline,
        cache_parity=[],
        serial_batch_pixels=[],
        uncached_seconds=[],
        cached_seconds=[],
        serial2_seconds=[],
        batch2_seconds=[],
    )
    evidence.data["summaries"].append(summary)
    acquire_start = time.perf_counter()
    binding = engine = None
    try:
        async with pool.acquire(model_id, image_pipeline=pipeline) as engine:
            summary["acquire_seconds"] = time.perf_counter() - acquire_start
            await evidence.run(engine, label + "-warmup", prompt, pipeline, refs)
            await engine.clear_prompt_caches(hot=True)
            binding = engine._backend._cache_bindings[id(engine._model)]
            for trial in range(evidence.data["trials"]):
                with binding.bypass():
                    old, item = await evidence.run(
                        engine, f"{label}-uncached-{trial}", prompt, pipeline, refs
                    )
                summary["uncached_seconds"].append(item["seconds"])
                await evidence.run(
                    engine, f"{label}-prime-{trial}", prompt, pipeline, refs
                )
                new, item = await evidence.run(
                    engine, f"{label}-cached-{trial}", prompt, pipeline, refs
                )
                summary["cached_seconds"].append(item["seconds"])
                summary["cache_parity"].append(compare(old[0], new[0]))
                evidence.save()
                require(old == new, "Cache changed PNG bytes")
            first_batch = None
            for trial in range(evidence.data["trials"]):
                one, first = await evidence.run(
                    engine, f"{label}-serial-{trial}-42", prompt, pipeline, refs
                )
                two, second = await evidence.run(
                    engine, f"{label}-serial-{trial}-43", prompt, pipeline, refs, (43,)
                )
                batch, item = await evidence.run(
                    engine, f"{label}-batch2-{trial}", prompt, pipeline, refs, (42, 43)
                )
                summary["serial2_seconds"].append(first["seconds"] + second["seconds"])
                summary["batch2_seconds"].append(item["seconds"])
                summary["serial_batch_pixels"].append(
                    [compare(one[0], batch[0]), compare(two[0], batch[1])]
                )
                evidence.save()
                require(
                    first_batch is None or batch == first_batch,
                    "Repeated batch changed PNG bytes",
                )
                first_batch = batch
            if batch4:
                await evidence.run(
                    engine, label + "-batch4", prompt, pipeline, refs, (42, 43, 44, 45)
                )
            summary.update(
                cache_speedup=statistics.median(summary["uncached_seconds"])
                / statistics.median(summary["cached_seconds"]),
                batch2_speedup=statistics.median(summary["serial2_seconds"])
                / statistics.median(summary["batch2_seconds"]),
                batch2_repeated_byte_identical=True
                if evidence.data["trials"] > 1
                else None,
                batch2_repetition_comparisons=evidence.data["trials"] - 1,
                cache_before_clear=engine.get_runtime_cache_stats(),
            )
            summary["clear_report"] = await engine.clear_prompt_caches(hot=True)
            summary["cache_after_clear"] = engine.get_runtime_cache_stats()
            require(
                summary["cache_after_clear"]["entries"]
                == summary["cache_after_clear"]["prediction_factory_entries"]
                == 0,
                "Cache clear left resident entries",
            )
            evidence.save()
    finally:
        # These references must not pin the old native model at a later switch/shutdown.
        binding = engine = None


async def api_proof(pool, evidence, model_id, reference):
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient
    from PIL import Image

    from omlx.api import image_routes

    previous = image_routes._get_engine_pool, image_routes._resolve_model
    image_routes._get_engine_pool = lambda: pool
    image_routes._resolve_model = lambda value: value
    app = FastAPI()
    app.include_router(image_routes.router)
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://offline-proof"
        ) as client:
            for label, route, extra, seeds in (
                (
                    "txt2img",
                    "generations",
                    {"pipeline": "flux2-klein-4b", "seed": 2**32 - 1},
                    [2**32 - 1, 0],
                ),
                (
                    "edit",
                    "edits",
                    {
                        "pipeline": "flux2-klein-4b/edit",
                        "seed": 42,
                        "image": [
                            base64.b64encode(reference.read_bytes()).decode("ascii")
                        ],
                    },
                    [42, 43],
                ),
            ):
                evidence.data["active_run"] = "api-" + label
                evidence.save()
                response = await client.post(
                    "/v1/images/" + route,
                    json=dict(
                        model=model_id,
                        prompt="A watercolor teapot",
                        size="256x256",
                        n=2,
                        batch_size=2,
                        steps=4,
                        **extra,
                    ),
                )
                row = dict(
                    operation=label, status=response.status_code, memory=memory()
                )
                evidence.data["api"].append(row)
                evidence.save()
                require(response.status_code == 200, response.text)
                row["seeds"] = [item["seed"] for item in response.json()["data"]]
                require(row["seeds"] == seeds, "API seed order changed")
                row["hashes"] = []
                for index, item in enumerate(response.json()["data"]):
                    png = base64.b64decode(item["b64_json"], validate=True)
                    with Image.open(io.BytesIO(png)) as image:
                        require(
                            image.size == (256, 256), "API output dimensions changed"
                        )
                    (evidence.output / f"api-{label}-{index}.png").write_bytes(png)
                    row["hashes"].append(hashlib.sha256(png).hexdigest())
                evidence.save()
    finally:
        image_routes._get_engine_pool, image_routes._resolve_model = previous


async def verify(source, reference, output, batch4, trials):
    import gc

    import mlx.core as mx

    from omlx.diffusion.preparation import _provenance, _source
    from omlx.engine_pool import EnginePool

    checkpoint = _source(source)
    require(
        checkpoint.base_model == "flux2-klein-4b",
        "Use a prepared FLUX.2 Klein 4B checkpoint",
    )
    provenance = _provenance(checkpoint)
    output.mkdir(parents=True, exist_ok=False)
    evidence = Evidence(output, source, trials)
    evidence.data.update(
        source_before=provenance,
        reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
        batch4_opt_in=batch4,
        comparison_scope="Singleton seed noise preserved; batched pixels need not equal serial pixels",
        timing_scope="Engine generation includes PNG serialization and request cleanup; serial2 sums two generation times",
    )
    evidence.save()
    error = None
    with tempfile.TemporaryDirectory(prefix="omlx-performance-discovery-") as directory:
        model_id = "local-flux2"
        (Path(directory) / model_id).symlink_to(source, target_is_directory=True)
        pool = EnginePool()
        try:
            pool.discover_models(directory)
            require(
                pool.get_model_ids() == [model_id],
                "Isolated discovery must contain only the supplied checkpoint",
            )
            for pipeline, ref in (
                ("flux2-klein-4b", None),
                ("flux2-klein-4b/edit", reference),
            ):
                await operation(pool, evidence, model_id, pipeline, ref, batch4)
            await api_proof(pool, evidence, model_id, reference)
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
            evidence.data["error"] = error
            # Do not retain native models through failed operation traceback frames.
            exc.__traceback__ = exc.__cause__ = exc.__context__ = None
        finally:
            try:
                await pool.shutdown()
                gc.collect()
                mx.synchronize()
                mx.clear_cache()
                evidence.data["after_shutdown_memory"] = memory()
            except BaseException as exc:
                error = error or f"{type(exc).__name__}: {exc}"
                evidence.data["cleanup_error"] = str(exc)
                exc.__traceback__ = exc.__cause__ = exc.__context__ = None
            try:
                evidence.data["source_after"] = _provenance(_source(source))
                unchanged = evidence.data["source_after"] == provenance
                evidence.data["source_header_size_provenance_unchanged"] = unchanged
                require(unchanged, "Source headers or file sizes changed")
            except BaseException as exc:
                error = error or f"{type(exc).__name__}: {exc}"
                evidence.data["provenance_error"] = str(exc)
                exc.__traceback__ = exc.__cause__ = exc.__context__ = None
            evidence.data.update(
                status="failed" if error else "complete", active_run=None
            )
            evidence.save()
    print(
        json.dumps(dict(output=str(output), status=evidence.data["status"])), flush=True
    )
    require(error is None, error)


if __name__ == "__main__":
    if sys.argv[1:] in (["--help"], ["-h"]):
        print(__doc__)
        sys.exit(0)
    require(
        len(sys.argv) in (4, 5) and (len(sys.argv) == 4 or sys.argv[4] == "--batch4"),
        __doc__,
    )
    source, reference, output = (
        Path(value).expanduser().resolve() for value in sys.argv[1:4]
    )
    require(
        source.is_dir() and reference.is_file(),
        "Provide an existing local checkpoint and PNG",
    )
    require(
        not output.exists()
        and source not in output.parents
        and output not in source.parents
        and output not in reference.parents
        and reference not in output.parents,
        "Use a new output directory separate from all inputs",
    )
    trials = int(os.environ.get("OMLX_DIFFUSION_PROOF_TRIALS", "3"))
    require(1 <= trials <= 10, "OMLX_DIFFUSION_PROOF_TRIALS must be 1..10")
    from PIL import Image

    require(reference.stat().st_size <= 8 * 1024 * 1024, "Reference PNG exceeds 8 MiB")
    with Image.open(reference) as image:
        require(
            image.format == "PNG"
            and getattr(image, "n_frames", 1) == 1
            and image.width * image.height <= 16 * 1024 * 1024
            and max(image.size) <= 8192,
            "Use a single-frame PNG within API input limits",
        )
        image.verify()
    asyncio.run(verify(source, reference, output, len(sys.argv) == 5, trials))
