"""Read-only runtime observations and explicitly requested cache diagnostics."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import re
from contextlib import suppress

from omlx_contracts.runtime import RuntimeOperationError
from omlx_runtime.server_metrics import get_server_metrics

from omlx_management.management import ManagementError


class MonitoringService:
    def __init__(self, context):
        self.context = context
        self.pool = context.engine_pool

    @property
    def metrics(self):
        return (
            getattr(self.context.runtime_state, "metrics", None)
            or getattr(self.context.runtime_state, "server_metrics", None)
            or get_server_metrics()
        )

    def activity(self):
        return self.pool.activity_snapshot(self.metrics)

    def usage(self, period, model, include_details):
        # Historical canonical IDs can outlive the discovered model registry.
        if model and (
            model.startswith("/") or "\\" in model or ".." in model.split("/")
        ):
            raise ManagementError("invalid_configuration", "Use a canonical model ID")
        history = self.metrics.usage_history
        if history is None:
            return {
                "state": "unavailable",
                "enabled": False,
                "available": False,
                "dropped_requests": 0,
                "range": period,
                "totals": None,
                "models": [],
                "heatmap": [],
            }
        try:
            result = history.query(period, model, include_details=include_details)
        except Exception:
            return {
                "state": "unavailable",
                "enabled": history.enabled,
                "available": False,
                "dropped_requests": history.dropped_requests,
                "range": period,
                "totals": None,
                "models": [],
                "heatmap": [],
            }
        return {
            **result,
            "state": "disabled"
            if not result["enabled"]
            else "available"
            if result["available"]
            else "unavailable",
        }

    def logs(self, lines=100, file=None, level=None):
        settings = self.context.global_settings
        if settings is None:
            raise ManagementError("unavailable", "Server settings unavailable")
        directory = settings.logging.get_log_dir(settings.base_path).resolve()
        files = sorted(
            p.name
            for p in directory.glob("server.log*")
            if p.is_file()
            and not p.is_symlink()
            and re.fullmatch(r"server\.log(?:\.\d+|\.\d{4}-\d{2}-\d{2})?", p.name)
        )
        name = file or "server.log"
        if not re.fullmatch(r"server\.log(?:\.\d+|\.\d{4}-\d{2}-\d{2})?", name):
            raise ManagementError("invalid_configuration", "Invalid log file name")
        path = directory / name
        if path.is_symlink() or path.resolve().parent != directory:
            raise ManagementError("invalid_configuration", "Invalid log file")
        if file and name not in files:
            raise ManagementError("not_found", "Log file not found")
        max_scan_bytes = 1024 * 1024
        data = b""
        end = 0
        scan_truncated = False
        try:
            if path.exists():
                with path.open("rb") as stream:
                    stream.seek(0, 2)
                    end = stream.tell()
                    position = end
                    chunks = []
                    remaining = max_scan_bytes
                    while position and remaining:
                        count = min(65536, position, remaining)
                        position -= count
                        stream.seek(position)
                        chunks.append(stream.read(count))
                        remaining -= count
                    data = b"".join(reversed(chunks))
                    scan_truncated = position > 0
                    if scan_truncated:
                        # The first fragment may begin inside a physical line.
                        boundary = data.find(b"\n")
                        data = data[boundary + 1 :] if boundary >= 0 else b""
        except OSError as exc:
            raise ManagementError("unavailable", "Log file unavailable") from exc
        physical_lines = data.decode("utf-8", errors="replace").splitlines(
            keepends=True
        )
        severity = {
            "TRACE": 5,
            "DEBUG": 10,
            "INFO": 20,
            "WARNING": 30,
            "ERROR": 40,
            "CRITICAL": 50,
        }
        matches = []
        record_level = None
        # Continuations inherit the most recent parsed record. An initial orphan
        # continuation is excluded from filtered output because its level is unknown.
        for line in physical_lines:
            parsed_level = None
            if line.startswith("{"):
                with suppress(ValueError, TypeError):
                    record = json.loads(line)
                    if isinstance(record, dict):
                        parsed_level = record.get("level")
            else:
                match = re.match(
                    r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} - [\w.]+ - (TRACE|DEBUG|INFO|WARNING|ERROR|CRITICAL) - ",
                    line,
                )
                if match:
                    parsed_level = match.group(1)
            if parsed_level in severity:
                record_level = parsed_level
            if level is None or (
                record_level is not None and severity[record_level] >= severity[level]
            ):
                matches.append(line)
        return {
            "logs": "".join(matches[-lines:]),
            "total_lines": None if scan_truncated else len(physical_lines),
            "matched_lines": len(matches),
            "scanned_lines": len(physical_lines),
            "scanned_bytes": min(end, max_scan_bytes),
            "scan_truncated": scan_truncated,
            "scan_message": "Only the newest 1 MiB was searched; older log content and an incomplete leading line were excluded."
            if scan_truncated
            else None,
            "log_file": name,
            "available_files": files,
        }

    def versions(self):
        engines = {}
        from omlx_runtime.version_sources import version_sources

        fallback, declared = version_sources()
        for package in (
            "omlx",
            "mlx",
            "mlx-lm",
            "mlx-vlm",
            "mlx-embeddings",
            "mlx-audio",
            "mflux",
        ):
            info = {
                "name": package,
                "version": None,
                "commit": None,
                "url": None,
                "source": None,
            }
            try:
                dist = importlib.metadata.distribution(package)
                info["version"] = dist.version
                direct = json.loads(dist.read_text("direct_url.json") or "{}")
                vcs = direct.get("vcs_info", {})
                info.update(
                    commit=vcs.get("commit_id"),
                    source="direct_url" if direct else "distribution",
                )
                # Do not expose credentials embedded in source URLs.
                url = direct.get("url", "")
                from urllib.parse import urlsplit, urlunsplit

                parsed = urlsplit(url)
                if parsed.scheme in ("https", "http"):
                    info["url"] = urlunsplit(
                        (parsed.scheme, parsed.hostname or "", parsed.path, "", "")
                    )
                if not info["commit"] and package in fallback:
                    info.update(commit=fallback[package].get("commit"), source="bundle")
            except (importlib.metadata.PackageNotFoundError, ValueError, OSError):
                pass
            if not info["commit"] and package in fallback:
                info.update(commit=fallback[package].get("commit"), source="bundle")
            info["declared_dependency"] = declared.get(package)
            engines[package] = info
        return {
            "engines": engines,
            "python": platform.python_version(),
            "platform": platform.platform(),
        }

    def reset_stats(self, scope):
        from omlx_runtime.observability import reset_metrics

        try:
            return reset_metrics(self.metrics, scope)
        except RuntimeOperationError as exc:
            raise ManagementError(exc.code, exc.detail) from exc

    def probe(self, request):
        settings = self.context.settings_manager.get_settings(request.model_id)
        try:
            return self.pool.probe_cache(request, settings)
        except RuntimeOperationError as exc:
            raise ManagementError(exc.code, exc.detail) from exc
