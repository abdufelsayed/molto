"""Process-owned managers and shared admission for management operations."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from molto_management.management import ManagementError, ManagementService
from molto_management.model_control import ModelControl


class ManagementRuntime:
    def __init__(self, context):
        self.context = context
        self.settings = context.global_settings
        base = (
            self.settings.base_path
            if self.settings
            else context.settings_manager.base_path
        )
        self.control = ModelControl(base)
        self.operation_lock = asyncio.Lock()
        self.mutation_lock = asyncio.Lock()
        self._managers: dict[str, Any] = {}
        self._closed = False
        self.path_activities: dict[str, list[Path]] = {}
        self._operation_path_owners: dict[str, str] = {}

    @property
    def roots(self) -> list[Path]:
        if self.settings is None:
            raise ManagementError("unavailable", "Server settings are unavailable")
        effective = getattr(self.settings, "get_effective_model_dirs", None)
        paths = (
            effective()
            if callable(effective)
            else self.settings.model.get_model_dirs(self.settings.base_path)
        )
        return [Path(p).expanduser().resolve() for p in paths]

    def normalized_paths(self, paths):
        """Reserve directory aliases and files reached through descendant links."""
        normalized = set()
        pending = []
        roots = self.roots
        try:
            for value in paths:
                path = Path(value).expanduser().absolute()
                resolved = path.resolve()
                normalized.update((path, resolved))
                pending.append(resolved)
            visited = set()
            while pending:
                directory = pending.pop()
                if directory in visited or not directory.is_dir():
                    continue
                visited.add(directory)
                # Include external link targets for validation, never walk them.
                if not any(directory.is_relative_to(root) for root in roots):
                    continue
                for child in directory.iterdir():
                    if child.is_symlink():
                        target = child.resolve()
                        normalized.update((child, target))
                        pending.append(target)
                    elif child.is_dir():
                        pending.append(child)
        except (OSError, RuntimeError) as exc:
            raise ManagementError(
                "invalid_configuration", "Cannot inspect model file ownership"
            ) from exc
        return list(normalized)

    def assert_paths_idle(self, paths, owner=None):
        for path in self.normalized_paths(paths):
            for active_owner, active_paths in self.path_activities.items():
                if active_owner == owner:
                    continue
                if any(
                    path.is_relative_to(active) or active.is_relative_to(path)
                    for active in active_paths
                ):
                    raise ManagementError(
                        "busy",
                        "Model files are in use by an active management operation",
                    )

    def assert_path_available(self, path):
        self.assert_paths_idle([path])

    def reserve_paths(self, paths, owner):
        paths = self.normalized_paths(paths)
        self.assert_paths_idle(paths, owner)
        self.path_activities[owner] = paths

    def release_paths(self, owner):
        self.path_activities.pop(owner, None)
        for operation_id, active_owner in list(self._operation_path_owners.items()):
            if active_owner == owner:
                self._operation_path_owners.pop(operation_id, None)

    def extend_operation_paths(self, operation_id, paths):
        owner = self._operation_path_owners[operation_id]
        self.reserve_paths([*self.path_activities[owner], *paths], owner)

    async def refresh(self):
        self.control.invalidate_storage_cache()
        return await ManagementService(self.context).refresh()

    async def refresh_when_idle(self):
        # Completed downloads must be discovered after a concurrent GPU job ends.
        while self.operation_lock.locked() or getattr(
            self.context.engine_pool, "preparation_active", False
        ):
            await asyncio.sleep(0.05)
        async with self.operation_lock:
            await self.refresh()

    def _manager(self, name):
        if self._closed:
            raise ManagementError("unavailable", "Management runtime is shutting down")
        roots = [str(p) for p in self.roots]
        if name not in self._managers:
            if name == "hf":
                from molto_management.hf_downloader import HFDownloader

                manager = HFDownloader(roots[0], on_complete=self.refresh_when_idle)
            elif name == "ms":
                from molto_management.ms_downloader import MSDownloader

                manager = MSDownloader(roots[0], on_complete=self.refresh_when_idle)
            elif name == "oq":
                from molto_management.oq_manager import OQManager

                manager = OQManager(roots)
            else:
                from molto_management.hf_uploader import HFUploader

                manager = HFUploader(roots)
            self._managers[name] = manager
        manager = self._managers[name]
        if name in {"hf", "ms"}:
            manager.update_model_dir(roots[0])
        else:
            manager.update_model_dirs(roots)
        return manager

    hf = property(lambda self: self._manager("hf"))
    ms = property(lambda self: self._manager("ms"))
    oq = property(lambda self: self._manager("oq"))
    uploader = property(lambda self: self._manager("uploader"))

    async def acquire_preparation(self):
        if self._closed:
            raise ManagementError("unavailable", "Management runtime is shutting down")
        if self.context.engine_pool.preparation_pending:
            raise ManagementError("busy", "Engine preparation is active")
        if self.operation_lock.locked():
            raise ManagementError(
                "busy", "Another preparation or diagnostic operation is active"
            )
        await self.operation_lock.acquire()

    @asynccontextmanager
    async def preparation_gate(self):
        await self.acquire_preparation()
        try:
            async with self.pool_preparation():
                yield
        finally:
            self.operation_lock.release()

    @asynccontextmanager
    async def pool_preparation(self):
        pool = self.context.engine_pool
        if getattr(pool, "preparation_active", False):
            raise ManagementError("busy", "Engine preparation is active")
        if hasattr(pool, "exclusive_preparation"):
            async with pool.exclusive_preparation(lambda: None):
                yield
        else:
            yield

    async def run_native(self, function, *args, **kwargs):
        """Cancellation waits for the native thread before releasing admission."""
        worker = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
        try:
            return await asyncio.shield(worker)
        except asyncio.CancelledError:
            # Repeated cancellation must not detach the native worker either.
            while not worker.done():
                try:
                    await asyncio.shield(worker)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
            if worker.done() and not worker.cancelled():
                worker.exception()
            raise

    async def start_exclusive(self, kind, runner, **kwargs):
        import uuid

        paths = kwargs.pop("paths", [])
        owner = uuid.uuid4().hex
        self.reserve_paths(paths, owner)
        try:
            await self.acquire_preparation()
        except BaseException:
            self.release_paths(owner)
            raise
        released = False

        def release(_=None):
            nonlocal released
            if not released:
                released = True
                self.operation_lock.release()
                self.release_paths(owner)

        async def guarded(operation_id):
            try:
                async with self.pool_preparation():
                    result = await runner(operation_id)
                await self.refresh()
                return result
            finally:
                release()

        try:
            operation = self.control.start_operation(kind, guarded, **kwargs)
        except BaseException:
            release()
            raise
        self._operation_path_owners[operation["id"]] = owner
        self.control._tasks[operation["id"]].add_done_callback(release)
        return operation

    async def cancel_quantization(self, manager, task_id):
        """Avoid the retained manager's unsafe timeout and thread detachment."""
        manager._cancelled.add(task_id)
        worker = manager._active_tasks.get(task_id)
        if worker:
            while not worker.done():
                try:
                    await asyncio.shield(worker)
                except asyncio.CancelledError:
                    continue
            from molto_management.oq_manager import QuantStatus

            task = manager._tasks[task_id]
            task.status = QuantStatus.CANCELLED
            output = Path(task.output_path)
            if (
                not output.is_symlink()
                and output.resolve().parent == self.roots[0]
                and output.is_dir()
            ):
                import shutil

                shutil.rmtree(output)
        return True

    async def shutdown(self):
        self._closed = True
        tasks = list(self.control._tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for manager in self._managers.values():
            await manager.shutdown()
