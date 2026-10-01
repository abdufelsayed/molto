"""Runtime graphics ledger regressions."""

import pytest


class TestMetalReleaseLag:
    GB = 1024**3

    def test_lag_is_graphics_above_the_settled_residual(self, monkeypatch):
        from molto_runtime.utils import metal_sync

        graphics = [16.1 * self.GB]
        monkeypatch.setattr(
            metal_sync, "get_graphics_footprint", lambda: int(graphics[0])
        )
        # 0.1 GB of Metal memory outside MLX is the settled level.
        assert metal_sync.unreleased_graphics_bytes(16 * self.GB) == 0
        # MLX dropped a 3 GB pool; the ledger still charges it.
        assert metal_sync.unreleased_graphics_bytes(13 * self.GB) == pytest.approx(
            3 * self.GB, abs=1
        )
        # The driver finished releasing it.
        graphics[0] = 13.1 * self.GB
        assert metal_sync.unreleased_graphics_bytes(13 * self.GB) == 0

    def test_settled_level_follows_the_recent_window(self, monkeypatch):
        from molto_runtime.utils import metal_sync

        now = [100.0]
        monkeypatch.setattr(metal_sync.time, "monotonic", lambda: now[0])
        monkeypatch.setattr(metal_sync, "get_graphics_footprint", lambda: 20 * self.GB)
        metal_sync.unreleased_graphics_bytes(19 * self.GB)
        # New Metal memory outside MLX reads as pending release until the
        # window settles it.
        now[0] += 1.0
        assert metal_sync.unreleased_graphics_bytes(18 * self.GB) == 1 * self.GB
        now[0] += metal_sync._RESIDUAL_WINDOW_S
        assert metal_sync.unreleased_graphics_bytes(18 * self.GB) == 0
