"""Server integration coverage for persisted CORS configuration."""

import tempfile
from pathlib import Path

from molto_config.settings import GlobalSettings


class TestCORSMiddleware:
    """Test that CORS middleware is correctly applied to the server."""

    def test_cors_preflight(self):
        """Test that CORS preflight requests get proper response headers."""
        from fastapi.testclient import TestClient
        from molto_server.server import create_app

        app = create_app()

        # Reset middleware stack so add_middleware works even if app was
        # already started by another test in the same process.
        app.middleware_stack = None

        with tempfile.TemporaryDirectory() as tmpdir:
            settings = GlobalSettings(base_path=Path(tmpdir))
            app.state.controller.initialize(
                model_dirs=[tmpdir],
                global_settings=settings,
            )

            client = TestClient(app)
            resp = client.options(
                "/v1/models",
                headers={
                    "Origin": "https://example.com",
                    "Access-Control-Request-Method": "GET",
                },
            )
            assert resp.status_code == 200
            assert "access-control-allow-origin" in resp.headers
            assert resp.headers["access-control-allow-origin"] == "*"
