"""Export oMLX routes without constructing engines or starting jobs."""
import json
from pathlib import Path
from fastapi import FastAPI
from omlx.api.management_routes import router
from omlx.api.management_setup_routes import router as setup_router
app = FastAPI(title="oMLX management API", version="1")
app.include_router(router)
app.include_router(setup_router)
destination = Path(__file__).resolve().parents[1] / "src/lib/openapi.json"
destination.write_text(json.dumps(app.openapi(), indent=2) + "\n")
