"""Filesystem validation for configured storage directories."""

from pathlib import Path


def model_directory_access_error(path: Path) -> str | None:
    """Return a user-facing error if a model directory cannot be scanned."""
    try:
        if not path.exists():
            return f"Model directory does not exist: {path}"
        if not path.is_dir():
            return f"Model directory is not a directory: {path}"
        next(path.iterdir(), None)
    except OSError as e:
        return f"Model directory is not readable: {path} ({type(e).__name__}: {e})"
    return None
