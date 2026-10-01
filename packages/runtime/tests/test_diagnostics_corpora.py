"""Bundled benchmark snapshots must retain their deterministic manifest values."""

import hashlib
import json
from importlib.resources import files

import pytest

CORPORA = files("molto_runtime.diagnostics").joinpath("bench_corpora")
MANIFEST = json.loads(CORPORA.joinpath("manifest.json").read_text())


@pytest.mark.parametrize("corpus", MANIFEST["corpora"], ids=lambda item: item["file"])
def test_bundled_snapshot_integrity(corpus):
    data = CORPORA.joinpath(corpus["file"]).read_bytes()
    assert len(data) == corpus["bytes"]
    assert len(data.decode("utf-8")) == corpus["characters"]
    assert hashlib.sha256(data).hexdigest() == corpus["sha256"]
