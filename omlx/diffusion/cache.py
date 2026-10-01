"""Bounded exact embeddings, reference conditioning and prediction factories."""

from __future__ import annotations

import gc
import hashlib
import inspect
from collections import OrderedDict
from collections.abc import MutableMapping
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from functools import wraps
from pathlib import Path

_MISSING = object()
_REFERENCE = object()


def _arrays(value):
    import mlx.core as mx

    found = {}

    def walk(item):
        if isinstance(item, mx.array):
            found[id(item)] = item
        elif isinstance(item, dict):
            for child in item.values():
                walk(child)
        elif isinstance(item, (tuple, list)):
            for child in item:
                walk(child)

    walk(value)
    return list(found.values())


def _reference_compute(**kwargs):
    from mflux.models.flux2.variants.edit.flux2_klein_edit_helpers import (
        _Flux2KleinEditHelpers,
    )

    return _Flux2KleinEditHelpers.prepare_reference_image_conditioning(**kwargs)


def _reference_digests(paths):
    digests = []
    for path in paths:
        digest = hashlib.sha256()
        with Path(path).open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        digests.append(digest.hexdigest())
    return tuple(digests)


def _broadcast_reference(result, batch_size):
    import mlx.core as mx

    if not isinstance(result, tuple) or len(result) != 2:
        raise ValueError("Native reference conditioning must return latents and IDs")
    expanded = []
    for value in result:
        if value is None:
            expanded.append(None)
        elif isinstance(value, mx.array) and value.ndim == 3 and value.shape[0] == 1:
            expanded.append(mx.broadcast_to(value, (batch_size, *value.shape[1:])))
        else:
            raise ValueError(
                "Native reference conditioning must have singleton batch dimensions"
            )
    return tuple(expanded)


class PromptCache(MutableMapping):
    """An LRU of materialized embeddings; stats never contain prompt data."""

    def __init__(self, max_entries=32, max_bytes=64 * 1024 * 1024):
        if max_entries < 1 or max_bytes < 1:
            raise ValueError("Prompt cache bounds must be positive")
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self.enabled = True
        self._items = OrderedDict()
        self._bytes = 0
        self._hits = self._misses = self._evictions = 0

    def __getitem__(self, key):
        if not self.enabled or key not in self._items:
            self._misses += 1
            raise KeyError(key)
        self._hits += 1
        self._items.move_to_end(key)
        return self._items[key][0]

    def __contains__(self, key):
        present = self.enabled and key in self._items
        if not present:
            self._misses += 1
        return present

    def __setitem__(self, key, value):
        if not self.enabled:
            return
        arrays = _arrays(value)
        size = sum(array.nbytes for array in arrays)
        if size > self.max_bytes:
            if key in self._items:
                del self[key]
            return
        if arrays:
            import mlx.core as mx

            mx.eval(*arrays)
        if key in self._items:
            del self[key]
        while self._items and (
            len(self._items) >= self.max_entries or self._bytes + size > self.max_bytes
        ):
            _, (_, old_size) = self._items.popitem(last=False)
            self._bytes -= old_size
            self._evictions += 1
        self._items[key] = value, size
        self._bytes += size

    def __delitem__(self, key):
        _, size = self._items.pop(key)
        self._bytes -= size

    def __iter__(self):
        return iter(self._items)

    def __len__(self):
        return len(self._items)

    def clear(self):
        self._items.clear()
        self._bytes = 0

    def stats(self):
        return {
            "entries": len(self),
            "bytes": self._bytes,
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
            "max_entries": self.max_entries,
            "max_bytes": self.max_bytes,
        }


def _semantic(value):
    """Keep complete scalar encoder arguments, without ambiguous string joins."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return type(value).__name__, value
    if isinstance(value, (tuple, list)):
        return type(value).__name__, tuple(_semantic(child) for child in value)
    if isinstance(value, dict):
        return "dict", tuple(
            sorted((key, _semantic(child)) for key, child in value.items())
        )
    raise TypeError("Encoder argument is not suitable for exact prompt caching")


def _prediction_signature(value):
    """Describe native specialization inputs without tensor identity or contents.

    Tensors contribute shape and dtype. Scalars and nested static containers
    contribute their complete values. Opaque mutable objects (including native
    KV-cache objects) bypass retention rather than assuming their state is fixed.
    """
    import mlx.core as mx

    if isinstance(value, mx.array):
        return "tensor", tuple(value.shape), str(value.dtype)
    if value is None or isinstance(value, (str, int, float, bool)):
        return type(value).__name__, value
    if isinstance(value, (tuple, list)):
        return type(value).__name__, tuple(
            _prediction_signature(child) for child in value
        )
    if isinstance(value, dict):
        return "dict", tuple(
            sorted((key, _prediction_signature(child)) for key, child in value.items())
        )
    raise TypeError(
        "Mutable prediction arguments cannot retain a compiled specialization"
    )


class _PredictionProxy:
    """Keep one native callable for the current invocation signature only.

    A single mx.compile callable otherwise retains every encountered shape.
    Reconstruct through the original native factory on a signature change, so
    native hardware selection and prediction formulas remain untouched.
    """

    def __init__(self, owner, factory, args, kwargs, native):
        self._owner = owner
        self._factory = factory
        self._args = args
        self._kwargs = kwargs
        self._native = native
        self._signature = _MISSING

    def clear(self):
        self._native = None
        self._signature = _MISSING

    def _build(self):
        native = self._factory(*self._args, **self._kwargs)
        self._owner._factory_builds += 1
        return native

    def __call__(self, *args, **kwargs):
        try:
            signature = _prediction_signature((args, kwargs))
        except TypeError:
            native = self._native if self._signature is _MISSING else None
            self.clear()
            gc.collect()
            if native is None:
                native = self._build()
            self._owner._signature_bypasses += 1
            # Do not retain this native callable after an opaque input call.
            return native(*args, **kwargs)
        if self._signature is not _MISSING and self._signature != signature:
            self.clear()
            # Release the previous compiled closure before building its
            # replacement; it may own materialized captured model tensors.
            gc.collect()
            self._owner._shape_rebuilds += 1
        elif self._signature == signature:
            self._owner._signature_reuses += 1
        if self._native is None:
            self._native = self._build()
        self._signature = signature
        return self._native(*args, **kwargs)


class PromptCacheBinding:
    """Instance-only native hooks, restored on release and bypassed in calibration."""

    def __init__(
        self,
        model,
        pipeline_id,
        base_model,
        *,
        max_entries=32,
        max_bytes=64 * 1024 * 1024,
    ):
        self.model = model
        self.pipeline_id = pipeline_id
        self.cache = PromptCache(max_entries, max_bytes)
        self._attributes = {}
        self._module_items = {}
        self._original_methods = {}
        self._factories = {}
        self._factory_builds = self._factory_reuses = 0
        self._shape_rebuilds = self._signature_reuses = self._signature_bypasses = 0
        self._factory_enabled = True
        self._reference_vae = None
        self._reference_hits = self._reference_misses = 0
        # mflux < Qwen-Image-2.1 joins positive/negative prompts with a delimiter,
        # so distinct semantic pairs can collide. Bypass that native cache.
        if base_model.startswith("qwen-image") and base_model != "qwen-image-2.1":
            self.cache.enabled = False
        if hasattr(model, "prompt_cache"):
            self._replace("prompt_cache", self.cache)
        if base_model.startswith("flux2-klein"):
            if hasattr(model, "_encode_prompt_pair"):
                self._wrap_encoder("_encode_prompt_pair")
            for name in ("_predict", "_cached_predict"):
                if hasattr(model, name):
                    self._wrap_factory(name)
        elif base_model.startswith("z-image") and hasattr(model, "_encode_prompts"):
            self._wrap_encoder("_encode_prompts")

    def _replace(self, name, value):
        self._attributes[name] = vars(self.model).get(name, _MISSING)
        # nn.Module stores dict-valued attributes as parameter-tree items.
        # Remove that entry as well as hiding the replacement from __setattr__.
        if isinstance(self.model, MutableMapping) and name in self.model:
            self._module_items[name] = self.model.pop(name)
        object.__setattr__(self.model, name, value)

    def _wrap_encoder(self, name):
        original = getattr(self.model, name)
        signature = inspect.signature(original)
        self._original_methods[name] = original

        @wraps(original)
        def encode(*args, **kwargs):
            if not self.cache.enabled:
                return original(*args, **kwargs)
            try:
                bound = signature.bind(*args, **kwargs)
                bound.apply_defaults()
                key = (self.pipeline_id, name, _semantic(dict(bound.arguments)))
            except TypeError:
                return original(*args, **kwargs)
            try:
                return self.cache[key]
            except KeyError:
                result = original(*args, **kwargs)
                self.cache[key] = result
                return result

        self._replace(name, encode)

    def _wrap_factory(self, name):
        original = getattr(self.model, name)
        signature = inspect.signature(original)
        self._original_methods[name] = original

        @wraps(original)
        def factory(*args, **kwargs):
            if not self._factory_enabled:
                return original(*args, **kwargs)
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            # Factories receive the actual transformer instance; retain only
            # their current argument identities, never a history of models.
            key = tuple(bound.arguments.items())
            existing = self._factories.get(name)
            if (
                existing is not None
                and len(existing[0]) == len(key)
                and all(
                    old_name == new_name and old_value is new_value
                    for (old_name, old_value), (new_name, new_value) in zip(
                        existing[0], key
                    )
                )
            ):
                self._factory_reuses += 1
                return existing[1]
            if existing is not None:
                existing[1].clear()
            native = original(*args, **kwargs)
            result = _PredictionProxy(self, original, args, kwargs, native)
            self._factories[name] = key, result
            self._factory_builds += 1
            return result

        self._replace(name, factory)

    def reference_conditioning(
        self, *, vae, tiling_config, image_paths=None, batch_size=1
    ):
        """Cache exact singleton VAE conditioning, then broadcast per request.

        Uploaded files may have different temporary names. Content hashes,
        ordered references, VAE identity and every tiling setting define reuse;
        output dimensions and generation seeds do not affect this native step.
        """
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("Reference conditioning batch size must be positive")
        paths = tuple(image_paths or ())
        arguments = dict(
            vae=vae, tiling_config=tiling_config, image_paths=paths, batch_size=1
        )
        if not self.cache.enabled or not paths:
            return _broadcast_reference(_reference_compute(**arguments), batch_size)
        try:
            snapshot = _semantic(
                asdict(tiling_config) if is_dataclass(tiling_config) else tiling_config
            )
        except TypeError:
            # Unknown mutable configuration cannot be represented exactly.
            return _broadcast_reference(_reference_compute(**arguments), batch_size)
        if self._reference_vae is not vae:
            for key in list(self.cache):
                if isinstance(key, tuple) and key and key[0] is _REFERENCE:
                    del self.cache[key]
            self._reference_vae = vae
        digests = _reference_digests(paths)
        key = (_REFERENCE, self.pipeline_id, id(vae), snapshot, digests)
        try:
            result = self.cache[key]
            self._reference_hits += 1
        except KeyError:
            self._reference_misses += 1
            result = _reference_compute(**arguments)
            # Materialization is performed by cache insertion. Recheck inputs
            # after native loading so a concurrently changed file is not admitted.
            try:
                unchanged = _reference_digests(paths) == digests
            except OSError:
                unchanged = False
            if unchanged:
                self.cache[key] = result
        return _broadcast_reference(result, batch_size)

    def original_method(self, name):
        return self._original_methods.get(name, getattr(self.model, name))

    def stats(self):
        # CPython builds this tuple under the GIL. The following Python-level
        # filtering can then safely run while the MLX worker mutates the LRU.
        items = tuple(self.cache._items.items())
        reference_items = [
            item
            for key, item in items
            if isinstance(key, tuple) and key and key[0] is _REFERENCE
        ]
        totals = self.cache.stats()
        return {
            **totals,
            "prompt_hits": totals["hits"] - self._reference_hits,
            "prompt_misses": totals["misses"] - self._reference_misses,
            "reference_hits": self._reference_hits,
            "reference_misses": self._reference_misses,
            "reference_entries": len(reference_items),
            "reference_bytes": sum(size for _, size in reference_items),
            "prediction_factory_entries": len(self._factories),
            "prediction_factory_builds": self._factory_builds,
            "prediction_factory_reuses": self._factory_reuses,
            "prediction_signatures": sum(
                proxy._signature is not _MISSING
                for _, proxy in tuple(self._factories.values())
            ),
            "prediction_shape_rebuilds": self._shape_rebuilds,
            "prediction_signature_reuses": self._signature_reuses,
            "prediction_signature_bypasses": self._signature_bypasses,
        }

    def clear(self):
        count = len(self.cache) + len(self._factories)
        self.cache.clear()
        for _, proxy in tuple(self._factories.values()):
            proxy.clear()
        self._factories.clear()
        self._reference_vae = None
        return count

    @contextmanager
    def bypass(self):
        enabled, factory_enabled = self.cache.enabled, self._factory_enabled
        self.clear()
        self.cache.enabled = self._factory_enabled = False
        try:
            yield
        finally:
            self.clear()
            self.cache.enabled, self._factory_enabled = enabled, factory_enabled

    def release(self):
        self.clear()
        for name, previous in self._attributes.items():
            if previous is _MISSING:
                object.__delattr__(self.model, name)
            else:
                object.__setattr__(self.model, name, previous)
            if name in self._module_items:
                previous_item = self._module_items[name]
                if name == "prompt_cache":
                    previous_item.clear()
                self.model[name] = previous_item
        self._attributes.clear()
        self._module_items.clear()
        self._original_methods.clear()
