"""Bounded exact prompt embeddings and native prediction factories per model."""

from __future__ import annotations

import inspect
from collections import OrderedDict
from collections.abc import MutableMapping
from contextlib import contextmanager
from functools import wraps

_MISSING = object()


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
        self._factory_enabled = True
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
            result = original(*args, **kwargs)
            self._factories[name] = key, result
            self._factory_builds += 1
            return result

        self._replace(name, factory)

    def original_method(self, name):
        return self._original_methods.get(name, getattr(self.model, name))

    def stats(self):
        return {
            **self.cache.stats(),
            "prediction_factory_entries": len(self._factories),
            "prediction_factory_builds": self._factory_builds,
            "prediction_factory_reuses": self._factory_reuses,
        }

    def clear(self):
        count = len(self.cache) + len(self._factories)
        self.cache.clear()
        self._factories.clear()
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
