"""Public Molto compatibility exports; implementation lives in workspace packages."""

from molto_config._version import __version__ as __version__


def __getattr__(name):
    import molto_runtime

    return getattr(molto_runtime, name)
