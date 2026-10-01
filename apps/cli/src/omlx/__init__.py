"""Public oMLX compatibility exports; implementation lives in workspace packages."""

from omlx_config._version import __version__ as __version__


def __getattr__(name):
    import omlx_runtime

    return getattr(omlx_runtime, name)
