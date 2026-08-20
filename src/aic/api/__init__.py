"""The HTTP layer — optional. ``fastapi`` is imported **lazily**.

The core ``aic`` package depends on no web framework; ``uv sync --extra backend`` adds
``fastapi`` and ``uvicorn``. That keeps `aic validate`, `aic build-index` and the whole test suite
runnable in an environment with no web packages at all.
"""

from __future__ import annotations

__all__ = ["SolveRequest", "SolveResponse", "create_app", "get_engine", "reset_engine"]

_APP_EXPORTS = frozenset({"create_app", "get_engine", "reset_engine"})
_SCHEMA_EXPORTS = frozenset({"SolveRequest", "SolveResponse"})


def __getattr__(name: str):
    # Lazy import: only touching aic.api.create_app pulls fastapi in.
    if name in _APP_EXPORTS:
        from . import app

        return getattr(app, name)
    if name in _SCHEMA_EXPORTS:
        from . import schemas

        return getattr(schemas, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
