"""Logging for the library. The library does **not** print to stdout.

Basis: when ``aic`` is embedded in a backend process, any library ``print`` pollutes the
server's stdout (and with some runtimes, the stream returned to the client). The library emits
through :mod:`logging`; the caller decides how to display it. The CLI attaches a handler writing
to stderr; a backend attaches its own.

One deliberate exception: the index-building functions take a ``verbose`` flag and log progress
directly. Those are hand-run operations lasting tens of minutes, and the caller asked for it
explicitly.
"""

from __future__ import annotations

import logging
import sys

__all__ = ["log", "setup_cli_logging"]

#: The package root logger. Submodules use this directly.
log = logging.getLogger("aic")

# No default handler: this is the convention for a library, and it avoids duplicate output when
# the application has already configured logging itself.
log.addHandler(logging.NullHandler())

#: Marker attribute identifying the handler installed by :func:`setup_cli_logging`, so repeated
#: calls stay idempotent.
_CLI_HANDLER_FLAG = "_aic_cli_handler"


def setup_cli_logging(level: int = logging.INFO) -> None:
    """Attach a handler writing to **stderr**, for the CLI. Idempotent."""
    if any(getattr(handler, _CLI_HANDLER_FLAG, False) for handler in log.handlers):
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(message)s"))
    setattr(handler, _CLI_HANDLER_FLAG, True)
    log.addHandler(handler)
    log.setLevel(level)
