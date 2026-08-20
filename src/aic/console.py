"""Make Vietnamese output printable on a Windows console.

The Windows console defaults to code page cp1252; any ``print`` containing Vietnamese
diacritics raises :class:`UnicodeEncodeError` and kills the process *mid-way* through a long
index build. Every entry point must call :func:`enable_utf8_stdio` before printing.

Vietnamese still reaches stdout after this rewrite — data such as query text, answer strings and
media titles is Vietnamese even though the code and comments are in English.
"""

from __future__ import annotations

import sys

__all__ = ["enable_utf8_stdio"]


def enable_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
