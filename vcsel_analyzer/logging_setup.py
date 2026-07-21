"""Lightweight logging setup.

Core modules use :func:`logging.getLogger` instead of ``print`` so that they
stay side-effect free and testable.  :func:`setup_logging` routes those records
to ``stdout`` using a bare ``%(message)s`` format, which reproduces the original
``print`` output text (including the check/cross/warning glyphs).

On terminals whose encoding cannot represent those glyphs, the handler falls
back to ASCII-safe replacements instead of raising ``UnicodeEncodeError``.
"""

import logging
import sys

_CONFIGURED = False


class _SafeStreamHandler(logging.StreamHandler):
    """StreamHandler that never crashes on non-encodable characters."""

    def emit(self, record):
        try:
            super().emit(record)
        except UnicodeEncodeError:
            try:
                msg = self.format(record)
                stream = self.stream
                enc = getattr(stream, "encoding", None) or "ascii"
                stream.write(msg.encode(enc, "replace").decode(enc) + self.terminator)
                self.flush()
            except Exception:
                self.handleError(record)


def setup_logging(level=logging.INFO):
    """Configure the ``vcsel_analyzer`` logger to emit plain text to stdout.

    Idempotent: repeated calls do not add duplicate handlers.
    """
    global _CONFIGURED
    logger = logging.getLogger("vcsel_analyzer")
    if not _CONFIGURED:
        handler = _SafeStreamHandler(stream=sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.propagate = False
        _CONFIGURED = True
    logger.setLevel(level)
    return logger
