"""Pytest bootstrap.

Placing a ``conftest.py`` at the repository root makes pytest add this
directory to ``sys.path`` (rootdir insertion), so tests can import the
``vcsel_analyzer`` package regardless of the working directory pytest is
invoked from.
"""

import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
