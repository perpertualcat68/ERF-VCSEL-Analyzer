"""
vcsel_analyzer
==============

Modular refactor of the original ``erf_vcsel_analyzer_combined.py`` tool.

The package separates concerns into:

- :mod:`vcsel_analyzer.config`        -- fitting configuration constants
- :mod:`vcsel_analyzer.logging_setup` -- stdout logging helper
- :mod:`vcsel_analyzer.core`          -- pure, GUI-free numerics (unit tested)
- :mod:`vcsel_analyzer.io`            -- DM3 loading and parameter JSON IO
- :mod:`vcsel_analyzer.export`        -- text report construction helpers

The interactive GUI class ``CombinedVCSELAnalyzer`` still lives in the original
top-level module and delegates its numerical kernels to this package.
"""

__all__ = ["config"]

__version__ = "1.0.0"
