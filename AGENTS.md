# Repository Guidelines

## Project Structure & Module Organization

`vcsel_analyzer/` contains the reusable Python package. Numerical logic belongs in `vcsel_analyzer/core/`, file parsing and DM3 loading in `vcsel_analyzer/io/`, and report generation in `vcsel_analyzer/export/`. Shared configuration and logging live at the package root.

`erf_vcsel_analyzer_combined.py` is the Tkinter/Matplotlib application entry point. Keep new reusable logic out of this large GUI module when it fits an existing package module.

Tests are under `tests/` and mirror package concerns, such as `test_fitting.py` and `test_params_io.py`. Root-level `.dm3` and `.txt` files are sample inputs or generated measurement data, not source modules.

## Build, Test, and Development Commands

- `python -m venv .venv` creates a local environment.
- `.venv\Scripts\Activate.ps1` activates it on Windows PowerShell.
- `python -m pip install -r requirements.txt` installs runtime and test dependencies.
- `python -m pytest` runs the complete suite configured by `pytest.ini`.
- `python -m pytest tests/test_fitting.py -q` runs one focused test module.
- `python erf_vcsel_analyzer_combined.py` launches the desktop analyzer.

There is no separate build step; run commands from the repository root so local package imports resolve correctly.

**Always activate the virtual environment and install dependencies before testing.** `pytest` is provided via `requirements.txt`, so a fresh shell will raise `No module named pytest` until you activate `.venv` and install. On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest
```

## Coding Style & Naming Conventions

Use four-space indentation and standard Python naming: `snake_case` for modules, functions, and variables; `PascalCase` for classes; and `UPPER_CASE` for constants. Add short docstrings to public functions and keep numerical operations vectorized with NumPy where practical. No formatter or linter is configured, so follow nearby style and keep diffs focused.

## Testing Guidelines

Tests use `pytest` with plain `test_*` functions. Name files `test_<module>.py`, cover invalid inputs and numerical edge cases, and use NumPy tolerance assertions for floating-point results. No coverage threshold is configured. Every behavioral change should include the smallest regression test that would fail without it.

## Commit & Pull Request Guidelines

This checkout contains no Git history, so no repository-specific commit convention can be verified. Use short imperative subjects such as `Fix unit conversion for reciprocal axes`. Pull requests should explain the behavior change, list verification commands, link relevant issues, and include screenshots only for visible GUI changes. Do not commit generated scan outputs or large microscopy files unless they are intentional test fixtures.
