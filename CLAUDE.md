# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Environment setup (Windows)
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

# Run all tests
python -m pytest

# Run a single test module
python -m pytest tests/test_fitting.py -q

# Launch the desktop analyzer
python erf_vcsel_analyzer_combined.py
```

Run all commands from the repository root so that `import vcsel_analyzer` resolves correctly (configured via `pytest.ini` `pythonpath = .`).

## Architecture

This project fits a **sum-of-error-function (ERF) model** to STEM intensity linescans of VCSEL cavities to extract individual layer thicknesses.

### Parameter layout

All numerical logic revolves around a flat parameter vector:

```
[k1, k2, k3, k4,  k5, k6, k7,  ...]
 ↑   ↑   ↑   ↑    ← next component →
 |   amp wid pos
 baseline
```

- `k1` — baseline intensity
- Every triplet `(k_{3i+2}, k_{3i+3}, k_{3i+4})` encodes one ERF component: amplitude, width, position
- Model: `y = k1 + Σ amplitude_i · erf(width_i · (x − position_i))`
- Default config (`ERF_CONFIG` in `vcsel_analyzer/config.py`): 97 total parameters → 32 ERF components

### Package layout (`vcsel_analyzer/`)

| Module | Responsibility |
|---|---|
| `config.py` | `ERF_CONFIG` dict — tolerances, figure sizes, layer counts |
| `core/erf_model.py` | Model evaluation (`build_erf_model`) and data-driven initial parameter guess (`data_driven_initial_params`) |
| `core/fitting.py` | `fit_erf()` → `FitResult` dataclass; wraps `scipy.optimize.curve_fit` with LM (default) or TRF (bounded widths) |
| `core/thickness.py` | Edge extraction from fitted params, consecutive-difference layer thicknesses, error propagation (covariance-based preferred, residual-based fallback) |
| `core/units.py` | Converts DM3 axis units (nm, µm, Å, …) to nm |
| `io/dm3_loader.py` | Loads DM3 microscopy files; tries HyperSpy first, ncempy fallback; returns `LoadedImage` dataclass |
| `io/params_io.py` | Reads/writes the `k<N>: <value>` parameters text format |
| `export/report.py` | Report generation |
| `logging_setup.py` | Logger configuration |

`erf_vcsel_analyzer_combined.py` is the Tkinter/Matplotlib GUI entry point. Keep reusable numerical logic in `vcsel_analyzer/core/` and file I/O in `vcsel_analyzer/io/`, not in the GUI file.

### Data flow

```
DM3 file ──► dm3_loader ──► pixel size (nm) + 2-D image array
                                     │
                              user draws linescan
                                     │
                          data_driven_initial_params
                                     │
                               fit_erf → FitResult
                                     │
                    edges_from_params → layer_thicknesses
                                     │
                         thickness_errors (covariance or residual)
                                     │
                              export / report
```

### Fitting modes

- **LM (default)**: `scipy.optimize.curve_fit` with Levenberg-Marquardt, unconstrained. Reproduces original behavior.
- **TRF (bounded)**: opt-in via `fit_erf(..., bounded=True)`; constrains ERF widths to be positive (eliminates amplitude/width sign degeneracy), uses trust-region-reflective algorithm.

### Tests

Tests mirror package modules: `tests/test_erf_model.py`, `tests/test_fitting.py`, `tests/test_thickness.py`, `tests/test_units.py`, `tests/test_params_io.py`. Use NumPy tolerance assertions (`np.testing.assert_allclose`) for floating-point results.
