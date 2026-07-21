"""High-precision ERF curve fitting (pure, GUI-free)."""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import curve_fit

from vcsel_analyzer.core.erf_model import build_erf_model

logger = logging.getLogger("vcsel_analyzer.fitting")


@dataclass
class FitResult:
    """Container for the outcome of :func:`fit_erf`."""
    params: list
    covariance: Optional[np.ndarray]
    mse: float
    rmse: float
    max_error: float
    elapsed_s: float
    param_errors: Optional[list]


def _curve_fit_model(x, *params):
    return build_erf_model(x, params)


def fit_erf(x, y, p0, config, *, bounded=False, sigma=None):
    """Fit the sum-of-ERF model to ``(x, y)`` starting from ``p0``.

    Default behavior (``bounded=False``) reproduces the original
    ``scipy.optimize.curve_fit`` call exactly: Levenberg-Marquardt with the
    tolerances from ``config``.

    ``bounded=True`` is an opt-in improvement that switches to the
    trust-region-reflective algorithm and constrains the ERF widths
    (``k3, k6, ...``) to be positive, removing the amplitude/width sign
    degeneracy.  ``sigma`` (per-point measurement noise) is passed through to
    ``curve_fit`` when provided.
    """
    import time

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    p0 = np.asarray(p0, dtype=np.float64)

    kwargs = dict(
        p0=p0,
        maxfev=config.get('maxfev', 20000),
        ftol=config.get('ftol', 1e-8),
        xtol=config.get('xtol', 1e-8),
        gtol=config.get('gtol', 1e-8),
    )
    if sigma is not None:
        kwargs['sigma'] = np.asarray(sigma, dtype=np.float64)
        kwargs['absolute_sigma'] = True

    if bounded:
        lower = np.full(p0.shape, -np.inf)
        upper = np.full(p0.shape, np.inf)
        # width parameters live at indices 2, 5, 8, ... -> must be positive
        lower[2::3] = 1e-12
        # ensure the initial guess respects the bounds: widths must be > 0
        p0 = p0.copy()
        widths0 = np.abs(p0[2::3])
        widths0[widths0 == 0] = 1e-6
        p0[2::3] = widths0
        kwargs['p0'] = p0
        kwargs['bounds'] = (lower, upper)
        kwargs['method'] = 'trf'
    else:
        kwargs['method'] = 'lm'

    start = time.time()
    fitted_params, covariance = curve_fit(_curve_fit_model, x, y, **kwargs)
    elapsed = time.time() - start

    y_fitted = build_erf_model(x, fitted_params)
    residuals = y - y_fitted
    mse = float(np.mean(residuals ** 2))
    rmse = float(np.sqrt(mse))
    max_error = float(np.max(np.abs(residuals)))

    param_errors = None
    if covariance is not None:
        param_errors = np.sqrt(np.diag(covariance)).tolist()

    return FitResult(
        params=fitted_params.tolist(),
        covariance=covariance,
        mse=mse,
        rmse=rmse,
        max_error=max_error,
        elapsed_s=elapsed,
        param_errors=param_errors,
    )
