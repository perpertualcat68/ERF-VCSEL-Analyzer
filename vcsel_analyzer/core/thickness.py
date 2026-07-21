"""Edge extraction, layer-thickness computation and error propagation (pure).

Faithful port of the numeric kernel of the original
``extract_fitting_results``.  Functions are side-effect free apart from
informational logging that reproduces the original console messages.
"""

import logging

import numpy as np

logger = logging.getLogger("vcsel_analyzer.thickness")


def edges_from_params(fitted_parameters):
    """Extract and sort ERF component descriptors from fitted parameters.

    Returns a tuple ``(edge_positions, param_indices, amplitudes, widths)`` with
    all four lists sorted by ascending edge position.
    """
    edge_positions = []
    param_indices = []
    amplitudes = []
    widths = []

    n_components = (len(fitted_parameters) - 1) // 3
    for i in range(n_components):
        if 3 * i + 3 < len(fitted_parameters):
            edge_positions.append(fitted_parameters[3 * i + 3])  # k4, k7, ...
            param_indices.append(3 * i + 3)
            amplitudes.append(fitted_parameters[3 * i + 1])      # k2, k5, ...
            widths.append(fitted_parameters[3 * i + 2])          # k3, k6, ...

    order = np.argsort(edge_positions)
    edge_positions = [edge_positions[j] for j in order]
    param_indices = [param_indices[j] for j in order]
    amplitudes = [amplitudes[j] for j in order]
    widths = [widths[j] for j in order]
    return edge_positions, param_indices, amplitudes, widths


def layer_thicknesses(edge_positions):
    """Consecutive differences between sorted edge positions."""
    return [edge_positions[i + 1] - edge_positions[i] for i in range(len(edge_positions) - 1)]


def thickness_errors(edge_positions, amplitudes, widths, param_indices,
                     covariance, fit_x, fit_y, model_fn):
    """Estimate per-layer thickness uncertainties.

    Method 1 (preferred): propagate the fit covariance of edge positions.
    Method 2 (fallback): residual-noise / local-ERF-gradient estimate.

    Returns ``(thickness_errors, error_source)`` where ``error_source`` is one
    of ``"covariance"``, ``"residual-based"`` or ``"none"``.
    """
    thickness_errs = []
    error_source = "none"
    n_layers = max(0, len(edge_positions) - 1)

    # --- Method 1: covariance-based -------------------------------------
    covariance_usable = False
    if covariance is not None:
        if np.isfinite(covariance).all():
            covariance_usable = True
        else:
            logger.info("  Covariance matrix contains inf/nan, using residual-based errors instead")

    if covariance_usable:
        try:
            for i in range(n_layers):
                idx_a = param_indices[i]
                idx_b = param_indices[i + 1]
                var_a = covariance[idx_a, idx_a]
                var_b = covariance[idx_b, idx_b]
                cov_ab = covariance[idx_a, idx_b]
                thickness_var = var_b + var_a - 2.0 * cov_ab
                if np.isfinite(thickness_var) and thickness_var > 0:
                    thickness_errs.append(float(np.sqrt(thickness_var)))
                else:
                    thickness_errs.append(0.0)
            if any(e > 0 for e in thickness_errs):
                error_source = "covariance"
                logger.info(f"  Thickness errors (covariance): {[f'{e:.4f}' for e in thickness_errs]}")
            else:
                thickness_errs = []  # fall through to method 2
        except Exception as cov_err:
            logger.info(f"  Warning: Covariance error propagation failed: {cov_err}")
            thickness_errs = []

    # --- Method 2: residual-based ---------------------------------------
    if not thickness_errs:
        try:
            x_data = np.asarray(fit_x, dtype=np.float64)
            y_data = np.asarray(fit_y, dtype=np.float64)
            y_fit = model_fn(x_data)
            residuals = y_data - y_fit

            data_range = float(x_data[-1] - x_data[0])
            half_window = max(data_range * 0.05, 1.0)

            edge_pos_errors = []
            for i in range(len(edge_positions)):
                pos = edge_positions[i]
                amp = amplitudes[i]
                wid = widths[i]

                mask = (x_data >= pos - half_window) & (x_data <= pos + half_window)
                if np.sum(mask) > 2:
                    sigma_local = float(np.std(residuals[mask]))
                else:
                    sigma_local = float(np.std(residuals))

                gradient = abs(amp * wid) * 2.0 / np.sqrt(np.pi)
                if gradient > 1e-12:
                    sigma_pos = sigma_local / gradient
                else:
                    sigma_pos = sigma_local
                edge_pos_errors.append(sigma_pos)

            for i in range(n_layers):
                sigma_a = edge_pos_errors[i]
                sigma_b = edge_pos_errors[i + 1]
                thickness_errs.append(float(np.sqrt(sigma_a ** 2 + sigma_b ** 2)))

            error_source = "residual-based"
            logger.info(f"  Edge position errors (residual-based): {[f'{e:.4f}' for e in edge_pos_errors]}")
            logger.info(f"  Thickness errors (residual-based): {[f'{e:.4f}' for e in thickness_errs]}")
        except Exception as res_err:
            logger.info(f"  Warning: Residual-based error estimation failed: {res_err}")
            thickness_errs = [0.0] * n_layers

    logger.info(f"  Error bar source: {error_source}")
    return thickness_errs, error_source


def assign_materials(thicknesses, errors):
    """Split thicknesses/errors into alternating Material_A / Material_B lists."""
    layer_thick = {'Material_A': [], 'Material_B': []}
    layer_err = {'Material_A': [], 'Material_B': []}
    for i, thickness in enumerate(thicknesses):
        err = errors[i] if i < len(errors) else 0.0
        key = 'Material_A' if i % 2 == 0 else 'Material_B'
        layer_thick[key].append(thickness)
        layer_err[key].append(err)
    return layer_thick, layer_err
