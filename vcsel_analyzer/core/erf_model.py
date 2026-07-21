"""The sum-of-ERF cavity model and data-driven initial guesses (pure).

Model:
    y = k1 + k2*erf(k3*(x-k4)) + k5*erf(k6*(x-k7)) + ...

Parameters are laid out as ``[baseline, (amp, width, pos) * n_components]``.
"""

import logging

import numpy as np
from scipy.special import erf
from scipy.signal import find_peaks

logger = logging.getLogger("vcsel_analyzer.erf_model")


def num_components(n_params):
    """Number of ERF components encoded by ``n_params`` parameters."""
    return (int(n_params) - 1) // 3


def build_erf_model(x, parameters):
    """Evaluate the ERF model at ``x`` for the given ``parameters``.

    Identical to the original ``build_erf_model_numpy``.
    """
    parameters = np.asarray(parameters, dtype=np.float64)
    result = parameters[0]  # k1 (baseline)

    n = num_components(len(parameters))
    for i in range(n):
        amplitude = parameters[3 * i + 1]  # k2, k5, k8, ...
        width = parameters[3 * i + 2]      # k3, k6, k9, ...
        position = parameters[3 * i + 3]   # k4, k7, k10, ...
        result = result + amplitude * erf(width * (x - position))

    return result


def data_driven_initial_params(positions, profile, total_params, seed=0):
    """Produce ``total_params`` initial ERF parameters from a linescan.

    Faithful, side-effect-light port of the numeric part of the original
    ``initialize_parameters``.  Uses a seeded RNG for reproducibility.
    """
    positions = np.asarray(positions, dtype=np.float64)
    profile = np.asarray(profile, dtype=np.float64)
    rng = np.random.RandomState(seed)

    baseline = float(np.mean(profile))
    intensity_range = float(np.max(profile) - np.min(profile))
    position_range = float(positions[-1] - positions[0])

    gradient = np.gradient(profile)
    gradient_abs = np.abs(gradient)

    try:
        peaks, _ = find_peaks(
            gradient_abs,
            height=np.std(gradient_abs),
            distance=max(1, len(gradient_abs) // 50),
        )
        detected_positions = positions[peaks]
    except Exception:
        num_c = (total_params - 1) // 3
        detected_positions = np.linspace(
            positions[0] + position_range * 0.1,
            positions[-1] - position_range * 0.1,
            num_c,
        )

    params = []
    for i in range(total_params):
        if i == 0:
            params.append(baseline)
            continue

        param_index = i - 1
        param_type = param_index % 3          # 0=amp, 1=width, 2=position
        component_index = param_index // 3

        if param_type == 0:  # amplitude
            value = intensity_range * (0.3 + 0.4 * rng.random_sample()) * ((-1) ** component_index)
        elif param_type == 1:  # width
            value = 0.5 + 1.5 * rng.random_sample()
        else:  # position
            if component_index < len(detected_positions):
                value = float(detected_positions[component_index])
            else:
                num_positions = (total_params - 1) // 3
                value = positions[0] + (component_index + 1) * position_range / (num_positions + 1)
        params.append(value)

    return params
