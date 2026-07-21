"""Configuration constants for ERF fitting.

The values are identical to the original in-file ``ERF_CONFIG`` dictionary so
that default behavior is preserved after the refactor.
"""

ERF_CONFIG = {
    'target_error': 1e-10,  # Target fitting error
    'maxfev': 20000,  # Maximum function evaluations (bounded to guarantee termination)
    'ftol': 1e-8,  # Function tolerance (reachable for noisy measured data)
    'xtol': 1e-8,  # Parameter tolerance
    'gtol': 1e-8,  # Gradient tolerance
    'figure_size': (15, 10),
    'thickness_figure_size': (12, 8),
    'default_num_layers': 32,
    'max_layers': 200,
    'default_total_params': 97,
    'random_seed': 0  # seed for reproducible random initial parameters
}
