"""Length-unit conversion helpers (pure, GUI-free)."""

import logging

logger = logging.getLogger("vcsel_analyzer.units")

# Factors that convert a given length unit into nanometers.
LENGTH_FACTORS = {
    'nm': 1.0, 'nanometer': 1.0, 'nanometers': 1.0,
    'µm': 1000.0, 'μm': 1000.0, 'um': 1000.0,
    'micron': 1000.0, 'microns': 1000.0,
    'micrometer': 1000.0, 'micrometers': 1000.0,
    'mm': 1e6, 'millimeter': 1e6, 'millimeters': 1e6,
    'm': 1e9, 'meter': 1e9, 'meters': 1e9,
    'pm': 1e-3, 'picometer': 1e-3, 'picometers': 1e-3,
    'a': 0.1, 'å': 0.1, 'ang': 0.1,
    'angstrom': 0.1, 'angstroms': 0.1, 'ångström': 0.1,
}


def convert_scale_to_nm(scale, units):
    """Convert an axis scale given in ``units`` to nanometers.

    Returns the scale value expressed in nm (float), or ``None`` if ``units``
    is not a usable length unit (e.g. reciprocal-space or angular units).
    Unrecognized units are assumed to already be in nm (with a warning),
    preserving backward-compatible behavior.
    """
    try:
        scale = float(scale)
    except (TypeError, ValueError):
        return None

    if units is None:
        return scale  # assume already in nm

    unit_str = str(units).strip().lower()

    # Reciprocal-space / angular units are not lengths
    if unit_str.startswith('1/') or unit_str.startswith('/') or 'rad' in unit_str:
        logger.info(f"Warning: axis units '{units}' are not a length unit; cannot convert to nm.")
        return None

    if unit_str in LENGTH_FACTORS:
        return scale * LENGTH_FACTORS[unit_str]

    logger.info(f"Warning: unrecognized axis units '{units}'; assuming nanometers.")
    return scale
