"""DM3 image loading and pixel-size extraction.

Heavy microscopy dependencies (HyperSpy / ncempy) are imported lazily inside
the functions so that importing this module never fails when they are absent.
"""

import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

from vcsel_analyzer.core.units import convert_scale_to_nm

logger = logging.getLogger("vcsel_analyzer.dm3")

DEFAULT_PIXEL_SIZE = 0.246  # nm per pixel (common for HRTEM)


@dataclass
class LoadedImage:
    """Result of :func:`load_dm3`."""
    data: Any
    metadata: dict
    signal: Optional[Any]  # HyperSpy signal (None when loaded via ncempy)


def load_dm3(file_path):
    """Load a DM3 file, trying HyperSpy first then ncempy.

    Returns a :class:`LoadedImage`, or ``None`` if the file cannot be read.

    Fixes over the original implementation: when ``hyperspy.load`` returns a
    list of signals (multi-dataset DM3), the first signal is used instead of
    crashing on ``list.data``.
    """
    if not file_path:
        return None
    if not os.path.exists(file_path):
        logger.info(f"Error: File {file_path} not found.")
        return None

    logger.info(f"Loading DM3 file: {file_path}")

    # --- Try HyperSpy ---------------------------------------------------
    try:
        import hyperspy.api as hs
        try:
            logger.info("Using HyperSpy...")
            signal = hs.load(file_path)
            if isinstance(signal, (list, tuple)):
                if not signal:
                    raise ValueError("HyperSpy returned an empty signal list")
                logger.info(f"  HyperSpy returned {len(signal)} datasets; using the first one.")
                signal = signal[0]
            loaded = LoadedImage(
                data=signal.data,
                metadata=signal.metadata.as_dictionary(),
                signal=signal,
            )
            logger.info("\u2713 Successfully loaded with HyperSpy")
            logger.info(f"  Image shape: {loaded.data.shape}")
            logger.info(f"  Data type: {loaded.data.dtype}")
            return loaded
        except Exception as e:
            logger.info(f"\u2717 HyperSpy failed: {e}")
    except ImportError:
        pass

    # --- Try ncempy -----------------------------------------------------
    try:
        import ncempy.io as ncempy_io
        try:
            logger.info("Using ncempy...")
            dm3_data = ncempy_io.dm.dmReader(file_path)
            loaded = LoadedImage(
                data=dm3_data['data'],
                metadata=dm3_data.get('tags', {}),
                signal=None,
            )
            logger.info("\u2713 Successfully loaded with ncempy")
            logger.info(f"  Image shape: {loaded.data.shape}")
            return loaded
        except Exception as e:
            logger.info(f"\u2717 ncempy failed: {e}")
    except ImportError:
        pass

    logger.info("\n\u2717 No DM3 reading libraries available!")
    logger.info("Please install: pip install hyperspy")
    return None


def get_pixel_size(signal=None, metadata=None):
    """Extract pixel size (nm/pixel) from a HyperSpy signal or DM3 metadata.

    Falls back to :data:`DEFAULT_PIXEL_SIZE` when nothing usable is found.
    """
    # First: HyperSpy axes manager (most reliable)
    if signal is not None:
        try:
            if hasattr(signal, 'axes_manager'):
                axes_manager = signal.axes_manager
                if hasattr(axes_manager, 'signal_axes') and len(axes_manager.signal_axes) > 0:
                    first_axis = axes_manager.signal_axes[0]
                    if hasattr(first_axis, 'scale') and hasattr(first_axis, 'units'):
                        units = first_axis.units
                        converted = convert_scale_to_nm(first_axis.scale, units)
                        if converted is not None:
                            logger.info(f"Found pixel size from HyperSpy axes: {first_axis.scale} {units} -> {converted} nm")
                            return converted
                        logger.info(f"Could not interpret HyperSpy axis units '{units}'; continuing to fallback.")
        except Exception as e:
            logger.info(f"Could not extract pixel size from HyperSpy axes: {e}")

    # Fallback: DM3 metadata (legacy)
    if metadata:
        try:
            if 'ImageList' in metadata:
                image_data = metadata['ImageList']['TagGroup0']['ImageData']
                if 'Calibrations' in image_data:
                    cal = image_data['Calibrations']['Dimension']
                    if len(cal) > 0:
                        pixel_size = float(cal[0].get('Scale', 1.0))
                        unit = cal[0].get('Units', 'nm')
                        converted = convert_scale_to_nm(pixel_size, unit)
                        if converted is not None:
                            logger.info(f"Found pixel size from metadata: {pixel_size} {unit} -> {converted} nm")
                            return converted
                        logger.info(f"Could not interpret metadata units '{unit}'; continuing to fallback.")
        except Exception as e:
            logger.info(f"Could not extract pixel size from metadata: {e}")

    logger.info(f"Using default pixel size: {DEFAULT_PIXEL_SIZE} nm/pixel")
    return DEFAULT_PIXEL_SIZE
