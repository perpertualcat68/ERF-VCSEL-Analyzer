"""Pure text-report construction helpers.

These build report strings from already-computed results, with no dependency on
matplotlib, Tk or instance state, so they can be unit tested directly.
"""

import numpy as np


def _stats(values):
    """Return ``(mean, std, min, max, count)`` for a sequence (safe on empty)."""
    if not values:
        return 0.0, 0.0, 0.0, 0.0, 0
    arr = np.asarray(values, dtype=np.float64)
    return (float(arr.mean()), float(arr.std()), float(arr.min()),
            float(arr.max()), int(arr.size))


def build_layer_summary_text(material_names, layer_thicknesses,
                             layer_thickness_errors=None, final_loss=None):
    """Build a concise per-material layer-thickness summary.

    Parameters
    ----------
    material_names : dict
        Mapping ``{'Material_A': name, 'Material_B': name}``.
    layer_thicknesses : dict
        Mapping ``{'Material_A': [...], 'Material_B': [...]}``.
    layer_thickness_errors : dict, optional
        Same shape as ``layer_thicknesses``; used to annotate mean +/- error.
    final_loss : float, optional
        Final fitting MSE to include in the header.
    """
    layer_thickness_errors = layer_thickness_errors or {}
    lines = []
    lines.append("=" * 60)
    lines.append("VCSEL LAYER THICKNESS SUMMARY")
    lines.append("=" * 60)
    if final_loss is not None:
        lines.append(f"Final fitting error (MSE): {final_loss:.2e}")
    total_layers = sum(len(v) for v in layer_thicknesses.values())
    lines.append(f"Total layers: {total_layers}")
    lines.append("")

    for key in ('Material_A', 'Material_B'):
        thicknesses = layer_thicknesses.get(key, [])
        name = material_names.get(key, key)
        mean, std, tmin, tmax, count = _stats(thicknesses)
        lines.append(f"[{name}]  ({count} layers)")
        if count:
            lines.append(f"  mean +/- std : {mean:.3f} +/- {std:.3f} nm")
            lines.append(f"  range        : [{tmin:.3f}, {tmax:.3f}] nm")
            errs = layer_thickness_errors.get(key, [])
            for i, t in enumerate(thicknesses):
                if i < len(errs) and errs[i] > 0:
                    lines.append(f"    Layer {i + 1}: {t:.3f} +/- {errs[i]:.3f} nm")
                else:
                    lines.append(f"    Layer {i + 1}: {t:.3f} nm")
        else:
            lines.append("  (no layers)")
        lines.append("")

    return "\n".join(lines)
