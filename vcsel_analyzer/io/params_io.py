"""Read/write ERF parameters in the tool's on-disk text format.

The format matches what the GUI produces/consumes::

    # VCSEL ERF Parameters - Saved 2026-07-04 19:11:00
    # Total parameters: 97
    ...
    k1: 1.000000000000
    k2: -0.500000000000
    ...

Comment lines start with ``#``; parameter lines are ``k<N>: <value>`` (a bare
``<value>`` per line is also accepted on read).  These functions are pure
(no dialogs), which makes them straightforward to unit test.
"""

import math
from datetime import datetime


def format_parameters_file(params, final_loss=None, timestamp=None):
    """Return the full text content for a parameters file."""
    params = list(params)
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    amplitudes = params[1::3]
    widths = params[2::3]
    positions = params[3::3]

    lines = []
    lines.append(f"# VCSEL ERF Parameters - Saved {timestamp}")
    lines.append(f"# Total parameters: {len(params)}")
    lines.append(f"# ERF components: {(len(params) - 1) // 3}")
    if final_loss is not None:
        lines.append(f"# Final fitting error: {final_loss:.2e}")
    lines.append("#")
    lines.append("# Parameter Range Summary:")
    if params:
        lines.append(f"#   Baseline (k1): {params[0]:.6f}")
    if amplitudes:
        lines.append(f"#   Amplitudes (k2, k5, k8, ...): [{min(amplitudes):.2f}, {max(amplitudes):.2f}]")
        lines.append(f"#     Count: {len(amplitudes)}")
    if widths:
        lines.append(f"#   Widths (k3, k6, k9, ...): [{min(widths):.4f}, {max(widths):.4f}]")
        lines.append(f"#     Count: {len(widths)}")
    if positions:
        lines.append(f"#   Positions (k4, k7, k10, ...): [{min(positions):.2f}, {max(positions):.2f}] nm")
        lines.append(f"#     Count: {len(positions)}")
        lines.append(f"#     Span: {max(positions) - min(positions):.2f} nm")
    lines.append("#")
    lines.append("# Format: k<N>: <value>")
    lines.append("# All values use consistent 12-digit precision")
    lines.append("")
    for i, param in enumerate(params):
        lines.append(f"k{i + 1}: {param:.12f}")
    return "\n".join(lines) + "\n"


def parse_parameters_text(text):
    """Parse parameter text, returning ``(params, parse_errors)``.

    Mirrors the GUI parser: ignores blank/``#`` lines, accepts ``k<N>: value``
    or a bare ``value``, and rejects non-finite numbers.
    """
    params = []
    parse_errors = []
    for line_number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        try:
            if ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    value = float(parts[1].strip())
                    if not math.isfinite(value):
                        parse_errors.append(f"Line {line_number}: non-finite value")
                        continue
                    params.append(value)
                else:
                    parse_errors.append(f"Line {line_number}: invalid format")
            else:
                value = float(line)
                if not math.isfinite(value):
                    parse_errors.append(f"Line {line_number}: non-finite value")
                    continue
                params.append(value)
        except ValueError:
            parse_errors.append(f"Line {line_number}: cannot parse '{line}'")
    return params, parse_errors


def write_parameters_file(path, params, final_loss=None, timestamp=None):
    """Write ``params`` to ``path`` in the standard text format."""
    content = format_parameters_file(params, final_loss=final_loss, timestamp=timestamp)
    with open(path, 'w') as f:
        f.write(content)


def read_parameters_file(path):
    """Read a parameters file, returning ``(params, parse_errors)``."""
    with open(path, 'r') as f:
        return parse_parameters_text(f.read())
