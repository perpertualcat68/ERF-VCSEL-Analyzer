# Choosing Good Initial ERF Parameters

Good starting parameters dramatically improve fit quality, speed, and
robustness. Because every parameter of the ERF model has a direct physical
meaning, you can **read a reasonable initial guess straight off the linescan
profile**. This guide explains how, and why a single parameter set can be reused
for linescans taken at different lateral positions of the same cavity.

## 1. The model and parameter layout

A linescan is modeled as a baseline plus a sum of error-function steps, one per
interface:

$$
y(x) = k_1 + \sum_{i=1}^{N} a_i \, \operatorname{erf}\!\big(b_i\,(x - c_i)\big)
$$

The flat parameter vector is

```
[ k1 | a_1, b_1, c_1 | a_2, b_2, c_2 | ... ]
   ↑     ↑    ↑    ↑
baseline amp width position
```

so `k1` is the baseline and each triplet `(k_{3i-1}, k_{3i}, k_{3i+1})` is one
component: amplitude, width, position. The positions `k4, k7, k10, …` are the
layer edges.

## 2. Physical meaning of each parameter

| Parameter | Symbol | Meaning | How to read it from the profile |
|---|---|---|---|
| Baseline | $k_1$ | Vertical offset of the whole curve | Mean level, or the intensity of the first plateau |
| Amplitude | $a_i$ | Half the height of step $i$; **sign = step direction** | `+` if the profile **rises** left→right at that edge, `−` if it **falls**; $\lvert a_i\rvert \approx \Delta_i/2$ |
| Width | $b_i$ | Edge steepness (larger = sharper) | From the edge's 10–90% rise width $W_i$: $b_i \approx 1.81 / W_i$ |
| Position | $c_i$ | Location of edge $i$ (its inflection point) | $x$-coordinate where the step is steepest (gradient peak) |

### Why the amplitude sign matters

A single term $a\,\operatorname{erf}(b(x-c))$ goes from $-a$ to $+a$ as $x$
increases. So:

- **Rising** edge → the term must increase → $a > 0$.
- **Falling** edge → the term must decrease → $a < 0$.

And since the total change across the edge is from $-a$ to $+a$, the step height
is $\Delta = 2\lvert a\rvert$, i.e. $\lvert a\rvert \approx \Delta/2$.

### Why $b \approx 1.81 / W$

$\operatorname{erf}(z) = \pm 0.8$ at $z \approx \pm 0.906$, so the 10%–90% rise
occupies an argument span of $2 \times 0.906 = 1.812$. If that rise spans a
physical width $W$ in $x$, then $b\,W \approx 1.812$, hence
$b \approx 1.81 / W$. Equivalently, from the peak slope $s = y'(c)$ at the edge,
$b = s\sqrt{\pi} / \Delta = s\sqrt{\pi} / (2\lvert a\rvert)$.

## 3. Step-by-step manual estimation

1. **Baseline $k_1$.** Take the mean of the profile, or the flat level on one
   side. (The fit is insensitive to small baseline errors.)
2. **Find the edges.** Compute the numerical gradient $y'(x)$; each interface
   shows up as a peak in $\lvert y'(x)\rvert$.
3. **Positions $c_i$.** Set each $c_i$ to the $x$-value of a gradient peak (the
   steepest point of that edge).
4. **Amplitude signs.** Set $a_i > 0$ for a rising edge and $a_i < 0$ for a
   falling edge. In a periodic multilayer the edges alternate, so the signs
   alternate $+,-,+,-,\dots$
5. **Amplitude magnitudes.** Estimate the step height $\Delta_i$ at each edge and
   use $\lvert a_i\rvert \approx \Delta_i/2$.
6. **Widths $b_i$.** Measure the 10–90% rise width $W_i$ of each edge and use
   $b_i \approx 1.81 / W_i$. If all edges look similarly sharp, one common $b$ is
   usually fine to start.

### Tiny worked example

Suppose the profile sits near $100$, rises to $180$ across an edge centered at
$x = 12$ nm over a $10$–$90\%$ width of about $3$ nm, then falls back to $100$ at
an edge centered at $x = 30$ nm with a similar sharpness. A good initial guess:

- $k_1 = 140$ (mean of $100$ and $180$)
- Edge 1 (rising): $a_1 = +40$ (half of the $80$ step), $b_1 = 1.81/3 \approx 0.60$, $c_1 = 12$
- Edge 2 (falling): $a_2 = -40$, $b_2 \approx 0.60$, $c_2 = 30$

## 4. What the tool does automatically

When a linescan is loaded, **Set Parameters** builds a data-driven guess
(`vcsel_analyzer.core.erf_model.data_driven_initial_params`) as follows:

- **Baseline** = mean of the profile.
- **Positions** = peaks of $\lvert\text{gradient}\rvert$ (via
  `scipy.signal.find_peaks`); if detection fails, positions are spread evenly.
- **Amplitudes** = `intensity_range × (0.3–0.7) × (−1)^i`, i.e. a fraction of the
  peak-to-peak intensity with **alternating sign** per component.
- **Widths** = a random value in roughly $0.5$–$2.0$.

This is a solid automatic start, but the manual refinements above — especially
**getting each amplitude sign to match the real step direction** and setting
**widths from the measured edge sharpness** — typically give the fitter a much
better starting point and a higher-quality fit.

## 5. Reusing one parameter set across linescans

Linescans taken at **different lateral positions of the same cavity** sample the
same epitaxial stack: the number of edges, their order, the step directions
(signs), the approximate step heights, and the edge sharpness are all nearly
identical. Only the absolute positions $c_i$ shift slightly and the overall
intensity may scale a little. Therefore **one carefully tuned parameter set is an
excellent initial guess for every linescan of that structure.**

To reuse a set:

1. Tune parameters once on a representative linescan.
2. **Set Parameters → Save to File** (writes the `k<N>: <value>` text format).
3. For a new linescan, **Set Parameters → Load from File** to load the same
   starting values, then **Fit ERF**.

Caveats: if a scan is laterally offset, the positions $c_i$ may need a small
uniform shift; a strong FIB thickness wedge can tilt the baseline/amplitudes. In
those cases, re-detect positions or nudge $k_1$ and the amplitudes, but the
signs, widths and edge count usually carry over unchanged.

## 6. Practical tips

- **Signs first.** The single biggest lever is making every amplitude sign match
  the local slope direction.
- **Consistent widths.** If edges are similarly sharp, start them all at the same
  $b$; let the fit specialize them.
- **Order and count.** Keep positions sorted and match the number of components
  to the number of visible edges (`N = (total_params − 1) / 3`).
- **Sanity-check with the residuals.** After fitting, use the residual plot to
  see whether any edge needs a better initial sign/width.
