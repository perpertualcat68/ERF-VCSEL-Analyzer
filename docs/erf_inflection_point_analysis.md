# When Are the ERF Centers $k_4, k_7, k_{10}, \dots$ Valid Inflection Points?

## Summary

The sum-of-error-function (ERF) model fits STEM intensity linescans with

$$
y(x) = k_1 + \sum_{m \ge 1} a_m \, \operatorname{erf}\!\big(b_m (x - c_m)\big),
$$

where each component $m$ has amplitude $a_m = k_{3m-1}$, width $b_m = k_{3m}$, and
center $c_m = k_{3m+1}$. Thus $k_4 = c_1,\; k_7 = c_2,\; k_{10} = c_3, \dots$ are the
component centers, which we use as **layer edges (inflection points)**.

For a single isolated ERF term the center is **exactly** the inflection point.
For the multi-term sum, neighboring transitions shift the true inflection point by
a small amount that is **exponentially suppressed** with edge separation. This note
derives that shift and gives a quantitative range in which treating
$k_4, k_7, k_{10}, \dots$ as inflection points is accurate.

---

## 1. Single ERF term: the center is exactly the inflection point

For one component $f_m(x) = a_m \operatorname{erf}\big(b_m (x - c_m)\big)$, using
$\operatorname{erf}'(z) = \tfrac{2}{\sqrt{\pi}} e^{-z^2}$:

$$
f_m'(x) = \frac{2}{\sqrt{\pi}}\, a_m b_m \, e^{-b_m^2 (x - c_m)^2},
$$

$$
f_m''(x) = -\frac{4}{\sqrt{\pi}}\, a_m b_m^{3} \, (x - c_m)\, e^{-b_m^2 (x - c_m)^2}.
$$

Hence $f_m''(x) = 0 \iff x = c_m$, and $f_m''$ changes sign there. Therefore the
inflection point of an isolated ERF term is **rigorously** at $x = c_m$
(i.e. $k_4, k_7, \dots$), with no approximation.

---

## 2. Superposition: origin and magnitude of the deviation

The full model is a sum, so $y''(x) = \sum_j f_j''(x)$. Evaluated at $x = c_i$:

$$
y''(c_i) = \underbrace{f_i''(c_i)}_{=\,0} + \sum_{j \ne i} f_j''(c_i)
         = \sum_{j \ne i} f_j''(c_i) \neq 0 .
$$

The component's own curvature vanishes at its center, but **neighboring components**
contribute nonzero curvature, so the true inflection point $x_i^\*$ (where
$y''(x_i^\*) = 0$) is shifted by $\delta_i = x_i^\* - c_i$. A first-order expansion
$y''(c_i + \delta_i) \approx y''(c_i) + \delta_i\, y'''(c_i) = 0$ gives

$$
\delta_i \approx -\frac{y''(c_i)}{y'''(c_i)}
= -\frac{\displaystyle\sum_{j \ne i} a_j b_j^{3} (c_i - c_j)\, e^{-b_j^{2}(c_i - c_j)^2}}
        {\displaystyle a_i b_i^{3} + \sum_{j \ne i} a_j b_j^{3}\big(1 - 2 b_j^{2}(c_i - c_j)^2\big) e^{-b_j^{2}(c_i - c_j)^2}} .
$$

The denominator is $y'''(c_i) = -\tfrac{4}{\sqrt{\pi}}\big[a_i b_i^3 + \cdots\big]$,
whose leading term is the sharp central component $a_i b_i^3$.

The crucial feature is the **Gaussian suppression factor**
$e^{-b_j^{2}(c_i - c_j)^2}$. When the separation $d = |c_i - c_j|$ to a neighbor is
large relative to the transition width, this factor vanishes exponentially, so
$\delta_i \to 0$.

Assuming comparable neighbor amplitudes and widths
($a_j \sim a_i,\; b_j \sim b_i = b$), the denominator $\approx a_i b_i^3$ and

$$
\frac{|\delta_i|}{d} \;\lesssim\; e^{-(b\,d)^2}.
$$

---

## 3. Dimensionless criterion $b\,d$ and quantitative range

Let $d$ be the spacing between adjacent edges ($\approx$ layer thickness) and $b$
the ERF width parameter of the edge. Define the coupling $\varepsilon = e^{-(bd)^2}$:

| $b\,d$ | $\varepsilon = e^{-(bd)^2}$ | Relative deviation $\|\delta\|/d$ |
|:---:|:---:|:---:|
| 1.0 | 0.37       | ~37%  (edges overlap heavily — not valid) |
| 1.5 | 0.105      | ~10%  |
| 2.0 | 1.8×10⁻²   | ~1.8% |
| 2.5 | 1.9×10⁻³   | ~0.19% |
| 3.0 | 1.2×10⁻⁴   | ~0.012% |
| 3.5 | 4.8×10⁻⁶   | ~5×10⁻⁴ % |

Requiring $|\delta|/d < \eta$ inverts to a required separation:

$$
b\,d \;\gtrsim\; \sqrt{\ln(1/\eta)}:\quad
\eta = 1\% \Rightarrow b d \ge 2.15,\quad
\eta = 0.1\% \Rightarrow b d \ge 2.63,\quad
\eta = 0.01\% \Rightarrow b d \ge 3.03 .
$$

---

## 4. Restatement via physical edge width

The derivative of an ERF edge is a Gaussian with standard deviation
$\sigma = \dfrac{1}{b\sqrt{2}}$, so

$$
\text{FWHM} = 2\sqrt{2\ln 2}\,\sigma = \frac{2\sqrt{\ln 2}}{b} \approx \frac{1.665}{b},
\qquad
W_{10\text{-}90} \approx \frac{1.812}{b}.
$$

Then $b\,d \ge 2$ is equivalent to

$$
d \;\gtrsim\; 2.83\,\sigma \;\approx\; 1.20\,\text{FWHM} \;\approx\; 1.10\, W_{10\text{-}90}.
$$

**Engineering conclusion.** As long as each layer thickness (adjacent-edge spacing)
is at least about $1.1$–$1.2\times$ the edge transition width (i.e. $b\,d \gtrsim 2$),
treating $k_4, k_7, k_{10}, \dots$ as inflection points has a relative deviation
$< 2\%$. If $b\,d \gtrsim 3$ (layer thickness $\gtrsim 1.7\times$ the 10–90% edge
width), the deviation is $< 0.01\%$ and is entirely negligible for engineering
purposes.

---

## 5. Additional remarks

- **Layer-thickness error is even smaller.** A thickness is a difference
  $L_k = c_{k+1} - c_k$, so its systematic error is $\delta_{k+1} - \delta_k$. In a
  VCSEL multilayer the neighboring amplitudes $a_j$ alternate in sign and spacings
  are nearly regular, so the leading terms of $\delta$ partially cancel. The table
  above is therefore a **conservative upper bound**; the actual thickness bias is
  usually smaller.

- **Two distinct notions of "edge."** The derivation above concerns the deviation of
  $c_m$ from the **inflection point of the model itself**. Whether that inflection
  point coincides with the **true physical interface** depends on imaging/contrast
  symmetry assumptions and is a separate modeling question outside this analysis.

## One-line takeaway

A single ERF center is exactly its inflection point; the shift caused by
overlapping neighbors is suppressed by $e^{-(bd)^2}$, so whenever adjacent edges are
separated by $b\,d \gtrsim 2\text{–}3$, the centers $k_4, k_7, k_{10}, \dots$ can be
safely regarded as layer edges (inflection points) in the engineering-mathematics
sense.
