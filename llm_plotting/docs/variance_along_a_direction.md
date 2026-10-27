# Variance along a fitted direction — definitions

What the variance row of the per-session SVM figure plots, and how its curves relate.
Implementation: `plotlib/variance.py`, `var_plotting_lib.py`.

## Samples

$n$ trials, $p$ units. Trial $i$'s sample is its mean over the `TIMEBINS_PER_FIT` native
40 ms bins of that fit window (L2-normalised per sample iff the fit did that):

$$\mathbf{x}_i \in \mathbb{R}^{p},\qquad \bar{\mathbf{x}} = \tfrac{1}{n}\textstyle\sum_i \mathbf{x}_i$$

## Covariance, trace, total variance

$$\widehat{\Sigma} = \frac{1}{n-1}\sum_{i=1}^{n}(\mathbf{x}_i-\bar{\mathbf{x}})(\mathbf{x}_i-\bar{\mathbf{x}})^{\!\top}
\qquad
T = \operatorname{tr}\widehat{\Sigma} = \sum_{j=1}^{p}\widehat{\operatorname{Var}}(x_{\cdot j}) = \sum_{k=1}^{p}\lambda_k$$

The trace is basis-independent, which is what makes it a legitimate denominator for a
direction chosen in any basis. Under per-sample L2 normalisation ($\lVert\mathbf{x}_i\rVert=1$):

$$T = \frac{n}{n-1}\bigl(1-\lVert\bar{\mathbf{x}}\rVert^{2}\bigr)$$

All variances use `ddof = 1`, so the $n/(n-1)$ factors are real, not cosmetic.

## Variance along a direction — the plotted quantity

$$\hat{\mathbf{u}} = \mathbf{w}/\lVert\mathbf{w}\rVert
\qquad
V(\mathbf{u}) = \mathbf{u}^{\!\top}\widehat{\Sigma}\,\mathbf{u} = \widehat{\operatorname{Var}}(X\mathbf{u})
\qquad
\boxed{\ \mathrm{VE}(\mathbf{u}) = \frac{\mathbf{u}^{\!\top}\widehat{\Sigma}\,\mathbf{u}}{\operatorname{tr}\widehat{\Sigma}}\ }$$

Dimensionless, in $[0,1]$. Unbiased at any $n$ **provided $\mathbf{u}$ was not fitted on
this $X$** — which is why the PC reference needs cross-validation to be comparable.

## The two reference bands

Ceiling — the top-$K$ principal directions, averaged:

$$\mathrm{VE}(\mathbf{v}_k) = \lambda_k/T
\qquad
\overline{\mathrm{VE}}_K = \frac{1}{K}\sum_{k=1}^{K}\frac{\lambda_k}{T}$$

Floor — $\mathbf{u}$ uniform on the unit sphere. Writing $\mathbf{u}=V\mathbf{q}$ with
$\mathbf{q}=\mathbf{z}/\lVert\mathbf{z}\rVert$, $\mathbf{z}\sim\mathcal{N}(0,I_p)$, the
squared coordinates are Dirichlet:

$$\mathrm{VE}(\mathbf{u}) = \frac{\sum_k \lambda_k c_k}{\sum_k \lambda_k},
\qquad (c_1,\dots,c_p)\sim\mathrm{Dirichlet}(\tfrac12,\dots,\tfrac12),
\qquad \mathbb{E}_{\mathbf{u}}[\mathrm{VE}] = \frac{1}{p}$$

So the band is percentiles of that law, not a line at $1/p$; it is wide whenever the
eigenvalues are unequal.

## Per-condition curves

Groups $g\in\{0,1\}$, index sets $I_g$, $n_g=|I_g|$:

$$\widehat{\Sigma}_g = \frac{1}{n_g-1}\sum_{i\in I_g}(\mathbf{x}_i-\bar{\mathbf{x}}_g)(\mathbf{x}_i-\bar{\mathbf{x}}_g)^{\!\top},
\qquad T_g = \operatorname{tr}\widehat{\Sigma}_g,
\qquad \boxed{\ \mathrm{VE}_g(\mathbf{u}) = \frac{\mathbf{u}^{\!\top}\widehat{\Sigma}_g\,\mathbf{u}}{T_g}\ }$$

Own denominator: *"of the variance this condition has, the share along $\mathbf{u}$"*.

## Do the two condition curves sum to the pooled one? **No**

The exact decomposition (law of total variance) is:

$$(n-1)\,\widehat{\Sigma}
= \underbrace{\sum_g (n_g-1)\widehat{\Sigma}_g}_{\text{within}}
+ \underbrace{\sum_g n_g(\bar{\mathbf{x}}_g-\bar{\mathbf{x}})(\bar{\mathbf{x}}_g-\bar{\mathbf{x}})^{\!\top}}_{\text{between}}$$

Projected onto $\mathbf{u}$ and divided by $T$:

$$\mathrm{VE}(\mathbf{u})
= \sum_g \frac{n_g-1}{n-1}\cdot\frac{T_g}{T}\cdot \mathrm{VE}_g(\mathbf{u})
\;+\; \frac{B(\mathbf{u})}{T},
\qquad
B(\mathbf{u}) = \frac{1}{n-1}\sum_g n_g\bigl(\mathbf{u}^{\!\top}(\bar{\mathbf{x}}_g-\bar{\mathbf{x}})\bigr)^{2}$$

The plotted curves carry neither the weights $\frac{n_g-1}{n-1}\frac{T_g}{T}$ nor the
between term, so they cannot sum to the pooled curve — and note
$\sum_g\frac{n_g-1}{n-1} = \frac{n-2}{n-1} < 1$, so even the weighted terms alone can't.

For two groups the between term is the projected class gap, squared:

$$\Delta(\mathbf{u}) = \mathbf{u}^{\!\top}(\bar{\mathbf{x}}_1-\bar{\mathbf{x}}_0),
\qquad
B(\mathbf{u}) = \frac{n_0 n_1}{n(n-1)}\,\Delta(\mathbf{u})^{2}
\;\xrightarrow[n_0=n_1]{}\;
\frac{n}{n-1}\Bigl(\frac{\Delta}{2}\Bigr)^{2}$$

**Measured** (S15, cue, $n_0=36$, $n_1=47$, 79 timebins): the identity holds to
$1.7\times10^{-17}$; $T_g/T = 0.99$ for both groups, so the two conventions differ almost
only by the weights; $\mathrm{VE}_0+\mathrm{VE}_1 = 1.43\times\mathrm{VE}$ (p10 1.15,
p90 1.71); and $B/\mathrm{VE} = 0.28$ — **that last number is the visible offset of the
black pooled curve above both condition curves.**

## Relative ratios worth quoting

**Which condition the axis fluctuates in.**
$\rho_{\mathrm{var}} = \mathrm{VE}_1(\mathbf{u})/\mathrm{VE}_0(\mathbf{u})$. If
$\mathbf{u}$ were fixed a priori and the samples Gaussian, $V_1/V_0 \sim F(n_1-1,n_0-1)$
scaled by the true ratio — it is not, since $\mathbf{u}$ was fitted on these trials, so use
a label-permutation null instead.

**How much of this axis's variance *is* the condition difference.**
$\rho_{\mathrm{between}}(\mathbf{u}) = B(\mathbf{u})/V(\mathbf{u}) \in [0,1]$; measured
median 0.24–0.32 across sessions and targets (max 0.53–0.67).

**Why a mean-difference readout of the same $\mathbf{w}$ is a different (and more
circular) thing.** An SVM maximises $\Delta$, so a gap readout is the most fitted quantity
available and its sign is fixed by which class is labelled 1. It does *not* optimise
$\rho_{\mathrm{var}}$. The only term the two share is $\Delta^2$, through $B$ — which is a
quarter to a third of $\mathrm{VE}$, hence the measured $r(\Delta,\mathrm{VE}) = 0.37$–$0.84$
rather than $\approx 1$.
