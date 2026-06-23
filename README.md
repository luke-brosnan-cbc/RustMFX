# RustMFX

[![Build Status](https://github.com/luke-brosnan-cbc/RustMFX/actions/workflows/CI.yml/badge.svg)](https://github.com/luke-brosnan-cbc/RustMFX/actions/workflows/CI.yml)
[![PyPI version](https://badge.fury.io/py/rustmfx.svg)](https://pypi.org/project/rustmfx/)

A Rust-backed Python package for computing average marginal effects (AMEs) and standard errors in binary choice models (Logit and Probit). The main motivation is performance: `statsmodels.get_margeff()` becomes slow and memory-intensive as $N$ and $K$ grow, primarily due to Python-level loops and intermediate array allocations. RustMFX replaces these with vectorised matrix operations and in-place computation via `ndarray` (BLAS backend).

---

## Installation

```bash
pip install rustmfx
```

---

## Why It's Faster

`statsmodels.get_margeff()` iterates over observations for continuous marginal effects and loops over each discrete variable, recomputing probabilities twice per observation. RustMFX computes both in single matrix operations. Memory usage is the more significant difference: `statsmodels` creates multiple intermediate arrays that grow with $K$, while RustMFX reuses memory in-place. The performance gap widens substantially as $K$ increases.

---

## Matching statsmodels

`rustmfx.mfx(model)` reproduces `statsmodels.get_margeff()` **exactly** — identical AMEs and standard errors to floating-point precision (typically within ~1e-15), for both Logit and Probit.

The only knob is `dummy`, which maps directly onto statsmodels' own flag:

| RustMFX call | Equivalent statsmodels call |
|---|---|
| `mfx(model)` | `model.get_margeff()` |
| `mfx(model, dummy=True)` | `model.get_margeff(dummy=True)` |

With `dummy=False` (the default) every non-constant regressor — including columns that happen to be 0/1 — is treated as continuous, exactly as statsmodels does by default. Set `dummy=True` to treat strictly-binary columns as discrete, using the finite-difference effect $F(\cdot\mid x{=}1) - F(\cdot\mid x{=}0)$.

---

## Standard Errors

Standard errors are computed by the delta method,

$$\text{Var}(\text{AME}) = J \cdot \text{Var}(\hat{\beta}) \cdot J^{T},$$

where $J$ is the exact analytic Jacobian of the average marginal effect with respect to $\beta$:

$$J_{jl} = \frac{\partial\, \text{AME}_j}{\partial \beta_l} = \delta_{jl}\,\frac{1}{N}\sum_{i=1}^{N} f(X_i\beta) \;+\; \beta_j\,\frac{1}{N}\sum_{i=1}^{N} f'(X_i\beta)\,x_{il}$$

for continuous variables (with $f'$ the derivative of the density: $f'(z)=f(z)(1-2F(z))$ for logit, $f'(z)=-z\,\phi(z)$ for probit). For discrete variables under `dummy=True`, $J$ is the difference of the predicted-probability gradients evaluated at $x=1$ and $x=0$.

This is the same Jacobian statsmodels obtains by numerical differentiation, which is why the standard errors are identical.

> **Note (v0.2.0):** earlier versions exposed a second `se_method='rust'` option that used a different, incorrect off-diagonal Jacobian term and did not match statsmodels (or Stata). It has been removed — there is now a single, correct method and no `se_method` argument.

---

## Methodology

### Average Marginal Effects

For observation $i$ and variable $j$, the marginal effect is:

$$\frac{\partial P(Y=1 \mid X_i)}{\partial X_{ij}} = \beta_j \cdot f(X_i \beta)$$

where $f(\cdot)$ is the PDF of the assumed distribution:

- **Logit:** $f(z) = \dfrac{e^z}{(1+e^z)^2}$

- **Probit:** $f(z) = \dfrac{1}{\sqrt{2\pi}} e^{-0.5 z^2}$

The AME averages this over all $N$ observations:

$$\text{AME}_j = \frac{1}{N} \sum_{i=1}^{N} \beta_j \cdot f(X_i \beta)$$

### Continuous vs. Discrete Variables

For **continuous** variables, the marginal effect is the derivative above.

For **discrete** (binary) variables, a finite-difference approach is used instead — but **only when `dummy=True`**:

$$\Delta P = P(Y=1 \mid X_{ij}=1) - P(Y=1 \mid X_{ij}=0)$$

This matches `statsmodels.get_margeff(dummy=True)`. With the default `dummy=False`, binary columns are treated as continuous, exactly as statsmodels does by default.

---

## Usage

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.datasets import make_classification
import rustmfx

nobs = 1_000_000
num_continuous = 4
num_dummy = 4
num_features = num_continuous + num_dummy
num_informative = np.random.randint(1, num_features + 1)

X, y = make_classification(
    n_samples=nobs,
    n_features=num_features,
    n_informative=num_informative,
    n_redundant=0,
    n_classes=2,
    random_state=42
)

continuous_names = [f"continuous_{i+1}" for i in range(num_continuous)]
dummy_names = [f"dummy_{i+1}" for i in range(num_dummy)]
X = pd.DataFrame(X, columns=continuous_names + dummy_names)
y = pd.DataFrame(y, columns=["outcome"])

X[dummy_names] = (X[dummy_names] > X[dummy_names].median()).astype(int)
X = sm.add_constant(X)

results_probit = sm.Probit(y, X).fit(disp=0)
results_logit = sm.Logit(y, X).fit(disp=0)
```

> **Note:** `y` and `X` passed to `sm.{Model}(y, X).fit()` must be `pandas.DataFrame` objects, and column names must be strings. This ensures the fit object output is a DataFrame, which is required by `rustmfx.mfx()`.

### Probit — default (`get_margeff()` equivalent)

```python
rustmfx.mfx(results_probit)
```

Output:
```
|              |        dy/dx |    Std. Err |           z |   Pr(>|z|) |   Conf. Int. Low |   Conf. Int. Hi | Significance   |
|:-------------|-------------:|------------:|------------:|-----------:|-----------------:|----------------:|:---------------|
| continuous_1 |  5.73956e-05 | 0.000431725 |    0.132945 |   0.894237 |     -0.000788785 |     0.000903577 |                |
| continuous_2 |  0.000241658 | 0.000432161 |    0.559185 |   0.576035 |     -0.000605377 |      0.00108869 |                |
| continuous_3 |    0.0349411 | 0.000302463 |     115.522 |          0 |        0.0343483 |       0.0355339 | ***            |
| continuous_4 |   -0.0317055 |  0.00033468 |    -94.7337 |          0 |       -0.0323614 |      -0.0310495 | ***            |
| dummy_1      |     0.217042 | 0.000857476 |     253.118 |          0 |         0.215362 |        0.218723 | ***            |
| dummy_2      |     0.283884 | 0.000900227 |     315.347 |          0 |         0.282119 |        0.285648 | ***            |
| dummy_3      |    -0.271684 | 0.000995668 |    -272.866 |          0 |        -0.273636 |       -0.269733 | ***            |
| dummy_4      | -0.000337121 | 0.000864605 |   -0.389913 |   0.696601 |      -0.00203175 |      0.00135751 |                |
```

Here `dummy=False` (the default), so the binary columns are treated as continuous — exactly as `results_probit.get_margeff()` does.

Significance: `*` p<0.1, `**` p<0.05, `***` p<0.01

### Logit — discrete dummies (`get_margeff(dummy=True)` equivalent)

```python
rustmfx.mfx(results_logit, dummy=True)
```

Output:
```
|              |        dy/dx |    Std. Err |           z |   Pr(>|z|) |   Conf. Int. Low |   Conf. Int. Hi | Significance   |
|:-------------|-------------:|------------:|------------:|-----------:|-----------------:|----------------:|:---------------|
| continuous_1 |  2.66309e-05 |  0.00043231 |   0.0616014 |    0.95088 |     -0.000820696 |     0.000873958 |                |
| continuous_2 |  0.000259106 | 0.000432796 |    0.598679 |   0.549387 |     -0.000589175 |      0.00110739 |                |
| continuous_3 |    0.0351363 | 0.000298504 |     117.708 |          0 |        0.0345512 |       0.0357213 | ***            |
| continuous_4 |   -0.0331457 | 0.000340525 |    -97.3373 |          0 |       -0.0338132 |      -0.0324783 | ***            |
| dummy_1      |     0.228592 | 0.000976438 |     234.108 |          0 |         0.226678 |        0.230506 | ***            |
| dummy_2      |     0.290481 |  0.00095629 |     303.758 |          0 |         0.288607 |        0.292356 | ***            |
| dummy_3      |    -0.275727 |  0.00102521 |    -268.946 |          0 |        -0.277736 |       -0.273717 | ***            |
| dummy_4      | -0.000336179 | 0.000866042 |   -0.388178 |   0.697884 |      -0.00203362 |      0.00136126 |                |
```

With `dummy=True`, each binary column's effect is the finite difference $F(\cdot\mid x{=}1)-F(\cdot\mid x{=}0)$, matching `results_logit.get_margeff(dummy=True)`.

The model type (Logit or Probit) is detected automatically from `model.__class__.__name__`.

### Parameters

| Argument | Type | Default | Description |
|---|---|---|---|
| `model` | statsmodels Logit/Probit result | — | A fitted results object. `params`, `cov_params()`, and `exog` are read from it. |
| `dummy` | `bool` | `False` | If `True`, strictly-0/1 columns are treated as discrete (finite-difference effect), matching `get_margeff(dummy=True)`. If `False`, all non-constant columns are treated as continuous, matching `get_margeff()`. |
| `chunk_size` | `int` | `None` | Process observations in chunks of this size to bound peak memory on very large $N$. Does not change the result. |

Constant/intercept columns (named `const` or `Intercept`, case-insensitive) are detected automatically and dropped from the output, mirroring statsmodels.

---

## Robust, Clustered, and Weighted Standard Errors

`rustmfx.mfx()` extracts the covariance matrix via `cov_params()` from the `statsmodels` fit object. Any adjustments made during fitting — robust SEs, clustered SEs, observation weights — are already reflected in that covariance matrix and will propagate automatically into the marginal effect SEs.

```python
results = sm.Probit(y, X).fit(cov_type='HC0')
rustmfx.mfx(results)  # uses the HC0 covariance matrix
```

---

## Memory Comparison

Peak memory usage of `.mfx()` vs. `.get_margeff()` across varying $N$ and $K$:

![Memory Usage Comparison](Memory%20Comparison%20.get_margeff()%20VS%20.mfx().png?raw=true&v=2)

---

## Contributing

Issues and pull requests welcome on [GitHub](https://github.com/luke-brosnan-cbc/RustMFX).

## License

MIT — see [LICENSE](LICENSE).
