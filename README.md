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

## Standard Error Methods

RustMFX offers two Jacobian formulations for the delta method variance $\text{Var}(\text{AME}) = J \cdot \text{Var}(\hat{\beta}) \cdot J^T$:

**`se_method='rust'` (default)**

$$J_{\text{rust}} = \frac{1}{N} \sum_{i=1}^{N} \frac{\partial \left( \beta_j f(X_i \beta) \right)}{\partial \beta}$$

The Jacobian is computed at the observation level and then averaged. This preserves individual-level variability in the gradient and is closer to how Stata computes marginal effect standard errors.

**`se_method='sm'`**

$$J_{\text{sm}} = \frac{1}{N} \sum_{i=1}^{N} X_i \, f(X_i \beta)$$

The derivative of the predicted probability with respect to $\beta$ is averaged over all observations. This smooths out individual-level variation and produces standard errors identical to `statsmodels.get_margeff()`. Useful for replication or when more stable SEs in small samples are preferred.

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

For **discrete** (binary) variables, a finite-difference approach is used instead:

$$\Delta P = P(Y=1 \mid X_{ij}=1) - P(Y=1 \mid X_{ij}=0)$$

This is consistent with how `statsmodels.get_margeff()` handles binary variables.

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

### Probit — default SE method

```python
rustmfx.mfx(results_probit)
```

Output:
```
|              |        dy/dx |    Std. Err |           z |   Pr(>|z|) |   Conf. Int. Low |   Conf. Int. Hi | Significance   |
|:-------------|-------------:|------------:|------------:|-----------:|-----------------:|----------------:|:---------------|
| continuous_1 | -0.000711967 | 0.000429515 |   -1.65761  |  0.0973968 |     -0.00155382  |     0.000129882 | *              |
| continuous_2 |  0.000143285 | 0.000429703 |    0.333452 |  0.738793  |     -0.000698933 |     0.000985503 |                |
| continuous_3 |  0.132766    | 0.000325385 |  408.028    |  0         |      0.132128    |     0.133404    | ***            |
| continuous_4 |  0.000172238 | 0.000429776 |    0.400763 |  0.688595  |     -0.000670123 |     0.0010146   |                |
| dummy_1      | -0.144887    | 0.000853498 | -169.757    |  0         |     -0.14656     |    -0.143214    | ***            |
| dummy_2      |  0.355616    | 0.000853642 |  416.586    |  0         |      0.353943    |     0.357289    | ***            |
| dummy_3      | -0.000114508 | 0.000858866 |   -0.133325 |  0.893936  |     -0.00179789  |     0.00156887  |                |
| dummy_4      |  0.193636    | 0.000841226 |  230.183    |  0         |      0.191987    |     0.195285    | ***            |
```

Significance: `*` p<0.1, `**` p<0.05, `***` p<0.01

### Logit — statsmodels-style SE

```python
rustmfx.mfx(results_logit, se_method='sm')
```

Output:
```
|              |        dy/dx |    Std. Err |            z |   Pr(>|z|) |   Conf. Int. Low |   Conf. Int. Hi | Significance   |
|:-------------|-------------:|------------:|-------------:|-----------:|-----------------:|----------------:|:---------------|
| continuous_1 | -0.000727353 | 0.000429072 |   -1.69518   |  0.0900422 |     -0.00156833  |     0.000113629 | *              |
| continuous_2 |  0.000160417 | 0.000429249 |    0.373717  |  0.708615  |     -0.00068091  |     0.00100174  |                |
| continuous_3 |  0.133292    | 0.000259576 |  513.501     |  0         |      0.132784    |     0.133801    | ***            |
| continuous_4 |  0.000194787 | 0.000429317 |    0.453715  |  0.650034  |     -0.000646674 |     0.00103625  |                |
| dummy_1      | -0.152622    | 0.00085115  | -179.313     |  0         |     -0.15429     |    -0.150954    | ***            |
| dummy_2      |  0.355251    | 0.000852706 |  416.616     |  0         |      0.35358     |     0.356923    | ***            |
| dummy_3      | -8.12991e-05 | 0.000857946 |   -0.0947601 |  0.924505  |     -0.00176287  |     0.00160027  |                |
| dummy_4      |  0.194628    | 0.000837786 |  232.313     |  0         |      0.192986    |     0.19627     | ***            |
```

The model type (Logit or Probit) is detected automatically from `model.__class__.__name__`.

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


