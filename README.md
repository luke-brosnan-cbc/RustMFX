# RustMFX

[![Build Status](https://github.com/luke-brosnan-cbc/RustMFX/workflows/CI.yml/badge.svg)](https://github.com/luke-brosnan-cbc/RustMFX/actions)
[![PyPI version](https://badge.fury.io/py/rustmfx.svg)](https://pypi.org/project/rustmfx/)

## Introduction

**RustMFX** is a high-performance Rust package for computing marginal effects (MEs) in binary choice models (Logit and Probit). 


As datasets in economics, social sciences, and data science continue to grow in size and complexity, traditional tools (like ```statsmodels```) can struggle with speed and memory efficiency.


RustMFX leverages ```Rust```’s efficiency and safety to deliver fast and memory-friendly computations of average marginal effects (AMEs) and their standard errors (SEs).



## 1. Introduction

In today's data-driven world, researchers and practitioners are faced with increasingly massive datasets.


In fields such as economics, social sciences, and data science, analyzing high-dimensional data efficiently is critical.


While libraries like statsmodels provide robust statistical tools, they can be limited by the speed and memory constraints of Python when handling large datasets.


**RustMFX** was designed to overcome these challenges by using ```Rust```—a language known for its high performance and low memory overhead—to compute marginal effects quickly and reliably.


## 2. Speed and Memory Efficiency

RustMFX outperforms Python-based `statsmodels` in several key areas:

### **Efficient Vectorization & Low-Level Optimizations**
The Rust code leverages libraries like `ndarray` (using a BLAS backend) to perform operations in a highly optimized, vectorized manner.  
This means calculations are executed on entire matrices at once, rather than iterating over rows with loops.  
This eliminates costly Python-level loops, allowing computations to be performed in a fraction of the time.

### **Loop-Based Computation in `statsmodels`**
- `statsmodels.get_margeff()` uses explicit **Python loops** in places where RustMFX performs **fully vectorized** operations.
- For **continuous variables**, `statsmodels` iterates over observations to compute marginal effects, while RustMFX applies **matrix operations across the entire dataset in one step**.
- For **discrete variables**, `statsmodels` **loops over each discrete feature**, recomputing probabilities for every observation twice (once for `X=0`, once for `X=1`).  
  In contrast, RustMFX **batch-computes these changes** in a single matrix operation.

### **Memory Efficiency & In-Place Computation**
Rust’s strict memory management and zero-cost abstractions help keep memory overhead low.  
- RustMFX **reuses memory** and avoids large temporary allocations, ensuring that memory usage scales efficiently with increasing variables.  
- `statsmodels.get_margeff()` **creates multiple intermediate arrays**, whose sizes grow significantly as the number of variables $K$ increases.  
  This leads to exponential memory growth in `statsmodels`, whereas RustMFX remains efficient even at large scales.

### **Compiled vs. Interpreted Code**
Rust is a compiled language with aggressive optimizations by LLVM, while `statsmodels` is written in Python (using `NumPy` for vectorization).  
Although `NumPy` is optimized for array computations, Python’s **dynamic memory management and interpreted nature** introduce **overhead**,  
especially when handling large models with many variables.


### **Why RustMFX Scales Better**
✔ **Fully vectorized operations** eliminate Python loops for both continuous and discrete marginal effects.  
✔ **Lower memory footprint** by avoiding large temporary allocations and using in-place computation.  
✔ **Efficient discrete variable handling** batch-computes discrete effects instead of looping over them.  
✔ **Scales efficiently with increasing $K$**, whereas `statsmodels.get_margeff()` suffers from excessive memory usage.

These features make RustMFX an ideal choice for analyzing large-scale data **quickly, efficiently, and with minimal memory overhead**.


## 3. Comparison: Statsmodels-Style vs. Rust Method

RustMFX provides two methods for calculating standard errors of the marginal effects:

### Rust Method (```"rust"```)
- **Description:**  
  This method calculates the gradient (Jacobian) of the marginal effects with respect to the coefficients by averaging the individual observation-level gradients.
- **Benefits:**  
  Captures detailed individual-level variability, which is beneficial when data heterogeneity is significant. Closer to the way Stata calculates Standard Errors.
- **Use Cases:**  
  Best used in applications where capturing the nuances of individual effects is crucial.

### Statsmodels-Style Method (```"sm"```)
- **Description:**  
  This approach computes the derivative of the predicted probabilities with respect to the coefficients and then averages this derivative over all observations.
- **Benefits:**  
  Produces smoother and often more stable SE estimates, especially useful in smaller samples.
- **Use Cases:**  
  Preferred when consistency with traditional ```statsmodels``` output is desired or when more stable SE estimates are needed.



## 4. Methodology: Calculating Marginal Effects and Standard Errors

RustMFX calculates Average Marginal Effects (AMEs) and uses the delta method to compute standard errors.

### Marginal Effects Calculation

For a given observation $i$ and variable $j$, the marginal effect is:

<div align="center"; margin: 0>
  
### $\frac{\partial P(Y=1 \mid \text{X}\_i)}{\partial \text{X}\_{ij}} = \beta_j \times f(\text{X}\_i \beta)$

</div>

where:
- $\beta_j$ is the coefficient for variable $j$,
- $f(\cdot)$ is the probability density function (PDF):
 
  - **Logit:**

  <div align="center"; margin: 0>
    
  ### $f(z) = \frac{e^z}{\left(1+e^z\right)^2}$

  </div>

  - **Probit:**

  <div align="center"; margin: 0>
    
  ### $f(z) = \frac{1}{\sqrt{2\pi}} e^{-0.5 z^2}$

  </div>

  <br>

The Average Marginal Effect (AME) is computed by averaging over all $N$ observations:

<div align="center"; margin: 0>

### $\text{AME}\_j = \frac{1}{N} \sum_{i=1}^{N} \beta_j \times f(\text{X}\_i \beta)$

</div>

<br>

### Standard Errors Calculation

Using the delta method, the variance of the AME is calculated as:

<div align="center"; margin: 0>
  
### $\text{Var}(\text{AME}) = \text{J} \cdot \text{Var}(\beta) \cdot \text{J}^T$

</div>

where $\text{J}$ is the Jacobian matrix of the transformation from coefficients to marginal effects. The Jacobian is computed differently depending on the method:

- **Rust Method:**

<div align="center"; margin: 0>
  
### $\text{J}\_{\text{rust}} = \frac{1}{N} \sum_{i=1}^{N} \frac{\partial \left( \beta_j f(\text{X}\_i \beta) \right)}{\partial \beta}$

</div>

#### **Explanation:**
- This method computes the **Jacobian by taking the derivative of the marginal effects function for each individual observation**.
- The gradient is **first computed at the observation level** and then **averaged over all $N$ observations**.
- This approach **captures individual-level variability** in marginal effects **more accurately**.
- The **partial derivative term** represents how a **small change in $\beta$** affects the probability distribution function $f(\text{X}_i \beta)$.
- **Closer to how Stata computes standard errors** and more **sensitive to variations across observations**.

#### **✅ Why Use This Method?**
✔ More precise in heterogeneous datasets (e.g., large-scale social science or labor market studies).  
✔ Preserves individual-level variability rather than smoothing it out.  
✔ Better for large datasets where heterogeneity matters. 

- **Statsmodels-Style Method:**

<div align="center"; margin: 0>
  
### $\text{J}\_{\text{sm}} = \frac{1}{N} \sum_{i=1}^{N} \text{X}\_i \ f(\text{X}\_i \beta)$

</div>

#### **Explanation:**
- Instead of differentiating at the individual level, this method **computes the derivative of the predicted probability function with respect to $\beta$, then averages over all observations**.
- The term $\text{X}_i$ represents the **independent variable values** for observation $i$, and it is multiplied by the **PDF** $f(\text{X}_i \beta)$.
- This method **smooths out variability across observations**, leading to **more stable standard error estimates**.
- While **slightly less precise at the individual level**, it is **computationally simpler and produces standard errors identical to `statsmodels`**.

#### **✅ Why Use This Method?**
✔ Produces more stable standard errors, especially in small samples.  
✔ Ensures consistency with traditional `statsmodels` outputs.  
✔ Useful when a more aggregated (less individual-specific) marginal effect estimate is preferred.  

<br>

These approaches allow you to choose the estimation method that best fits your application needs.

### Handling of Continuous vs. Discrete Variables

RustMFX distinguishes between **continuous** and **discrete** variables when computing marginal effects and their standard errors (in an identical way to ```sm.get_margeff()```):

- **Continuous Variables:**  
  For a continuous variable $\text{X}\_j$, the marginal effect is computed as:

<div align="center"; margin: 0>
  
### $\frac{\partial P(Y=1 \mid \text{X}\_i)}{\partial \text{X}\_{ij}} = \beta_j \times f(\text{X}\_i \beta)$

</div>
  
This formula directly measures how a small change in $\text{X}_j$ affects the probability $\text{P}(Y=1)$. The associated Jacobian and standard errors are calculated using the derivative of this expression.

- **Discrete Variables:**  
  For a discrete (binary) variable $\text{X}\_j$, the marginal effect is computed using a **finite-difference approach**:

<div align="center"; margin: 0>
  
### $\Delta P = P(Y=1 \mid \text{X}\_{ij}=1) - P(Y=1 \mid \text{X}\_{ij}=0)$

</div>

Rather than using a derivative (which isn't defined for variables that only take two values), this method measures the change in the predicted probability when the variable switches from ```0``` to ```1```.


The gradient for discrete variables is computed by comparing the probability densities for both states, ensuring that the binary nature of the variable is appropriately handled.

## 5. How to Use RustMFX

Below is an example of how to integrate RustMFX into your workflow. First install ```rustmfx``` on your system.

```python
pip install rustmfx
```


Here we will construct a dataset using ```make_classification``` from the ```sklearn``` library. Then we fit ```sm.Probit``` and ```sm.Logit``` models.

__Note__:
- It is important that ```y``` and ```X``` fed to the ```sm.{Model}(y,X).fit()``` function are ```pandas.DataFrame``` type, as this will result in the output objects being DFs, which is required by ```rustmfx.mfx()```.
- Column names must also be strings/non-numeric.


```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.datasets import make_classification

# Import RustMFX
import rustmfx  

# Number of rows
nobs = 1_000_000

# Define feature counts
num_continuous = 4
num_dummy = 4
num_features = num_continuous + num_dummy

# Randomly decide how many variables should be predictive
num_informative = np.random.randint(1, num_features + 1)  # At least 1 variable is predictive

# Generate dataset with random informative variables
X, y = make_classification(
    n_samples=nobs,
    n_features=num_features,
    n_informative=num_informative,  # Randomly chosen number of informative variables
    n_redundant=0,  
    n_classes=2,  
    random_state=42
)

# Convert to Pandas DataFrame with proper naming
continuous_names = [f"continuous_{i+1}" for i in range(num_continuous)]
dummy_names = [f"dummy_{i+1}" for i in range(num_dummy)]
column_names = continuous_names + dummy_names
X = pd.DataFrame(X, columns=column_names)
y = pd.DataFrame(y, columns=["outcome"])

# Convert last 4 columns (dummy variables) into 0/1 binary variables
X[dummy_names] = (X[dummy_names] > X[dummy_names].median()).astype(int)  

# Add intercept column
X = sm.add_constant(X)

# Fit Logit and Probit models using Statsmodels
results_probit = sm.Probit(y, X).fit(disp=0)
results_logit = sm.Logit(y, X).fit(disp=0)
```


Below we run ```rustmfx.mfx()``` on the Probit model. ```se_method``` defaults to ```"rust"```.


This produces a ```pandas.DataFrame``` object with:\
```variable names```;\
```dy/dx``` (average marginal effects);\
```Standard Errors```;\
```z score```;\
```p-values```;\
```95% confidence interval```;\
```Significance level``` (```*``` = p<0.1, ```**``` = p<0.05, ```***``` = p<0.01).



```python
# get marginal effects for Probit model using rustmfx.mfx()
# By default, <option: se_method='rust'>
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


This time we run ```.mfx()``` on the Logit model.


The ```.mfx()``` function automatically detects if the ```sm.{Model}(y,X).fit()``` is Probit or Logit by extracting ```{Model}.model.__class__.__name__```


Here I set ```se_method='sm'``` to mimic ```statsmodel```'s method for getting standard errors.


```python
# get marginal effects for Logit model using rustmfx.mfx()
# Use <option: se_method='sm'> instead. This will produce SE identical to sm.get_margeff()
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

## Accounting for Robust SE, Clustered SE, and Weights

One of the key advantages of RustMFX is that it automatically uses the model’s covariance matrix—obtained via the ```cov_params()``` method from the ```statsmodels``` fit object—in the delta method to compute the standard errors of the marginal effects.

**What does this mean for you?**  
If you fit your model with additional parameters such as robust standard errors, clustered standard errors, or observation weights (for example, by using options like ```cov_type='HC0'```, ```cov_kwds={'groups': clusters}```, or specifying weights), these adjustments will be captured in the covariance matrix output of your ```sm.{Model}(y, X).fit()``` call.

For instance, if you fit a Probit model with robust standard errors:

```python
results = sm.Probit(y, X).fit(cov_type='HC0')
```
Then call
```python
mfx_results = rustmfx.mfx(results)
```
RustMFX will automatically extract and use the robust covariance matrix in the marginal effects calculations. In other words, as long as these parameters (robust SE, clustered SE, weights, etc.) are specified in your ``statsmodels``` ```.fit()```, the ```.mfx()``` function will automatically account for them.

This integration ensures that your marginal effects and their standard errors reflect any adjustments made during model fitting, providing you with accurate and reliable inference.


## Performance Comparison between ```.mfx()``` and ```.get_margeff()```
Below is a graph showing the difference in peak memory usage of ```.mfx()``` and ```.get_margeff()``` across datasets of differing number of observations $N$ and degrees of freedom (number of parameters) $k$.\
RustMFX performs exponentially better than ```statsmodels``` as the number of parameters increases.

<div align="center"; margin: 0>
  
### Memory Usage Comparison of ```.get_margeff()``` VS ```.mfx()```

</div>

![Memory Usage Comparison of .get_margeff() VS .mfx()](Memory%20Comparison%20.get_margeff()%20VS%20.mfx().png?raw=true&v=2)


## Contributing

Contributions are welcome! Feel free to submit issues and pull requests on [GitHub](https://github.com/luke-brosnan-cbc/RustMFX).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

