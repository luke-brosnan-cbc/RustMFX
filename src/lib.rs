// src/lib.rs

use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyAny, PyDict, PyList, PyType};
use pyo3::wrap_pyfunction;

use numpy::{PyArray1, PyArray2, IntoPyArray};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, s, Axis};
use statrs::distribution::{Continuous, Normal, ContinuousCDF};
use std::collections::HashMap;
use std::fmt::Write as _;  // We use this for building cluster keys (if needed)

/// Returns significance stars based on the p-value.
/// 
/// This is a helper function to annotate significance levels.  
/// For example, p < 0.01 gets three stars, p < 0.05 gets two, etc.
fn add_significance_stars(p: f64) -> &'static str {
    if p < 0.01 {
        "***"
    } else if p < 0.05 {
        "**"
    } else if p < 0.1 {
        "*"
    } else {
        ""
    }
}

/// Checks the type of the model to see whether it's Logit or Probit.
/// 
/// This helper extracts the model’s class name from the Python object and converts it
/// to lowercase for comparison. It returns `true` for Logit and `false` for Probit.
/// If the model type is not one of these, it returns an error.
fn detect_model_type(model: &PyAny) -> Result<bool, PyErr> {
    // Try to get the underlying model attribute (or use the model itself if not available)
    let model_obj = model.getattr("model").unwrap_or(model);
    // Extract the class name as a string
    let cls: String = model_obj
        .getattr("__class__")?
        .getattr("__name__")?
        .extract()?;
    let lc = cls.to_lowercase();
    // Check whether the class name is "logit" or "probit"
    if lc == "logit" {
        Ok(true)  // true indicates a Logit model
    } else if lc == "probit" {
        Ok(false) // false indicates a Probit model
    } else {
        Err(pyo3::exceptions::PyValueError::new_err(
            format!("ame: only Logit or Probit supported, got {cls}"),
        ))
    }
}

/// Converts a Python object (expected to be a NumPy array) into an ndarray ArrayView2<f64>.
///
/// # Safety
/// This function is marked unsafe because it relies on the assumption that the Python memory is valid.
fn as_array2_f64<'py>(obj: &'py PyAny) -> PyResult<ArrayView2<'py, f64>> {
    let pyarray = obj.downcast::<PyArray2<f64>>()?;
    let view = unsafe { pyarray.as_array() };
    Ok(view)
}

/// Similar to `as_array2_f64`, but for 1D arrays.
/// Converts a Python object into an ndarray ArrayView1<f64>.
fn as_array1_f64<'py>(obj: &'py PyAny) -> PyResult<ArrayView1<'py, f64>> {
    let pyarray = obj.downcast::<PyArray1<f64>>()?;
    let view = unsafe { pyarray.as_array() };
    Ok(view)
}

/// Computes the cumulative distribution function for either the logistic or the normal distribution.
///
/// If the model is logistic, it uses the logistic CDF; otherwise, it uses the normal CDF.
fn cdf_logit_probit(is_logit: bool, z: f64) -> f64 {
    if is_logit {
        // For logistic regression, compute the logistic CDF
        1.0 / (1.0 + (-z).exp())
    } else {
        // For probit regression, compute the normal CDF using a standard Normal distribution
        let dist = Normal::new(0.0, 1.0).unwrap();
        dist.cdf(z)
    }
}

/// Computes the probability density function for either the logistic or the normal distribution.
///
/// This is used later when calculating marginal effects.
fn pdf_logit_probit(is_logit: bool, z: f64) -> f64 {
    if is_logit {
        // For logistic: f(z) = exp(z) / (1+exp(z))^2
        let e = z.exp();
        e / (1.0 + e).powi(2)
    } else {
        // For normal: use the standard normal density formula
        (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt()
    }
}

/// Computes the derivative of the probability density function with respect to z.
///
/// This derivative is needed for the statsmodels-style standard error calculation.
fn pdf_deriv_logit_probit(is_logit: bool, z: f64) -> f64 {
    if is_logit {
        // For logistic regression, the derivative is f(z) * (1 - 2 * F(z))
        pdf_logit_probit(is_logit, z) * (1.0 - 2.0 * cdf_logit_probit(is_logit, z))
    } else {
        // For probit regression, the derivative is -z * phi(z)
        -z * pdf_logit_probit(is_logit, z)
    }
}

/// Computes the average marginal effects (AMEs) for Logit/Probit models.
///
/// This function mimics the style of the original Probit code, but with a twist:  
/// you can choose between two methods for calculating the standard errors (SEs):
/// 
/// - "rust": Uses the original gradient (Jacobian) calculation based on per-observation effects.
/// - "sm": Uses a statsmodels-style gradient for continuous variables.
///
/// The function accepts a model (Logit or Probit), an optional chunk size (to process data in parts),
/// and an optional `se_method` string which defaults to "rust".
#[pyfunction]
fn mfx<'py>(
    py: Python<'py>,
    model: &'py PyAny,              // This could be either a Logit or Probit model from statsmodels.
    chunk_size: Option<usize>,      // Optional: process the data in chunks to save on memory.
    se_method: Option<&str>,        // Choose "rust" (default) or "sm" for SE calculation.
) -> PyResult<&'py PyAny> {
    // Decide which standard error method to use. Default to "rust" if not specified.
    let se_method_str = se_method.unwrap_or("rust");

    // 1) Figure out if we have a Logit or a Probit model.
    let is_logit = detect_model_type(model)?;

    // 2) Read in the model parameters. These might come from a pandas Series or directly as a NumPy array.
    let params_obj = model.getattr("params")?;
    let params_pyarray = if let Ok(values) = params_obj.getattr("values") {
        values.downcast::<PyArray1<f64>>()?
    } else {
        params_obj.downcast::<PyArray1<f64>>()?
    };
    // Convert the parameters to an ndarray for easy manipulation.
    let beta = unsafe { params_pyarray.as_array() };

    // 3) Get the covariance matrix for the parameters. Again, handle both pandas DataFrame and NumPy array cases.
    let cov_obj = model.call_method0("cov_params")?;
    let cov_pyarray = if let Ok(values) = cov_obj.getattr("values") {
        values.downcast::<PyArray2<f64>>()?
    } else {
        cov_obj.downcast::<PyArray2<f64>>()?
    };
    let cov_beta = unsafe { cov_pyarray.as_array() };

    // 4) Extract the exogenous variables (X) and their names.
    // This code supports both pandas DataFrames and NumPy arrays.
    let model_obj = model.getattr("model").unwrap_or(model);
    let exog_py = model_obj.getattr("exog")?;
    let (x_pyarray, exog_names) = if let Ok(values) = exog_py.getattr("values") {
        (
            values.downcast::<PyArray2<f64>>()?,
            exog_py.getattr("columns")?.extract::<Vec<String>>()?
        )
    } else {
        (
            exog_py.downcast::<PyArray2<f64>>()?,
            model_obj.getattr("exog_names")?.extract::<Vec<String>>()?
        )
    };

    // Convert the exogenous variables to an ndarray and get its dimensions.
    let X = unsafe { x_pyarray.as_array() };
    let (n, k) = (X.nrows(), X.ncols());
    // If no chunk size is provided, process all observations at once.
    let chunk = chunk_size.unwrap_or(n);

    // 5) Identify which columns are intercepts.
    // Here we look for column names that are "const" or "intercept" (case-insensitive).
    let intercept_indices: Vec<usize> = exog_names
        .iter()
        .enumerate()
        .filter_map(|(i, nm)| {
            let ln = nm.to_lowercase();
            if ln == "const" || ln == "intercept" {
                Some(i)
            } else {
                None
            }
        })
        .collect();

    // 6) Identify discrete columns.
    // We assume that discrete columns are strictly 0/1 values, ignoring intercepts.
    let is_discrete: Vec<usize> = exog_names
        .iter()
        .enumerate()
        .filter_map(|(j, _)| {
            if intercept_indices.contains(&j) {
                None
            } else {
                let col_j = X.column(j);
                if col_j.iter().all(|&v| v == 0.0 || v == 1.0) {
                    Some(j)
                } else {
                    None
                }
            }
        })
        .collect();

    // 7) Prepare accumulators for the average marginal effects and their gradients.
    // We maintain two sets: one for our original "rust" method and one for the statsmodels "sm" method.
    let mut sum_ame = vec![0.0; k];               // Accumulator for marginal effects (rust)
    let mut partial_jl_sums = vec![0.0; k * k];      // Accumulator for the Jacobian (rust)
    let mut sm_sum_ame = vec![0.0; k];              // Accumulator for marginal effects (sm)
    let mut sm_partial_jl_sums = vec![0.0; k * k];     // Accumulator for the Jacobian (sm)
    let normal = Normal::new(0.0, 1.0).unwrap();

    // 8) Process the data in chunks.
    // This loop handles the data piecewise, which can be helpful with very large datasets.
    let mut idx_start = 0;
    while idx_start < n {
        let idx_end = (idx_start + chunk).min(n);
        // Slice the chunk from the full dataset.
        let x_chunk = X.slice(s![idx_start..idx_end, ..]);
        // Compute the linear predictor z = X * beta.
        let z_chunk = x_chunk.dot(&beta);
        // Evaluate the pdf at each z value.
        let pdf_chunk = z_chunk.mapv(|z| pdf_logit_probit(is_logit, z));

        // Handle discrete variables.
        // For each discrete variable, we adjust the linear predictor for the two possible values (0 and 1)
        // and then compute the change in the predicted probability.
        for &j in &is_discrete {
            let xj_col = x_chunk.column(j);
            let b_j = beta[j];
            // Adjust z for when x_j is 1 (or 0) using a finite difference approach.
            let delta_j1 = (1.0 - &xj_col).mapv(|x| x * b_j);
            let delta_j0 = xj_col.mapv(|x| -x * b_j);
            let z_j1 = &z_chunk + &delta_j1;
            let z_j0 = &z_chunk + &delta_j0;

            // Compute the cdf for the adjusted z values.
            let cdf_j1 = z_j1.mapv(|z| cdf_logit_probit(is_logit, z));
            let cdf_j0 = z_j0.mapv(|z| cdf_logit_probit(is_logit, z));
            // The effect for variable j is the difference in the summed probabilities.
            let effect_sum = cdf_j1.sum() - cdf_j0.sum();
            sum_ame[j] += effect_sum;
            sm_sum_ame[j] += effect_sum;

            // Also update the Jacobian accumulators.
            let pdf_j1 = z_j1.mapv(|z| pdf_logit_probit(is_logit, z));
            let pdf_j0 = z_j0.mapv(|z| pdf_logit_probit(is_logit, z));
            for l in 0..k {
                let grad = if l == j {
                    pdf_j1.sum()
                } else {
                    let x_l = x_chunk.column(l);
                    let diff_pdf = &pdf_j1 - &pdf_j0;
                    diff_pdf.dot(&x_l)
                };
                partial_jl_sums[j * k + l] += grad;
                sm_partial_jl_sums[j * k + l] += grad;
            }
        }

        // Handle continuous variables.
        // The calculation differs based on which SE method is selected.
        for j in 0..k {
            if intercept_indices.contains(&j) || is_discrete.contains(&j) {
                continue;
            }
            let b_j = beta[j];
            if se_method_str == "rust" {
                // Original method: simply multiply b_j with the pdf sum.
                sum_ame[j] += b_j * pdf_chunk.sum();
                for l in 0..k {
                    let grad = if j == l {
                        pdf_chunk.sum()
                    } else {
                        let x_l = x_chunk.column(l);
                        // Negative sign comes from the derivation of the gradient.
                        -b_j * (&z_chunk * &x_l).dot(&pdf_chunk)
                    };
                    partial_jl_sums[j * k + l] += grad;
                }
            } else if se_method_str == "sm" {
                // Statsmodels-style: incorporate the derivative of the pdf.
                let fprime_chunk = z_chunk.mapv(|z| pdf_deriv_logit_probit(is_logit, z));
                sm_sum_ame[j] += b_j * pdf_chunk.sum();
                for l in 0..k {
                    let x_l = x_chunk.column(l);
                    let term = (&x_l * &fprime_chunk).sum();
                    let grad = if j == l {
                        pdf_chunk.sum() + b_j * term
                    } else {
                        b_j * term
                    };
                    sm_partial_jl_sums[j * k + l] += grad;
                }
            }
        }

        // Move to the next chunk.
        idx_start = idx_end;
    }

    // 9) Average the accumulated marginal effects and gradients over all observations.
    let final_ame: Vec<f64> = if se_method_str == "sm" {
        sm_sum_ame.iter().map(|v| v / (n as f64)).collect()
    } else {
        sum_ame.iter().map(|v| v / (n as f64)).collect()
    };

    // Build the Jacobian matrix from the accumulated partial derivatives.
    let mut grad_ame = Array2::<f64>::zeros((k, k));
    for j in 0..k {
        for l in 0..k {
            let value = if se_method_str == "sm" {
                sm_partial_jl_sums[j * k + l] / (n as f64)
            } else {
                partial_jl_sums[j * k + l] / (n as f64)
            };
            grad_ame[[j, l]] = value;
        }
    }

    // Use the delta method to obtain the covariance of the AME:
    // cov_ame = grad_ame * cov_beta * grad_ame^T
    let cov_ame = grad_ame.dot(&cov_beta).dot(&grad_ame.t());
    // Extract the diagonal (variances) and compute standard errors.
    let var_ame: Vec<f64> = cov_ame.diag().iter().map(|&v| v.max(0.0)).collect();
    let se_ame: Vec<f64> = var_ame.iter().map(|&v| v.sqrt()).collect();

    // 10) Build the final result vectors for the marginal effects, SEs, z-values, p-values, and significance stars.
    // Also compute the lower and upper bounds of the 95% confidence interval.
    let (mut dy_dx, mut se_err, mut z_vals, mut p_vals, mut sig) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let (mut conf_low, mut conf_high) = (Vec::new(), Vec::new());
    let mut names_out = Vec::new();

    for j in 0..k {
        // Skip intercept columns.
        if intercept_indices.contains(&j) {
            continue;
        }
        let dy = final_ame[j];
        let s = se_ame[j];
        dy_dx.push(dy);
        se_err.push(s);
        if s > 1e-15 {
            let z = dy / s;
            let p = 2.0 * (1.0 - normal.cdf(z.abs()));
            z_vals.push(z);
            p_vals.push(p);
            sig.push(add_significance_stars(p));
            // 95% confidence interval: dy ± 1.96 * SE
            conf_low.push(dy - 1.96 * s);
            conf_high.push(dy + 1.96 * s);
        } else {
            // If SE is extremely small, set default values.
            z_vals.push(0.0);
            p_vals.push(1.0);
            sig.push("");
            conf_low.push(dy);
            conf_high.push(dy);
        }
        // Keep the original column name for the output.
        names_out.push(exog_names[j].clone());
    }

    // 11) Build a Pandas DataFrame from the results.
    // The DataFrame includes the marginal effects, standard errors, z-values, p-values, 
    // confidence interval lower/upper bounds, and significance stars.
    let pd = py.import("pandas")?;
    let data = PyDict::new(py);
    data.set_item("dy/dx", &dy_dx)?;
    data.set_item("Std. Err", &se_err)?;
    data.set_item("z", &z_vals)?;
    data.set_item("Pr(>|z|)", &p_vals)?;
    data.set_item("Conf. Int. Low", &conf_low)?;
    data.set_item("Conf. Int. Hi", &conf_high)?;
    data.set_item("Significance", &sig)?;

    let kwargs = PyDict::new(py);
    kwargs.set_item("data", data)?;
    kwargs.set_item("index", &names_out)?;

    let df = pd.call_method("DataFrame", (), Some(kwargs))?;
    Ok(df)
}

#[pymodule]
fn rustmfx(_py: Python, m: &PyModule) -> PyResult<()> {
    // Register our mfx function so it can be called from Python.
    m.add_function(wrap_pyfunction!(mfx, m)?)?;
    Ok(())
}


