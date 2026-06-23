// src/lib.rs

use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use pyo3::wrap_pyfunction;

use numpy::{PyArray1, PyArray2};
use ndarray::{Array2, ArrayView1, ArrayView2, s};
use statrs::distribution::{Continuous, Normal, ContinuousCDF};

/// Returns significance stars based on the p-value.
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

/// Detect whether the statsmodels model is Logit (`true`) or Probit (`false`).
fn detect_model_type(model: &PyAny) -> Result<bool, PyErr> {
    let model_obj = model.getattr("model").unwrap_or(model);
    let cls: String = model_obj
        .getattr("__class__")?
        .getattr("__name__")?
        .extract()?;
    let lc = cls.to_lowercase();
    if lc == "logit" {
        Ok(true)
    } else if lc == "probit" {
        Ok(false)
    } else {
        Err(pyo3::exceptions::PyValueError::new_err(format!(
            "mfx: only Logit or Probit supported, got {cls}"
        )))
    }
}

/// CDF of the logistic (logit) or standard normal (probit) distribution.
fn cdf_logit_probit(is_logit: bool, z: f64) -> f64 {
    if is_logit {
        1.0 / (1.0 + (-z).exp())
    } else {
        let dist = Normal::new(0.0, 1.0).unwrap();
        dist.cdf(z)
    }
}

/// PDF of the logistic (logit) or standard normal (probit) distribution.
fn pdf_logit_probit(is_logit: bool, z: f64) -> f64 {
    if is_logit {
        let e = z.exp();
        e / (1.0 + e).powi(2)
    } else {
        (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt()
    }
}

/// Derivative f'(z) of the PDF, used for the delta-method Jacobian.
///
/// - Logit:  f'(z) = f(z) * (1 - 2 F(z))
/// - Probit: f'(z) = -z * phi(z)
fn pdf_deriv_logit_probit(is_logit: bool, z: f64) -> f64 {
    if is_logit {
        pdf_logit_probit(is_logit, z) * (1.0 - 2.0 * cdf_logit_probit(is_logit, z))
    } else {
        -z * pdf_logit_probit(is_logit, z)
    }
}

/// Computes average marginal effects (AMEs) and delta-method standard errors
/// for fitted statsmodels Logit/Probit models. Results are identical to
/// `statsmodels.get_margeff()` to floating-point precision.
///
/// Parameters
/// ----------
/// model : a fitted statsmodels Logit or Probit results object.
/// chunk_size : optional, process observations in chunks to bound peak memory.
///     Has no effect on the result, only on memory/throughput.
/// dummy : optional bool, default False. Mirrors statsmodels' `dummy` flag.
///     - False: every non-constant regressor (including 0/1 columns) is treated
///              as continuous. This is the statsmodels DEFAULT, so a bare
///              `mfx(model)` reproduces a bare `model.get_margeff()`.
///     - True:  any strictly 0/1 column is treated as discrete and its effect is
///              the finite difference F(.,x=1) - F(.,x=0), reproducing
///              `model.get_margeff(dummy=True)`.
///
/// Notes
/// -----
/// The delta-method Jacobian is the exact analytic derivative of the average
/// marginal effect with respect to beta:
///
///     d AME_j / d beta_l = delta_{jl} * mean(f(Xb)) + beta_j * mean(f'(Xb) * x_l)
///
/// for continuous variables, and the difference of predicted-probability
/// gradients at x=1 and x=0 for discrete variables. This matches the Jacobian
/// statsmodels computes by numerical differentiation, hence the identical SEs.
#[pyfunction]
#[pyo3(signature = (model, chunk_size=None, dummy=None))]
fn mfx<'py>(
    py: Python<'py>,
    model: &'py PyAny,
    chunk_size: Option<usize>,
    dummy: Option<bool>,
) -> PyResult<&'py PyAny> {
    let use_dummy = dummy.unwrap_or(false);

    // 1) Logit vs Probit.
    let is_logit = detect_model_type(model)?;

    // 2) Model parameters (pandas Series or numpy array).
    let params_obj = model.getattr("params")?;
    let params_pyarray = if let Ok(values) = params_obj.getattr("values") {
        values.downcast::<PyArray1<f64>>()?
    } else {
        params_obj.downcast::<PyArray1<f64>>()?
    };
    let beta: ArrayView1<f64> = unsafe { params_pyarray.as_array() };

    // 3) Parameter covariance matrix.
    let cov_obj = model.call_method0("cov_params")?;
    let cov_pyarray = if let Ok(values) = cov_obj.getattr("values") {
        values.downcast::<PyArray2<f64>>()?
    } else {
        cov_obj.downcast::<PyArray2<f64>>()?
    };
    let cov_beta: ArrayView2<f64> = unsafe { cov_pyarray.as_array() };

    // 4) Exogenous design matrix X and its column names.
    let model_obj = model.getattr("model").unwrap_or(model);
    let exog_py = model_obj.getattr("exog")?;
    let (x_pyarray, exog_names) = if let Ok(values) = exog_py.getattr("values") {
        (
            values.downcast::<PyArray2<f64>>()?,
            exog_py.getattr("columns")?.extract::<Vec<String>>()?,
        )
    } else {
        (
            exog_py.downcast::<PyArray2<f64>>()?,
            model_obj.getattr("exog_names")?.extract::<Vec<String>>()?,
        )
    };
    let x_mat: ArrayView2<f64> = unsafe { x_pyarray.as_array() };
    let (n, k) = (x_mat.nrows(), x_mat.ncols());
    if n == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "mfx: model has no observations",
        ));
    }
    let chunk = chunk_size.unwrap_or(n).max(1);

    // 5) Intercept columns ("const"/"intercept", case-insensitive). Dropped from output.
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

    // 6) Discrete columns: only when dummy=True, and only strictly-0/1 non-intercept
    //    columns. With dummy=False (the statsmodels default) this is empty and every
    //    regressor is treated as continuous.
    let is_discrete: Vec<usize> = if use_dummy {
        (0..k)
            .filter(|&j| {
                !intercept_indices.contains(&j)
                    && x_mat.column(j).iter().all(|&v| v == 0.0 || v == 1.0)
            })
            .collect()
    } else {
        Vec::new()
    };

    // 7) Accumulators (summed over observations; averaged at the end).
    let mut sum_ame = vec![0.0; k];
    let mut jac_sums = vec![0.0; k * k]; // row-major k x k

    // 8) Stream over observation chunks.
    let mut idx_start = 0;
    while idx_start < n {
        let idx_end = (idx_start + chunk).min(n);
        let x_chunk = x_mat.slice(s![idx_start..idx_end, ..]);
        let z_chunk = x_chunk.dot(&beta);
        let pdf_chunk = z_chunk.mapv(|z| pdf_logit_probit(is_logit, z));

        // --- Discrete (dummy) variables: finite-difference effect, matching
        //     statsmodels get_margeff(dummy=True). ---
        for &j in &is_discrete {
            let xj_col = x_chunk.column(j);
            let b_j = beta[j];
            // z with x_j forced to 1 and to 0.
            let z_j1 = &z_chunk + &(1.0 - &xj_col).mapv(|d| d * b_j);
            let z_j0 = &z_chunk + &xj_col.mapv(|x| -x * b_j);

            let cdf_j1 = z_j1.mapv(|z| cdf_logit_probit(is_logit, z));
            let cdf_j0 = z_j0.mapv(|z| cdf_logit_probit(is_logit, z));
            sum_ame[j] += cdf_j1.sum() - cdf_j0.sum();

            // Jacobian row j: mean( f(z1)*X|x_j=1  -  f(z0)*X|x_j=0 ).
            // Only column j of X is altered (to 1 and 0 respectively); all other
            // columns keep their observed values.
            let pdf_j1 = z_j1.mapv(|z| pdf_logit_probit(is_logit, z));
            let pdf_j0 = z_j0.mapv(|z| pdf_logit_probit(is_logit, z));
            for l in 0..k {
                let grad = if l == j {
                    // f(z1)*1 - f(z0)*0
                    pdf_j1.sum()
                } else {
                    let x_l = x_chunk.column(l);
                    (&pdf_j1 - &pdf_j0).dot(&x_l)
                };
                jac_sums[j * k + l] += grad;
            }
        }

        // --- Continuous variables (everything not intercept and not discrete). ---
        // AME_j      = mean( beta_j * f(Xb) )
        // dAME_j/db_l = delta_{jl} * mean(f(Xb)) + beta_j * mean(f'(Xb) * x_l)
        for j in 0..k {
            if intercept_indices.contains(&j) || is_discrete.contains(&j) {
                continue;
            }
            let b_j = beta[j];
            sum_ame[j] += b_j * pdf_chunk.sum();

            let fprime_chunk = z_chunk.mapv(|z| pdf_deriv_logit_probit(is_logit, z));
            for l in 0..k {
                let x_l = x_chunk.column(l);
                let term = (&x_l * &fprime_chunk).sum();
                let grad = if j == l {
                    pdf_chunk.sum() + b_j * term
                } else {
                    b_j * term
                };
                jac_sums[j * k + l] += grad;
            }
        }

        idx_start = idx_end;
    }

    // 9) Average over observations.
    let inv_n = 1.0 / (n as f64);
    let final_ame: Vec<f64> = sum_ame.iter().map(|v| v * inv_n).collect();
    let mut grad_ame = Array2::<f64>::zeros((k, k));
    for j in 0..k {
        for l in 0..k {
            grad_ame[[j, l]] = jac_sums[j * k + l] * inv_n;
        }
    }

    // 10) Delta method: Var(AME) = J * Var(beta) * J^T.
    let cov_ame = grad_ame.dot(&cov_beta).dot(&grad_ame.t());
    let se_ame: Vec<f64> = cov_ame
        .diag()
        .iter()
        .map(|&v| v.max(0.0).sqrt())
        .collect();

    // 11) Assemble output rows (intercepts dropped).
    let normal = Normal::new(0.0, 1.0).unwrap();
    let (mut dy_dx, mut se_err, mut z_vals, mut p_vals, mut sig) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let (mut conf_low, mut conf_high) = (Vec::new(), Vec::new());
    let mut names_out = Vec::new();

    for j in 0..k {
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
            conf_low.push(dy - 1.96 * s);
            conf_high.push(dy + 1.96 * s);
        } else {
            z_vals.push(0.0);
            p_vals.push(1.0);
            sig.push("");
            conf_low.push(dy);
            conf_high.push(dy);
        }
        names_out.push(exog_names[j].clone());
    }

    // 12) Build the pandas DataFrame result.
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
    m.add_function(wrap_pyfunction!(mfx, m)?)?;
    Ok(())
}
