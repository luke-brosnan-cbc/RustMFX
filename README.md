# FeBOLT-

[![Build Status](https://github.com/luke-brosnan-cbc/FeBOLT/workflows/build/badge.svg)](https://github.com/luke-brosnan-cbc/FeBOLT/actions)
[![PyPI version](https://badge.fury.io/py/febolt.svg)](https://pypi.org/project/febolt/)

## Introduction

As datasets continue to grow in size, economists, social scientists, and data analysts require more efficient tools for statistical modeling and inference. Traditional Python libraries like `statsmodels` provide robust inference capabilities but can be slow and memory-intensive, making them impractical for large datasets. Meanwhile, `scikit-learn` offers efficient machine learning tools but lacks the depth of statistical inference needed for rigorous empirical research.

Enter `Febolt`: a high-performance statistical modeling package built with Rust to provide **fast, memory-efficient**, and **fully-featured inference** capabilities. `FeBOLT` is designed to bridge the gap between performance and analytical depth, making it an ideal choice for researchers working with large-scale data.

## Features

- **Probit, Logit, and OLS Models**: Supports fundamental regression models with additional enhancements.
- **Weighted Regression**: Apply observation weights to models.
- **Clustered and Robust Standard Errors**: More reliable inference with robust and cluster-adjusted SEs.
- **Average Marginal Effects (AMEs)**: Compute AMEs for Logit and Probit models.
- **Rust-Powered Performance**: Significantly faster computations compared to Python-based alternatives.
- **Optimized for 32-bit and 64-bit Floats**: Choose between improved memory efficiency with 32-bit floats or higher precision with 64-bit floats.

## Why FeBOLT?

### **Performance Meets Inference**
Unlike `scikit-learn`, which focuses on machine learning without comprehensive inference support, `FeBOLT` is built specifically for statistical modeling while maintaining **speed and efficiency**. Unlike `statsmodels`, which can be bulky and slow for large datasets, `FeBOLT` leverages **Rust’s performance optimizations** to provide rapid computations without sacrificing analytical power.

### **Memory Efficiency for Large Datasets**
Economists and social scientists often deal with panel datasets and large-scale survey data, where traditional inference models become infeasible due to memory constraints. `FeBOLT` allows the use of **32-bit floats** to **significantly reduce memory usage**, while still offering **64-bit float precision** for cases where accuracy is paramount.

### **Inference Without Compromise**
While `scikit-learn` lacks built-in inference tools like **robust and clustered standard errors**, `FeBOLT` incorporates these essential statistical features to support rigorous empirical research. Whether you need **fast OLS regression** or **efficient Probit/Logit estimation with AMEs**, `FeBOLT` delivers both speed and accuracy in one package.

## Installation

```bash
pip install febolt
```

## Quick Start

```python
import febolt

# Example usage (to be filled in)
```

## Performance

`FeBOLT` outperforms `statsmodels` and `scikit-learn` by leveraging Rust’s speed and memory efficiency. This results in significantly faster execution times, especially for large datasets and models requiring robust standard errors.

## Contributing

Contributions are welcome! Feel free to submit issues and pull requests on [GitHub](https://github.com/luke-brosnan-cbc/FeBOLT).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

