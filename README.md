# CAO — Curvature-Aware Optimization

[![CI](https://github.com/zetta55byte/cao/actions/workflows/ci.yml/badge.svg)](https://github.com/zetta55byte/cao/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/cao)](https://pypi.org/project/cao/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**CAO** provides practical second-order optimization using **exact Hessians**
computed via [`hcderiv`](https://pypi.org/project/hcderiv/).

> Make trust-region and Newton-style optimization as easy to use as Adam or SGD.

No finite differences. No approximations. No step-size tuning.

---

## Install

```bash
pip install cao                  # NumPy backend
pip install "cao[jax]"           # + JAX/XLA backend (recommended)
```

---

## Quick start

```python
from cao import TrustRegionOptimizer
import numpy as np

def f(X):
    return (X[1] - X[0]**2)**2 * 100.0 + (X[0]*0.0 + 1.0 - X[0])**2

opt = TrustRegionOptimizer(f, backend="jax-xla")
result = opt.minimize([-1.2, 1.0])

print(result)
# OptimizeResult(f=1.5e-15, ‖g‖∞=3.4e-08, nit=23, converged)

print(result.x)
# [1. 1.]
```

---

## How it works

At each iteration CAO calls `hcderiv.grad_and_hessian(f, x, backend=...)` to
compute the exact gradient and full Hessian in a single forward pass.  It then
solves the trust-region subproblem:

```
min  g·p + ½ p·H·p   s.t.  ‖p‖ ≤ Δ
```

using the **Cauchy point** as a safe fallback and the **regularised Newton
step** when it lies inside the trust region.  The radius Δ is adapted each
iteration based on the ratio of actual to predicted reduction.

**Why exact Hessians?**

Quasi-Newton methods (L-BFGS) and diagonal approximations miss off-diagonal
curvature.  `hcderiv`'s XLA backend computes the full d×d Hessian in one
forward pass with NumPy-class speed, making exact second-order steps practical
for the first time at this scale.

---

## Architecture

```
hcderiv (exact curvature engine)
    ↓  grad_and_hessian(f, x, backend="jax-xla")
cao (trust-region solver)
    ↓  TrustRegionOptimizer.minimize(x0)
x*
```

---

## Performance

Rosenbrock from `(-1.2, 1.0)` — convergence to `f < 1e-14`:

| Method | Iterations | Hessian evals |
|---|---|---|
| CAO trust-region (exact H) | **23** | 23 |
| scipy L-BFGS-B | ~120 | 0 (approx) |
| scipy Newton-CG | ~35 | 35 (FD) |

CAO converges in fewer iterations because it uses the exact curvature at every step.

---

## API

```python
from cao import TrustRegionOptimizer, TrustRegionConfig, OptimizeResult

# Configure
config = TrustRegionConfig(
    max_iters=200,
    initial_radius=1.0,
    tol_grad=1e-6,
    verbose=True,       # print iteration log
)

# Run
opt = TrustRegionOptimizer(f, config=config, backend="jax-xla")
result = opt.minimize(x0)

# Result fields
result.x           # final iterate
result.f           # function value
result.grad_norm   # ‖∇f‖∞ at x*
result.nit         # iterations
result.nfev        # Hessian evaluations
result.converged   # bool
result.f_history   # f at each accepted step
```

---

## Ecosystem

```
hcderiv          →  exact Hessians (NumPy + JAX-XLA)
    ↓
cao              →  curvature-aware optimizer
    ↓
constitutional-os + governed-research-lab-v2  →  curvature-governed agents
```

---

## Roadmap

- v0.2.0 — cubic regularization (ARC)
- v0.3.0 — PyTorch backend
- v0.4.0 — curvature-aware learning rate schedules for ML models

---

## Citation

A JOSS paper will accompany CAO after the 6-month public availability window.
Until then, please cite the hcderiv engine:

```bibtex
@misc{byte2026d,
  author = {Byte, Zetta},
  title  = {hcderiv v0.4.0 — JAX-XLA backend for exact one-pass Hessians},
  year   = {2026},
  doi    = {10.5281/zenodo.19433812},
  url    = {https://doi.org/10.5281/zenodo.19433812}
}
```
