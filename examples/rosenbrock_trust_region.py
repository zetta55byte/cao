"""
Rosenbrock minimization via CAO TrustRegionOptimizer.

True minimum: f(1, 1) = 0.
Starting point: (-1.2, 1.0) — the classic hard start.
"""

import numpy as np
from cao import TrustRegionOptimizer


def rosenbrock(X):
    """f(x0, x1) = 100(x1 - x0^2)^2 + (1 - x0)^2"""
    return (X[1] - X[0] ** 2) ** 2 * 100.0 + (X[0] * 0.0 + 1.0 - X[0]) ** 2


if __name__ == "__main__":
    x0 = np.array([-1.2, 1.0])

    print("=== CAO Trust-Region Optimizer — Rosenbrock ===\n")
    print(f"Start:  x0 = {x0},  f(x0) = {rosenbrock(list(x0)):.4f}")
    print(f"Target: x* = [1.0, 1.0],  f(x*) = 0.0\n")

    opt = TrustRegionOptimizer(
        rosenbrock,
        backend="jax-xla",
    )
    result = opt.minimize(x0)

    print(result)
    print(f"x* = {result.x}")
    print(f"f(x*) = {result.f:.2e}")
    print(f"Iterations: {result.nit}")
    print(f"Hessian evaluations: {result.nfev}")
    print(f"Converged: {result.converged}")
