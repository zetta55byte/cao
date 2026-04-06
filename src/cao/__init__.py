"""
cao — Curvature-Aware Optimization
====================================
Practical second-order optimization using exact Hessians from hcderiv.

Quick start
-----------
    from cao import TrustRegionOptimizer
    import numpy as np

    def f(X):
        return (X[1] - X[0]**2)**2 * 100.0 + (X[0]*0.0 + 1.0 - X[0])**2

    opt = TrustRegionOptimizer(f, backend="jax-xla")
    x_star = opt.minimize([-1.2, 1.0])
    # array([1., 1.])

References
----------
Byte, Z. (2026). hcderiv: exact one-pass Hessians via hypercomplex perturbation.
DOI: 10.5281/zenodo.19433812
"""

from .trust_region import OptimizeResult, TrustRegionConfig, TrustRegionOptimizer

__version__ = "0.1.0"
__author__ = "Zetta Byte"

__all__ = [
    "TrustRegionOptimizer",
    "TrustRegionConfig",
    "OptimizeResult",
]
