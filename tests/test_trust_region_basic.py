"""
tests/test_trust_region_basic.py
---------------------------------
Basic tests for TrustRegionOptimizer.

Tests are deterministic and do not require network access.
JAX is required for the jax-xla backend tests.
"""

from __future__ import annotations

import numpy as np
import pytest
from cao import OptimizeResult, TrustRegionConfig, TrustRegionOptimizer
from cao.trust_region import _cauchy_point, _newton_step, _quadratic_model

# ── test functions ────────────────────────────────────────────────────────────


def rosenbrock(X):
    """f(x0, x1) = 100(x1-x0^2)^2 + (1-x0)^2.  Min at (1,1), f=0."""
    return (X[1] - X[0] ** 2) ** 2 * 100.0 + (X[0] * 0.0 + 1.0 - X[0]) ** 2


def bowl(X):
    """f(x) = x0^2 + 2*x1^2.  Min at (0,0), f=0. Convex."""
    return X[0] ** 2 + X[1] ** 2 * 2.0


# ── subproblem solvers ────────────────────────────────────────────────────────


class TestCauchyPoint:
    def test_zero_gradient(self):
        g = np.zeros(3)
        H = np.eye(3)
        p = _cauchy_point(g, H, delta=1.0)
        assert np.allclose(p, np.zeros(3))

    def test_within_trust_region(self):
        g = np.array([1.0, 0.0])
        H = np.eye(2)
        delta = 2.0
        p = _cauchy_point(g, H, delta)
        assert np.linalg.norm(p) <= delta + 1e-12

    def test_descent_direction(self):
        g = np.array([1.0, 1.0])
        H = np.eye(2)
        p = _cauchy_point(g, H, delta=1.0)
        assert g @ p < 0  # descent

    def test_indefinite_H_steps_to_boundary(self):
        g = np.array([1.0, 0.0])
        H = -np.eye(2)  # negative definite
        delta = 1.0
        p = _cauchy_point(g, H, delta)
        assert abs(np.linalg.norm(p) - delta) < 1e-10


class TestNewtonStep:
    def test_exact_for_quadratic(self):
        """For f = ½ x^T H x + g^T x, Newton step = -H^{-1} g."""
        g = np.array([2.0, -1.0])
        H = np.array([[4.0, 0.0], [0.0, 2.0]])
        p = _newton_step(g, H, delta=10.0, reg=0.0)
        p_true = -np.linalg.solve(H, g)
        assert np.allclose(p, p_true, atol=1e-10)

    def test_clipped_to_trust_region(self):
        g = np.array([1.0, 0.0])
        H = np.eye(2) * 1e-6  # nearly flat — Newton step is huge
        delta = 0.5
        p = _newton_step(g, H, delta, reg=0.0)
        assert np.linalg.norm(p) <= delta + 1e-10


class TestQuadraticModel:
    def test_zero_step(self):
        g = np.array([1.0, 2.0])
        H = np.eye(2)
        p = np.zeros(2)
        assert _quadratic_model(g, H, p) == 0.0

    def test_value(self):
        g = np.array([1.0, 0.0])
        H = np.array([[2.0, 0.0], [0.0, 2.0]])
        p = np.array([1.0, 0.0])
        # m(p) = 1*1 + 0.5*1*2*1 = 1 + 1 = 2
        assert _quadratic_model(g, H, p) == pytest.approx(2.0)


# ── optimizer ─────────────────────────────────────────────────────────────────


class TestTrustRegionOptimizer:
    def test_result_type(self):
        opt = TrustRegionOptimizer(bowl, backend="numpy")
        result = opt.minimize([1.0, 1.0])
        assert isinstance(result, OptimizeResult)

    def test_bowl_converges(self):
        """Convex bowl — should converge fast to (0,0)."""
        opt = TrustRegionOptimizer(bowl, backend="numpy")
        result = opt.minimize([2.0, -3.0])
        assert result.converged
        assert np.allclose(result.x, [0.0, 0.0], atol=1e-4)
        assert result.f < 1e-8

    def test_rosenbrock_decreases(self):
        """Must reduce the objective from the starting point."""
        opt = TrustRegionOptimizer(rosenbrock, backend="numpy")
        x0 = np.array([-1.2, 1.0])
        result = opt.minimize(x0)
        assert result.f < rosenbrock(list(x0))

    def test_rosenbrock_approaches_minimum(self):
        """Should get within distance 0.1 of (1,1)."""
        opt = TrustRegionOptimizer(
            rosenbrock,
            config=TrustRegionConfig(max_iters=300, tol_grad=1e-6),
            backend="numpy",
        )
        result = opt.minimize([-1.2, 1.0])
        assert np.linalg.norm(result.x - np.array([1.0, 1.0])) < 0.1

    def test_f_history_decreasing(self):
        """Accepted steps should decrease f."""
        opt = TrustRegionOptimizer(bowl, backend="numpy")
        result = opt.minimize([3.0, 3.0])
        hist = result.f_history
        # Each accepted step should not increase f
        for i in range(1, len(hist)):
            assert hist[i] <= hist[i - 1] + 1e-10

    def test_nfev_positive(self):
        opt = TrustRegionOptimizer(bowl, backend="numpy")
        result = opt.minimize([1.0, 1.0])
        assert result.nfev > 0

    def test_config_max_iters_respected(self):
        opt = TrustRegionOptimizer(
            rosenbrock,
            config=TrustRegionConfig(max_iters=5),
            backend="numpy",
        )
        result = opt.minimize([-1.2, 1.0])
        assert result.nit <= 5

    def test_already_at_minimum(self):
        """Starting at the minimum should converge immediately."""
        opt = TrustRegionOptimizer(
            bowl,
            config=TrustRegionConfig(tol_grad=1e-6),
            backend="numpy",
        )
        result = opt.minimize([0.0, 0.0])
        assert result.converged

    @pytest.mark.skipif(
        not pytest.importorskip("jax", reason="JAX not installed"),
        reason="JAX not installed",
    )
    def test_jax_xla_backend_runs(self):
        """XLA backend should produce same result as numpy."""
        result_np = TrustRegionOptimizer(bowl, backend="numpy").minimize([2.0, -1.0])
        result_xla = TrustRegionOptimizer(bowl, backend="jax-xla").minimize([2.0, -1.0])
        assert np.allclose(result_np.x, result_xla.x, atol=1e-4)
