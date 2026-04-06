"""
cao.trust_region
----------------
Trust-region optimizer using exact Hessians from hcderiv.

Algorithm
---------
At each iteration:
  1. Compute exact f(x), ∇f(x), ∇²f(x) via hcderiv (one forward pass).
  2. Solve the trust-region subproblem:
       min  g·p + ½ p·H·p   s.t.  ‖p‖ ≤ Δ
     using the Cauchy point as a fallback and the Newton step when feasible.
  3. Compute the ratio ρ = actual_reduction / predicted_reduction.
  4. Accept or reject the step; adapt the trust-region radius Δ.

The Cauchy point is always safe (guarantees descent for positive definite H).
The Newton step is projected onto the trust region when it overshoots.

References
----------
Nocedal, J. & Wright, S. J. (2006). Numerical Optimization (2nd ed.), ch. 4.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TrustRegionConfig:
    """
    Hyperparameters for the trust-region optimizer.

    Parameters
    ----------
    max_iters : int
        Maximum number of iterations. Default 200.
    initial_radius : float
        Initial trust-region radius Δ₀. Default 1.0.
    max_radius : float
        Maximum allowed radius. Default 100.0.
    eta1 : float
        Step acceptance threshold (ρ ≥ η₁ → accept). Default 0.1.
    eta2 : float
        Radius expansion threshold (ρ ≥ η₂ → expand). Default 0.75.
    gamma1 : float
        Radius shrink factor. Default 0.25.
    gamma2 : float
        Radius expand factor. Default 2.0.
    tol_grad : float
        Gradient infinity-norm stopping tolerance. Default 1e-6.
    reg : float
        Regularisation added to H before Newton solve. Default 1e-8.
    backend : str
        hcderiv backend. Default "jax-xla".
    verbose : bool
        Print iteration log. Default False.
    """

    max_iters: int = 200
    initial_radius: float = 1.0
    max_radius: float = 100.0
    eta1: float = 0.1
    eta2: float = 0.75
    gamma1: float = 0.25
    gamma2: float = 2.0
    tol_grad: float = 1e-6
    reg: float = 1e-8
    backend: str = "jax-xla"
    verbose: bool = False


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class OptimizeResult:
    """
    Result of a trust-region optimization run.

    Attributes
    ----------
    x : np.ndarray
        Final iterate.
    f : float
        Function value at x.
    grad_norm : float
        Infinity norm of the gradient at x.
    nit : int
        Number of iterations taken.
    nfev : int
        Number of function/gradient/Hessian evaluations.
    converged : bool
        True if gradient tolerance was met.
    f_history : list[float]
        Function value at each accepted iterate.
    """

    x: np.ndarray
    f: float
    grad_norm: float
    nit: int
    nfev: int
    converged: bool
    f_history: List[float] = field(default_factory=list)

    def __repr__(self) -> str:
        status = "converged" if self.converged else "max_iters"
        return (
            f"OptimizeResult(f={self.f:.6g}, ‖g‖∞={self.grad_norm:.2e}, "
            f"nit={self.nit}, {status})"
        )


# ---------------------------------------------------------------------------
# Core subproblem solvers
# ---------------------------------------------------------------------------


def _cauchy_point(g: np.ndarray, H: np.ndarray, delta: float) -> np.ndarray:
    """
    Compute the Cauchy point: steepest-descent step clipped to trust region.

    Always gives a reduction in the quadratic model when g ≠ 0.
    """
    g_norm = np.linalg.norm(g)
    if g_norm < 1e-300:
        return np.zeros_like(g)
    gHg = g @ H @ g
    if gHg <= 0:
        # Indefinite or negative-definite: step to boundary
        tau = 1.0
    else:
        tau = min(g_norm**3 / (delta * gHg), 1.0)
    return -tau * delta * g / g_norm


def _newton_step(g: np.ndarray, H: np.ndarray, delta: float, reg: float) -> np.ndarray:
    """
    Compute the Newton step, regularised and projected onto trust region.

    Solves (H + reg·I) p = -g, then clips to ‖p‖ ≤ delta.
    """
    n = len(g)
    H_reg = H + reg * np.eye(n)
    try:
        p = np.linalg.solve(H_reg, -g)
    except np.linalg.LinAlgError:
        # Singular — fall back to gradient direction
        p = -g / (np.linalg.norm(g) + 1e-300) * delta
    norm_p = np.linalg.norm(p)
    if norm_p > delta:
        p = p * (delta / norm_p)
    return p


def _quadratic_model(g: np.ndarray, H: np.ndarray, p: np.ndarray) -> float:
    """m(p) = g·p + ½ p·H·p"""
    return float(g @ p + 0.5 * p @ H @ p)


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------


class TrustRegionOptimizer:
    """
    Trust-region optimizer using exact Hessians from hcderiv.

    Parameters
    ----------
    f : callable
        Scalar objective function.
        For backend="jax-xla": f takes a list of JAXHyperArray objects.
        For backend="numpy": f takes a list of Hyper objects.
        Write f using the same operators as hcderiv examples
        (e.g., X[0]**2 + X[1].sin()).
    config : TrustRegionConfig, optional
        Optimisation hyperparameters.
    backend : str, optional
        hcderiv backend override. Overrides config.backend if given.

    Examples
    --------
    >>> from cao import TrustRegionOptimizer
    >>> def f(X): return (X[1] - X[0]**2)**2 * 100.0 + (X[0]*0.0+1.0-X[0])**2
    >>> opt = TrustRegionOptimizer(f, backend="jax-xla")
    >>> result = opt.minimize([-1.2, 1.0])
    >>> result.x
    array([1., 1.])
    """

    def __init__(
        self,
        f: Callable,
        config: Optional[TrustRegionConfig] = None,
        backend: Optional[str] = None,
    ):
        self.f_raw = f
        self.config = config or TrustRegionConfig()
        if backend is not None:
            self.config.backend = backend

    def _eval(self, x: np.ndarray):
        """Return (f_val, grad, H) via hcderiv."""
        from hypercomplex import grad_and_hessian

        g, H = grad_and_hessian(self.f_raw, x, backend=self.config.backend)
        # Also need f value — evaluate directly
        # Get f value cheaply: just call the raw function on plain floats if possible
        try:
            f_val = float(self.f_raw(list(x)))
        except Exception:
            # Fallback: use the primal from hcderiv (needs grad_and_hessian to return it)
            # For now use a plain numpy evaluation
            f_val = float(np.nan)

        return f_val, g, H

    def minimize(
        self,
        x0,
        return_result: bool = True,
    ) -> OptimizeResult:
        """
        Run the trust-region optimizer from x0.

        Parameters
        ----------
        x0 : array-like
            Starting point.
        return_result : bool
            If True return OptimizeResult. If False return just the final x.

        Returns
        -------
        OptimizeResult
        """
        cfg = self.config
        x = np.asarray(x0, dtype=np.float64)
        delta = cfg.initial_radius
        nfev = 0
        f_history = []
        converged = False

        # Initial evaluation
        f_val, g, H = self._eval(x)
        nfev += 1

        # If f_val is nan, try direct evaluation
        if np.isnan(f_val):
            try:
                import jax.numpy as jnp
                from hypercomplex.backends.jax_xla import hessian_xla

                x_jax = jnp.array(x)
                p_xla, g_xla, H_xla = hessian_xla(self.f_raw, x_jax)
                f_val = float(p_xla)
                g = np.array(g_xla)
                H = np.array(H_xla)
            except Exception:
                pass

        f_history.append(f_val)

        if cfg.verbose:
            print(f"{'iter':>6}  {'f':>14}  {'‖g‖∞':>10}  {'Δ':>10}  {'ρ':>8}  status")
            print("-" * 65)

        for i in range(cfg.max_iters):
            grad_norm = float(np.linalg.norm(g, ord=np.inf))

            if cfg.verbose:
                print(f"{i:>6}  {f_val:>14.6g}  {grad_norm:>10.2e}  " f"{delta:>10.4f}  {'---':>8}")

            if grad_norm < cfg.tol_grad:
                converged = True
                break

            # Subproblem: compute candidate steps
            p_c = _cauchy_point(g, H, delta)
            p_n = _newton_step(g, H, delta, cfg.reg)

            m_c = _quadratic_model(g, H, p_c)
            m_n = _quadratic_model(g, H, p_n)

            # Pick better model reduction
            if m_n < m_c:
                p = p_n
                pred_red = -m_n
            else:
                p = p_c
                pred_red = -m_c

            if pred_red <= 0:
                # No predicted reduction — shrink and continue
                delta *= cfg.gamma1
                continue

            # Evaluate at candidate
            x_new = x + p
            f_new, g_new, H_new = self._eval(x_new)
            nfev += 1

            # Primal from XLA if nan
            if np.isnan(f_new):
                try:
                    import jax.numpy as jnp
                    from hypercomplex.backends.jax_xla import hessian_xla

                    p_xla, g_xla, H_xla = hessian_xla(self.f_raw, jnp.array(x_new))
                    f_new = float(p_xla)
                    g_new = np.array(g_xla)
                    H_new = np.array(H_xla)
                except Exception:
                    pass

            actual_red = f_val - f_new
            rho = actual_red / (pred_red + 1e-300)

            # Radius adaptation
            step_norm = np.linalg.norm(p)
            if rho < cfg.eta1:
                delta = cfg.gamma1 * delta
            elif rho >= cfg.eta2 and step_norm >= 0.9 * delta:
                delta = min(cfg.gamma2 * delta, cfg.max_radius)

            # Step acceptance
            if rho >= cfg.eta1:
                x = x_new
                f_val = f_new
                g = g_new
                H = H_new
                f_history.append(f_val)

            if cfg.verbose:
                status = "accept" if rho >= cfg.eta1 else "reject"
                print(
                    f"{'':>6}  {f_new:>14.6g}  {'':>10}  " f"{delta:>10.4f}  {rho:>8.3f}  {status}"
                )

        grad_norm = float(np.linalg.norm(g, ord=np.inf))

        return OptimizeResult(
            x=x,
            f=f_val,
            grad_norm=grad_norm,
            nit=i + 1,
            nfev=nfev,
            converged=converged,
            f_history=f_history,
        )
