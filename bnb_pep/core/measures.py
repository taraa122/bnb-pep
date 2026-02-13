"""
Performance measures and initial conditions for PEP.

Performance measure E: what we're bounding in the worst case.
Initial condition C: constraint on the starting point.
"""

from __future__ import annotations
import numpy as np
from enum import Enum, auto
from core.gram import GramBuilder


class PerformanceMeasure(Enum):
    FUNCTION_VALUE = auto()      # f(x_N) - f(x_*)
    GRADIENT_NORM = auto()       # ||∇f(x_N)||^2
    MIN_GRADIENT_NORM = auto()   # min_{i∈[0:N]} ||∇f(x_i)||^2
    DISTANCE_TO_OPT = auto()     # ||x_N - x_*||^2


class InitialCondition(Enum):
    DISTANCE_BOUND = auto()      # ||x_0 - x_*||^2 ≤ R^2
    FUNCTION_VALUE_BOUND = auto()  # f(x_0) - f(x_*) ≤ R^2


def build_objective(
    measure: PerformanceMeasure,
    gb: GramBuilder,
    iterates: dict[str, np.ndarray],
    grads: dict[str, np.ndarray],
    fvals: dict[str, np.ndarray],
    mu: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Build the objective tr(G @ Q) + F @ q + constant to MAXIMIZE.
    
    Returns (Q_gram, q_fval, constant).
    """
    n = gb.gram_size
    m = gb.fval_size
    N = gb.N

    if measure == PerformanceMeasure.FUNCTION_VALUE:
        # f(x_N) - f(x_*) = F @ a_{*,N}
        Q = np.zeros((n, n))
        q = gb.a(gb.fstar(), gb.f(N))  # f_N - f_* = f_N
        return Q, q, 0.0

    elif measure == PerformanceMeasure.GRADIENT_NORM:
        # ||∇f(x_N)||^2 = tr(G @ C_{N,*})
        # After reparam for F_{μ,L}: ||g_N||^2 + μ^2||x_N||^2 + 2μ⟨g_N, x_N⟩
        # but if mu=0 it's just ||g_N||^2
        gN = grads['g_N']
        gstar = gb.gstar()
        Q = gb.C(gN, gstar)  # ||g_N - 0||^2
        q = np.zeros(m)
        
        if mu > 0:
            xN = iterates['x_N']
            xstar = gb.xstar()
            Q = Q + mu**2 * gb.B(xN, xstar) - 2 * mu * GramBuilder.sym_outer(gstar - gN, xstar - xN)
        
        return Q, q, 0.0

    elif measure == PerformanceMeasure.MIN_GRADIENT_NORM:
        # min_{i∈[0:N]} ||∇f(x_i)||^2  — handled via epigraph: t ≤ ||g_i||^2 ∀i
        # The objective becomes just t (scalar), handled specially
        # Return placeholder; actual handling is in the PEP builder
        Q = np.zeros((n, n))
        q = np.zeros(m)
        return Q, q, 0.0

    elif measure == PerformanceMeasure.DISTANCE_TO_OPT:
        xN = iterates['x_N']
        xstar = gb.xstar()
        Q = gb.B(xN, xstar)
        q = np.zeros(m)
        return Q, q, 0.0

    else:
        raise ValueError(f"Unknown measure: {measure}")


def build_initial_condition(
    condition: InitialCondition,
    gb: GramBuilder,
    R: float,
    iterates: dict[str, np.ndarray],
    fvals: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Build constraint tr(G @ A_ic) + F @ a_ic + constant ≤ 0.
    """
    if condition == InitialCondition.DISTANCE_BOUND:
        # ||x_0 - x_*||^2 ≤ R^2  →  tr(G @ B_{0,*}) - R^2 ≤ 0
        A_ic = gb.B(gb.x0(), gb.xstar())
        a_ic = np.zeros(gb.fval_size)
        return A_ic, a_ic, -R**2

    elif condition == InitialCondition.FUNCTION_VALUE_BOUND:
        # f(x_0) - f(x_*) ≤ R^2  →  F @ (f_0 - f_*) - R^2 ≤ 0
        A_ic = np.zeros((gb.gram_size, gb.gram_size))
        a_ic = gb.f(0)  # f_0 (since f_* = 0)
        return A_ic, a_ic, -R**2

    else:
        raise ValueError(f"Unknown initial condition: {condition}")
