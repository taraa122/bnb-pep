"""
Function classes and their interpolation constraints.

Each function class defines what constraints a set of triples {(x_i, g_i, f_i)}
must satisfy to be interpolable by some function in that class.

Supported classes:
    F_{0,L}    : L-smooth convex
    F_{μ,L}    : L-smooth μ-strongly convex  
    F_{-L,L}   : L-smooth nonconvex
    W_{ρ,L}    : ρ-weakly convex with L-bounded subgradients
"""

from __future__ import annotations
import numpy as np
from enum import Enum, auto
from dataclasses import dataclass
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from core.gram import GramBuilder


def _sym_outer(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Symmetric outer product u ⊙ v = (uv^T + vu^T)/2"""
    return 0.5 * (np.outer(u, v) + np.outer(v, u))


class FunctionClassType(Enum):
    SMOOTH_CONVEX = auto()              # F_{0,L}
    SMOOTH_STRONGLY_CONVEX = auto()     # F_{μ,L}
    SMOOTH_NONCONVEX = auto()           # F_{-L,L}
    WEAKLY_CONVEX_BOUNDED = auto()      # W_{ρ,L}


@dataclass
class FunctionClass:
    """
    A function class with its interpolation constraints.
    
    The interpolation constraint for a class F says:
    A set {(x_i, g_i, f_i)}_{i∈I} is F-interpolable if and only if
    certain inequalities hold for all pairs (i,j).
    """
    name: str
    class_type: FunctionClassType
    mu: float = 0.0      # strong convexity parameter
    L: float = 1.0       # smoothness parameter
    rho: float = 0.0     # weak convexity parameter
    M: float = float('inf')  # subgradient bound

    def interpolation_constraint(
        self,
        gb,
        xi: np.ndarray, xj: np.ndarray,
        gi: np.ndarray, gj: np.ndarray,
        fi: np.ndarray, fj: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Returns (A_gram, a_fval, constant) such that the interpolation
        constraint is:
            tr(G @ A_gram) + F @ a_fval + constant <= 0
        
        The specific form depends on the function class.
        """
        if self.class_type == FunctionClassType.SMOOTH_CONVEX:
            return self._smooth_convex_constraint(gb, xi, xj, gi, gj, fi, fj)
        elif self.class_type == FunctionClassType.SMOOTH_STRONGLY_CONVEX:
            return self._smooth_strongly_convex_constraint(gb, xi, xj, gi, gj, fi, fj)
        elif self.class_type == FunctionClassType.SMOOTH_NONCONVEX:
            return self._smooth_nonconvex_constraint(gb, xi, xj, gi, gj, fi, fj)
        elif self.class_type == FunctionClassType.WEAKLY_CONVEX_BOUNDED:
            return self._weakly_convex_constraint(gb, xi, xj, gi, gj, fi, fj)
        else:
            raise ValueError(f"Unknown function class: {self.class_type}")

    def _smooth_convex_constraint(self, gb, xi, xj, gi, gj, fi, fj):
        """
        F_{0,L} interpolation (Lemma 2 / Theorem from Taylor et al.):
        
        f_i ≥ f_j + ⟨g_j, x_i - x_j⟩ + (1/2L)||g_i - g_j||^2
        
        Rearranged to ≤ 0 form:
        f_j - f_i + ⟨g_j, x_i - x_j⟩ + (1/2L)||g_i - g_j||^2 ≤ 0
        """
        A_gram = gb.A(xi, xj, gj) + (1.0 / (2.0 * self.L)) * gb.C(gi, gj)
        a_fval = gb.a(fi, fj)  # f_j - f_i
        return A_gram, a_fval, 0.0

    def _smooth_strongly_convex_constraint(self, gb, xi, xj, gi, gj, fi, fj):
        """
        F_{μ,L} interpolation (Theorem 1 from Taylor, Hendrickx, Glineur 2017):
        
        f_i ≥ f_j + ⟨g_j, x_i - x_j⟩ 
              + 1/(2(1-μ/L)) [ (1/L)||g_i - g_j||^2 + μ||x_i - x_j||^2 
                                - (2μ/L)⟨g_j - g_i, x_j - x_i⟩ ]
        
        Rearranged to: (stuff) ≤ 0 form for the constraint.
        """
        if abs(self.mu) < 1e-15:
            return self._smooth_convex_constraint(gb, xi, xj, gi, gj, fi, fj)
        
        inv_factor = 1.0 / (2.0 * (1.0 - self.mu / self.L))
        
        # ⟨g_j, x_i - x_j⟩
        A1 = gb.A(xi, xj, gj)
        # (1/L)||g_i - g_j||^2
        A2 = (1.0 / self.L) * gb.C(gi, gj)
        # μ||x_i - x_j||^2
        A3 = self.mu * gb.B(xi, xj)
        # -(2μ/L)⟨g_j - g_i, x_j - x_i⟩ = (2μ/L)⟨g_i - g_j, x_i - x_j⟩
        # = (2μ/L) * sym_outer(g_i - g_j, x_i - x_j) as trace with G
        A4 = -(2.0 * self.mu / self.L) * _sym_outer(gi - gj, xi - xj)
        
        A_gram = A1 + inv_factor * (A2 + A3 + A4)
        a_fval = gb.a(fi, fj)  # f_j - f_i
        return A_gram, a_fval, 0.0

    def _smooth_nonconvex_constraint(self, gb, xi, xj, gi, gj, fi, fj):
        """
        F_{-L,L} interpolation (Lemma 5 / Drori-Shamir Theorem 7):
        
        f_i ≥ f_j - (L/4)||x_i - x_j||^2 + (1/2)⟨g_i + g_j, x_i - x_j⟩ 
              + (1/(4L))||g_i - g_j||^2
        """
        Atilde = _sym_outer(gi + gj, xi - xj)  # ⟨g_i+g_j, x_i-x_j⟩
        A_gram = (
            0.5 * Atilde
            - (self.L / 4.0) * gb.B(xi, xj)
            + (1.0 / (4.0 * self.L)) * gb.C(gi, gj)
        )
        a_fval = gb.a(fi, fj)
        return A_gram, a_fval, 0.0

    def _weakly_convex_constraint(self, gb, xi, xj, gi, gj, fi, fj):
        """
        W_{ρ,L} necessary conditions:
        
        f_i ≥ f_j + ⟨g_j, x_i - x_j⟩ - (ρ/2)||x_i - x_j||^2
        ||g_i|| ≤ L  (handled separately as bounded subgradient constraint)
        
        Note: No tight interpolation result exists for W_{ρ,L}.
        """
        A_gram = gb.A(xi, xj, gj) - (self.rho / 2.0) * gb.B(xi, xj)
        a_fval = gb.a(fi, fj)
        return A_gram, a_fval, 0.0

    def subgradient_bound_constraint(self, gb, gi: np.ndarray) -> np.ndarray:
        """
        ||g_i||^2 ≤ M^2
        Returns A_gram such that tr(G @ A_gram) ≤ M^2
        """
        return gb.C(gi, gb.gstar())  # ||g_i - 0||^2 = ||g_i||^2

    def optimality_constraint(self, gb, fi: np.ndarray, gi: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """
        For nonconvex: f_* ≤ f_i - (1/(2L))||g_i||^2
        Rearranged: -(f_i) + (1/(2L))||g_i||^2 ≤ 0  (since f_*=0)
        """
        A_gram = (1.0 / (2.0 * self.L)) * gb.C(gi, gb.gstar())
        a_fval = -fi  # -f_i (since f_* = 0)
        return A_gram, a_fval, 0.0


# ── Factory functions ─────────────────────────────────────────────────

def SmoothConvex(L: float = 1.0) -> FunctionClass:
    return FunctionClass("F_{0," + f"{L}" + "}", FunctionClassType.SMOOTH_CONVEX, L=L)

def SmoothStronglyConvex(mu: float, L: float) -> FunctionClass:
    assert 0 <= mu < L, f"Need 0 ≤ μ < L, got μ={mu}, L={L}"
    return FunctionClass(f"F_{{{mu},{L}}}", FunctionClassType.SMOOTH_STRONGLY_CONVEX, mu=mu, L=L)

def SmoothNonconvex(L: float = 1.0) -> FunctionClass:
    return FunctionClass(f"F_{{-{L},{L}}}", FunctionClassType.SMOOTH_NONCONVEX, L=L)

def WeaklyConvexBounded(rho: float, M: float) -> FunctionClass:
    return FunctionClass(f"W_{{{rho},{M}}}", FunctionClassType.WEAKLY_CONVEX_BOUNDED, rho=rho, M=M)
