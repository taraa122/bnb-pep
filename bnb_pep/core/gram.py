"""
Gram matrix construction for PEP.

The Gram matrix G = H^T H where H = [x_0 | g_0 | g_1 | ... | g_N].
Size: (N+2) × (N+2). x_* = 0, g_* = 0 by convention.

All PEP quantities reduce to tr(G @ M) for some coefficient matrix M,
plus affine terms in function values.
"""

from __future__ import annotations
import numpy as np


class GramBuilder:
    """Builds selector vectors and coefficient matrices for PEP Gram formulation."""

    def __init__(self, N: int):
        self.N = N
        self.gram_size = N + 2      # [x_0, g_0, g_1, ..., g_N]
        self.fval_size = N + 1      # [f_0, f_1, ..., f_N]; f_*=0

    # ── Selector vectors ─────────────────────────────────────────────

    def e(self, i: int) -> np.ndarray:
        v = np.zeros(self.gram_size); v[i] = 1.0; return v

    def x0(self) -> np.ndarray:
        return self.e(0)

    def xstar(self) -> np.ndarray:
        return np.zeros(self.gram_size)

    def g(self, i: int) -> np.ndarray:
        """g_i selector: column i+1 in Gram."""
        return self.e(i + 1)

    def gstar(self) -> np.ndarray:
        return np.zeros(self.gram_size)

    def f(self, i: int) -> np.ndarray:
        v = np.zeros(self.fval_size); v[i] = 1.0; return v

    def fstar(self) -> np.ndarray:
        return np.zeros(self.fval_size)

    def x_from_alpha(self, alpha: dict[int, float], mu: float = 0.0, L: float = 1.0) -> np.ndarray:
        """
        Iterate selector after F_{μ,L} → F_{0,L-μ} reparametrization.
        x_i = x_0 (1 - μ/L Σα_{i,j}) - (1/L) Σ α_{i,j} g_j
        """
        v = np.zeros(self.gram_size)
        asum = sum(alpha.values())
        v[0] = 1.0 - (mu / L) * asum
        for j, aij in alpha.items():
            v[j + 1] = -aij / L
        return v

    def x_from_h(self, h: dict[int, float], L: float = 1.0) -> np.ndarray:
        """
        Iterate selector using CUMULATIVE step coefficients (s̄ in eq. 27).
        x_i = x_0 - (1/L) Σ_j s̄_{i,j} g_j
        
        h[j] = s̄_{i,j} is the total coefficient of g_j in x_i's expression.
        For GD: s̄_{i,j} = 1 for all j < i.
        """
        v = np.zeros(self.gram_size)
        v[0] = 1.0
        for j, hij in h.items():
            v[j + 1] = -hij / L
        return v

    # ── Coefficient matrices (for tr(G @ M) expressions) ─────────────

    @staticmethod
    def sym_outer(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """u ⊙ v = (uv^T + vu^T)/2"""
        return 0.5 * (np.outer(u, v) + np.outer(v, u))

    def A(self, xi: np.ndarray, xj: np.ndarray, gj: np.ndarray) -> np.ndarray:
        """A_{i,j} = g_j ⊙ (x_i - x_j)  →  ⟨g_j, x_i - x_j⟩ = tr(G A_{i,j})"""
        return self.sym_outer(gj, xi - xj)

    def B(self, xi: np.ndarray, xj: np.ndarray) -> np.ndarray:
        """B_{i,j} = (x_i-x_j) ⊙ (x_i-x_j)  →  ||x_i-x_j||^2 = tr(G B_{i,j})"""
        d = xi - xj
        return self.sym_outer(d, d)

    def C(self, gi: np.ndarray, gj: np.ndarray) -> np.ndarray:
        """C_{i,j} = (g_i-g_j) ⊙ (g_i-g_j)  →  ||g_i-g_j||^2 = tr(G C_{i,j})"""
        d = gi - gj
        return self.sym_outer(d, d)

    def a(self, fi: np.ndarray, fj: np.ndarray) -> np.ndarray:
        """a_{i,j} = f_j - f_i  →  f_j - f_i = F @ a_{i,j}"""
        return fj - fi
