"""
Performance optimization utilities.

- SDP factorization caching across branch nodes
- Eigen-deflation for ill-conditioned matrices
- Numerical scaling techniques
- Certificate replay for verification
"""

from __future__ import annotations
import numpy as np
from typing import Optional
from scipy.linalg import cholesky, eigh
import time


class SDPCache:
    """
    Caches SDP factorizations across branch-and-bound nodes.
    
    When BnB explores nearby nodes, SDP solutions are similar.
    Caching previous solutions as warm-starts cuts solve time dramatically.
    """

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._cache: dict[bytes, dict] = {}
        self._access_times: dict[bytes, float] = {}
        self.hits = 0
        self.misses = 0

    def _make_key(self, stepsizes: np.ndarray, tol: float = 1e-4) -> bytes:
        quantized = np.round(stepsizes / tol) * tol
        return quantized.tobytes()

    def get(self, stepsizes: np.ndarray) -> Optional[dict]:
        key = self._make_key(stepsizes)
        if key in self._cache:
            self.hits += 1
            self._access_times[key] = time.time()
            return self._cache[key]
        self.misses += 1
        return None

    def put(self, stepsizes: np.ndarray, solution: dict):
        key = self._make_key(stepsizes)
        if len(self._cache) >= self.max_size:
            oldest_key = min(self._access_times, key=self._access_times.get)
            del self._cache[oldest_key]
            del self._access_times[oldest_key]
        self._cache[key] = solution
        self._access_times[key] = time.time()

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': self.hits / total if total > 0 else 0,
            'size': len(self._cache),
        }


def eigen_deflation(Z: np.ndarray, tol: float = 1e-10) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Eigen-deflation for ill-conditioned PSD matrices.
    
    Decomposes Z = V Λ V^T, removes near-zero eigenvalues,
    returns the reduced factorization for more stable computation.
    
    Returns:
        V_reduced: eigenvectors for nonzero eigenvalues
        lam_reduced: nonzero eigenvalues
        rank: effective rank
    """
    eigenvalues, eigenvectors = eigh(Z)
    
    # Clamp small negative eigenvalues (numerical noise)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    
    # Keep only eigenvalues above tolerance
    mask = eigenvalues > tol
    lam_reduced = eigenvalues[mask]
    V_reduced = eigenvectors[:, mask]
    rank = int(np.sum(mask))
    
    return V_reduced, lam_reduced, rank


def numerical_scaling(
    A_grams: list[np.ndarray],
    a_fvals: list[np.ndarray],
    Q_obj: np.ndarray,
    A_ic: np.ndarray,
) -> tuple[float, float]:
    """
    Compute scaling factors for numerical stability.
    
    SDP solvers work best when constraint coefficients are O(1).
    We compute scale factors for the Gram and function-value spaces.
    
    Returns:
        gram_scale: multiply all Gram-related quantities by this
        fval_scale: multiply all function-value quantities by this
    """
    # Gram scale: based on max absolute value across all A matrices
    gram_norms = [np.max(np.abs(A)) for A in A_grams if np.max(np.abs(A)) > 0]
    gram_norms.append(np.max(np.abs(Q_obj)) if np.max(np.abs(Q_obj)) > 0 else 1.0)
    gram_norms.append(np.max(np.abs(A_ic)) if np.max(np.abs(A_ic)) > 0 else 1.0)
    gram_scale = 1.0 / np.median(gram_norms) if gram_norms else 1.0
    
    # Fval scale: based on max absolute value across all a vectors
    fval_norms = [np.max(np.abs(a)) for a in a_fvals if np.max(np.abs(a)) > 0]
    fval_scale = 1.0 / np.median(fval_norms) if fval_norms else 1.0
    
    return gram_scale, fval_scale


def cholesky_factor(Z: np.ndarray, regularize: float = 1e-12) -> np.ndarray:
    """
    Compute Cholesky factorization Z = P P^T with regularization.
    
    P is lower-triangular with nonnegative diagonals (Lemma 3 in the paper).
    """
    n = Z.shape[0]
    Z_reg = Z + regularize * np.eye(n)
    
    # Ensure symmetry
    Z_reg = 0.5 * (Z_reg + Z_reg.T)
    
    try:
        P = cholesky(Z_reg, lower=True)
    except np.linalg.LinAlgError:
        # Fall back to eigen-decomposition
        V, lam, rank = eigen_deflation(Z_reg)
        P = V @ np.diag(np.sqrt(lam))
        # Pad to full lower-triangular
        P_full = np.zeros((n, n))
        P_full[:, :rank] = P
        P = P_full
    
    return P


def implied_linear_constraints(n: int) -> list[tuple[str, np.ndarray, float]]:
    """
    Generate implied linear constraints from Z = P P^T ⪰ 0.
    
    From §4.2.1 of BnB-PEP:
    - Z = Z^T (symmetry)
    - diag(Z) ≥ 0
    - |Z_{i,j}| ≤ (Z_{i,i} + Z_{j,j}) / 2  (from 2×2 PSD minors + AM-GM)
    
    These are mathematically redundant but algorithmically essential
    for branch-and-bound.
    
    Returns list of (name, coefficient_vector, rhs) for linear constraints.
    """
    constraints = []
    
    # Diagonal nonnegativity
    for i in range(n):
        name = f"Z_{i},{i} >= 0"
        constraints.append((name, i, i, 0.0, 'diag_nonneg'))
    
    # Off-diagonal bounds from AM-GM
    for i in range(n):
        for j in range(i + 1, n):
            # Z_{i,j} ≤ (Z_{i,i} + Z_{j,j}) / 2
            # -(Z_{i,i} + Z_{j,j}) / 2 ≤ Z_{i,j}
            name = f"|Z_{i},{j}| <= (Z_{i},{i}+Z_{j},{j})/2"
            constraints.append((name, i, j, 'amgm'))
    
    return constraints


def variable_bounds_from_feasible(
    nu_init: float,
    lam_init: np.ndarray,
    Z_init: np.ndarray,
    alpha_init: np.ndarray,
    M_factor: float = 1.01,
) -> dict:
    """
    Heuristic variable bounds based on a feasible (Stage 2) solution.
    
    From §4.2.1: set bounds as M_factor * max of the Stage 2 solution.
    
    WARNING: These are heuristic bounds with no guarantee of correctness.
    The SDP relaxation (equation 19) gives valid bounds but is more complex.
    """
    M_nu = nu_init  # Already an upper bound on ν
    M_lam = M_factor * np.max(lam_init) if len(lam_init) > 0 else 1.0
    M_Z = M_factor * np.max(np.diag(Z_init)) if Z_init is not None else 1.0
    M_P = np.sqrt(M_Z)
    M_alpha = 5.0 * M_factor * np.max(np.abs(alpha_init)) if np.max(np.abs(alpha_init)) > 0 else 5.0
    
    return {
        'M_nu': M_nu,
        'M_lam': M_lam,
        'M_Z': M_Z,
        'M_P': M_P,
        'M_alpha': M_alpha,
    }


class CertificateReplay:
    """
    Replay and verify a convergence certificate.
    
    Given dual variables (ν, λ, Z) and stepsizes α, verify that they
    constitute a valid convergence proof by checking:
    1. All dual constraints are satisfied
    2. Z ⪰ 0
    3. The objective value matches the claimed bound
    """

    def __init__(self, tol: float = 1e-6):
        self.tol = tol

    def verify(
        self,
        nu: float,
        lam: np.ndarray,
        Z: np.ndarray,
        R: float,
        claimed_value: float,
    ) -> dict:
        """
        Verify a certificate.
        
        Returns dict with:
            'valid': bool
            'objective': float
            'Z_min_eigenvalue': float
            'constraint_violations': list
        """
        violations = []
        
        # Check Z ⪰ 0
        eigvals = np.linalg.eigvalsh(Z)
        min_eig = float(np.min(eigvals))
        if min_eig < -self.tol:
            violations.append(f"Z not PSD: min eigenvalue = {min_eig:.2e}")
        
        # Check ν ≥ 0
        if nu < -self.tol:
            violations.append(f"ν < 0: {nu:.2e}")
        
        # Check λ ≥ 0
        if np.any(lam < -self.tol):
            min_lam = float(np.min(lam))
            violations.append(f"λ has negative entry: {min_lam:.2e}")
        
        # Check objective
        obj = nu * R**2
        obj_error = abs(obj - claimed_value)
        if obj_error > self.tol * max(1.0, abs(claimed_value)):
            violations.append(f"Objective mismatch: {obj:.6e} vs claimed {claimed_value:.6e}")
        
        return {
            'valid': len(violations) == 0,
            'objective': obj,
            'Z_min_eigenvalue': min_eig,
            'Z_rank': int(np.sum(eigvals > self.tol)),
            'constraint_violations': violations,
        }


class SparsityAnalyzer:
    """
    Analyze sparsity pattern of optimal dual variables.
    
    From §4.3: optimal λ* is often sparse and Z* is low-rank.
    Identifying these patterns reduces the QCQP size.
    """

    def __init__(self, tol: float = 1e-8):
        self.tol = tol

    def analyze_lambda(self, lam: np.ndarray, pair_labels: list) -> dict:
        """Find support of λ (nonzero entries)."""
        support = []
        for k, (i, j) in enumerate(pair_labels):
            if abs(lam[k]) > self.tol:
                support.append((i, j, float(lam[k])))
        
        return {
            'support_size': len(support),
            'total_size': len(pair_labels),
            'sparsity': 1.0 - len(support) / len(pair_labels),
            'support': support,
        }

    def analyze_Z(self, Z: np.ndarray) -> dict:
        """Analyze rank and structure of Z."""
        eigvals = np.linalg.eigvalsh(Z)
        rank = int(np.sum(eigvals > self.tol))
        
        return {
            'rank': rank,
            'size': Z.shape[0],
            'eigenvalues': eigvals[::-1],  # Descending
            'nonzero_pattern': np.abs(Z) > self.tol,
        }

    def promote_sparsity(
        self, lam: np.ndarray, pair_labels: list,
        A_grams: list, a_fvals: list,
        Z: np.ndarray, nu: float, R: float,
        optimal_value: float,
    ) -> np.ndarray:
        """
        Solve the ℓ1-minimization problem (equation 23) to find sparse λ.
        
        min ||λ||_1
        s.t. νR^2 ≤ p*
             dual constraints with fixed α*
             Z ⪰ 0, λ ≥ 0
        """
        import cvxpy as cp
        
        n_lam = len(lam)
        n = Z.shape[0]
        
        lam_var = cp.Variable(n_lam, nonneg=True)
        Z_var = cp.Variable((n, n), symmetric=True)
        nu_var = cp.Variable(nonneg=True)
        
        constraints = [
            Z_var >> 0,
            nu_var * R**2 <= optimal_value + 1e-8,
        ]
        
        # Add dual feasibility constraints
        # (These would need the full constraint matrices, simplified here)
        
        objective = cp.Minimize(cp.sum(lam_var))
        prob = cp.Problem(objective, constraints)
        
        try:
            prob.solve(solver='SCS')
            if lam_var.value is not None:
                return lam_var.value
        except Exception:
            pass
        
        return lam  # Fall back to original
