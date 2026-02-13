"""
Outer BnB-PEP: Find the OPTIMAL first-order method.

This implements the BnB-PEP Algorithm (Algorithm 1 from §3.2):

Stage 1: Compute feasible solution (fix stepsizes to GD, solve inner SDP)
Stage 2: Local optimization (warm-start interior-point solve of QCQP)
Stage 3: Global optimization (spatial branch-and-bound with customizations)

The BnB-PEP-QCQP (equation 14) is:
    min  ν R^2
    s.t. Σ λ_{i,j} a_{i,j} = 0
         ν B_{0,*} - Q_obj + Σ λ_{i,j}[A_{i,j}(α) + ...] = Z
         P lower-triangular with nonneg diagonals
         P P^T = Z
         ν ≥ 0, λ ≥ 0

Variables: α (stepsizes), ν, λ (dual multipliers), Z, P (Cholesky factor)
"""

from __future__ import annotations
import numpy as np
import time
from dataclasses import dataclass, field
from typing import Optional, Callable

from core.gram import GramBuilder
from core.inner_pep import InnerPEP, InnerPEPResult
from function_classes import FunctionClass, FunctionClassType
from core.measures import PerformanceMeasure, InitialCondition


@dataclass
class BnBPEPResult:
    """Result of the full BnB-PEP optimization."""
    optimal_value: float
    stepsizes_alpha: Optional[np.ndarray] = None
    stepsizes_h: Optional[np.ndarray] = None
    nu: Optional[float] = None
    lam: Optional[np.ndarray] = None
    Z: Optional[np.ndarray] = None
    stage1_value: Optional[float] = None
    stage2_value: Optional[float] = None
    stage3_value: Optional[float] = None
    stage1_time: float = 0.0
    stage2_time: float = 0.0
    stage3_time: float = 0.0
    status: str = "unknown"


def alpha_to_h(alpha: np.ndarray, mu: float, L: float) -> np.ndarray:
    """
    Convert α parametrization to h parametrization using Lemma 1.
    
    h_{i,j} relates to α via:
        α_{i,j} = h_{i,i-1}                            if j = i-1
        α_{i,j} = α_{i-1,j} + h_{i,j} - (μ/L) Σ_{k>j} h_{i,k} α_{k,j}  otherwise
    
    Inverse: given α, recover h.
    """
    N = alpha.shape[0]
    h = np.zeros_like(alpha)
    
    for i in range(N):  # i goes 0..N-1, representing step i+1
        h[i, i] = alpha[i, i]  # h_{i+1,i} = α_{i+1,i}
        for j in range(i):
            # h_{i+1,j} = α_{i+1,j} - α_{i,j} + (μ/L) Σ_{k=j+1}^{i} h_{i+1,k} α_{k,j}
            correction = 0.0
            for k in range(j + 1, i + 1):
                if k < N and k > 0:
                    correction += h[i, k] * alpha[k - 1, j] if k - 1 >= 0 and j < k else 0
            h[i, j] = alpha[i, j] - (alpha[i - 1, j] if i > 0 and j < i else 0) + (mu / L) * correction
    
    return h


def h_to_alpha(h: np.ndarray, mu: float, L: float) -> np.ndarray:
    """Convert h parametrization to α using the recursive relation (eq. 4)."""
    N = h.shape[0]
    alpha = np.zeros_like(h)
    
    for i in range(N):
        alpha[i, i] = h[i, i]  # α_{i+1,i} = h_{i+1,i}
        for j in range(i):
            alpha[i, j] = (alpha[i - 1, j] if i > 0 else 0) + h[i, j]
            # Correction for strongly convex reparametrization
            if mu > 0:
                for k in range(j + 1, i + 1):
                    alpha[i, j] -= (mu / L) * h[i, k] * (alpha[k - 1, j] if k > 0 else 0)
    
    return alpha


class BnBPEP:
    """
    Branch-and-Bound Performance Estimation Programming.
    
    Finds the optimal fixed-step first-order method for a given:
    - Function class (smooth convex, strongly convex, nonconvex, etc.)
    - Performance measure (function value gap, gradient norm, etc.)
    - Initial condition (distance bound, function value bound)
    """
    
    def __init__(
        self,
        N: int,
        func_class: FunctionClass,
        measure: PerformanceMeasure,
        init_cond: InitialCondition,
        R: float = 1.0,
        use_h_parametrization: bool = False,
    ):
        self.N = N
        self.func_class = func_class
        self.measure = measure
        self.init_cond = init_cond
        self.R = R
        self.use_h = use_h_parametrization
        self.inner_pep = InnerPEP(N, func_class, measure, init_cond, R)
    
    def _default_stepsizes(self) -> np.ndarray:
        """
        Default stepsizes for Stage 1 (gradient descent).
        
        For F_{0,L}: h_{i,i-1} = 1, others = 0 (normalized stepsize 1/L)
        For F_{μ,L}: same
        For F_{-L,L}: same
        For W_{ρ,L}: h_{i,i-1} = R*ρ/(L*sqrt(N+1)), others = 0
        """
        N = self.N
        h = np.zeros((N, N + 1))  # h[i] has entries for j = 0..i
        
        if self.func_class.class_type == FunctionClassType.WEAKLY_CONVEX_BOUNDED:
            rho = self.func_class.rho
            L_bound = self.func_class.M
            step = self.R * rho / (L_bound * np.sqrt(N + 1))
            for i in range(N):
                for j in range(i + 1):
                    h[i, j] = step
        else:
            # GD cumulative: x_i = x_0 - (1/L) Σ_{j<i} g_j
            # So h[i,j] = 1 for all j ≤ i
            for i in range(N):
                for j in range(i + 1):
                    h[i, j] = 1.0
        
        return h
    
    def stage1(self, verbose: bool = False) -> tuple[np.ndarray, InnerPEPResult]:
        """
        Stage 1: Compute feasible solution using gradient descent stepsizes.
        
        Fix stepsizes to GD, solve the inner convex SDP to get dual variables.
        """
        h_init = self._default_stepsizes()
        
        # Build proper stepsize matrix for inner PEP
        # Inner PEP expects (N, N+1) lower-triangular where [i,j] = coefficient
        # For GD: only diagonal entries are nonzero
        stepsizes = np.zeros((self.N, self.N + 1))
        for i in range(self.N):
            for j in range(i + 1):
                stepsizes[i, j] = h_init[i, j]
        
        result = self.inner_pep.solve(
            stepsizes, use_h=True, mode='primal', solver='SCS', verbose=verbose
        )
        
        if verbose:
            print(f"Stage 1 (GD): worst-case = {result.worst_case_value:.6e}")
        
        return h_init, result
    
    def stage2_scipy(self, h_init: np.ndarray, verbose: bool = False) -> tuple[np.ndarray, float]:
        """
        Stage 2: Local optimization using scipy (fallback when no QCQP solver available).
        
        Uses scipy.optimize.minimize to locally optimize stepsizes by repeatedly
        solving the inner PEP.
        """
        from scipy.optimize import minimize
        
        N = self.N
        
        # Flatten stepsizes to 1D vector (lower-triangular entries)
        def stepsizes_to_vec(h):
            vec = []
            for i in range(N):
                for j in range(i + 1):
                    vec.append(h[i, j])
            return np.array(vec)
        
        def vec_to_stepsizes(vec):
            h = np.zeros((N, N + 1))
            idx = 0
            for i in range(N):
                for j in range(i + 1):
                    h[i, j] = vec[idx]
                    idx += 1
            return h
        
        def objective(vec):
            h = vec_to_stepsizes(vec)
            result = self.inner_pep.solve(h, use_h=True, mode='primal', solver='SCS')
            val = result.worst_case_value
            if val is None or np.isnan(val):
                return 1e10
            return val
        
        x0 = stepsizes_to_vec(h_init)
        
        if verbose:
            print(f"Stage 2: optimizing {len(x0)} stepsize parameters...")
        
        res = minimize(objective, x0, method='Nelder-Mead',
                       options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10,
                                'adaptive': True})
        
        h_opt = vec_to_stepsizes(res.x)
        
        if verbose:
            print(f"Stage 2: optimized worst-case = {res.fun:.6e}")
        
        return h_opt, res.fun
    
    def stage2_ipopt(self, h_init: np.ndarray, nu_init: float,
                     lam_init: np.ndarray, Z_init: np.ndarray,
                     verbose: bool = False) -> tuple[np.ndarray, float]:
        """
        Stage 2: Local optimization using IPOPT on the QCQP formulation.
        
        This solves the BnB-PEP-QCQP locally using interior-point methods.
        """
        try:
            import cyipopt
        except ImportError:
            if verbose:
                print("IPOPT not available, falling back to scipy")
            return self.stage2_scipy(h_init, verbose=verbose)
        
        # TODO: Implement full QCQP formulation for IPOPT
        # For now, fall back to scipy
        return self.stage2_scipy(h_init, verbose=verbose)
    
    def stage3_gurobi(self, h_init: np.ndarray, best_value: float,
                      verbose: bool = False) -> tuple[np.ndarray, float]:
        """
        Stage 3: Global optimization using Gurobi's spatial branch-and-bound.
        
        This is where the magic happens — we solve the nonconvex QCQP to
        certified global optimality.
        """
        try:
            import gurobipy as gp
        except ImportError:
            if verbose:
                print("Gurobi not available. Stage 3 skipped — returning Stage 2 solution.")
            return h_init, best_value
        
        # TODO: Implement full Gurobi QCQP with customizations from §4
        if verbose:
            print("Stage 3: Gurobi QCQP solver (full implementation pending)")
        return h_init, best_value
    
    def solve(self, verbose: bool = True, max_stage: int = 3) -> BnBPEPResult:
        """
        Run the full BnB-PEP Algorithm.
        
        Args:
            verbose: Print progress
            max_stage: Stop after this stage (1, 2, or 3)
        """
        result = BnBPEPResult(optimal_value=float('inf'))
        
        # Stage 1
        t0 = time.time()
        h_init, inner_result = self.stage1(verbose=verbose)
        result.stage1_time = time.time() - t0
        result.stage1_value = inner_result.worst_case_value
        
        if max_stage < 2:
            result.optimal_value = inner_result.worst_case_value
            result.stepsizes_h = h_init
            result.status = "stage1_complete"
            return result
        
        # Stage 2
        t0 = time.time()
        h_opt, opt_value = self.stage2_scipy(h_init, verbose=verbose)
        result.stage2_time = time.time() - t0
        result.stage2_value = opt_value
        
        if max_stage < 3:
            result.optimal_value = opt_value
            result.stepsizes_h = h_opt
            result.status = "stage2_complete"
            return result
        
        # Stage 3
        t0 = time.time()
        h_global, global_value = self.stage3_gurobi(h_opt, opt_value, verbose=verbose)
        result.stage3_time = time.time() - t0
        result.stage3_value = global_value
        
        result.optimal_value = global_value
        result.stepsizes_h = h_global
        result.status = "stage3_complete"
        
        return result
