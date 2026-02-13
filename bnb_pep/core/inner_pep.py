"""
Inner PEP: Given a fixed first-order method (fixed stepsizes), compute its
worst-case performance by solving a convex SDP.

This implements §3.1.1 of the BnB-PEP paper:
1. Write the infinite-dimensional worst-case problem
2. Apply interpolation to get finite-dimensional QCQP
3. Introduce Gram matrix to get SDP
4. Solve primal or dual SDP

The dual SDP is:
    minimize  ν R^2
    s.t.      Σ λ_{i,j} a_{i,j} = 0                          (function value)
              ν B_{0,*} - Q_obj + Σ λ_{i,j}[A_{i,j} + ...] = Z   (Gram)
              Z ⪰ 0, ν ≥ 0, λ_{i,j} ≥ 0
"""

from __future__ import annotations
import numpy as np
import cvxpy as cp
from dataclasses import dataclass, field
from typing import Optional

from core.gram import GramBuilder
from function_classes import FunctionClass, FunctionClassType
from core.measures import (
    PerformanceMeasure, InitialCondition,
    build_objective, build_initial_condition,
)


@dataclass
class InnerPEPResult:
    """Result of solving the inner PEP (worst-case analysis of a fixed method)."""
    worst_case_value: float
    status: str
    # Dual variables (needed for outer problem)
    nu: Optional[float] = None
    lam: Optional[np.ndarray] = None
    Z: Optional[np.ndarray] = None
    # Primal variables (worst-case instance)
    G: Optional[np.ndarray] = None
    F: Optional[np.ndarray] = None


class InnerPEP:
    """
    Constructs and solves the inner PEP for a fixed method.
    
    Given:
        - Function class F (with interpolation constraints)
        - Method M defined by stepsizes h or α
        - Performance measure E
        - Initial condition C with parameter R
    
    Computes: max_f E subject to f ∈ F, algorithm, initial condition.
    """
    
    def __init__(
        self,
        N: int,
        func_class: FunctionClass,
        measure: PerformanceMeasure,
        init_cond: InitialCondition,
        R: float = 1.0,
    ):
        self.N = N
        self.func_class = func_class
        self.measure = measure
        self.init_cond = init_cond
        self.R = R
        self.gb = GramBuilder(N)
    
    def _build_iterate_selectors(
        self, stepsizes: np.ndarray, use_h: bool = False
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Build all selector vectors for iterates, gradients, function values.
        
        Args:
            stepsizes: Lower-triangular matrix of step coefficients.
                       stepsizes[i, j] = α_{i+1, j} or h_{i+1, j} for 0 ≤ j < i+1
            use_h: If True, use direct h formulation. If False, use α (reparametrized).
        """
        gb = self.gb
        mu = self.func_class.mu
        L = self.func_class.L
        
        iterates = {'x_0': gb.x0(), 'x_*': gb.xstar()}
        grads = {'g_*': gb.gstar()}
        fvals = {'f_*': gb.fstar()}
        
        for i in range(self.N + 1):
            grads[f'g_{i}'] = gb.g(i)
            fvals[f'f_{i}'] = gb.f(i)
        
        for i in range(1, self.N + 1):
            coeffs = {}
            for j in range(i):
                val = stepsizes[i - 1, j]
                if abs(val) > 1e-15:
                    coeffs[j] = val
            if use_h:
                iterates[f'x_{i}'] = gb.x_from_h(coeffs, L=L)
            else:
                iterates[f'x_{i}'] = gb.x_from_alpha(coeffs, mu=mu, L=L)
        
        return iterates, grads, fvals
    
    def solve_primal(self, stepsizes: np.ndarray, use_h: bool = False,
                     solver: str = 'SCS', verbose: bool = False) -> InnerPEPResult:
        """
        Solve the primal (maximization) SDP.
        
        max  tr(G @ Q_obj) + F @ q_obj
        s.t. interpolation constraints
             initial condition
             G ⪰ 0
        """
        gb = self.gb
        n = gb.gram_size
        m = gb.fval_size
        
        iterates, grads, fvals = self._build_iterate_selectors(stepsizes, use_h)
        
        # Decision variables
        G = cp.Variable((n, n), symmetric=True)
        Fv = cp.Variable(m)
        
        constraints = [G >> 0]  # G ⪰ 0
        
        # Index set I_N^* = {*, 0, 1, ..., N}
        labels = ['*'] + list(range(self.N + 1))
        
        # Interpolation constraints for all pairs (i, j) with i ≠ j
        for i_label in labels:
            for j_label in labels:
                if i_label == j_label:
                    continue
                
                xi_key = f'x_{i_label}' if i_label != '*' else 'x_*'
                xj_key = f'x_{j_label}' if j_label != '*' else 'x_*'
                gi_key = f'g_{i_label}' if i_label != '*' else 'g_*'
                gj_key = f'g_{j_label}' if j_label != '*' else 'g_*'
                fi_key = f'f_{i_label}' if i_label != '*' else 'f_*'
                fj_key = f'f_{j_label}' if j_label != '*' else 'f_*'
                
                A_gram, a_fval, const = self.func_class.interpolation_constraint(
                    gb,
                    iterates[xi_key], iterates[xj_key],
                    grads[gi_key], grads[gj_key],
                    fvals[fi_key], fvals[fj_key],
                )
                constraints.append(
                    cp.trace(G @ A_gram) + Fv @ a_fval + const <= 0
                )
        
        # Additional constraints for nonconvex classes
        if self.func_class.class_type == FunctionClassType.SMOOTH_NONCONVEX:
            for i in range(self.N + 1):
                A_opt, a_opt, c_opt = self.func_class.optimality_constraint(
                    gb, fvals[f'f_{i}'], grads[f'g_{i}']
                )
                constraints.append(
                    cp.trace(G @ A_opt) + Fv @ a_opt + c_opt <= 0
                )
        
        # Bounded subgradient constraints
        if self.func_class.M < float('inf'):
            for i in range(self.N + 1):
                Cgi = self.func_class.subgradient_bound_constraint(gb, grads[f'g_{i}'])
                constraints.append(cp.trace(G @ Cgi) <= self.func_class.M**2)
        
        # Initial condition
        A_ic, a_ic, c_ic = build_initial_condition(
            self.init_cond, gb, self.R, iterates, fvals
        )
        constraints.append(cp.trace(G @ A_ic) + Fv @ a_ic + c_ic <= 0)
        
        # Objective
        Q_obj, q_obj, c_obj = build_objective(
            self.measure, gb, iterates, grads, fvals, self.func_class.mu
        )
        
        if self.measure == PerformanceMeasure.MIN_GRADIENT_NORM:
            # Epigraph: max t s.t. t ≤ ||g_i||^2 for all i
            t = cp.Variable()
            for i in range(self.N + 1):
                Ci_star = gb.C(grads[f'g_{i}'], gb.gstar())
                constraints.append(t <= cp.trace(G @ Ci_star))
            objective = cp.Maximize(t)
        else:
            objective = cp.Maximize(cp.trace(G @ Q_obj) + Fv @ q_obj + c_obj)
        
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=solver, verbose=verbose)
        
        return InnerPEPResult(
            worst_case_value=prob.value if prob.value is not None else float('nan'),
            status=prob.status,
            G=G.value,
            F=Fv.value,
        )
    
    def solve_dual(self, stepsizes: np.ndarray, use_h: bool = False,
                   solver: str = 'SCS', verbose: bool = False) -> InnerPEPResult:
        """
        Solve the dual (minimization) SDP.
        
        min  ν R^2
        s.t. Σ λ_{i,j} a_{i,j} [+ extra terms] = 0     (fval constraint)
             ν B_{0,*} - Q_obj + Σ λ_{i,j} M_{i,j} = Z  (Gram constraint)
             Z ⪰ 0, ν ≥ 0, λ ≥ 0
        
        This gives us the dual variables (ν, λ, Z) needed for the outer BnB-PEP.
        """
        gb = self.gb
        n = gb.gram_size
        m = gb.fval_size
        
        iterates, grads, fvals = self._build_iterate_selectors(stepsizes, use_h)
        
        # Count constraint pairs
        labels = ['*'] + list(range(self.N + 1))
        pairs = [(i, j) for i in labels for j in labels if i != j]
        n_interp = len(pairs)
        
        # Build all interpolation constraint matrices
        A_grams = []
        a_fvals = []
        for (i_label, j_label) in pairs:
            xi_key = f'x_{i_label}' if i_label != '*' else 'x_*'
            xj_key = f'x_{j_label}' if j_label != '*' else 'x_*'
            gi_key = f'g_{i_label}' if i_label != '*' else 'g_*'
            gj_key = f'g_{j_label}' if j_label != '*' else 'g_*'
            fi_key = f'f_{i_label}' if i_label != '*' else 'f_*'
            fj_key = f'f_{j_label}' if j_label != '*' else 'f_*'
            
            Ag, af, _ = self.func_class.interpolation_constraint(
                gb, iterates[xi_key], iterates[xj_key],
                grads[gi_key], grads[gj_key],
                fvals[fi_key], fvals[fj_key],
            )
            A_grams.append(Ag)
            a_fvals.append(af)
        
        # Dual variables
        lam = cp.Variable(n_interp, nonneg=True)
        nu = cp.Variable(nonneg=True)
        Z = cp.Variable((n, n), symmetric=True)
        
        # Additional dual vars for nonconvex/min-gradient
        extra_constraints = []
        extra_gram = np.zeros((n, n))
        extra_fval = np.zeros(m)
        
        if self.func_class.class_type == FunctionClassType.SMOOTH_NONCONVEX:
            tau = cp.Variable(self.N + 1, nonneg=True)
            tau_grams = []
            tau_fvals = []
            for i in range(self.N + 1):
                Ag, af, _ = self.func_class.optimality_constraint(
                    gb, fvals[f'f_{i}'], grads[f'g_{i}']
                )
                tau_grams.append(Ag)
                tau_fvals.append(af)
        
        if self.measure == PerformanceMeasure.MIN_GRADIENT_NORM:
            eta = cp.Variable(self.N + 1, nonneg=True)
            extra_constraints.append(cp.sum(eta) == 1)
        
        # Objective and initial condition
        Q_obj, q_obj, _ = build_objective(
            self.measure, gb, iterates, grads, fvals, self.func_class.mu
        )
        A_ic, a_ic, c_ic = build_initial_condition(
            self.init_cond, gb, self.R, iterates, fvals
        )
        
        # Build dual constraints
        constraints = [Z >> 0] + extra_constraints
        
        # Function value constraint: Σ λ_{i,j} a_{i,j} [+ other terms] = [rhs]
        fval_sum = sum(lam[k] * a_fvals[k] for k in range(n_interp))
        if self.measure == PerformanceMeasure.FUNCTION_VALUE:
            fval_rhs = q_obj  # a_{*,N}
        else:
            fval_rhs = np.zeros(m)
        
        if self.func_class.class_type == FunctionClassType.SMOOTH_NONCONVEX:
            fval_sum = fval_sum + sum(tau[i] * tau_fvals[i] for i in range(self.N + 1))
        
        fval_sum = fval_sum + nu * a_ic  # from initial condition if it has fval terms
        constraints.append(fval_sum == fval_rhs)
        
        # Gram constraint: ν B_{0,*} + Σ λ_{i,j} M_{i,j} [+...] - Q_obj = Z
        gram_expr = nu * A_ic
        for k in range(n_interp):
            gram_expr = gram_expr + lam[k] * A_grams[k]
        
        if self.func_class.class_type == FunctionClassType.SMOOTH_NONCONVEX:
            for i in range(self.N + 1):
                gram_expr = gram_expr + tau[i] * tau_grams[i]
        
        if self.measure == PerformanceMeasure.MIN_GRADIENT_NORM:
            for i in range(self.N + 1):
                Ci_star = gb.C(grads[f'g_{i}'], gb.gstar())
                gram_expr = gram_expr - eta[i] * Ci_star
        else:
            gram_expr = gram_expr - Q_obj
        
        constraints.append(gram_expr == Z)
        
        # Objective
        objective_val = nu * self.R**2
        if self.init_cond == InitialCondition.FUNCTION_VALUE_BOUND:
            # The ν appears differently
            pass  # already handled through A_ic
        
        prob = cp.Problem(cp.Minimize(objective_val), constraints)
        prob.solve(solver=solver, verbose=verbose)
        
        result = InnerPEPResult(
            worst_case_value=prob.value if prob.value is not None else float('nan'),
            status=prob.status,
            nu=nu.value,
            Z=Z.value,
        )
        if lam.value is not None:
            result.lam = lam.value
        
        return result
    
    def solve(self, stepsizes: np.ndarray, use_h: bool = False,
              mode: str = 'primal', solver: str = 'SCS', verbose: bool = False) -> InnerPEPResult:
        """Convenience method to solve either primal or dual."""
        if mode == 'primal':
            return self.solve_primal(stepsizes, use_h, solver, verbose)
        elif mode == 'dual':
            return self.solve_dual(stepsizes, use_h, solver, verbose)
        else:
            raise ValueError(f"mode must be 'primal' or 'dual', got '{mode}'")
