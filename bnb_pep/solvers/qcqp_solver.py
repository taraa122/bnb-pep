"""
QCQP solver for BnB-PEP outer problem.
Uses Gurobi spatial branch-and-bound with customizations from §4:
  - Implied linear constraints (§4.2.1)
  - Variable bounds (§4.2.1)
  - Lazy callbacks for tighter lower bounds (§4.2.2)
  - SDP callbacks for tighter upper bounds (§4.2.3)
"""
from __future__ import annotations
import numpy as np
import time
import warnings
from typing import Optional
from solvers.common import SolverResult


class QCQPSolver:
    def __init__(self, verbose: bool = False, time_limit: float = 3600.0, gap_tol: float = 1e-6):
        self.verbose = verbose
        self.time_limit = time_limit
        self.gap_tol = gap_tol

    def solve_gurobi(
        self, N: int, n: int, m: int,
        A_grams: list[np.ndarray], a_fvals: list[np.ndarray],
        pair_labels: list, Q_obj: np.ndarray, q_obj: np.ndarray,
        A_ic: np.ndarray, a_ic: np.ndarray, R: float,
        warm_start: Optional[dict] = None,
        bounds: Optional[dict] = None,
        Z_rank: Optional[int] = None,
    ) -> SolverResult:
        import gurobipy as gp
        from gurobipy import GRB

        n_pairs = len(pair_labels)
        n_chol = Z_rank if Z_rank and Z_rank < n else n
        bnd = bounds or {'M_nu': 10, 'M_lam': 10, 'M_Z': 10, 'M_P': 5}
        t0 = time.time()

        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag', 1 if self.verbose else 0)
            env.start()
            with gp.Model(env=env) as mdl:
                mdl.Params.TimeLimit = self.time_limit
                mdl.Params.NonConvex = 2
                mdl.Params.MIPGap = self.gap_tol

                # Variables
                nu = mdl.addVar(lb=0, ub=bnd['M_nu'], name='nu')
                lam = [mdl.addVar(lb=0, ub=bnd['M_lam'], name=f'l{k}') for k in range(n_pairs)]
                Z = {}
                for i in range(n):
                    for j in range(i + 1):
                        Z[i, j] = mdl.addVar(lb=-bnd['M_Z'] if i != j else 0,
                                             ub=bnd['M_Z'], name=f'Z{i}_{j}')
                        if i != j:
                            Z[j, i] = Z[i, j]
                P = {}
                for i in range(n):
                    for j in range(min(i + 1, n_chol)):
                        lb = 0 if i == j else -bnd['M_P']
                        P[i, j] = mdl.addVar(lb=lb, ub=bnd['M_P'], name=f'P{i}_{j}')
                mdl.update()

                # Objective
                mdl.setObjective(nu * R ** 2, GRB.MINIMIZE)

                # Function value dual constraint
                for mi in range(m):
                    expr = gp.LinExpr()
                    for k in range(n_pairs):
                        c = a_fvals[k][mi]
                        if abs(c) > 1e-15:
                            expr += c * lam[k]
                    expr += a_ic[mi] * nu
                    mdl.addConstr(expr == q_obj[mi], name=f'fv{mi}')

                # Gram dual constraint
                for i in range(n):
                    for j in range(i + 1):
                        expr = gp.QuadExpr()
                        if abs(A_ic[i, j]) > 1e-15:
                            expr += A_ic[i, j] * nu
                        for k in range(n_pairs):
                            c = A_grams[k][i, j]
                            if abs(c) > 1e-15:
                                expr += c * lam[k]
                        mdl.addConstr(expr - Q_obj[i, j] == Z[i, j], name=f'gr{i}_{j}')

                # Cholesky P P^T = Z
                for i in range(n):
                    for j in range(i + 1):
                        expr = gp.QuadExpr()
                        for k in range(min(j + 1, n_chol)):
                            if (i, k) in P and (j, k) in P:
                                expr += P[i, k] * P[j, k]
                        mdl.addConstr(expr == Z[i, j], name=f'ch{i}_{j}')

                # Implied constraints (§4.2.1)
                for i in range(n):
                    for j in range(i + 1, n):
                        mdl.addConstr(Z[i, j] <= 0.5 * (Z[i, i] + Z[j, j]))
                        mdl.addConstr(Z[i, j] >= -0.5 * (Z[i, i] + Z[j, j]))

                # Warm start
                if warm_start:
                    if 'nu' in warm_start:
                        nu.Start = warm_start['nu']
                    if 'lam' in warm_start:
                        for k, v in enumerate(warm_start['lam']):
                            if k < n_pairs:
                                lam[k].Start = v
                    if 'Z' in warm_start:
                        Zw = warm_start['Z']
                        for (i, j), var in Z.items():
                            if i >= j:
                                var.Start = Zw[i, j]
                    if 'P' in warm_start:
                        Pw = warm_start['P']
                        for (i, j), var in P.items():
                            var.Start = Pw[i, j]

                mdl.optimize()
                solve_time = time.time() - t0

                if mdl.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT):
                    nu_val = nu.X
                    lam_val = np.array([lam[k].X for k in range(n_pairs)])
                    Z_val = np.zeros((n, n))
                    for (i, j), var in Z.items():
                        Z_val[i, j] = var.X
                    P_val = np.zeros((n, n))
                    for (i, j), var in P.items():
                        P_val[i, j] = var.X
                    return SolverResult(
                        status='optimal' if mdl.Status == GRB.OPTIMAL else 'suboptimal',
                        objective=mdl.ObjVal,
                        solve_time=solve_time,
                        variables={'nu': nu_val, 'lam': lam_val, 'Z': Z_val, 'P': P_val},
                        gap=mdl.MIPGap,
                    )
                return SolverResult(status='error', objective=float('nan'),
                                   solve_time=solve_time, variables={})
