"""SDP solver wrapper using cvxpy with best available backend."""
from __future__ import annotations
import warnings
import time
from solvers.common import SolverResult, available_solvers, SolverType


def best_sdp_solver() -> str:
    avail = available_solvers()
    if SolverType.MOSEK in avail:
        return 'MOSEK'
    return 'SCS'


class SDPSolver:
    def __init__(self, solver: str | None = None, verbose: bool = False):
        self.solver = solver or best_sdp_solver()
        self.verbose = verbose

    def solve(self, problem) -> SolverResult:
        import cvxpy as cp
        t0 = time.time()
        try:
            problem.solve(solver=self.solver, verbose=self.verbose)
        except cp.SolverError:
            if self.solver != 'SCS':
                warnings.warn(f"{self.solver} failed, falling back to SCS")
                problem.solve(solver='SCS', verbose=self.verbose)
            else:
                raise
        return SolverResult(
            status='optimal' if problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) else problem.status,
            objective=problem.value if problem.value is not None else float('nan'),
            solve_time=time.time() - t0,
            variables={},
        )
