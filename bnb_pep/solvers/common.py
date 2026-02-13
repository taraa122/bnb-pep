from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Any
from enum import Enum, auto
import numpy as np


class SolverType(Enum):
    MOSEK = auto()
    GUROBI = auto()
    IPOPT = auto()
    SDPA = auto()
    SCS = auto()
    SCIPY = auto()


@dataclass
class SolverResult:
    status: str
    objective: float
    solve_time: float
    variables: dict[str, Any]
    dual_variables: Optional[dict[str, Any]] = None
    gap: Optional[float] = None
    iterations: int = 0


def available_solvers() -> list[SolverType]:
    available = [SolverType.SCS, SolverType.SCIPY]
    try:
        import mosek; available.append(SolverType.MOSEK)
    except ImportError:
        pass
    try:
        import gurobipy; available.append(SolverType.GUROBI)
    except ImportError:
        pass
    try:
        import cyipopt; available.append(SolverType.IPOPT)
    except ImportError:
        pass
    return available
