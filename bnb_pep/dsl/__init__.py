"""
Domain-Specific Language for specifying PEP problems.

Allows writing problems in a natural mathematical style:

    problem = PEP()
    f = problem.declare_function(SmoothStronglyConvex(mu=0.1, L=1.0))
    x0 = problem.starting_point()
    xs, fs = f.optimal_point()
    
    problem.initial_condition(f.value(x0) - fs <= 1)
    
    # Algorithm
    x = x0
    for i in range(N):
        x = x - gamma[i] * f.gradient(x)
    
    problem.performance_metric(f.gradient(x) ** 2)
    problem.solve()
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Union
from dataclasses import dataclass, field

from function_classes import (
    FunctionClass, SmoothConvex, SmoothStronglyConvex, 
    SmoothNonconvex, WeaklyConvexBounded,
)
from core.inner_pep import InnerPEP
from core.outer_bnb import BnBPEP, BnBPEPResult
from core.measures import PerformanceMeasure, InitialCondition


class DSLPoint:
    """A point in the DSL."""
    def __init__(self, name: str, index: int):
        self.name = name
        self.index = index
    def __repr__(self):
        return f"Point({self.name})"


class DSLGradient:
    """A gradient evaluation in the DSL."""
    def __init__(self, name: str, at_point: DSLPoint):
        self.name = name
        self.at_point = at_point
    
    def __pow__(self, exp):
        if exp == 2:
            return DSLGradNormSq(self)
        raise ValueError("Only **2 supported for gradients")


class DSLFunctionValue:
    """A function value in the DSL."""
    def __init__(self, name: str, at_point: DSLPoint, value_index: int):
        self.name = name
        self.at_point = at_point
        self.value_index = value_index
    
    def __sub__(self, other):
        return DSLFunctionGap(self, other)
    
    def __le__(self, bound):
        return DSLCondition(self, bound)


class DSLFunctionGap:
    """f(x) - f(y)"""
    def __init__(self, left: DSLFunctionValue, right: DSLFunctionValue):
        self.left = left
        self.right = right
    
    def __le__(self, bound):
        return DSLCondition(self, bound)


class DSLGradNormSq:
    """||∇f(x)||^2"""
    def __init__(self, grad: DSLGradient):
        self.grad = grad


class DSLDistanceSq:
    """||x - y||^2"""
    def __init__(self, p1: DSLPoint, p2: DSLPoint):
        self.p1 = p1
        self.p2 = p2
    
    def __le__(self, bound):
        return DSLCondition(self, bound)


class DSLCondition:
    """A constraint expression <= bound."""
    def __init__(self, expr, bound: float):
        self.expr = expr
        self.bound = bound


class DSLFunction:
    """A function in the DSL that can be queried for gradients and values."""
    
    def __init__(self, func_class: FunctionClass, problem: 'PEP'):
        self.func_class = func_class
        self.problem = problem
        self._grad_count = 0
        self._value_count = 0
        self._optimal_point = None
    
    def gradient(self, point: DSLPoint) -> DSLGradient:
        g = DSLGradient(f"g_{self._grad_count}", point)
        self._grad_count += 1
        return g
    
    def value(self, point: DSLPoint) -> DSLFunctionValue:
        fv = DSLFunctionValue(f"f_{self._value_count}", point, self._value_count)
        self._value_count += 1
        return fv
    
    def optimal_point(self) -> tuple[DSLPoint, DSLFunctionValue]:
        xs = DSLPoint("x_*", -1)
        fs = DSLFunctionValue("f_*", xs, -1)
        self._optimal_point = (xs, fs)
        return xs, fs


class PEP:
    """
    Top-level PEP problem specification using the DSL.
    
    Example:
        pep = PEP()
        f = pep.declare_function(SmoothStronglyConvex(mu=0.1, L=1.0))
        x0 = pep.starting_point()
        xs, fs = f.optimal_point()
        pep.initial_condition(f.value(x0) - fs <= 1)
        
        x = x0
        for i in range(N):
            x = x - gamma[i] * f.gradient(x)
        xN = x
        
        pep.performance_metric(f.gradient(xN) ** 2)
        result = pep.solve()
    """
    
    def __init__(self):
        self._function = None
        self._starting_point = None
        self._init_condition = None
        self._performance_metric = None
        self._algorithm_steps = []
        self._point_count = 0
    
    def declare_function(self, func_class: FunctionClass) -> DSLFunction:
        self._function = DSLFunction(func_class, self)
        return self._function
    
    def starting_point(self) -> DSLPoint:
        p = DSLPoint("x_0", self._point_count)
        self._point_count += 1
        self._starting_point = p
        return p
    
    def initial_condition(self, condition: DSLCondition):
        self._init_condition = condition
    
    def performance_metric(self, metric: Union[DSLGradNormSq, DSLFunctionGap]):
        self._performance_metric = metric
    
    def _infer_measure(self) -> PerformanceMeasure:
        if isinstance(self._performance_metric, DSLGradNormSq):
            return PerformanceMeasure.GRADIENT_NORM
        elif isinstance(self._performance_metric, DSLFunctionGap):
            return PerformanceMeasure.FUNCTION_VALUE
        else:
            raise ValueError(f"Cannot infer measure from {type(self._performance_metric)}")
    
    def _infer_init_condition(self) -> tuple[InitialCondition, float]:
        cond = self._init_condition
        if isinstance(cond.expr, DSLDistanceSq):
            return InitialCondition.DISTANCE_BOUND, cond.bound
        elif isinstance(cond.expr, DSLFunctionGap):
            return InitialCondition.FUNCTION_VALUE_BOUND, cond.bound
        elif isinstance(cond.expr, DSLFunctionValue):
            return InitialCondition.FUNCTION_VALUE_BOUND, cond.bound
        else:
            raise ValueError(f"Cannot infer initial condition from {type(cond.expr)}")


def distance_sq(p1: DSLPoint, p2: DSLPoint) -> DSLDistanceSq:
    """||p1 - p2||^2"""
    return DSLDistanceSq(p1, p2)


def analyze_method(
    N: int,
    func_class: FunctionClass,
    stepsizes: np.ndarray,
    measure: PerformanceMeasure = PerformanceMeasure.GRADIENT_NORM,
    init_cond: InitialCondition = InitialCondition.DISTANCE_BOUND,
    R: float = 1.0,
    solver: str = 'SCS',
    verbose: bool = False,
) -> float:
    """
    Analyze the worst-case performance of a given method.
    
    Quick-use function: provide stepsizes, get worst-case bound.
    """
    pep = InnerPEP(N, func_class, measure, init_cond, R)
    result = pep.solve(stepsizes, use_h=True, mode='primal', solver=solver, verbose=verbose)
    return result.worst_case_value


def find_optimal_method(
    N: int,
    func_class: FunctionClass,
    measure: PerformanceMeasure = PerformanceMeasure.GRADIENT_NORM,
    init_cond: InitialCondition = InitialCondition.DISTANCE_BOUND,
    R: float = 1.0,
    max_stage: int = 2,
    verbose: bool = True,
) -> BnBPEPResult:
    """
    Find the optimal N-step first-order method.
    
    Quick-use function: specify the setup, get optimal stepsizes.
    """
    bnb = BnBPEP(N, func_class, measure, init_cond, R)
    return bnb.solve(verbose=verbose, max_stage=max_stage)
