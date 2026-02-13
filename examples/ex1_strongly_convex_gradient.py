#!/usr/bin/env python3
"""
Example: Optimal method for reducing gradient of smooth strongly convex functions.

Reproduces the setup from §3 and Table 2 of the BnB-PEP paper:
    Function class: F_{μ,L} with μ=0.1, L=1.0
    Performance measure: ||∇f(x_N)||^2
    Initial condition: ||x_0 - x_*||^2 ≤ R^2 = 1

Compares:
    - Gradient Descent (GD)
    - ITEM (Taylor & Drori 2022)
    - Optimal method from BnB-PEP

Expected results (Table 2):
    N=1: Optimal=0.1473, GD=0.2244
    N=2: Optimal=0.0409, GD=0.0893
    N=5: Optimal=0.002459, GD=0.0159
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from core.gram import GramBuilder
from core.inner_pep import InnerPEP, InnerPEPResult
from core.outer_bnb import BnBPEP, BnBPEPResult
from core.measures import PerformanceMeasure, InitialCondition
from function_classes import SmoothStronglyConvex, SmoothConvex


def analyze_gradient_descent(N: int, mu: float, L: float, R: float) -> float:
    """Worst-case ||∇f(x_N)||^2 for N steps of GD with step 1/L."""
    func_class = SmoothStronglyConvex(mu=mu, L=L)
    pep = InnerPEP(N, func_class, PerformanceMeasure.GRADIENT_NORM,
                   InitialCondition.DISTANCE_BOUND, R=R)
    
    # GD stepsizes: h_{i+1,i} = 1 (normalized), everything else 0
    h = np.zeros((N, N + 1))
    for i in range(N):
        h[i, i] = 1.0
    
    result = pep.solve(h, use_h=True, mode='primal')
    return result.worst_case_value


def find_optimal_method(N: int, mu: float, L: float, R: float,
                        verbose: bool = True) -> BnBPEPResult:
    """Find optimal N-step method using BnB-PEP."""
    func_class = SmoothStronglyConvex(mu=mu, L=L)
    bnb = BnBPEP(N, func_class, PerformanceMeasure.GRADIENT_NORM,
                 InitialCondition.DISTANCE_BOUND, R=R)
    return bnb.solve(verbose=verbose, max_stage=2)


def main():
    mu, L, R = 0.1, 1.0, 1.0
    
    print("=" * 70)
    print("BnB-PEP: Optimal method for ||∇f(x_N)||^2")
    print(f"Function class: F_{{{mu},{L}}} (smooth strongly convex)")
    print(f"Initial condition: ||x_0 - x*||^2 ≤ {R}")
    print("=" * 70)
    print()
    
    print(f"{'N':>3}  {'GD':>12}  {'Optimal':>12}  {'Improvement':>12}")
    print("-" * 50)
    
    for N in [1, 2, 3, 4, 5]:
        gd_value = analyze_gradient_descent(N, mu, L, R)
        
        result = find_optimal_method(N, mu, L, R, verbose=False)
        opt_value = result.optimal_value
        
        if gd_value is not None and opt_value is not None and not np.isnan(opt_value):
            improvement = (gd_value - opt_value) / gd_value * 100
            print(f"{N:>3}  {gd_value:>12.6f}  {opt_value:>12.6f}  {improvement:>10.1f}%")
        else:
            print(f"{N:>3}  {gd_value:>12.6f}  {'FAILED':>12}")
    
    print()
    
    # Detailed output for N=3
    print("=" * 70)
    print("Detailed result for N=3:")
    print("=" * 70)
    result = find_optimal_method(3, mu, L, R, verbose=True)
    if result.stepsizes_h is not None:
        print(f"\nOptimal stepsizes h:")
        print(result.stepsizes_h)
        print(f"\nOptimal worst-case: {result.optimal_value:.6e}")
        print(f"Stage 1 time: {result.stage1_time:.3f}s")
        print(f"Stage 2 time: {result.stage2_time:.3f}s")


if __name__ == '__main__':
    main()
