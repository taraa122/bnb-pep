#!/usr/bin/env python3
"""
Quick analysis example — like PESTO toolbox but in Python.

Reproduces Figure 1 from the PESTO paper:
    Gradient method for minimizing a 1-smooth 0.1-strongly convex function.
    Worst-case ||∇f(x_N)||^2 as a function of step size γ ∈ [0, 2/L].
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from core.inner_pep import InnerPEP
from core.measures import PerformanceMeasure, InitialCondition
from function_classes import SmoothStronglyConvex


def main():
    mu, L = 0.1, 1.0
    R = 1.0  # F(x_0) - F(x_*) ≤ 1
    
    print("=" * 70)
    print("PESTO-style analysis: gradient method worst-case vs step size")
    print(f"Function: F_{{{mu},{L}}}, Initial: F(x_0)-F(x*) ≤ {R}")
    print("=" * 70)
    
    for N in [2, 5, 10]:
        print(f"\nN = {N}:")
        print(f"  {'gamma':>8}  {'||∇f(x_N)||^2':>15}")
        print(f"  {'-'*8}  {'-'*15}")
        
        func_class = SmoothStronglyConvex(mu=mu, L=L)
        pep = InnerPEP(N, func_class, PerformanceMeasure.GRADIENT_NORM,
                       InitialCondition.FUNCTION_VALUE_BOUND, R=R)
        
        gammas = np.linspace(0.1, 2.0 / L, 10)
        for gamma in gammas:
            h = np.zeros((N, N + 1))
            for i in range(N):
                h[i, i] = gamma * L  # normalized: h = γL
            
            result = pep.solve(h, use_h=True, mode='primal')
            val = result.worst_case_value
            if val is not None and not np.isnan(val):
                print(f"  {gamma:>8.4f}  {val:>15.6f}")
            else:
                print(f"  {gamma:>8.4f}  {'FAILED':>15}")


if __name__ == '__main__':
    main()
