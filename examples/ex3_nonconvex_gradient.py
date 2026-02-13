#!/usr/bin/env python3
"""
Example: Optimal method for reducing gradient of smooth nonconvex functions.

From §6.2: min_{i ∈ [0:N]} ||∇f(x_i)||^2 for f ∈ F_{-L,L}.

Compares against:
  - GD: h_{i,i-1} = 1  
  - AKZ (Abbaszadehpeivasti, de Klerk, Zamani): h_{i,i-1} = 2/√3

The optimal FSFOM admits a "momentum form" (equation 29).
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from core.inner_pep import InnerPEP
from core.outer_bnb import BnBPEP
from core.measures import PerformanceMeasure, InitialCondition
from function_classes import SmoothNonconvex


def main():
    L, R = 1.0, 1.0
    
    print("=" * 70)
    print("Optimal method for smooth NONCONVEX gradient reduction")
    print(f"Function class: F_{{-{L},{L}}}")
    print(f"Measure: min_i ||∇f(x_i)||^2")
    print(f"Initial: f(x_0) - f(x*) ≤ R^2/2 = {R**2/2}")
    print("=" * 70)
    print()
    
    print(f"{'N':>3}  {'GD':>12}  {'AKZ':>12}  {'BnB-PEP':>12}")
    print("-" * 50)
    
    for N in [1, 2, 3, 5]:
        func_class = SmoothNonconvex(L=L)
        pep = InnerPEP(N, func_class, PerformanceMeasure.MIN_GRADIENT_NORM,
                       InitialCondition.FUNCTION_VALUE_BOUND, R=R)
        
        # GD: h = 1
        h_gd = np.zeros((N, N + 1))
        for i in range(N):
            h_gd[i, i] = 1.0
        res_gd = pep.solve(h_gd, use_h=True, mode='primal')
        
        # AKZ: h = 2/√3
        h_akz = np.zeros((N, N + 1))
        for i in range(N):
            h_akz[i, i] = 2.0 / np.sqrt(3.0)
        res_akz = pep.solve(h_akz, use_h=True, mode='primal')
        
        # BnB-PEP
        bnb = BnBPEP(N, func_class, PerformanceMeasure.MIN_GRADIENT_NORM,
                     InitialCondition.FUNCTION_VALUE_BOUND, R=R)
        res_opt = bnb.solve(verbose=False, max_stage=2)
        
        gd_val = res_gd.worst_case_value if res_gd.worst_case_value else float('nan')
        akz_val = res_akz.worst_case_value if res_akz.worst_case_value else float('nan')
        opt_val = res_opt.optimal_value if res_opt.optimal_value else float('nan')
        
        print(f"{N:>3}  {gd_val:>12.7f}  {akz_val:>12.7f}  {opt_val:>12.7f}")


if __name__ == '__main__':
    main()
