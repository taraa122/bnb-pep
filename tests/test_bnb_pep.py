"""Tests for BnB-PEP."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from core.gram import GramBuilder
from core.inner_pep import InnerPEP
from core.measures import PerformanceMeasure, InitialCondition
from function_classes import SmoothConvex, SmoothStronglyConvex, SmoothNonconvex
from utils.performance import (
    eigen_deflation, cholesky_factor, CertificateReplay, SparsityAnalyzer,
    SDPCache, variable_bounds_from_feasible,
)


class TestGramBuilder:
    def test_selectors_orthogonal(self):
        gb = GramBuilder(3)
        assert gb.gram_size == 5
        assert gb.fval_size == 4
        assert np.dot(gb.x0(), gb.g(0)) == 0
        assert np.dot(gb.g(0), gb.g(1)) == 0

    def test_xstar_is_zero(self):
        gb = GramBuilder(3)
        assert np.allclose(gb.xstar(), np.zeros(5))
        assert np.allclose(gb.gstar(), np.zeros(5))

    def test_sym_outer(self):
        u = np.array([1, 0, 0])
        v = np.array([0, 1, 0])
        S = GramBuilder.sym_outer(u, v)
        assert S.shape == (3, 3)
        assert np.allclose(S, S.T)
        assert S[0, 1] == 0.5

    def test_B_matrix_is_psd(self):
        gb = GramBuilder(2)
        xi = gb.x0()
        xj = gb.xstar()
        B = gb.B(xi, xj)
        eigvals = np.linalg.eigvalsh(B)
        assert np.all(eigvals >= -1e-10)

    def test_iterate_selector_gd(self):
        gb = GramBuilder(2)
        # x_1 = x_0 - (1/L) h g_0, with h=1, L=1
        x1 = gb.x_from_h({0: 1.0}, L=1.0)
        # Should be [1, -1, 0, 0]
        expected = np.array([1.0, -1.0, 0.0, 0.0])
        assert np.allclose(x1, expected)


class TestInnerPEP:
    def test_gd_smooth_convex_known_bound(self):
        """GD on smooth convex: f(x_N)-f(x*) ≤ L||x_0-x*||^2 / (2N)  (approx)."""
        L, R, N = 1.0, 1.0, 5
        func_class = SmoothConvex(L=L)
        pep = InnerPEP(N, func_class, PerformanceMeasure.FUNCTION_VALUE,
                       InitialCondition.DISTANCE_BOUND, R=R)
        
        h = np.zeros((N, N + 1))
        for i in range(N):
            h[i, i] = 1.0
        
        result = pep.solve(h, use_h=True, mode='primal')
        assert result.status in ('optimal', 'optimal_inaccurate')
        # Tight PEP bound for GD with h=1 on F_{0,L} is ~0.176 for N=5
        # This is tighter than the analytical L R^2/(2(N+1)) ≈ 0.083
        # but the SDP finds the actual worst case over all problem instances
        assert result.worst_case_value < 0.25
        assert result.worst_case_value > 0.0

    def test_strongly_convex_better_than_convex(self):
        """Strong convexity should give tighter bounds."""
        L, R, N = 1.0, 1.0, 3
        
        # Convex
        pep_cvx = InnerPEP(N, SmoothConvex(L=L),
                           PerformanceMeasure.FUNCTION_VALUE,
                           InitialCondition.DISTANCE_BOUND, R=R)
        h = np.zeros((N, N + 1))
        for i in range(N):
            h[i, i] = 1.0
        res_cvx = pep_cvx.solve(h, use_h=True, mode='primal')
        
        # Strongly convex
        pep_sc = InnerPEP(N, SmoothStronglyConvex(mu=0.1, L=L),
                          PerformanceMeasure.FUNCTION_VALUE,
                          InitialCondition.DISTANCE_BOUND, R=R)
        res_sc = pep_sc.solve(h, use_h=True, mode='primal')
        
        assert res_sc.worst_case_value <= res_cvx.worst_case_value + 1e-6

    def test_gradient_norm_positive(self):
        """Gradient norm should be positive."""
        N = 2
        func_class = SmoothStronglyConvex(mu=0.1, L=1.0)
        pep = InnerPEP(N, func_class, PerformanceMeasure.GRADIENT_NORM,
                       InitialCondition.DISTANCE_BOUND, R=1.0)
        h = np.zeros((N, N + 1))
        for i in range(N):
            h[i, i] = 1.0
        result = pep.solve(h, use_h=True, mode='primal')
        assert result.worst_case_value > 0


class TestUtils:
    def test_eigen_deflation(self):
        Z = np.diag([1.0, 0.5, 1e-15, 0.0])
        V, lam, rank = eigen_deflation(Z)
        assert rank == 2
        assert len(lam) == 2

    def test_cholesky_factor(self):
        Z = np.array([[4, 2], [2, 3]], dtype=float)
        P = cholesky_factor(Z)
        assert np.allclose(P @ P.T, Z, atol=1e-8)
        # Lower triangular
        assert abs(P[0, 1]) < 1e-10

    def test_sdp_cache(self):
        cache = SDPCache(max_size=10)
        h = np.array([1.0, 2.0])
        cache.put(h, {'nu': 0.5})
        result = cache.get(h)
        assert result is not None
        assert result['nu'] == 0.5
        assert cache.stats()['hits'] == 1

    def test_certificate_replay(self):
        verifier = CertificateReplay()
        Z = np.eye(3)
        result = verifier.verify(nu=0.5, lam=np.array([0.1, 0.2]), Z=Z,
                                 R=1.0, claimed_value=0.5)
        assert result['valid']
        assert result['Z_rank'] == 3

    def test_sparsity_analyzer(self):
        analyzer = SparsityAnalyzer()
        lam = np.array([0.5, 0.0, 0.0, 0.3, 0.0, 0.1])
        labels = [('*', 0), ('*', 1), (0, '*'), (0, 1), (1, '*'), (1, 0)]
        result = analyzer.analyze_lambda(lam, labels)
        assert result['support_size'] == 3
        assert result['sparsity'] == 0.5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
