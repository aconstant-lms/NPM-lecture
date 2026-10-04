"""Exercise skew (ch1) of the lecture notes "Nonlinear Problems in Mechanics".
Run: python3 python/examples/ch1/exo_skew.py
"""
# Exercise 1.15: a non-symmetric but coercive matrix K + b S is still invertible
# (Lax-Milgram without symmetry, Section 1.7).
import numpy as np
# n interior nodes, mesh size h = 1/(n+1).
n, h = 5, 1/6
# Diffusion: second-difference matrix K = tridiag(-1, 2, -1) / h^2 (symmetric).
K = (2*np.eye(n) - np.eye(n,k=1) - np.eye(n,k=-1))/h**2
# Convection: centred first difference S_ij = (delta_{i+1,j} - delta_{i-1,j}) / (2h).
S = (np.eye(n,k=1) - np.eye(n,k=-1))/(2*h)       # S^T = -S
rng = np.random.default_rng(1)
# For each convection coefficient b: symmetry of A = K + b S, the quadratic form
# u^T A u against u^T K u (the skew part gives u^T S u = 0), and the norm of the
# solution of A u = 1 (finite: A is invertible).
for b in [0.0, 5.0, 50.0]:
    A = K + b*S; u = rng.normal(size=n)
    print(b, np.allclose(A, A.T), np.isclose(u@A@u, u@K@u),
          np.linalg.norm(np.linalg.solve(A, np.ones(n))))
# symmetric only for b = 0; quadratic form unchanged; always solvable
