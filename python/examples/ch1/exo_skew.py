"""Exercise skew (ch1) of the lecture notes "Nonlinear Problems in Mechanics".
Run: python3 python/examples/ch1/exo_skew.py
"""
import numpy as np
n, h = 5, 1/6
K = (2*np.eye(n) - np.eye(n,k=1) - np.eye(n,k=-1))/h**2
S = (np.eye(n,k=1) - np.eye(n,k=-1))/(2*h)       # S^T = -S
rng = np.random.default_rng(1)
for b in [0.0, 5.0, 50.0]:
    A = K + b*S; u = rng.normal(size=n)
    print(b, np.allclose(A, A.T), np.isclose(u@A@u, u@K@u),
          np.linalg.norm(np.linalg.solve(A, np.ones(n))))
# symmetric only for b = 0; quadratic form unchanged; always solvable
