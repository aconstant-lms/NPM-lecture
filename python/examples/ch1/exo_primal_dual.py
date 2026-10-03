"""Exercise primal-dual (ch1) of the lecture notes "Nonlinear Problems in Mechanics".
Run: python3 python/examples/ch1/exo_primal_dual.py
"""
import numpy as np
B = np.array([[1,0,0],[-1,1,0],[0,-1,1],[0,0,-1]], dtype=float)
K, f = B.T @ B, np.ones(3)
u = np.linalg.solve(K, f); s = B @ u
print(0.5*u@K@u - f@u, -0.5*s@s)   # -2.5 -2.5
