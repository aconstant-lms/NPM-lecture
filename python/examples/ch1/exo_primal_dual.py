"""Exercise primal-dual (ch1) of the lecture notes "Nonlinear Problems in Mechanics".
Run: python3 python/examples/ch1/exo_primal_dual.py
"""
# Exercise 1.12: for the spring chain with unit stiffnesses (C = I), the minimum
# of the potential energy W(u*) equals the complementary energy -1/2 |sigma*|^2.
import numpy as np
# Compatibility matrix B: elongations e = B u of the four springs from the three
# displacements (Section 1.2.2). Stiffness K = B^T C B = B^T B, load f = (1,1,1).
B = np.array([[1,0,0],[-1,1,0],[0,-1,1],[0,0,-1]], dtype=float)
K, f = B.T @ B, np.ones(3)
# Equilibrium K u* = f, then the tensions sigma* = C e* = B u*.
u = np.linalg.solve(K, f); s = B @ u
# Potential energy W(u*) = 1/2 u^T K u - f^T u, and the complementary energy.
print(0.5*u@K@u - f@u, -0.5*s@s)   # -2.5 -2.5
