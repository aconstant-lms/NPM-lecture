"""Exercise gd (ch1) of the lecture notes "Nonlinear Problems in Mechanics".
Run: python3 python/examples/ch1/exo_gd.py
"""
# Exercise 1.9: minimize W(u) = 1/2 u^T K u - f^T u for the chain of springs
# (Section 1.2.2) by gradient descent, and compare with the direct solve K u = f.
import numpy as np
# Stiffness matrix K = B^T C B of the chain with unit springs: tridiag(-1, 2, -1);
# load f = (1, 1, 1) on the three masses.
K = np.array([[2,-1,0],[-1,2,-1],[0,-1,2]], dtype=float)
f = np.array([1,1,1], dtype=float)
# Start from u = 0 with a fixed step. The error is multiplied at each step by
# I - step K, so it contracts at the rate rho = max_i |1 - step lambda_i(K)|
# (Figure 1.16); here rho = 1 - 0.2 (2 - sqrt 2) = 0.88.
u, step = np.zeros(3), 0.2
for _ in range(500):
    u -= step * (K @ u - f)              # gradient of W is K u - f
# Gradient descent (left) and direct solution of K u = f (right).
print(np.round(u, 4), np.round(np.linalg.solve(K, f), 4))
# [1.5 2.  1.5] [1.5 2.  1.5]   max difference ~ 1e-15
