"""Exercise gd (ch1) of the lecture notes "Nonlinear Problems in Mechanics".
Run: python3 python/examples/ch1/exo_gd.py
"""
import numpy as np
K = np.array([[2,-1,0],[-1,2,-1],[0,-1,2]], dtype=float)
f = np.array([1,1,1], dtype=float)
u, step = np.zeros(3), 0.2
for _ in range(500):
    u -= step * (K @ u - f)              # gradient of W is K u - f
print(np.round(u, 4), np.round(np.linalg.solve(K, f), 4))
# [1.5 2.  1.5] [1.5 2.  1.5]   max difference ~ 1e-15
