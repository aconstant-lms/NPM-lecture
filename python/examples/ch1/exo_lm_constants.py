"""Exercise lm-constants (ch1) of the lecture notes "Nonlinear Problems in Mechanics".
Run: python3 python/examples/ch1/exo_lm_constants.py
"""
import numpy as np
K = np.array([[2,-1,0],[-1,2,-1],[0,-1,2]], dtype=float)
lam = np.linalg.eigvalsh(K); alpha, M = lam[0], lam[-1]
rng = np.random.default_rng(0)
for _ in range(5):
    u = rng.normal(size=3); a = u @ K @ u
    print(alpha*u@u <= a <= M*u@u)          # True, five times
