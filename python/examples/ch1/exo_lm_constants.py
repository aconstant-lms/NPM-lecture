"""Exercise lm-constants (ch1) of the lecture notes "Nonlinear Problems in Mechanics".
Run: python3 python/examples/ch1/exo_lm_constants.py
"""
# Exercise 1.11: the two constants of the Lax-Milgram theorem (Section 1.6) for
# the bilinear form a(u, v) = u^T K v of the spring chain.
import numpy as np
K = np.array([[2,-1,0],[-1,2,-1],[0,-1,2]], dtype=float)
# K is symmetric: its eigenvalues 2 - sqrt 2, 2, 2 + sqrt 2 give the sharp bounds
#   alpha |u|^2 <= u^T K u <= M |u|^2,
# alpha = smallest eigenvalue (coercivity), M = largest one (boundedness).
lam = np.linalg.eigvalsh(K); alpha, M = lam[0], lam[-1]
# Test the two bounds on random vectors u (Figure 1.17 plots 300 of them).
rng = np.random.default_rng(0)
for _ in range(5):
    u = rng.normal(size=3); a = u @ K @ u
    print(alpha*u@u <= a <= M*u@u)          # True, five times
