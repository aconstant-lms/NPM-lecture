"""Exercise conditioning (ch2) of the lecture notes "Nonlinear Problems in Mechanics".
Run: python3 python/examples/ch2/exo_conditioning.py
"""
# Exercise 2.20: condition number of the Ritz stiffness matrix of the bar
# (Exercise 2.19, EA = L = 1, fixed at x = 0, free at x = 1) for two bases:
# the monomials w_m = x^m and n hat functions (linear finite elements).
import numpy as np
for n in [3, 5, 7, 9]:
    # Monomials: K_ml = int_0^1 w_m' w_l' dx = m l / (m + l - 1), Hilbert-like.
    K = np.array([[m * l / (m + l - 1) for l in range(1, n + 1)] for m in range(1, n + 1)])
    # Hats: (1/h) tridiag(-1, 2, -1); the last node, at the free end, has one element.
    H = n * (2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1))   # n hats, h = 1/n
    H[-1, -1] = n                                                 # free end x = 1
    # Both matrices are symmetric positive definite: cond = lambda_max / lambda_min.
    ev, eh = np.linalg.eigvalsh(K), np.linalg.eigvalsh(H)
    print(n, f"monomials {ev[-1] / ev[0]:.1e}", f"hats {eh[-1] / eh[0]:.1f}")
# 3 monomials 2.8e+02 hats 16.4
# 5 monomials 1.9e+05 hats 45.5
# 7 monomials 1.7e+08 hats 87.6
# 9 monomials 1.6e+11 hats 142.7
# monomials: x 30 per function (geometric); hats: grows like 1/h^2 = n^2
