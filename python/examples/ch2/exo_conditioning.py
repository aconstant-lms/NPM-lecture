"""Exercise conditioning (ch2) of the lecture notes "Nonlinear Problems in Mechanics".
Run: python3 python/examples/ch2/exo_conditioning.py
"""
import numpy as np
for n in [3, 5, 7, 9]:
    K = np.array([[m * l / (m + l - 1) for l in range(1, n + 1)] for m in range(1, n + 1)])
    H = n * (2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1))   # n hats, h = 1/n
    H[-1, -1] = n                                                 # free end x = 1
    ev, eh = np.linalg.eigvalsh(K), np.linalg.eigvalsh(H)
    print(n, f"monomials {ev[-1] / ev[0]:.1e}", f"hats {eh[-1] / eh[0]:.1f}")
# 3 monomials 2.8e+02 hats 16.4
# 5 monomials 1.9e+05 hats 45.5
# 7 monomials 1.7e+08 hats 87.6
# 9 monomials 1.6e+11 hats 142.7
