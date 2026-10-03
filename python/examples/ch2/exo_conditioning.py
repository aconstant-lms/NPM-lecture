"""Exercise conditioning (ch2) of the lecture notes "Nonlinear Problems in Mechanics".
Run: python3 python/examples/ch2/exo_conditioning.py
"""
import numpy as np
for n in [3,5,7,9]:
    K = np.array([[m*l/(m+l-1) for l in range(1,n+1)] for m in range(1,n+1)])
    ev = np.linalg.eigvalsh(K); print(n, ev[-1]/ev[0])
# 3: 2.8e2   5: 1.9e5   7: 1.7e8   9: 1.6e11
# hat functions (tridiagonal K): 5.8, 13.9, 25.3, 39.9
