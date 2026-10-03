"""Exercise conv (ch1) of the lecture notes "Nonlinear Problems in Mechanics".
Run: python3 python/examples/ch1/exo_conv.py
"""
import numpy as np
def err(n):
    h = 1.0/(n+1); x = np.linspace(h, 1-h, n)
    K = (2*np.eye(n) - np.eye(n,k=1) - np.eye(n,k=-1))/h**2
    u = np.linalg.solve(K, np.sin(np.pi*x))
    return h, np.max(np.abs(u - np.sin(np.pi*x)/np.pi**2))
for n in [3, 7, 15, 31, 63]: print(*err(n))
# the error is divided by 4 when h is halved: O(h^2)
