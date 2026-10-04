"""Exercise ibp-num (ch1) of the lecture notes "Nonlinear Problems in Mechanics".
Run: python3 python/examples/ch1/exo_ibp_num.py
"""
# Exercise 1.10: check integration by parts on (0, 1),
#   int u' v' dx = - int u'' v dx + [u' v]_0^1,
# for u = sin(pi x) and v = x (1 - x), by numerical quadrature.
from scipy.integrate import quad
import numpy as np
# Derivatives of u = sin(pi x), and v with its derivative.
du  = lambda x: np.pi*np.cos(np.pi*x)
d2u = lambda x: -np.pi**2*np.sin(np.pi*x)
v   = lambda x: x*(1-x);  dv = lambda x: 1-2*x
# Left: the weak (symmetric) form. Right: the strong form plus the boundary
# term u' v at x = 1 and x = 0, which vanishes here since v(0) = v(1) = 0.
lhs = quad(lambda x: du(x)*dv(x), 0, 1)[0]
rhs = quad(lambda x: -d2u(x)*v(x), 0, 1)[0] + du(1)*v(1) - du(0)*v(0)
print(lhs, rhs)        # 1.27323954 1.27323954  (= 4/pi)
