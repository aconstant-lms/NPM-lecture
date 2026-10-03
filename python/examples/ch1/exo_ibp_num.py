"""Exercise ibp-num (ch1) of the lecture notes "Nonlinear Problems in Mechanics".
Run: python3 python/examples/ch1/exo_ibp_num.py
"""
from scipy.integrate import quad
import numpy as np
du  = lambda x: np.pi*np.cos(np.pi*x)
d2u = lambda x: -np.pi**2*np.sin(np.pi*x)
v   = lambda x: x*(1-x);  dv = lambda x: 1-2*x
lhs = quad(lambda x: du(x)*dv(x), 0, 1)[0]
rhs = quad(lambda x: -d2u(x)*v(x), 0, 1)[0] + du(1)*v(1) - du(0)*v(0)
print(lhs, rhs)        # 1.27323954 1.27323954  (= 4/pi)
