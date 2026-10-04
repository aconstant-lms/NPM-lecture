"""Exercise "Consistent against continuum tangent" (ch5) of the lecture notes.
Run: cd python/examples/ch5 && python3 exo_tangent.py
"""
# Global Newton method of Box 5.3 on the thick-walled cylinder with Voce
# hardening: consistent tangent (5.14) against continuum tangent (Figure 5.6).
import numpy as np
from ci_core import Voce
from ci_fem import solve
E, nu = 200e3, 0.3                             # MPa, -
mu, kappa = E / (2 * (1 + nu)), E / (3 * (1 - 2 * nu))
hard = Voce(250.0, 600.0, 30.0)                # s0, s_inf (MPa), delta
# a = 100, b = 200 mm, p = 320 MPa in 10 steps, 80 elements
for tg in ("alg", "cont"):
    u, r, al, sig, its, res = solve(100.0, 200.0, 320.0, 10, 80, mu, kappa, hard,
                                    tangent=tg)
    print(f"{tg:4s}: iterations per step {its}, total {sum(its)}, u(a) = {u[0]:.10f} mm")
    # quadratic convergence with "alg" (the exponent doubles), linear with "cont"
    print("      residuals of step 6:", " ".join(f"{x:.2e}" for x in res[5]))
# alg : iterations per step [2, 2, 2, 5, 6, 6, 6, 6, 6, 6], total 47, u(a) = 3.4526072858 mm
#       residuals of step 6: 3.20e+03 6.29e+03 3.84e+03 8.71e+02 3.15e+00 3.50e-05
# cont: iterations per step [2, 2, 2, 7, 8, 9, 11, 11, 10, 10], total 72, u(a) = 3.4526072857 mm
#       residuals of step 6: 3.20e+03 6.29e+03 3.84e+03 8.85e+02 1.22e+01 3.62e-01 1.58e-02 7.56e-04 3.95e-05
