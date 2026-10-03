"""Exercise "The plane-stress projected return" (ch5) of the lecture notes.
Run: cd python/examples/ch5 && python3 exo_plane_stress.py
"""
import numpy as np
from scipy.optimize import brentq
from ci_core import plane_stress_return, Linear
E, nu, sY, K = 200e3, 0.3, 250.0, 2e3
hard, C = Linear(sY, K), E / (1 - nu**2) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])
P = np.array([[2, -1, 0], [-1, 2, 0], [0, 0, 6]]) / 3.0
print("eigenvalues of CP:", np.round(np.sort(np.linalg.eigvals(C @ P).real), 2),
      f"| E/(3(1-nu)) = {E/(3*(1-nu)):.2f}, 2 mu = {E/(1+nu):.2f}")
z = (np.zeros(3), 0.0)
# (a) uniaxial stress, eps11 = 0.01 in ONE step (eps22 such that s22 = 0)
s22 = lambda e22: plane_stress_return([0.01, e22, 0], *z, E, nu, hard)[0][1]
s = plane_stress_return([0.01, brentq(s22, -0.02, 0.02, xtol=1e-14), 0], *z, E, nu, hard)[0]
print(f"uniaxial, 1 step: s11 = {s[0]:.6f}, closed form {270/1.01:.6f} MPa")
# (b) balanced biaxial, eps11 = eps22 = 0.005 in one step
s = plane_stress_return([0.005, 0.005, 0], *z, E, nu, hard)[0]
print(f"balanced biaxial, 1 step: s11 = s22 = {s[0]:.6f}, closed form {270/1.014:.6f} MPa")


# (c) strain-driven path: plane strain (eps22 = 0) to eps11 = 0.01, then balanced
#     biaxial increments to (0.02, 0.01); n steps per stage
def two_stage(n, stages=2):
    ep, al = z
    pts = [np.zeros(3), np.array([0.01, 0, 0]), np.array([0.02, 0.01, 0])]
    for k in range(stages):
        for t in np.linspace(0, 1, n + 1)[1:]:
            s, ep, al, _ = plane_stress_return((1 - t) * pts[k] + t * pts[k + 1], ep, al,
                                               E, nu, hard)
    return s, al


for stages in (1, 2):
    sr, ar = two_stage(20000, stages)
    errs = [np.abs(two_stage(n, stages)[0] - sr).max() for n in (1, 10, 100, 1000)]
    print(f"{stages} stage(s): reference s = {np.round(sr[:2], 4)} MPa;"
          f" max error for n = 1, 10, 100, 1000: " + ", ".join(f"{e:.2e}" for e in errs))
# eigenvalues of CP: [ 95238.1  153846.15 153846.15] | E/(3(1-nu)) = 95238.10, 2 mu = 153846.15
# uniaxial, 1 step: s11 = 267.326733, closed form 267.326733 MPa
# balanced biaxial, 1 step: s11 = s22 = 266.272189, closed form 266.272189 MPa
# 1 stage(s): reference s = [311.4365 155.0717] MPa; max error for n = 1, 10, 100, 1000: 6.91e+00, 2.18e-01, 1.16e-02, 1.02e-03
# 2 stage(s): reference s = [308.8066 308.8066] MPa; max error for n = 1, 10, 100, 1000: 5.57e+00, 8.37e-02, 1.23e-02, 1.25e-03
