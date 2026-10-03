"""Exercise "A non-monotone step: first order" (ch5) of the lecture notes.
Run: cd python/examples/ch5 && python3 exo_nonmonotone_1d.py
"""
import numpy as np
from ci_core import return_map_1d
E, sY, K, H = 200e3, 250.0, 2e3, 1e3
tp = 1 / np.sqrt(2)                            # turning point, never on the grid
eps = lambda t: np.where(t <= tp, 0.008 * t / tp, 0.008 - 0.010 * (t - tp) / (2 - tp))


def run(n):
    s = ep = al = q = 0.0
    for t in np.linspace(0, 2, n + 1)[1:]:
        s, ep, al, q, _ = return_map_1d(eps(t), ep, al, q, E, sY, K, H)
    return s, al


sr, ar = run(400001)
print(f"reference: sigma = {sr:.5f} MPa, alpha = {ar:.6e}")
for n in (1, 2, 8, 32, 128, 1024):
    s, a = run(n)
    print(f"n = {n:5d}: |sigma error| = {abs(s - sr):.3e} MPa, |alpha error| = {abs(a - ar):.3e}")
# reference: sigma = -278.42457 MPa, alpha = 1.390835e-02
# n =     1: |sigma error| = 2.621e+01 MPa, |alpha error| = 1.317e-02
# n =     2: |sigma error| = 8.796e+00 MPa, |alpha error| = 4.420e-03
# n =     8: |sigma error| = 1.288e+00 MPa, |alpha error| = 6.473e-04
# n =    32: |sigma error| = 8.612e-01 MPa, |alpha error| = 4.328e-04
# n =   128: |sigma error| = 1.749e-01 MPa, |alpha error| = 8.787e-05
# n =  1024: |sigma error| = 3.274e-03 MPa, |alpha error| = 1.645e-06
