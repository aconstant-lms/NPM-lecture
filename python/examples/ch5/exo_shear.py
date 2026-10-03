"""Exercise "Simple shear and a non-proportional path" (ch5) of the lecture notes.
Run: cd python/examples/ch5 && python3 exo_shear.py
"""
import numpy as np
from ci_core import radial_return, Linear
E, nu, sY, K = 200e3, 0.3, 250.0, 2e3
mu, kappa = E / (2 * (1 + nu)), E / (3 * (1 - 2 * nu))
hard, z0 = Linear(sY, K), (np.zeros(6), 0.0, np.zeros(6))


def drive(points, n):                          # piecewise linear strain path
    ep, al, beta = z0
    for e0, e1 in zip(points[:-1], points[1:]):
        for t in np.linspace(0, 1, n + 1)[1:]:
            s, ep, al, beta, *_ = radial_return((1 - t) * e0 + t * e1, ep, al, beta,
                                                mu, kappa, hard)
    return s, al


# (a) simple shear, gamma = 2 e12 = 0.01: one step against the closed form
k, Gep = sY / np.sqrt(3), mu * K / (3 * mu + K)
g = 0.01
exact = k + Gep * (g - k / mu)
O, S = np.zeros(6), np.array([0, 0, 0, g / 2, 0, 0])
print(f"shear: closed form {exact:.6f}, 1 step {drive([O, S], 1)[0][3]:.6f},"
      f" 1000 steps {drive([O, S], 1000)[0][3]:.6f} MPa")
# (b) shear to e12 = 0.002 (fine steps), then e11: 0 -> 0.004 at fixed e12
A = np.array([0, 0, 0, 0.002, 0, 0])
B = A + np.array([0.004, 0, 0, 0, 0, 0])


def two_stage(n):
    ep, al, beta = z0
    for t in np.linspace(0, 1, 501)[1:]:
        _, ep, al, beta, *_ = radial_return(t * A, ep, al, beta, mu, kappa, hard)
    for t in np.linspace(0, 1, n + 1)[1:]:
        s, ep, al, beta, *_ = radial_return(A + t * (B - A), ep, al, beta, mu, kappa, hard)
    return s, al


sr, ar = two_stage(200000)
print(f"two-stage reference: s11 = {sr[0]:.4f}, s12 = {sr[3]:.4f} MPa, alpha = {ar:.6e}")
for n in (1, 10, 100, 1000):
    s, a = two_stage(n)
    print(f"n = {n:4d}: s11 err {abs(s[0]-sr[0]):.3e}, s12 err {abs(s[3]-sr[3]):.3e},"
          f" alpha err {abs(a-ar):.3e}")
# shear: closed form 149.706775, 1 step 149.706775, 1000 steps 149.706775 MPa
# two-stage reference: s11 = 834.8243, s12 = 26.0140 MPa, alpha = 3.114637e-03
# n =    1: s11 err 1.027e+01, s12 err 3.007e+01, alpha err 1.260e-04
# n =   10: s11 err 1.160e+00, s12 err 4.941e+00, alpha err 2.987e-05
# n =  100: s11 err 1.148e-01, s12 err 5.281e-01, alpha err 3.504e-06
# n = 1000: s11 err 1.141e-02, s12 err 5.293e-02, alpha err 3.548e-07
