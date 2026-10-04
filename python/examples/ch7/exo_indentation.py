"""Exercise 7.x -- indentation of a membrane on an elastic foundation by a rigid
parabolic punch: adjoint derivative of the force (Dirichlet data on the contact
zone) against finite differences, identification of (k, T) from two depths, and
the kinks of the discrete problem."""
import numpy as np
from scipy.optimize import least_squares
from id_core import Membrane

# Exercise 7.10, Section 7.5. Membrane on [-1, 1] with 79 free nodes, punch radius
# R = 0.5; the parameters p = (k_f, T_m) are dimensionless, true values (4, 1).
Mb = Membrane(n=81)
p_true = np.array([4.0, 1.0])                     # foundation modulus k, tension T

# --- 1. adjoint derivative against central finite differences
# Table 7.4: dF/dp by (7.15) with the unit datum, against central differences
# with the step 1e-6 (relative, since p is of order one)
print("   U      k     T      F       dF/dk (adj, FD)        dF/dT (adj, FD)   contact nodes")
for U, p in [(0.02, p_true), (0.2, p_true), (0.2, [2.0, 3.0]), (0.1, [6.0, 0.5])]:
    p = np.array(p, float)
    g, act = Mb.dforce_adjoint(p, U)
    fd = [(Mb.force(p + d, U) - Mb.force(p - d, U)) / (2 * d.max()) for d in 1e-6 * np.eye(2)]
    print("%5.2f %5.1f %5.1f %7.4f  %9.6f %9.6f   %9.6f %9.6f   %d" %
          (U, p[0], p[1], Mb.force(p, U), g[0], fd[0], g[1], fd[1], act.sum()))

# --- 2. which depth informs which parameter: d log F / d log p
# F is homogeneous of degree one in (k_f, T_m): each row adds up to one
# (Exercise 7.5)
for U in (0.02, 0.2):
    g, act = Mb.dforce_adjoint(p_true, U)
    print("U = %.2f: d log F / d log (k, T) = %s" %
          (U, np.round(g * p_true / Mb.force(p_true, U), 3)))

# --- 3. identification of (k, T) from the forces at two depths
# Levenberg-Marquardt in q = log(p / p_true), residuals F(p)/F^m - 1, Jacobian
# from the adjoint derivatives (noise-free data)
Us = np.array([0.02, 0.2])
Fm = np.array([Mb.force(p_true, U) for U in Us])
resid = lambda q: np.array([Mb.force(p_true * np.exp(q), U) for U in Us]) / Fm - 1.0
jac = lambda q: np.array([Mb.dforce_adjoint(p_true * np.exp(q), U)[0] * p_true * np.exp(q)
                          for U in Us]) / Fm[:, None]
for f0 in ([0.2, 0.2], [0.2, 5.0], [5.0, 0.2], [5.0, 5.0]):
    r = least_squares(resid, np.log(f0), jac=jac, method="lm", xtol=1e-14, ftol=1e-15)
    print("start p/p_true %s -> %s in %d iterations" %
          (f0, np.round(np.exp(r.x), 6), r.njev))

# --- 4. kinks: jump of dF/dk where a node enters the contact zone (U = 0.1)
# Section 7.5.3: a biactive node gives a kink of F(k_f); the jump of the
# derivative is the contribution of one node and shrinks with the mesh size h
for n in (41, 81, 161, 321):
    M = Membrane(n=n)
    ks = np.linspace(2.0, 6.0, 401)
    out = [M.dforce_adjoint([k, 1.0], 0.1) for k in ks]
    g = np.array([o[0][0] for o in out])
    na = np.array([o[1].sum() for o in out])
    jumps = np.where(np.diff(na) != 0)[0]
    rel = max([abs(g[i + 1] - g[i]) / abs(g[i]) for i in jumps], default=0.0)
    print("h = %.4f: %d changes of the contact zone for k in [2, 6], largest jump of"
          " dF/dk %.1f%%" % (M.h, len(jumps), 100 * rel))
