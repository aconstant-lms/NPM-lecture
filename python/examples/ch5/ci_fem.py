"""ci_fem.py -- thick-walled cylinder, axisymmetric plane-strain elements and
global Newton with the consistent or continuum tangent (ch5). Run:
cd python/examples/ch5 && python3 ci_fem.py"""
import numpy as np
from ci_core import radial_return, C_elastic, C_algorithmic, C_continuum, Linear

# Finite element driver of Box 5.3 for the thick-walled cylinder a <= r <= b
# under an internal pressure p (plane strain, axisymmetry).
#   B_matrix   strain-displacement matrix of a 2-node element (one Gauss point)
#   solve      load steps, global Newton iterations, call of radial_return at
#              every Gauss point from the converged variables z_n, assembly of
#              F_int and of the tangent stiffness (consistent or continuum)
#   p_of_c     closed-form pressure for a plastic front at r = c
# Unknown: the radial displacement u at the nodes. Strain
# [e_rr, e_tt, e_zz, 0, 0, 0] = [du/dr, u/r, 0, 0, 0, 0]; the factor 2 pi of
# the volume element cancels and is dropped.
# Units: lengths in mm, stresses and pressures in MPa.


def B_matrix(r1, r2):
    """B (6x2), Gauss-point radius rg and length L of the element [r1, r2]."""
    # Linear element, one Gauss point at the mid-point: eps = B [u1, u2]
    L, rg = r2 - r1, 0.5 * (r1 + r2)
    B = np.zeros((6, 2))
    B[0, :] = [-1.0 / L, 1.0 / L]                 # e_rr = du/dr
    B[1, :] = [0.5 / rg, 0.5 / rg]                # e_tt = u/r
    return B, rg, L


def solve(a, b, p_max, n_steps, n_el, mu, kappa, hard, tangent="alg",
          tol=1e-8, max_it=40):
    """Internal pressure p_max in n_steps equal increments (Box 5.3).
    Returns u, r, alpha, sigma at the Gauss points, iterations per step and the
    residual norms of every step."""
    # a, b: inner and outer radii; n_el elements; mu, kappa (MPa); hard: law of
    # ci_core; tangent "alg" (consistent, (5.14)) or "cont" (continuum);
    # tol: relative tolerance on ||R||; max_it: Newton iterations per step.
    r = np.linspace(a, b, n_el + 1)
    u = np.zeros(n_el + 1)
    # internal variables z = (eps^p, beta, alpha) and stress, one Gauss point per element
    ep, beta, al = np.zeros((n_el, 6)), np.zeros((n_el, 6)), np.zeros(n_el)
    sig = np.zeros((n_el, 6))
    Cel = C_elastic(mu, kappa)
    its, res = [], []
    # Box 5.3 step 1: loop on the load steps t_n -> t_{n+1}
    for step in range(n_steps):
        p = p_max * (step + 1) / n_steps
        Fext = np.zeros(n_el + 1)
        Fext[0] = p * a                           # pressure on the bore
        ep_n, al_n, beta_n = ep.copy(), al.copy(), beta.copy()   # converged z_n
        hist = []
        # Newton loop at fixed t_{n+1}, starting from u^(0) = u_n
        for it in range(max_it):
            K = np.zeros((n_el + 1, n_el + 1))
            Fint = np.zeros(n_el + 1)
            for e in range(n_el):
                # Box 5.3 step 2: local stage, eps = B u, return map from z_n
                B, rg, L = B_matrix(r[e], r[e + 1])
                out = radial_return(B @ u[e:e + 2], ep_n[e], al_n[e], beta_n[e],
                                    mu, kappa, hard)
                sig[e], ep[e], al[e], beta[e], dg, n, nx, _ = out
                # tangent at the Gauss point: elastic, consistent or continuum
                if dg == 0.0:
                    C = Cel
                elif tangent == "alg":
                    C = C_algorithmic(dg, nx, n, mu, kappa, hard.slope(al[e]))
                else:
                    C = C_continuum(n, mu, kappa, hard.slope(al[e]))
                # Box 5.3 step 3: assembly, weight w = r_g L (2 pi dropped)
                w = rg * L
                Fint[e:e + 2] += (B.T @ sig[e]) * w
                K[e:e + 2, e:e + 2] += (B.T @ C @ B) * w
            # Box 5.3 step 4: residual R = F_ext - F_int and convergence test
            R = Fext - Fint
            hist.append(np.linalg.norm(R))
            if hist[-1] < tol * max(1.0, np.linalg.norm(Fext)):
                break
            # Box 5.3 step 5: Newton update K du = R, equation (5.15)
            u += np.linalg.solve(K, R)
        its.append(it + 1)
        res.append(hist)
    return u, r, al, sig, its, res


def p_of_c(c, a, b, k):
    """Pressure for a plastic front at r = c (Tresca-type solution, k = 2 sY/sqrt3)."""
    return k * np.log(c / a) + 0.5 * k * (1.0 - c**2 / b**2)


if __name__ == "__main__":
    from scipy.optimize import brentq
    E, nu, sY = 200e3, 0.3, 250.0                 # MPa, -, MPa
    mu, kappa = E / (2 * (1 + nu)), E / (3 * (1 - 2 * nu))
    a, b = 100.0, 200.0                           # inner and outer radii (mm)
    # limit pressure p_L = (2 sY / sqrt3) ln(b/a)
    k = 2 * sY / np.sqrt(3.0)
    pL = k * np.log(b / a)
    print(f"p_L = {pL:.4f} MPa")
    for frac in (0.80, 0.95):
        # perfect plasticity (K = 0), 20 load steps, 200 elements
        u, r, al, sig, its, _ = solve(a, b, pL * frac, 20, 200, mu, kappa,
                                      Linear(sY, 0.0))
        # plastic front: outer node of the last element with alpha > 0
        c_fe = r[np.where(al > 1e-12)[0][-1] + 1]
        # closed form: root of p_of_c(c) = p
        c_an = brentq(lambda c: p_of_c(c, a, b, k) - pL * frac, a + 1e-9, b)
        print(f"p/pL = {frac:.2f}: c_FE = {c_fe:.2f}, c_Tresca-type = {c_an:.2f} mm,"
              f" u(a) = {u[0]:.5f} mm, Newton iterations {sum(its)}")
# p_L = 200.0944 MPa
# p/pL = 0.80: c_FE = 131.00, c_Tresca-type = 130.77 mm, u(a) = 0.18305 mm, Newton iterations 60
# p/pL = 0.95: c_FE = 165.50, c_Tresca-type = 164.00 mm, u(a) = 0.30536 mm, Newton iterations 70
