"""Exercise "Hill anisotropy and the symmetry of the tangent" (ch5) of the lecture notes.
Run: cd python/examples/ch5 && python3 exo_hill.py
"""
# General backward Euler step (closest-point projection, Section 5.3) for Hill's
# criterion, and the symmetry of its consistent tangent, computed by finite
# differences: symmetric for associative flow, unsymmetric otherwise.
import numpy as np
E, nu, sY, K = 200e3, 0.3, 250.0, 2e3          # MPa, -, MPa, MPa (isotropic hardening)
lam, mu = E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))   # Lame constants
# Mandel vectors (shear components times sqrt 2): C is then a symmetric 6x6 matrix
C = lam * np.outer([1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0]) + 2 * mu * np.eye(6)   # Mandel


def hill(F, G, H, L=1.5, M=1.5, N=1.5):         # sigma_eq^2 = s.P s (Mandel vectors)
    """Hill matrix P_H; F = G = H = 1/2, L = M = N = 3/2 is von Mises."""
    P = np.zeros((6, 6))
    for (i, j), c in zip([(1, 2), (2, 0), (0, 1)], [F, G, H]):
        e = np.zeros(6); e[i], e[j] = 1, -1
        P += c * np.outer(e, e)                  # F (s22 - s33)^2 + G (s33 - s11)^2 + ...
    P[3, 3], P[4, 4], P[5, 5] = N, M, L          # Mandel shear components carry sqrt 2
    return P


def update(eps, Pf, Pg):
    """Backward Euler, yield sqrt(s.Pf s) - (sY + K a), flow along grad of sqrt(s.Pg s);
    Newton on (sigma, dgamma)."""
    # elastic predictor from the virgin state: sigma_trial = C eps
    s_tr = C @ eps
    feq = lambda s: np.sqrt(s @ Pf @ s)
    if feq(s_tr) <= sY:
        return s_tr
    s, dg, S = s_tr.copy(), 0.0, np.linalg.inv(C)   # S = C^-1, compliance
    for _ in range(50):
        # residuals: S (sigma - sigma_trial) + dg n_g = 0 and f(sigma, alpha) = 0
        ng = Pg @ s / np.sqrt(s @ Pg @ s)
        r = np.r_[S @ (s - s_tr) + dg * ng, feq(s) - sY - K * dg]   # alpha = dgamma
        if np.linalg.norm(r[:6]) < 1e-15 and abs(r[6]) < 1e-9:
            break
        # Jacobian of the residual with respect to (sigma, dgamma)
        q = np.sqrt(s @ Pg @ s)
        J = np.zeros((7, 7))
        J[:6, :6] = S + dg * (Pg / q - np.outer(Pg @ s, Pg @ s) / q**3)
        J[:6, 6] = ng
        J[6, :6] = Pf @ s / feq(s)
        J[6, 6] = -K
        dx = np.linalg.solve(J, -r)
        s, dg = s + dx[:6], dg + dx[6]
    return s


def fd_tangent(eps, Pf, Pg, h=1e-9):
    """Consistent tangent d sigma / d eps by centred finite differences."""
    return np.column_stack([(update(eps + h * e, Pf, Pg) - update(eps - h * e, Pf, Pg)) / (2 * h)
                            for e in np.eye(6)])


# a generic total strain beyond first yield (Mandel components)
eps = np.array([0.004, -0.001, -0.0015, 0.0012, 0.0004, -0.0007])
mises, aniso = hill(.5, .5, .5), hill(.25, .35, .65)
# relative asymmetry ||T - T^T|| / ||T||: about 1e-9 (finite-difference error) when
# the flow is associative, of order 1e-2 for flow along von Mises with Hill's yield
for name, Pf, Pg in [("von Mises", mises, mises), ("Hill, associative", aniso, aniso),
                     ("Hill, non-associative", aniso, mises)]:
    T = fd_tangent(eps, Pf, Pg)
    print(f"{name:22s}: |T - T^T| / |T| = {np.linalg.norm(T - T.T) / np.linalg.norm(T):.1e}")
# von Mises             : |T - T^T| / |T| = 1.0e-09
# Hill, associative     : |T - T^T| / |T| = 1.1e-09
# Hill, non-associative : |T - T^T| / |T| = 2.5e-02
