"""ci_core.py -- return maps of Chapter 5 (plasticity: integration algorithms).
Imported by the exercise and figure scripts of Chapter 5.
"""
import numpy as np

# Voigt convention (6 components), TENSOR shear strains:
#     a = [a11, a22, a33, a12, a13, a23],
# so that ||a||^2 = a11^2 + a22^2 + a33^2 + 2 (a12^2 + a13^2 + a23^2).
R23 = np.sqrt(2.0 / 3.0)
ONE = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
W = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])     # weights of the scalar product
# The 6x6 tangents below are used by ci_fem.py on normal components only.


def dev(a):
    d = np.asarray(a, dtype=float).copy()
    d[:3] -= (d[0] + d[1] + d[2]) / 3.0
    return d


def norm(a):
    a = np.asarray(a, dtype=float)
    return np.sqrt(np.sum(W * a * a))


def IDEV():
    P = np.eye(6)
    P[:3, :3] -= 1.0 / 3.0
    return P


# ------------------------------------------------------------ hardening laws
class Linear:
    """R(alpha) = sY + K alpha."""
    def __init__(self, sY, K):
        self.sY, self.K = sY, K

    def value(self, a):
        return self.sY + self.K * a

    def slope(self, a):
        return self.K


class Voce:
    """R(alpha) = sinf - (sinf - s0) exp(-delta alpha)."""
    def __init__(self, s0, sinf, delta):
        self.s0, self.sinf, self.delta = s0, sinf, delta

    def value(self, a):
        return self.sinf - (self.sinf - self.s0) * np.exp(-self.delta * a)

    def slope(self, a):
        return self.delta * (self.sinf - self.s0) * np.exp(-self.delta * a)


# --------------------------------------------------------------- filament
def return_map_1d(eps, ep_n, al_n, q_n, E, sY, K, H):
    """Backward Euler (Box 5.1). Returns sigma, ep, alpha, q, dgamma."""
    s_tr = E * (eps - ep_n)                       # elastic predictor
    xi = s_tr - q_n
    f_tr = abs(xi) - (sY + K * al_n)
    if f_tr <= 0.0:
        return s_tr, ep_n, al_n, q_n, 0.0
    dg = f_tr / (E + K + H)                       # plastic corrector
    sg = np.sign(xi)
    return (s_tr - E * dg * sg, ep_n + dg * sg, al_n + dg, q_n + H * dg * sg, dg)


def forward_euler_1d(eps_n, deps, ep_n, al_n, q_n, E, sY, K, H):
    """Forward Euler with the continuum tangent: the rates are evaluated at t_n."""
    s = E * (eps_n - ep_n)
    xi = s - q_n
    f = abs(xi) - (sY + K * al_n)
    sg = np.sign(xi)
    dg = 0.0
    if f >= -1e-10 * sY and sg * deps > 0.0:      # plastic loading at t_n
        dg = E * sg * deps / (E + K + H)
    ep, al, q = ep_n + dg * sg, al_n + dg, q_n + H * dg * sg
    return E * (eps_n + deps - ep), ep, al, q, dg


# ------------------------------------------------------ J2 radial return
def radial_return(eps, ep_n, al_n, beta_n, mu, kappa, hard, H=0.0,
                  eta_dt=0.0, tol=1e-12, maxit=50):
    """Radial return (Box 5.2), linear kinematic hardening H, isotropic law
    `hard`; eta_dt = eta/dt > 0 gives the Perzyna update (Section 5.7).
    Returns sigma, ep, alpha, beta, dgamma, n, ||xi_trial||, local iterations."""
    s_tr = 2.0 * mu * dev(eps - ep_n)             # trial deviator
    xi_tr = s_tr - beta_n
    nx = norm(xi_tr)
    p = kappa * (eps[0] + eps[1] + eps[2])        # pressure: elastic
    f_tr = nx - R23 * hard.value(al_n)
    if f_tr <= 0.0:
        return s_tr + p * ONE, ep_n, al_n, beta_n, 0.0, None, nx, 0
    hp = 2.0 * mu + 2.0 / 3.0 * H + eta_dt
    if isinstance(hard, Linear):                  # closed form
        dg, it = f_tr / (hp + 2.0 / 3.0 * hard.K), 0
    else:                                         # scalar Newton
        dg, it = 0.0, 0
        for it in range(1, maxit + 1):
            g = nx - hp * dg - R23 * hard.value(al_n + R23 * dg)
            if abs(g) < tol * max(1.0, nx):
                break
            dg += g / (hp + 2.0 / 3.0 * hard.slope(al_n + R23 * dg))
    n = xi_tr / nx                                # n_{n+1} = n_trial
    return (s_tr - 2.0 * mu * dg * n + p * ONE, ep_n + dg * n,
            al_n + R23 * dg, beta_n + 2.0 / 3.0 * H * dg * n, dg, n, nx, it)


def C_elastic(mu, kappa):
    return kappa * np.outer(ONE, ONE) + 2.0 * mu * IDEV()


def C_algorithmic(dg, nx, n, mu, kappa, Kh, H=0.0, eta_dt=0.0):
    """Consistent tangent of Simo and Taylor (1985)."""
    theta = 1.0 - 2.0 * mu * dg / nx
    theta_bar = 1.0 / (1.0 + (Kh + H + 1.5 * eta_dt) / (3.0 * mu)) - (1.0 - theta)
    return (kappa * np.outer(ONE, ONE) + 2.0 * mu * theta * IDEV()
            - 2.0 * mu * theta_bar * np.outer(n, n))


def C_continuum(n, mu, kappa, Kh, H=0.0):
    """Continuum elastoplastic tangent C^ep (limit dgamma -> 0)."""
    return (kappa * np.outer(ONE, ONE) + 2.0 * mu * IDEV()
            - 2.0 * mu / (1.0 + (Kh + H) / (3.0 * mu)) * np.outer(n, n))


# ---------------------------------------------- plane stress (Box 5.5)
def plane_stress_return(eps, ep_n, al_n, E, nu, hard):
    """Projected return of Simo and Taylor (1986), isotropic hardening.
    Vectors [e11, e22, 2 e12] and [s11, s22, s12].
    Returns sigma, ep, alpha, dgamma."""
    mu = E / (2.0 * (1.0 + nu))
    C = E / (1 - nu**2) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])
    P = np.array([[2, -1, 0], [-1, 2, 0], [0, 0, 6]]) / 3.0
    s_tr = C @ (np.asarray(eps) - np.asarray(ep_n))

    def modes(dg):                                # decoupled in the eigenbasis of CP
        A = 1.0 + E * dg / (3.0 * (1 - nu))
        B = 1.0 + 2.0 * mu * dg
        return (s_tr[0] + s_tr[1]) / A, (s_tr[0] - s_tr[1]) / B, s_tr[2] / B

    def phi(dg):                                  # phi^2 = s^T P s = 2 J2
        S, D, T = modes(dg)
        return np.sqrt(S**2 / 6.0 + D**2 / 2.0 + 2.0 * T**2)

    def g(dg):                                    # scalar consistency condition
        return 0.5 * phi(dg)**2 - hard.value(al_n + R23 * dg * phi(dg))**2 / 3.0

    if g(0.0) <= 0.0:
        return s_tr, np.asarray(ep_n, float), al_n, 0.0
    lo, hi = 0.0, 1e-6
    while g(hi) > 0.0:
        hi *= 2.0
    for _ in range(80):                           # bisection: robust
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if g(mid) > 0.0 else (lo, mid)
    dg = 0.5 * (lo + hi)
    S, D, T = modes(dg)
    s = np.array([0.5 * (S + D), 0.5 * (S - D), T])
    return s, ep_n + dg * P @ s, al_n + R23 * dg * phi(dg), dg


# ------------------------------------------------- Tresca (Box 5.4)
def tresca_return(s_tr, mu, sY):
    """Perfect Tresca plasticity in principal stresses s1 >= s2 >= s3:
    active-set search over the main face and the two adjacent faces.
    Returns the principal stresses, the mode and the multipliers."""
    s1, s2, s3 = s_tr
    f1 = s1 - s3 - sY                             # main face
    if f1 <= 0.0:
        return np.array(s_tr, float), "elastic", np.zeros(2)
    dg = f1 / (4.0 * mu)                          # one-face return
    s = np.array([s1 - 2 * mu * dg, s2, s3 + 2 * mu * dg])
    if s[0] >= s[1] >= s[2]:
        return s, "face", np.array([dg, 0.0])
    upper = s[1] > s[0]                           # s2 > s1: corner s1 = s2
    f2 = (s2 - s3 - sY) if upper else (s1 - s2 - sY)
    g1, g2 = np.linalg.solve(2 * mu * np.array([[2.0, 1.0], [1.0, 2.0]]), [f1, f2])
    if upper:                                     # N1 = e1 - e3, N2 = e2 - e3
        s = np.array([s1 - 2 * mu * g1, s2 - 2 * mu * g2, s3 + 2 * mu * (g1 + g2)])
    else:                                         # N1 = e1 - e3, N2 = e1 - e2
        s = np.array([s1 - 2 * mu * (g1 + g2), s2 + 2 * mu * g2, s3 + 2 * mu * g1])
    return s, "corner", np.array([g1, g2])
