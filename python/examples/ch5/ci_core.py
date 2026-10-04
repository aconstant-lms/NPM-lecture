"""ci_core.py -- return maps of Chapter 5 (plasticity: integration algorithms).
Imported by the exercise and figure scripts of Chapter 5.
"""
import numpy as np

# Contents (one function per box of the chapter; stresses in MPa):
#   Linear, Voce            isotropic hardening laws R(alpha) and R'(alpha)
#   return_map_1d           filament, combined linear hardening (Box 5.1)
#   forward_euler_1d        filament, explicit update with the continuum tangent
#   radial_return           J2, isotropic + linear kinematic hardening (Box 5.2),
#                           Perzyna viscoplasticity when eta_dt = eta/dt > 0
#   C_elastic, C_algorithmic, C_continuum
#                           elastic, consistent (5.14) and continuum tangents
#   plane_stress_return     projected return in plane stress (Box 5.5)
#   tresca_return           Tresca with faces and corners (Box 5.4)
# Every return map is a function of the strain at t_{n+1} and of the internal
# variables at t_n only; it returns the stress and the updated variables.
# This is the map (sigma, z)_{n+1} = R(eps_{n+1}; z_n) of equation (5.3).
#
# Voigt convention (6 components), TENSOR shear strains:
#     a = [a11, a22, a33, a12, a13, a23],
# so that ||a||^2 = a11^2 + a22^2 + a33^2 + 2 (a12^2 + a13^2 + a23^2).
R23 = np.sqrt(2.0 / 3.0)                         # the factor sqrt(2/3) of J2 plasticity
ONE = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])   # identity tensor I
W = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])     # weights of the scalar product
# The 6x6 tangents below are used by ci_fem.py on normal components only.


def dev(a):
    """Deviatoric part of a Voigt vector."""
    d = np.asarray(a, dtype=float).copy()
    d[:3] -= (d[0] + d[1] + d[2]) / 3.0
    return d


def norm(a):
    """Tensor norm sqrt(a:a) of a Voigt vector (shear terms counted twice)."""
    a = np.asarray(a, dtype=float)
    return np.sqrt(np.sum(W * a * a))


def IDEV():
    """Deviatoric projector I_dev as a 6x6 matrix."""
    P = np.eye(6)
    P[:3, :3] -= 1.0 / 3.0
    return P


# ------------------------------------------------------------ hardening laws
# Isotropic hardening laws R(alpha) of Section 5.4: the radius of the yield
# surface is sqrt(2/3) R(alpha). value() gives R, slope() gives R'(alpha).
class Linear:
    """R(alpha) = sY + K alpha."""
    def __init__(self, sY, K):
        self.sY, self.K = sY, K                   # yield stress, hardening modulus (MPa)

    def value(self, a):
        return self.sY + self.K * a

    def slope(self, a):
        return self.K


class Voce:
    """R(alpha) = sinf - (sinf - s0) exp(-delta alpha)."""
    def __init__(self, s0, sinf, delta):
        # initial and saturation yield stresses (MPa), saturation rate delta (-)
        self.s0, self.sinf, self.delta = s0, sinf, delta

    def value(self, a):
        return self.sinf - (self.sinf - self.s0) * np.exp(-self.delta * a)

    def slope(self, a):
        return self.delta * (self.sinf - self.s0) * np.exp(-self.delta * a)


# --------------------------------------------------------------- filament
def return_map_1d(eps, ep_n, al_n, q_n, E, sY, K, H):
    """Backward Euler (Box 5.1). Returns sigma, ep, alpha, q, dgamma.
    eps: strain at t_{n+1}; ep_n, al_n, q_n: plastic strain, accumulated plastic
    strain and back stress at t_n; E, sY, K, H in MPa (K isotropic, H kinematic)."""
    # Box 5.1 step 2: trial state, internal variables frozen at t_n
    s_tr = E * (eps - ep_n)                       # elastic predictor
    xi = s_tr - q_n                               # relative stress xi = sigma - q
    f_tr = abs(xi) - (sY + K * al_n)
    # Box 5.1 step 3: check; the trial state is admissible, the step is elastic
    if f_tr <= 0.0:
        return s_tr, ep_n, al_n, q_n, 0.0
    # Box 5.1 step 4: Delta gamma = f_trial / (E + K + H), equation (5.7)
    dg = f_tr / (E + K + H)                       # plastic corrector
    sg = np.sign(xi)                              # s = sign(xi_trial), kept by corrector
    return (s_tr - E * dg * sg, ep_n + dg * sg, al_n + dg, q_n + H * dg * sg, dg)


def forward_euler_1d(eps_n, deps, ep_n, al_n, q_n, E, sY, K, H):
    """Forward Euler with the continuum tangent: the rates are evaluated at t_n.
    Section 5.2, "Forward Euler" (Figure 5.2). No return to the yield surface:
    the stress may drift outside it. Same arguments as return_map_1d, with the
    strain eps_n at t_n and its increment deps."""
    s = E * (eps_n - ep_n)
    xi = s - q_n
    f = abs(xi) - (sY + K * al_n)
    sg = np.sign(xi)
    dg = 0.0
    # Plastic multiplier of the rate problem, (4.14), evaluated at t_n:
    # gamma dt = E sign(xi) deps / (E + K + H) if on the surface and loading
    if f >= -1e-10 * sY and sg * deps > 0.0:      # plastic loading at t_n
        dg = E * sg * deps / (E + K + H)
    ep, al, q = ep_n + dg * sg, al_n + dg, q_n + H * dg * sg
    return E * (eps_n + deps - ep), ep, al, q, dg


# ------------------------------------------------------ J2 radial return
def radial_return(eps, ep_n, al_n, beta_n, mu, kappa, hard, H=0.0,
                  eta_dt=0.0, tol=1e-12, maxit=50):
    """Radial return (Box 5.2), linear kinematic hardening H, isotropic law
    `hard`; eta_dt = eta/dt > 0 gives the Perzyna update (Section 5.8).
    Returns sigma, ep, alpha, beta, dgamma, n, ||xi_trial||, local iterations."""
    # Arguments: eps (Voigt) at t_{n+1}; ep_n, al_n, beta_n = eps^p, alpha and
    # back stress beta at t_n; mu, kappa shear and bulk moduli (MPa); hard a law
    # Linear or Voce; H kinematic modulus (MPa).
    # Box 5.2 step 2: elastic predictor
    s_tr = 2.0 * mu * dev(eps - ep_n)             # trial deviator
    xi_tr = s_tr - beta_n                         # relative stress (eta in the book)
    nx = norm(xi_tr)
    p = kappa * (eps[0] + eps[1] + eps[2])        # pressure: elastic
    f_tr = nx - R23 * hard.value(al_n)
    # Box 5.2 step 3: elastic step, internal variables unchanged
    if f_tr <= 0.0:
        return s_tr + p * ONE, ep_n, al_n, beta_n, 0.0, None, nx, 0
    # Box 5.2 step 4: solve g(Delta gamma) = 0, equation (5.13);
    # with Perzyna, eta/dt is added to the slope, equation (5.19)
    hp = 2.0 * mu + 2.0 / 3.0 * H + eta_dt
    if isinstance(hard, Linear):                  # closed form
        dg, it = f_tr / (hp + 2.0 / 3.0 * hard.K), 0
    else:                                         # scalar Newton
        dg, it = 0.0, 0
        for it in range(1, maxit + 1):
            # g = ||xi_trial|| - (2 mu + 2/3 H) dg - sqrt(2/3) R(alpha_n + sqrt(2/3) dg)
            g = nx - hp * dg - R23 * hard.value(al_n + R23 * dg)
            if abs(g) < tol * max(1.0, nx):
                break
            dg += g / (hp + 2.0 / 3.0 * hard.slope(al_n + R23 * dg))
    # Box 5.2 step 5: update along the trial normal, equation (5.12)
    n = xi_tr / nx                                # n_{n+1} = n_trial
    return (s_tr - 2.0 * mu * dg * n + p * ONE, ep_n + dg * n,
            al_n + R23 * dg, beta_n + 2.0 / 3.0 * H * dg * n, dg, n, nx, it)


def C_elastic(mu, kappa):
    """Isotropic elasticity kappa I x I + 2 mu I_dev."""
    return kappa * np.outer(ONE, ONE) + 2.0 * mu * IDEV()


def C_algorithmic(dg, nx, n, mu, kappa, Kh, H=0.0, eta_dt=0.0):
    """Consistent tangent (5.14) of Simo and Taylor (1985), from the output
    (dg, ||xi_trial||, n) of radial_return; Kh = R'(alpha_{n+1})."""
    # Box 5.2 step 6 (Section 5.4.1, "Consistent tangent of the radial return"):
    # C_alg = kappa I x I + 2 mu theta I_dev - 2 mu theta_bar n x n, with
    # theta = 1 - 2 mu dg / ||xi_trial||; Perzyna adds 3/2 eta/dt to R' + H.
    theta = 1.0 - 2.0 * mu * dg / nx
    theta_bar = 1.0 / (1.0 + (Kh + H + 1.5 * eta_dt) / (3.0 * mu)) - (1.0 - theta)
    return (kappa * np.outer(ONE, ONE) + 2.0 * mu * theta * IDEV()
            - 2.0 * mu * theta_bar * np.outer(n, n))


def C_continuum(n, mu, kappa, Kh, H=0.0):
    """Continuum elastoplastic tangent C^ep (limit dgamma -> 0)."""
    # Equation (4.38) of Chapter 4: C_alg with theta = 1; it linearizes the
    # rate equations, not the discrete step (Section 5.5, Figure 5.6).
    return (kappa * np.outer(ONE, ONE) + 2.0 * mu * IDEV()
            - 2.0 * mu / (1.0 + (Kh + H) / (3.0 * mu)) * np.outer(n, n))


# ---------------------------------------------- plane stress (Box 5.5)
def plane_stress_return(eps, ep_n, al_n, E, nu, hard):
    """Projected return of Simo and Taylor (1986), isotropic hardening.
    Vectors [e11, e22, 2 e12] and [s11, s22, s12].
    Returns sigma, ep, alpha, dgamma."""
    # Section 5.7: plane-stress moduli C and von Mises matrix P, s^T P s = 2 J2
    mu = E / (2.0 * (1.0 + nu))
    C = E / (1 - nu**2) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])
    P = np.array([[2, -1, 0], [-1, 2, 0], [0, 0, 6]]) / 3.0
    # Box 5.5 step 1: predictor
    s_tr = C @ (np.asarray(eps) - np.asarray(ep_n))

    # Box 5.5 step 2: sigma = (I + dg C P)^-1 sigma_trial is diagonal in the common
    # eigenbasis of C and P: s11 + s22 is divided by A, (s11 - s22, s12) by B
    def modes(dg):                                # decoupled in the eigenbasis of CP
        A = 1.0 + E * dg / (3.0 * (1 - nu))
        B = 1.0 + 2.0 * mu * dg
        return (s_tr[0] + s_tr[1]) / A, (s_tr[0] - s_tr[1]) / B, s_tr[2] / B

    def phi(dg):                                  # phi^2 = s^T P s = 2 J2
        S, D, T = modes(dg)
        return np.sqrt(S**2 / 6.0 + D**2 / 2.0 + 2.0 * T**2)

    # Box 5.5 step 3: 1/2 phi^2 - 1/3 R^2(alpha_n + sqrt(2/3) dg phi) = 0
    def g(dg):                                    # scalar consistency condition
        return 0.5 * phi(dg)**2 - hard.value(al_n + R23 * dg * phi(dg))**2 / 3.0

    # Box 5.5 step 1: elastic step if g(0) <= 0
    if g(0.0) <= 0.0:
        return s_tr, np.asarray(ep_n, float), al_n, 0.0
    # bracket the root of g (g(0) > 0), doubling the upper bound
    lo, hi = 0.0, 1e-6
    while g(hi) > 0.0:
        hi *= 2.0
    for _ in range(80):                           # bisection: robust
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if g(mid) > 0.0 else (lo, mid)
    dg = 0.5 * (lo + hi)
    # Box 5.5 step 4: update; back from (S, D, T) to (s11, s22, s12),
    # eps^p_{n+1} = eps^p_n + dg P sigma_{n+1}
    S, D, T = modes(dg)
    s = np.array([0.5 * (S + D), 0.5 * (S - D), T])
    return s, ep_n + dg * P @ s, al_n + R23 * dg * phi(dg), dg


# ------------------------------------------------- Tresca (Box 5.4)
def tresca_return(s_tr, mu, sY):
    """Perfect Tresca plasticity in principal stresses s1 >= s2 >= s3:
    active-set search over the main face and the two adjacent faces.
    Returns the principal stresses, the mode and the multipliers."""
    # s_tr: ordered principal trial stresses (Box 5.4 step 1 is done by the
    # caller). The plastic strain update of step 5 is left to the caller.
    s1, s2, s3 = s_tr
    f1 = s1 - s3 - sY                             # main face
    if f1 <= 0.0:
        return np.array(s_tr, float), "elastic", np.zeros(2)
    # Box 5.4 step 2: face return, N_a : C : N_a = 4 mu
    dg = f1 / (4.0 * mu)                          # one-face return
    s = np.array([s1 - 2 * mu * dg, s2, s3 + 2 * mu * dg])
    if s[0] >= s[1] >= s[2]:
        return s, "face", np.array([dg, 0.0])
    # Box 5.4 steps 3-4: corner; the order is violated, two faces are active.
    # 2x2 system 2 mu [[2, 1], [1, 2]] (dg_a, dg_b) = (f_a, f_b), since
    # N_a : C : N_b = 2 mu for adjacent faces: dg_a = (2 f_a - f_b) / (6 mu)
    upper = s[1] > s[0]                           # s2 > s1: corner s1 = s2
    f2 = (s2 - s3 - sY) if upper else (s1 - s2 - sY)
    g1, g2 = np.linalg.solve(2 * mu * np.array([[2.0, 1.0], [1.0, 2.0]]), [f1, f2])
    if upper:                                     # N1 = e1 - e3, N2 = e2 - e3
        s = np.array([s1 - 2 * mu * g1, s2 - 2 * mu * g2, s3 + 2 * mu * (g1 + g2)])
    else:                                         # N1 = e1 - e3, N2 = e1 - e2
        s = np.array([s1 - 2 * mu * (g1 + g2), s2 + 2 * mu * g2, s3 + 2 * mu * g1])
    return s, "corner", np.array([g1, g2])
