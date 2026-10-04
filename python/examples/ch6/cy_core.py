"""cy_core.py -- model problems and solvers of Chapter 6 (global--local methods).
Imported by the exercise and figure scripts of Chapter 6.
"""
import numpy as np

# Contents (stresses in MPa, strains dimensionless):
#   Structure          discrete structure made of uniaxial fibres: strains eps = B u,
#                      equilibrium B^T W sigma = F, fibre law of the filament with
#                      linear kinematic hardening (modulus H, H = 0: perfect plasticity)
#   three_bar_truss    three-bar truss of Maitournam (bars 1, 2 at 45 deg, bar 3)
#   bree_vessel        thin vessel wall in layers: uniform hoop strain, pressure
#                      stress sigma_P, linear temperature through the wall (Bree)
#   melan              projection P and residual-stress operator Z (eps = P ep + ...)
#   local_stage        return map along a history with the local direction G
#                      (G = E: total strain fixed, the return map of Chapter 5)
#   global_stage       linear admissible problem at every instant, direction h
#                      (h = 0: elastic problem with initial strain)
#   incremental        step by step, Newton (consistent tangent) or initial strain
#   global_local       iteration on the whole history (direct cyclic method when
#                      closure = "periodic", G = E, h = 0)
#   dcm_sweep_map      one direct cyclic iteration as a map on histories
#   latin              LATIN with conjugate directions on the rates, relaxation mu
#   cycle, period_map  one cycle computed incrementally, the map z(0) -> z(T)
#   fixed_point        Picard, Krasnoselskii--Mann or Anderson on the period map
#   zarka              Zarka's estimate of the elastic shakedown state
# The internal variable of a fibre is its plastic strain ep; the back stress is
# q = H ep (the hardening variable of Chapter 5, filament with K = 0).
#
# Where in the book. The three steps eps = B u, sigma = E (eps - ep - eth),
# B^T W sigma = F are (6.2) of Section 6.1 (Strang's A^T C A). The solvers follow
# the boxes of the chapter: Box 6.1 (initial strain, step by step), Box 6.2 (whole
# history; periodic closure = direct cyclic method, Section 6.6), Box 6.3 (LATIN,
# Section 6.4), Box 6.5 (Zarka, Section 6.7); the period map is Section 6.5.
# Histories are arrays of shape (N+1, nf): one row per instant t_0..t_N, one
# column per fibre.


class Structure:
    """Uniaxial fibres i = 1..nf, nodal unknowns u (nd), eps = B u.

    w[i] is the volume (weight) of fibre i, so that the internal virtual work is
    sum_i w_i sigma_i deps_i and equilibrium reads B^T W sigma = F(t).

    Arguments: B (nf x nd), w (nf), E, sY, H (scalars or one value per fibre,
    in MPa), F(t) the nodal loads, eth(t) the thermal strain of each fibre."""

    def __init__(self, B, w, E, sY, H, F, eth, name=""):
        self.B = np.atleast_2d(np.asarray(B, float))
        self.w = np.asarray(w, float)
        nf = self.B.shape[0]
        # E (modulus), sY (yield stress), H (kinematic hardening): one value per fibre
        self.E = np.full(nf, float(E)) if np.isscalar(E) else np.asarray(E, float)
        self.sY = np.full(nf, float(sY)) if np.isscalar(sY) else np.asarray(sY, float)
        self.H = np.full(nf, float(H)) if np.isscalar(H) else np.asarray(H, float)
        self.F, self.eth, self.name = F, eth, name        # functions of the time t
        self.nf, self.nd = self.B.shape

    def stiffness(self, Et):
        """B^T W diag(Et) B for a vector of fibre moduli Et: with Et = E it is
        K = B^T W E B of (6.3); with the tangent moduli, the Newton matrix."""
        return self.B.T @ ((self.w * Et)[:, None] * self.B)

    def solve_elastic(self, t, ep):
        """Elastic problem with initial strain ep + eth at time t: returns u, eps, sig.
        K u = F + B^T W E (ep + eth): the plastic and thermal strains act as loads."""
        Kel = self.stiffness(self.E)
        rhs = self.F(t) + self.B.T @ (self.w * self.E * (ep + self.eth(t)))
        u = np.linalg.solve(Kel, rhs)
        eps = self.B @ u
        return u, eps, self.E * (eps - ep - self.eth(t))


def three_bar_truss(E=200e3, sY=200.0, H=0.0, Q=lambda t: np.zeros(2), l=1.0):
    """Maitournam's truss: node O, bar 3 along e1 (length l), bars 1 and 2 at
    +-45 deg (length l sqrt2), unit cross-section; Q(t) = (Q1, Q2) in N for a
    section of 1 mm^2, i.e. directly the normal forces in units of sY.
    Section 6.1.1 and Figure 6.2: E = 200 GPa, sY = 200 MPa by default, so that
    N0 = S sY = 200 N. Rows of B are t_i / l_i, the volumes W are the lengths."""
    r = 1.0 / np.sqrt(2.0)
    t = np.array([[r, r], [r, -r], [1.0, 0.0]])        # unit vectors towards O
    L = np.array([l * np.sqrt(2.0), l * np.sqrt(2.0), l])
    B = t / L[:, None]                                   # eps_i = t_i . u / l_i
    return Structure(B, L, E, sY, H, Q, lambda t_: np.zeros(3), "truss")


def bree_vessel(sP, sT, lam, E=200e3 / 0.7, sY=280.0, H=0.0, nlay=200):
    """Wall of a thin vessel (Maitournam, Sec. 4.5), modelled by nlay layers in the
    thickness, x in (-1/2, 1/2) (x = (r - r_m)/e). Uniform hoop strain eps (one
    unknown), mean hoop stress sigma_P, temperature tau = -tau0 x lam(t) so that
    the elastic stress is sigma_P + 2 sigma_T x lam(t), sigma_T = E alpha tau0 / 2.
    E is the plane modulus E/(1 - nu).
    Section 6.1.2, (6.8)-(6.9): sP = sigma_P = X sY, sT = sigma_T = Y sY (MPa);
    defaults are the sphere of the book, E = 200 GPa, nu = 0.3, sY = 280 MPa."""
    x = (np.arange(nlay) + 0.5) / nlay - 0.5            # mid-points of the layers
    w = np.full(nlay, 1.0 / nlay)                        # layer thicknesses (sum 1)
    B = np.ones((nlay, 1))                               # same hoop strain everywhere
    # thermal strain chosen so that the elastic stress is sigma_P + 2 sigma_T x lam
    eth = lambda t: -(2.0 * sT / E) * x * lam(t)
    S = Structure(B, w, E, sY, H, lambda t: np.array([sP]), eth, "bree")
    S.x = x
    return S


def melan(S):
    """Matrices P and Z: for an initial strain ep at zero load,
    eps = P ep and the residual stress is rho = -Z ep.
    Melan decomposition (Section 6.2.2, (6.7) for the truss):
    P = B K^-1 B^T W E keeps the compatible part of ep, Z = E (I - P)."""
    Kel = S.stiffness(S.E)
    P = S.B @ np.linalg.solve(Kel, S.B.T * (S.w * S.E))
    Z = S.E[:, None] * (np.eye(S.nf) - P)
    return P, Z


# ------------------------------------------------------------ local stage
def return_map(sig_ad, ep_ad, ep_prev, G, S):
    """Return map of the filament with local search direction G (vectors over the
    fibres). Finds (sig, ep) with ep - ep_ad = -(sig - sig_ad)/G and the backward
    Euler flow rule from ep_prev. With G = E and sig_ad = E(eps - ep_ad - eth) this
    is Box 5.1 at fixed total strain.
    The search directions are those of (6.14), Section 6.4.1; LATIN uses G = 1/h.
    Returns the stress and the plastic strain at the end of the step."""
    s_tr = sig_ad - G * (ep_prev - ep_ad)               # elastic predictor
    # trial value of the yield function |sigma - q| - sY, back stress q = H ep
    xi = s_tr - S.H * ep_prev
    f_tr = np.abs(xi) - S.sY
    # plastic multiplier Delta gamma = <f_trial> / (G + H) (Box 5.1, step 4)
    dg = np.where(f_tr > 0.0, f_tr / (G + S.H), 0.0)    # plastic corrector
    sg = np.sign(xi)
    return s_tr - G * dg * sg, ep_prev + dg * sg


def local_stage(sig_ad, ep_ad, ep0, G, S):
    """Sweep in time: histories sig_ad, ep_ad of shape (N+1, nf) on Ad; ep0 is the
    plastic strain at t_0 (initial state or periodic closure). Returns the
    histories (sig_hat, ep_hat) on Gamma, ep_hat[0] = ep0.
    Box 6.2, step 3: each fibre integrates its own flow rule along the history."""
    sig_h, ep_h = np.empty_like(sig_ad), np.empty_like(ep_ad)
    ep_h[0] = ep0
    sig_h[0] = sig_ad[0] - G * (ep0 - ep_ad[0])
    for n in range(1, len(sig_ad)):
        # the step n starts from the plastic strain just computed at t_{n-1}
        sig_h[n], ep_h[n] = return_map(sig_ad[n], ep_ad[n], ep_h[n - 1], G, S)
    return sig_h, ep_h


# ----------------------------------------------------------- global stage
def global_stage(times, sig_h, ep_h, h, S):
    """Admissible fields (u, eps, sig, ep) at every instant with the global search
    direction ep - ep_hat = h (sig - sig_hat); h = 0: elastic problem with initial
    strain ep_hat. The instants are independent; one matrix for all of them.
    Box 6.2, step 2 (h = 0). With h > 0 the fibres have the modulus E/(1 + E h).
    Returns the histories eps, sig, ep of shape (N+1, nf)."""
    Eh = S.E / (1.0 + S.E * h)
    Kh = S.stiffness(Eh)                     # assembled once for all the instants
    out_eps, out_sig, out_ep = [], [], []
    for n, t in enumerate(times):
        # K_h u = F + B^T W E_h (ep_hat - h sig_hat + eth): initial strain as a load
        rhs = S.F(t) + S.B.T @ (S.w * Eh * (ep_h[n] - h * sig_h[n] + S.eth(t)))
        u = np.linalg.solve(Kh, rhs)
        eps = S.B @ u
        sig = Eh * (eps - ep_h[n] + h * sig_h[n] - S.eth(t))
        out_eps.append(eps)
        out_sig.append(sig)
        out_ep.append(ep_h[n] + h * (sig - sig_h[n]))
    return np.array(out_eps), np.array(out_sig), np.array(out_ep)


def energy_gap(sig, sig_h, S):
    """Error indicator: distance between the admissible and the constitutive stress
    histories in the complementary energy norm, relative to the latter.
    It is the test of Box 6.2 (step 5) and the indicator of Box 6.3 (step 5):
    sqrt( sum w (sig - sig_hat)^2 / E ) / sqrt( sum w sig_hat^2 / E )."""
    num = np.sum(S.w * (sig - sig_h) ** 2 / S.E)
    den = np.sum(S.w * sig_h ** 2 / S.E) + 1e-300
    return np.sqrt(num / den)


# --------------------------------------------------------- incremental
def incremental(S, times, ep0, method="newton", tol=1e-10, kmax=500, record=False):
    """Step-by-step solution (Box 5.3 with the consistent tangent, or the
    initial-strain iteration with the elastic stiffness). Returns the histories
    eps, sig, ep (N+1, nf), the number of iterations per step and, if record,
    the list of iterates of ep at each step.
    method = "newton": Box 5.3; any other value: Box 6.1 (initial strain).
    The loops are "for n, for k" (Figure 6.5a): each step is converged before the
    next one starts. One iteration = one local evaluation (return map)."""
    N = len(times) - 1
    eps_h = np.zeros((N + 1, S.nf))
    sig_h = np.zeros((N + 1, S.nf))
    ep_hist = np.zeros((N + 1, S.nf))
    ep_hist[0] = ep0
    u0, eps_h[0], sig_h[0] = S.solve_elastic(times[0], ep0)
    u = u0
    iters, rec = [], []
    Kel = S.stiffness(S.E)
    for n in range(1, N + 1):                   # Box 6.1, step 1: loop on the steps
        t = times[n]
        Fext = S.F(t)
        steps = []
        for k in range(kmax):
            # local stage: return map at fixed strain from ep(t_{n-1}) (Box 5.1)
            eps = S.B @ u
            sig_tr = S.E * (eps - ep_hist[n - 1] - S.eth(t))
            sig, ep = return_map(sig_tr, ep_hist[n - 1], ep_hist[n - 1], S.E, S)
            steps.append(ep.copy())
            # residual of equilibrium R = F - B^T W sigma, and the test (step 4)
            R = Fext - S.B.T @ (S.w * sig)
            if np.linalg.norm(R) <= tol * max(np.linalg.norm(Fext), S.sY.max()):
                break
            # global stage: correction of the displacement, delta u = K^-1 R
            if method == "newton":
                # consistent tangent of the filament: E H / (E + H) if plastic, else E
                plastic = np.abs(ep - ep_hist[n - 1]) > 0
                Et = np.where(plastic, S.E * S.H / (S.E + S.H), S.E)
                # tiny shift: the tangent is singular when every bar flows with H = 0
                Kt = S.stiffness(Et) + 1e-12 * np.eye(S.nd) * Kel.max()
                u = u + np.linalg.solve(Kt, R)
            else:
                # elastic stiffness, the same at every iteration (Box 6.1, step 3)
                u = u + np.linalg.solve(Kel, R)
        iters.append(k + 1)
        if record:
            rec.append(np.array(steps))
        eps_h[n], sig_h[n], ep_hist[n] = eps, sig, ep
    return eps_h, sig_h, ep_hist, np.array(iters), rec


def cycle(S, times, ep0, method="newton", tol=1e-10):
    """One period computed incrementally from ep0. Returns ep(T) and the histories."""
    eps, sig, ep, it, _ = incremental(S, times, ep0, method, tol)
    return ep[-1], eps, sig, ep, it


def period_map(S, times, method="newton"):
    """The map Pi: ep(0) -> ep(T) over one period.
    Section 6.5.2 ("The period map"): its fixed points are the periodic states,
    and iterating it is the cycle-by-cycle computation (Picard)."""
    return lambda z: cycle(S, times, z, method)[0]


def fixed_point(Pi, z0, scheme="picard", theta=0.5, m=3, kmax=200, tol=1e-10):
    """Iterations on the period map: Picard z <- Pi z, Krasnoselskii--Mann
    z <- (1-theta) z + theta Pi z, or Anderson(m). Returns the iterates and the
    residuals |Pi z - z| (max norm).
    The three schemes of Appendix E; Anderson is Algorithm 6.1 (Exercise 6.10).
    Pi may be the period map or one direct cyclic sweep (dcm_sweep_map)."""
    z = np.array(z0, float)
    Zs, res = [z.copy()], []
    Gs, Fs = [], []
    for k in range(kmax):
        g = Pi(z)                                     # one cycle (or one sweep)
        f = g - z                                     # residual f_k = g_k - z_k
        res.append(np.max(np.abs(f)))
        if res[-1] < tol:
            break
        if scheme == "picard":
            z = g
        elif scheme == "km":
            z = (1 - theta) * z + theta * g           # averaged iteration
        else:                                         # Anderson, type II
            # keep the last m+1 values of g and f (depth m)
            Gs.append(g)
            Fs.append(f)
            Gs, Fs = Gs[-(m + 1):], Fs[-(m + 1):]
            if len(Fs) == 1:
                z = g
            else:
                # differences Delta F, Delta G; gamma = argmin |f - Delta F gamma|
                dF = np.array([Fs[i + 1] - Fs[i] for i in range(len(Fs) - 1)]).T
                dG = np.array([Gs[i + 1] - Gs[i] for i in range(len(Gs) - 1)]).T
                gam = np.linalg.lstsq(dF, f, rcond=None)[0]
                z = g - dG @ gam
        Zs.append(z.copy())
    return np.array(Zs), np.array(res)


# ------------------------------------------------ iteration on the history
def global_local(S, times, ep_init=None, G=None, h=0.0, closure="initial", ep0=None,
                 mu=0.0, kmax=300, tol=1e-8, record=False):
    """Alternate the local stage (sweep in time) and the global stage (all instants)
    on the whole history. closure = "initial": ep(t_0) = ep0 (LATIN-type window);
    closure = "periodic": ep(t_0) <- ep(t_N) (direct cyclic method). G = E and
    h = 0 give the elastic predictor / plastic corrector on the whole history;
    G = 1/h gives conjugate directions. mu: relaxation of the global stage.
    Returns a dict with the final histories, the error indicators and, if
    record, the iterates of ep (after each local stage).
    Box 6.2; the loops are "for k, for n" (Figure 6.5b). With the periodic closure
    it is the direct cyclic method of Section 6.6.1."""
    N = len(times) - 1
    G = S.E if G is None else G
    ep0 = np.zeros(S.nf) if ep0 is None else np.array(ep0, float)
    # Box 6.2, step 1: initial guess of the plastic strain history (zero by default)
    ep_ad = np.zeros((N + 1, S.nf)) if ep_init is None else np.array(ep_init, float)
    eps_ad, sig_ad, _ = global_stage(times, np.zeros_like(ep_ad), ep_ad, 0.0, S)
    start = ep0.copy()
    errs, rec_ep, rec_sig = [], [], []
    for k in range(kmax):
        # local stage (step 3): sweep t_0 -> t_N from the closure state
        sig_h, ep_h = local_stage(sig_ad, ep_ad, start, G, S)
        if record:
            rec_ep.append(ep_h.copy())
            rec_sig.append(sig_h.copy())
        if closure == "periodic":                     # closure (step 4)
            start = ep_h[-1].copy()                    # z(t_0) <- z(t_N)
        # global stage (step 2 of the next iteration), with an optional relaxation
        eps_n, sig_n, ep_n = global_stage(times, sig_h, ep_h, h, S)
        if mu > 0.0:
            sig_n = (1 - mu) * sig_n + mu * sig_ad
            ep_n = (1 - mu) * ep_n + mu * ep_ad
            eps_n = (1 - mu) * eps_n + mu * eps_ad
        # test (step 5): stresses of the two stages agree, and periodicity if asked
        errs.append(energy_gap(sig_n, sig_h, S))
        gap_closure = (np.max(np.abs(ep_h[-1] - ep_h[0])) if closure == "periodic" else 0.0)
        eps_ad, sig_ad, ep_ad = eps_n, sig_n, ep_n
        if errs[-1] < tol and gap_closure < tol * 10:
            break
    return dict(eps=eps_ad, sig=sig_h, ep=ep_h, err=np.array(errs), iters=k + 1,
                rec_ep=rec_ep, rec_sig=rec_sig)


def dcm_sweep_map(S, times):
    """One iteration of the direct cyclic method as a map on the plastic strain
    history (flattened, shape (N+1)*nf): global stage at all instants, then local
    sweep started from the last instant (periodic closure). Its fixed points are
    the periodic solutions; it can be accelerated with fixed_point.
    Used for "direct cyclic + Anderson" (Section 6.6.1, Exercise 6.10)."""
    N = len(times) - 1

    def M(flat):
        ep = flat.reshape(N + 1, S.nf)
        _, sig, _ = global_stage(times, np.zeros_like(ep), ep, 0.0, S)   # step 2
        _, ep_h = local_stage(sig, ep, ep[-1], S.E, S)                   # steps 3-4
        return ep_h.ravel()
    return M


# ------------------------------------------- LATIN (directions on the rates)
def latin(S, times, h, mu=0.2, ep0=None, closure="initial", kmax=300, tol=1e-8,
          record=False):
    """LATIN with conjugate directions written on the plastic strain increments
    (rates times dt): local  d(ep_hat) - d(ep) = -h (sig_hat - sig),
                      global d(ep) - d(ep_hat) =  h (sig - sig_hat),
    h > 0 a scalar (or a vector over the fibres). The local stage is a return map
    with stiffness 1/h at each instant; the global stage is a linear problem with
    the stiffness E/(1 + E h) marched in time (Maxwell-type: it couples the
    instants). mu is the relaxation of Ladeveze's theorem. closure = "periodic"
    imposes ep(t_0) = ep(t_N) with a lag of one iteration.
    Box 6.3 and (6.15), Section 6.4.2; the convergence statement is the box
    "Convergence of the LATIN method" (mu > 0 needed in statics)."""
    N = len(times) - 1
    ep0 = np.zeros(S.nf) if ep0 is None else np.array(ep0, float)
    Eh = S.E / (1.0 + S.E * h)
    Kh = S.stiffness(Eh)
    # Box 6.3, step 1: elastic solution at every instant, ep = ep0
    ep = np.tile(ep0, (N + 1, 1))
    sig = np.array([S.solve_elastic(t, ep0)[2] for t in times])
    G = 1.0 / h
    errs, rec_ep = [], []
    start = ep0.copy()
    for k in range(kmax):
        # local stage: sweep in time (hardening couples the instants)
        # Box 6.3, step 2: trial stress sigma_k + Delta ep_k / h, stiffness 1/h
        sig_h, ep_h = np.empty_like(sig), np.empty_like(ep)
        ep_h[0], sig_h[0] = start, sig[0]
        for n in range(1, N + 1):
            dep = ep[n] - ep[n - 1]
            sig_h[n], ep_h[n] = return_map(sig[n] + G * dep, ep_h[n - 1], ep_h[n - 1], G, S)
        if record:
            rec_ep.append(ep_h.copy())
        # global stage: linear, marched in time
        # Box 6.3, step 3: modulus E/(1 + E h), initial strain
        # a = ep(t_{n-1}) + Delta ep_hat(t_n) - h sig_hat(t_n)
        sig_n, ep_n = np.empty_like(sig), np.empty_like(ep)
        ep_n[0] = start
        sig_n[0] = S.solve_elastic(times[0], start)[2]
        for n in range(1, N + 1):
            t = times[n]
            dep_h = ep_h[n] - ep_h[n - 1]
            a = ep_n[n - 1] + dep_h - h * sig_h[n] + S.eth(t)
            u = np.linalg.solve(Kh, S.F(t) + S.B.T @ (S.w * Eh * a))
            sig_n[n] = Eh * (S.B @ u - a)
            ep_n[n] = ep_n[n - 1] + dep_h + h * (sig_n[n] - sig_h[n])
        # Box 6.3, step 5: indicator, measured before the relaxation
        errs.append(energy_gap(sig_n, sig_h, S))
        # Box 6.3, step 4: relaxation s_{k+1} = (1 - mu) s_bar + mu s_k
        sig = (1 - mu) * sig_n + mu * sig
        ep = (1 - mu) * ep_n + mu * ep
        if closure == "periodic":
            start = ep[-1].copy()
        if errs[-1] < tol:
            break
    return dict(sig=sig_h, ep=ep_h, err=np.array(errs), iters=k + 1, rec_ep=rec_ep)


# ----------------------------------------------------------------- Zarka
def zarka(S, sig_el_hist, Y1):
    """Zarka's estimate of the elastic shakedown state for linear kinematic
    hardening (H > 0). Transformed parameter Y = H ep - rho. Elastic shakedown at a
    fibre iff the intervals [sig_el(t) - sY, sig_el(t) + sY] intersect; Y is then
    projected onto the intersection, and ep, rho follow from (H I + Z) ep = Y.
    Returns ep, rho, Y and a flag per fibre (True: elastic shakedown possible).
    Box 6.5, Section 6.7; Y = (H I + Z) ep is (6.17). sig_el_hist is the elastic
    stress over one cycle (N+1, nf), Y1 the value after the first half-cycle."""
    _, Z = melan(S)
    # step 2: admissible set [max_t sig_el - sY, min_t sig_el + sY] at each fibre
    lo = sig_el_hist.max(axis=0) - S.sY
    hi = sig_el_hist.min(axis=0) + S.sY
    ok = lo <= hi
    # step 4: projection of Y1 onto the interval (its middle if the set is empty)
    Y = np.where(ok, np.clip(Y1, lo, hi), 0.5 * (lo + hi))
    # step 5: one linear problem for the plastic strain, then rho = -Z ep
    ep = np.linalg.solve(np.diag(S.H) + Z, Y)
    rho = -Z @ ep
    return ep, rho, Y, ok
