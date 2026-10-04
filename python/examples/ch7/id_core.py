"""id_core.py -- model problems, sensitivities and adjoints of Chapter 7.
Imported by the exercise and figure scripts of Chapter 7.
"""
import sys
import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0] + "/../ch6")
from cy_core import three_bar_truss  # noqa: E402  (the truss of Chapter 6)

# Contents (stresses in MPa, strains dimensionless, times in s):
#   filament_kin      return map of the filament with linear kinematic hardening,
#                     parameters p = (E, sY, H), and all its partial derivatives
#   Truss             the three-bar truss of Chapter 6 under forces Q(t); forward
#                     solution step by step (Newton), cost J, gradient by finite
#                     differences, direct differentiation (DDM) and adjoint
#   norton_relax      Norton-Hoff filament under a strain history (ramp, hold,
#                     unload); stress history and its sensitivities (DDM)
#   Membrane          a membrane on an elastic foundation indented by a rigid
#                     parabolic punch (frictionless contact, active sets); force
#                     F(U), its derivative by the adjoint (Dirichlet data on the
#                     contact zone) and by finite differences
#   gauss_newton      sensitivity matrix -> Gauss-Newton matrix, eigenvalues and
#                     correlations (identifiability)
#   adjoint_with_storage, active_sets   checkpoints and plastic patterns (7.6.4)
#
# Where in the book: the three model problems are Section 7.1.2; the sensitivity
# and Gauss-Newton matrices Section 7.2.2; the incremental chain and its
# derivatives Section 7.6 (Box 7.3 direct differentiation, Box 7.4 adjoint);
# contact Section 7.5. The notation follows the chain (7.16): at step n,
# sig_n = sig_hat(eps_n, z_{n-1}; p), z_n = z_hat(eps_n, z_{n-1}; p), where z is
# the plastic strain of each bar (one Gauss point per bar).


# ------------------------------------------------------------ filament
def filament_kin(eps, z, p):
    """Return map (Box 5.1 with K = 0) of fibres with strain eps and plastic strain
    z at the start of the step; p = (E, sY, H). Returns sigma, the new plastic
    strain and the derivatives used by the DDM and the adjoint:
      sig_e = d sig/d eps (consistent tangent), sig_z = d sig/d z,
      sig_p = d sig/d p (shape nf x 3), r_e, r_z, r_p for the new plastic strain.
    These are the six derivatives of Section 7.6.1 ("The filament"): sig_e is
    C^alg, sig_z, sig_p, r_e, r_z, r_p are sig_hat,z, sig_hat,p, z_hat,eps, ...
    They are those of the active branch (box "The derivative of the return map
    freezes the active set"): elastic if f_trial <= 0, plastic otherwise."""
    E, sY, H = p
    eps, z = np.asarray(eps, float), np.asarray(z, float)
    nf = eps.size
    # elastic predictor: xi = sigma_trial - q, back stress q = H z (Box 5.1, step 2)
    xi = E * (eps - z) - H * z
    f = np.abs(xi) - sY
    s = np.sign(xi)
    pl = f > 0.0                               # active set: plastic fibres
    # plastic corrector: Delta gamma = <f_trial> / (E + H) (Box 5.1, step 4)
    dg = np.where(pl, f / (E + H), 0.0)
    zn = z + s * dg
    sig = E * (eps - zn)
    # derivatives of dg (zero in elastic points)
    # from dg = (s (E (eps - z) - H z) - sY) / (E + H) on the plastic branch
    dg_e = np.where(pl, s * E / (E + H), 0.0)
    dg_z = np.where(pl, -s, 0.0)
    dg_p = np.zeros((nf, 3))
    dg_p[:, 0] = np.where(pl, (s * (eps - z) - dg) / (E + H), 0.0)   # d/dE
    dg_p[:, 1] = np.where(pl, -1.0 / (E + H), 0.0)                  # d/dsY
    dg_p[:, 2] = np.where(pl, (-s * z - dg) / (E + H), 0.0)         # d/dH
    # chain rule: z_n = z + s dg and sig = E (eps - z_n)
    r_e, r_z, r_p = s * dg_e, 1.0 + s * dg_z, s[:, None] * dg_p
    sig_e = E * (1.0 - r_e)           # E elastic, E H / (E + H) plastic
    sig_z = -E * r_z                  # -E elastic, 0 plastic (history forgotten)
    sig_p = -E * r_p
    sig_p[:, 0] += eps - zn           # explicit dependence of E (eps - z_n) on E
    return dict(sig=sig, z=zn, sig_e=sig_e, sig_z=sig_z, sig_p=sig_p,
                r_e=r_e, r_z=r_z, r_p=r_p, plastic=pl, f_trial=f)


# ------------------------------------------------------------ truss
class Truss:
    """Three-bar truss (B, W of Chapter 6) loaded by forces Q(t_n), n = 0..N.
    The observations are the displacements u_n of the node at all instants,
    and the cost is J = 1/2 sum_n |u_n - u^m_n|^2 / u_ref^2.
    Section 7.1.2 ("The truss"); output least squares (7.3) with the scale u_ref
    for every observation (u_ref = sY l / E in the exercises, l = 1)."""

    def __init__(self, Q, times):
        S = three_bar_truss()                  # only its matrices B and W are used
        self.B, self.w, self.Q, self.t = S.B, S.w, Q, np.asarray(times)
        self.N = len(times) - 1
        self.nsolve = 0            # counter of forward solves
        self._last = (None, None)  # cache of the last forward solution

    def K(self, Et):
        """Stiffness B^T W diag(Et) B for the fibre moduli Et (tangent if Et = sig_e)."""
        return self.B.T @ ((self.w * Et)[:, None] * self.B)

    def forward(self, p, tol=1e-13):
        """Newton with the consistent tangent at each step. Returns u (N+1, 2),
        z (N+1, 3) and the list of return-map derivatives of each step. The last
        solution is cached: cost and gradient at the same p share one solve.
        This is the direct computation y(p) of Box 5.3 (Box 7.4, step 1)."""
        key = tuple(np.asarray(p, float))
        if self._last[0] == key:
            return self._last[1]
        self.nsolve += 1
        B, w = self.B, self.w
        u = np.zeros((self.N + 1, 2))
        z = np.zeros((self.N + 1, 3))
        steps = [None]
        un = np.zeros(2)
        for n in range(1, self.N + 1):
            F = self.Q(self.t[n])
            for _ in range(60):
                # return map in each bar, residual R = F - B^T W sigma
                rm = filament_kin(B @ un, z[n - 1], p)
                R = F - B.T @ (w * rm["sig"])
                if np.linalg.norm(R) <= tol * max(np.linalg.norm(F), p[1]):
                    break
                un = un + np.linalg.solve(self.K(rm["sig_e"]), R)
            # rm holds the derivatives at the converged state (u_n, z_{n-1}):
            # the operator of both exact methods (box "The consistent tangent ...")
            u[n], z[n] = un, rm["z"]
            steps.append(rm)
        self._last = (key, (u, z, steps))
        return u, z, steps

    def cost(self, p, um, uref):
        """J(p) = 1/2 sum_n |u_n(p) - um_n|^2 / uref^2 (um: measured displacements)."""
        u = self.forward(p)[0]
        return 0.5 * np.sum((u - um) ** 2) / uref ** 2

    def ddm(self, p):
        """Direct differentiation: u and du/dp (N+1, 2, 3); one linear solve per
        step and per parameter with the converged tangent.
        Box 7.3; the three parameters are three right-hand sides."""
        u, z, st = self.forward(p)
        B, w = self.B, self.w
        du = np.zeros((self.N + 1, 2, 3))
        dz = np.zeros((3, 3))                  # d z_{n-1}/d p (fibre x param)
        for n in range(1, self.N + 1):         # forward in time, with the steps
            rm = st[n]
            Kn = self.K(rm["sig_e"])           # K^alg_n at the converged state
            # Box 7.3, step 2: pseudo-load -B^T W (sig_hat,z dz_{n-1} + sig_hat,p)
            pseudo = rm["sig_z"][:, None] * dz + rm["sig_p"]     # d sig at du = 0
            dun = np.linalg.solve(Kn, -B.T @ (w[:, None] * pseudo))
            de = B @ dun
            # Box 7.3, step 3: local update dz_n = z_hat,eps de + z_hat,z dz + z_hat,p
            dz = rm["r_e"][:, None] * de + rm["r_z"][:, None] * dz + rm["r_p"]
            du[n] = dun
        return u, du

    def gradient_ddm(self, p, um, uref):
        """dJ/dp = sum_n (u_n - um_n) . du_n/dp / uref^2, from the DDM sensitivities."""
        u, du = self.ddm(p)
        return np.einsum("ni,nij->j", u - um, du) / uref ** 2

    def gradient_adjoint(self, p, um, uref):
        """Discrete adjoint: backward recursion with the transposed tangent.
        Lagrangian L = J - sum lam_n.(B^T W sig_n - F_n) - sum mu_n.(z_n - r_n).
        Box 7.4 and Algorithm 7.1; the Lagrangian is (7.17). lam_n is the nodal
        adjoint displacement, mu_n the local multiplier of each bar."""
        u, z, st = self.forward(p)
        B, w = self.B, self.w
        g = np.zeros(3)
        mu = np.zeros(3)                       # mu_N = 0: nothing after t_N
        for n in range(self.N, 0, -1):         # Box 7.4, step 3: backward in time
            rm = st[n]
            Kn = self.K(rm["sig_e"])           # symmetric here; K^T in general
            # adjoint equilibrium: K^T lam = dj_n/du_n + B^T (z_hat,eps mu_n)
            rhs = (u[n] - um[n]) / uref ** 2 + B.T @ (rm["r_e"] * mu)
            lam = np.linalg.solve(Kn.T, rhs)
            Wl = w * (B @ lam)                 # W B lam, one value per fibre
            # Box 7.4, step 4: gradient += -sig_hat,p^T W B lam + z_hat,p^T mu
            g += -Wl @ rm["sig_p"] + mu @ rm["r_p"]
            # Box 7.4, step 5: local adjoint update (the memory, carried backward)
            mu = rm["r_z"] * mu - rm["sig_z"] * Wl   # mu_{n-1}
        return g

    def gradient_fd(self, p, um, uref, h=1e-6, central=True):
        """Finite differences with the relative step h (Section 7.4.1): central
        (J(p + dp) - J(p - dp)) / (2 dp) or forward (J(p + dp) - J(p)) / dp."""
        p = np.asarray(p, float)
        g = np.zeros(3)
        for j in range(3):
            dp = np.zeros(3)
            dp[j] = h * p[j]
            if central:
                g[j] = (self.cost(p + dp, um, uref) - self.cost(p - dp, um, uref)) / (2 * dp[j])
            else:
                g[j] = (self.cost(p + dp, um, uref) - self.cost(p, um, uref)) / dp[j]
        return g


# ------------------------------------------------------------ Norton filament
def norton_history(times, eps):
    """Strain history sampled at times (arrays of the same length)."""
    return np.asarray(times, float), np.asarray(eps, float)


def norton_relax(p, times, eps, sens=True):
    """Norton-Hoff filament  sigma = E (eps - e_vp),
    d e_vp/dt = <(|sigma| - sY)/K>^m sign(sigma), backward Euler.
    p = (E, sY, K, m). Returns sigma (N+1) and d sigma/d p (N+1, 4).
    The law is (7.2); the sensitivities are those of Exercise 7.11 (a): with
    x = |sigma_n|, the step solves phi(x) = x - x_trial + E dt <(x - sY)/K>^m = 0,
    and differentiating phi gives (1 + E g_x) dx = dx_trial - g dE - E dg/dp."""
    E, sY, K, m = p
    N = len(times) - 1
    sig = np.zeros(N + 1)
    dsig = np.zeros((N + 1, 4))
    ev, dev = 0.0, np.zeros(4)         # e_vp and its derivative d e_vp / d p
    for n in range(1, N + 1):
        dt = times[n] - times[n - 1]
        # elastic predictor x_trial = |E (eps_n - e_vp_{n-1})| and its derivative
        st = E * (eps[n] - ev)
        s = np.sign(st) if st != 0 else 1.0
        xt = abs(st)
        dxt = s * np.array([eps[n] - ev, 0, 0, 0]) - s * E * dev
        if xt <= sY:
            # below the threshold: no viscoplastic flow in the step
            x, gv = xt, 0.0
            dx = dxt
            dg = np.zeros(4)
        else:
            x = 0.5 * (xt + sY)                       # Newton on phi(x) = 0
            # safeguarded Newton: the root lies in (sY, x_trial); bisection fallback
            lo, hi = sY, xt
            for _ in range(200):
                y = (x - sY) / K
                phi = x - xt + E * dt * y ** m
                if abs(phi) < 1e-13 * xt:
                    break
                if phi > 0:
                    hi = x
                else:
                    lo = x
                dphi = 1 + E * dt * m * y ** (m - 1) / K
                xn = x - phi / dphi
                x = xn if lo < xn < hi else 0.5 * (lo + hi)
            # g = dt y^m is the viscoplastic increment |Delta e_vp|, y = (x - sY)/K
            y = (x - sY) / K
            gv = dt * y ** m
            gx = dt * m * y ** (m - 1) / K
            # partial derivatives of g with respect to (E, sY, K, m) at fixed x
            gp = np.array([0.0, -gx, -gx * y, dt * y ** m * np.log(y)])
            # phi = x - xt + E g(x, p) = 0
            rhs = dxt - gv * np.array([1, 0, 0, 0]) - E * gp
            dx = rhs / (1 + E * gx)
            dg = gx * dx + gp
        ev = ev + s * gv
        dev = dev + s * dg
        sig[n] = s * x
        dsig[n] = s * dx
    return (sig, dsig) if sens else sig


def relaxation_test(t_ramp=10.0, t_hold=1000.0, eps_max=3e-3, n_ramp=40, n_hold=120,
                    n_unload=20, t_unload=10.0):
    """Ramp to eps_max in t_ramp, hold until t_ramp + t_hold (log-spaced steps),
    unload to zero in t_unload.
    The test of Section 7.1.2 (Figure 7.1d): 3e-3 in 10 s, hold t_h, unload in
    10 s. Returns the times (s) and the imposed strains."""
    t1 = np.linspace(0, t_ramp, n_ramp + 1)
    t2 = t_ramp + np.geomspace(0.05, t_hold, n_hold)
    t3 = t2[-1] + np.linspace(0, t_unload, n_unload + 1)[1:]
    times = np.concatenate([t1, t2, t3])
    eps = np.concatenate([eps_max * t1 / t_ramp, np.full(n_hold, eps_max),
                          eps_max * (1 - (t3 - t2[-1]) / t_unload)])
    return times, eps


# ------------------------------------------------------------ contact
class Membrane:
    """Membrane of tension T on a foundation of modulus k (per unit length), on
    [-L, L], u = 0 at both ends, nodes x_i. A rigid punch of profile
    g(x) = x^2 / (2R) is pressed to the depth U: u_i >= U - g_i (u downwards).
    Energy 1/2 u^T K u, K = k M + T A (lumped M, finite-difference A);
    contact forces p >= 0, K u = p, complementarity; F = sum p_i.
    The quadratic program (7.11) and its Kuhn-Tucker conditions (7.12); k and T
    are k_f and T_m of the book, p = (k_f, T_m), and the contact forces are t_i."""

    def __init__(self, n=81, L=1.0, R=0.5):
        self.x = np.linspace(-L, L, n)[1:-1]   # n - 2 free nodes (79 for n = 81)
        self.h = 2 * L / (n - 1)               # mesh size
        self.g = self.x ** 2 / (2 * R)         # punch profile g(x_i)
        m = len(self.x)
        self.M = self.h * np.eye(m)            # lumped foundation matrix
        A = 2 * np.eye(m) - np.eye(m, k=1) - np.eye(m, k=-1)
        self.A = A / self.h                    # membrane matrix
        self.dK = [self.M, self.A]             # dK/dk, dK/dT

    def K(self, p):
        """K = k_f M + T_m A, linear in the parameters."""
        return p[0] * self.M + p[1] * self.A

    def solve(self, p, U):
        """Primal-dual active set: returns u, contact forces, active set.
        Section 7.5.1: on the active set the node follows the punch, u_i = d_i;
        on the inactive set the force vanishes; repeat until the set is stable."""
        K = self.K(p)
        d = U - self.g                          # d_i = U - g(x_i)
        act = d > 0                             # first guess: nodes under the punch
        for _ in range(100):
            # linear problem with u = d imposed on the active set A
            u = np.zeros_like(d)
            I, A = ~act, act
            u[A] = d[A]
            if I.any():
                u[I] = np.linalg.solve(K[np.ix_(I, I)], -K[np.ix_(I, A)] @ d[A])
            pc = K @ u                          # contact forces t = K u on A
            pc[I] = 0.0
            # new active set: positive force or penetration (complementarity test)
            new = (pc + 1e3 * (d - u)) > 1e-14 * max(1.0, abs(pc).max())
            if np.array_equal(new, act):
                break
            act = new
        return u, pc, act

    def force(self, p, U):
        """Force on the punch F = sum_i t_i at the depth U."""
        u, pc, act = self.solve(p, U)
        return pc.sum()

    def dforce_adjoint(self, p, U):
        """dF/dp = w^T (dK/dp) u, with w = 1 on the contact zone and the
        elastic solution elsewhere (Dirichlet data on the active set).
        Box "The adjoint of a contact problem is not a contact problem"
        (Section 7.5.2), (7.14)-(7.15) with the unit datum: w is v_hat."""
        u, pc, act = self.solve(p, U)
        K = self.K(p)
        w = np.zeros_like(u)
        w[act] = 1.0                            # unit settlement of the contact zone
        I = ~act
        if I.any():
            # (K w)_i = 0 on the inactive set: no force outside the contact zone
            w[I] = np.linalg.solve(K[np.ix_(I, I)], -K[np.ix_(I, act)] @ w[act])
        return np.array([w @ (dK @ u) for dK in self.dK]), act


# ------------------------------------------------------------ identifiability
def gauss_newton(S, scale=None):
    """S: sensitivity matrix (observations x parameters). Optional column scaling
    (e.g. the parameter values, for log-parameters). Returns the Gauss-Newton
    matrix, its eigenvalues (ascending) and eigenvectors, and the correlation
    matrix of the parameters.
    Section 7.2.2: S of (7.4), G = S^T S of (7.5); small eigenvalues are the
    directions the experiment hardly sees, the covariance is G^-1."""
    S = np.asarray(S, float)
    if scale is not None:
        S = S * np.asarray(scale)[None, :]     # d/d log p_j = p_j d/d p_j
    G = S.T @ S
    lam, V = np.linalg.eigh(G)
    C = np.linalg.pinv(G)                      # pseudo-inverse: G may be singular
    d = np.sqrt(np.maximum(np.diag(C), 1e-300))
    return G, lam, V, C / np.outer(d, d)


# ------------------------------------------------ storage for the adjoint
def adjoint_with_storage(T, p, um, uref, every=1, mode="recompute"):
    """Adjoint gradient of the truss cost when the forward run keeps the state
    (u_n, z_n) only at the checkpoints n = 0, every, 2 every, ...
      mode = "recompute": between checkpoints, the steps are recomputed by Newton
             from the checkpoint (exact gradient; extra step solves are counted);
      mode = "interpolate": the states between checkpoints are interpolated
             linearly in time (no extra solve; approximate gradient).
    In both cases the return-map derivatives and the consistent tangent are
    recomputed locally from the states (u_n, z_{n-1}).
    Returns the gradient, the number of stored states and of recomputed steps.
    Section 7.6.4 ("What the adjoint must store"), Exercise 7.12."""
    u, z, _ = T.forward(p)
    N = T.N
    ck = list(range(0, N + 1, every)) + ([N] if N % every else [])   # checkpoints
    us, zs = np.zeros_like(u), np.zeros_like(z)
    extra = 0
    if mode == "recompute":
        B, w = T.B, T.w
        for a, b in zip(ck[:-1], ck[1:]):          # rerun each segment from its checkpoint
            un, zn = u[a].copy(), z[a].copy()
            us[a], zs[a] = un, zn
            for n in range(a + 1, b + 1):
                F = T.Q(T.t[n])
                for _ in range(60):
                    rm = filament_kin(B @ un, zn, p)
                    R = F - B.T @ (w * rm["sig"])
                    if np.linalg.norm(R) <= 1e-13 * max(np.linalg.norm(F), p[1]):
                        break
                    un = un + np.linalg.solve(T.K(rm["sig_e"]), R)
                zn = rm["z"]
                us[n], zs[n] = un, zn
                extra += 1
    else:
        # linear interpolation between checkpoints a and b: a state may fall on
        # the wrong side of the yield condition, hence a wrong tangent
        for a, b in zip(ck[:-1], ck[1:]):
            for n in range(a, b + 1):
                th = (n - a) / (b - a)
                us[n] = (1 - th) * u[a] + th * u[b]
                zs[n] = (1 - th) * z[a] + th * z[b]
    # backward sweep of Box 7.4 from the stored (or rebuilt) states
    B, w = T.B, T.w
    g, mu = np.zeros(3), np.zeros(3)
    for n in range(N, 0, -1):
        rm = filament_kin(B @ us[n], zs[n - 1], p)       # tangent recomputed from states
        lam = np.linalg.solve(T.K(rm["sig_e"]).T, (us[n] - um[n]) / uref ** 2
                              + B.T @ (rm["r_e"] * mu))
        Wl = w * (B @ lam)
        g += -Wl @ rm["sig_p"] + mu @ rm["r_p"]
        mu = rm["r_z"] * mu - rm["sig_z"] * Wl
    return g, len(ck), extra


def active_sets(T, p):
    """Distinct plastic patterns (one bit per Gauss point) along the history:
    for linear hardening the consistent tangent depends on the pattern only.
    Section 7.6.4, "Piecewise-constant tangents"."""
    st = T.forward(p)[2]
    pats = [tuple(st[n]["plastic"].astype(int)) for n in range(1, T.N + 1)]
    return pats, sorted(set(pats))
