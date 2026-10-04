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


# ------------------------------------------------------------ filament
def filament_kin(eps, z, p):
    """Return map (Box 5.1 with K = 0) of fibres with strain eps and plastic strain
    z at the start of the step; p = (E, sY, H). Returns sigma, the new plastic
    strain and the derivatives used by the DDM and the adjoint:
      sig_e = d sig/d eps (consistent tangent), sig_z = d sig/d z,
      sig_p = d sig/d p (shape nf x 3), r_e, r_z, r_p for the new plastic strain."""
    E, sY, H = p
    eps, z = np.asarray(eps, float), np.asarray(z, float)
    nf = eps.size
    xi = E * (eps - z) - H * z
    f = np.abs(xi) - sY
    s = np.sign(xi)
    pl = f > 0.0
    dg = np.where(pl, f / (E + H), 0.0)
    zn = z + s * dg
    sig = E * (eps - zn)
    # derivatives of dg (zero in elastic points)
    dg_e = np.where(pl, s * E / (E + H), 0.0)
    dg_z = np.where(pl, -s, 0.0)
    dg_p = np.zeros((nf, 3))
    dg_p[:, 0] = np.where(pl, (s * (eps - z) - dg) / (E + H), 0.0)
    dg_p[:, 1] = np.where(pl, -1.0 / (E + H), 0.0)
    dg_p[:, 2] = np.where(pl, (-s * z - dg) / (E + H), 0.0)
    r_e, r_z, r_p = s * dg_e, 1.0 + s * dg_z, s[:, None] * dg_p
    sig_e = E * (1.0 - r_e)
    sig_z = -E * r_z
    sig_p = -E * r_p
    sig_p[:, 0] += eps - zn
    return dict(sig=sig, z=zn, sig_e=sig_e, sig_z=sig_z, sig_p=sig_p,
                r_e=r_e, r_z=r_z, r_p=r_p, plastic=pl, f_trial=f)


# ------------------------------------------------------------ truss
class Truss:
    """Three-bar truss (B, W of Chapter 6) loaded by forces Q(t_n), n = 0..N.
    The observations are the displacements u_n of the node at all instants,
    and the cost is J = 1/2 sum_n |u_n - u^m_n|^2 / u_ref^2."""

    def __init__(self, Q, times):
        S = three_bar_truss()
        self.B, self.w, self.Q, self.t = S.B, S.w, Q, np.asarray(times)
        self.N = len(times) - 1
        self.nsolve = 0            # counter of forward solves
        self._last = (None, None)  # cache of the last forward solution

    def K(self, Et):
        return self.B.T @ ((self.w * Et)[:, None] * self.B)

    def forward(self, p, tol=1e-13):
        """Newton with the consistent tangent at each step. Returns u (N+1, 2),
        z (N+1, 3) and the list of return-map derivatives of each step. The last
        solution is cached: cost and gradient at the same p share one solve."""
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
                rm = filament_kin(B @ un, z[n - 1], p)
                R = F - B.T @ (w * rm["sig"])
                if np.linalg.norm(R) <= tol * max(np.linalg.norm(F), p[1]):
                    break
                un = un + np.linalg.solve(self.K(rm["sig_e"]), R)
            u[n], z[n] = un, rm["z"]
            steps.append(rm)
        self._last = (key, (u, z, steps))
        return u, z, steps

    def cost(self, p, um, uref):
        u = self.forward(p)[0]
        return 0.5 * np.sum((u - um) ** 2) / uref ** 2

    def ddm(self, p):
        """Direct differentiation: u and du/dp (N+1, 2, 3); one linear solve per
        step and per parameter with the converged tangent."""
        u, z, st = self.forward(p)
        B, w = self.B, self.w
        du = np.zeros((self.N + 1, 2, 3))
        dz = np.zeros((3, 3))                  # d z_{n-1}/d p (fibre x param)
        for n in range(1, self.N + 1):
            rm = st[n]
            Kn = self.K(rm["sig_e"])
            pseudo = rm["sig_z"][:, None] * dz + rm["sig_p"]     # d sig at du = 0
            dun = np.linalg.solve(Kn, -B.T @ (w[:, None] * pseudo))
            de = B @ dun
            dz = rm["r_e"][:, None] * de + rm["r_z"][:, None] * dz + rm["r_p"]
            du[n] = dun
        return u, du

    def gradient_ddm(self, p, um, uref):
        u, du = self.ddm(p)
        return np.einsum("ni,nij->j", u - um, du) / uref ** 2

    def gradient_adjoint(self, p, um, uref):
        """Discrete adjoint: backward recursion with the transposed tangent.
        Lagrangian L = J - sum lam_n.(B^T W sig_n - F_n) - sum mu_n.(z_n - r_n)."""
        u, z, st = self.forward(p)
        B, w = self.B, self.w
        g = np.zeros(3)
        mu = np.zeros(3)                       # mu_N = 0: nothing after t_N
        for n in range(self.N, 0, -1):
            rm = st[n]
            Kn = self.K(rm["sig_e"])           # symmetric here; K^T in general
            rhs = (u[n] - um[n]) / uref ** 2 + B.T @ (rm["r_e"] * mu)
            lam = np.linalg.solve(Kn.T, rhs)
            Wl = w * (B @ lam)                 # W B lam, one value per fibre
            g += -Wl @ rm["sig_p"] + mu @ rm["r_p"]
            mu = rm["r_z"] * mu - rm["sig_z"] * Wl   # mu_{n-1}
        return g

    def gradient_fd(self, p, um, uref, h=1e-6, central=True):
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
    p = (E, sY, K, m). Returns sigma (N+1) and d sigma/d p (N+1, 4)."""
    E, sY, K, m = p
    N = len(times) - 1
    sig = np.zeros(N + 1)
    dsig = np.zeros((N + 1, 4))
    ev, dev = 0.0, np.zeros(4)
    for n in range(1, N + 1):
        dt = times[n] - times[n - 1]
        st = E * (eps[n] - ev)
        s = np.sign(st) if st != 0 else 1.0
        xt = abs(st)
        dxt = s * np.array([eps[n] - ev, 0, 0, 0]) - s * E * dev
        if xt <= sY:
            x, gv = xt, 0.0
            dx = dxt
            dg = np.zeros(4)
        else:
            x = 0.5 * (xt + sY)                       # Newton on phi(x) = 0
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
            y = (x - sY) / K
            gv = dt * y ** m
            gx = dt * m * y ** (m - 1) / K
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
    unload to zero in t_unload."""
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
    contact forces p >= 0, K u = p, complementarity; F = sum p_i."""

    def __init__(self, n=81, L=1.0, R=0.5):
        self.x = np.linspace(-L, L, n)[1:-1]
        self.h = 2 * L / (n - 1)
        self.g = self.x ** 2 / (2 * R)
        m = len(self.x)
        self.M = self.h * np.eye(m)
        A = 2 * np.eye(m) - np.eye(m, k=1) - np.eye(m, k=-1)
        self.A = A / self.h
        self.dK = [self.M, self.A]             # dK/dk, dK/dT

    def K(self, p):
        return p[0] * self.M + p[1] * self.A

    def solve(self, p, U):
        """Primal-dual active set: returns u, contact forces, active set."""
        K = self.K(p)
        d = U - self.g
        act = d > 0
        for _ in range(100):
            u = np.zeros_like(d)
            I, A = ~act, act
            u[A] = d[A]
            if I.any():
                u[I] = np.linalg.solve(K[np.ix_(I, I)], -K[np.ix_(I, A)] @ d[A])
            pc = K @ u
            pc[I] = 0.0
            new = (pc + 1e3 * (d - u)) > 1e-14 * max(1.0, abs(pc).max())
            if np.array_equal(new, act):
                break
            act = new
        return u, pc, act

    def force(self, p, U):
        u, pc, act = self.solve(p, U)
        return pc.sum()

    def dforce_adjoint(self, p, U):
        """dF/dp = w^T (dK/dp) u, with w = 1 on the contact zone and the
        elastic solution elsewhere (Dirichlet data on the active set)."""
        u, pc, act = self.solve(p, U)
        K = self.K(p)
        w = np.zeros_like(u)
        w[act] = 1.0
        I = ~act
        if I.any():
            w[I] = np.linalg.solve(K[np.ix_(I, I)], -K[np.ix_(I, act)] @ w[act])
        return np.array([w @ (dK @ u) for dK in self.dK]), act


# ------------------------------------------------------------ identifiability
def gauss_newton(S, scale=None):
    """S: sensitivity matrix (observations x parameters). Optional column scaling
    (e.g. the parameter values, for log-parameters). Returns the Gauss-Newton
    matrix, its eigenvalues (ascending) and eigenvectors, and the correlation
    matrix of the parameters."""
    S = np.asarray(S, float)
    if scale is not None:
        S = S * np.asarray(scale)[None, :]
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
    Returns the gradient, the number of stored states and of recomputed steps."""
    u, z, _ = T.forward(p)
    N = T.N
    ck = list(range(0, N + 1, every)) + ([N] if N % every else [])
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
        for a, b in zip(ck[:-1], ck[1:]):
            for n in range(a, b + 1):
                th = (n - a) / (b - a)
                us[n] = (1 - th) * u[a] + th * u[b]
                zs[n] = (1 - th) * z[a] + th * z[b]
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
    for linear hardening the consistent tangent depends on the pattern only."""
    st = T.forward(p)[2]
    pats = [tuple(st[n]["plastic"].astype(int)) for n in range(1, T.N + 1)]
    return pats, sorted(set(pats))
