"""
Generates the figures of Chapter 7:
  truss_id.pdf     three-bar truss, identification of (E, sY, H): (a) measured,
                   initial and identified response Q1 against u1; (b) cost against
                   the number of forward solves for Nelder-Mead, BFGS with the
                   adjoint gradient and Levenberg-Marquardt with DDM sensitivities
  indentation.pdf  membrane on a foundation indented by a rigid punch: (a) the
                   displacement and the adjoint field (unit Dirichlet datum on the
                   contact zone); (b) force-depth curve and contact width; (c) dF/dk
                   by the adjoint against k, two meshes (kinks of the discrete problem)
  norton_id.pdf    Norton-Hoff relaxation test: (a) data and fits; (b) cost in the
                   (K, m) plane at the true E, sY, with the weakest Gauss-Newton direction
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize, least_squares
sys.path.insert(0, "../../examples/ch7")
from id_core import Truss, Membrane, norton_relax, relaxation_test, gauss_newton

blue, green, orange, red = "#1F5AC8", "#14963C", "#D9822B", "#B03030"
OUT = "../../../figures/ch7/"

# ------------------------------------------------------------------ truss
E, sY, H = 200e3, 200.0, 10e3
p_true = np.array([E, sY, H])
Q = lambda t: np.array([2.3 * sY * np.sin(t), 0.8 * sY * np.sin(2 * t)])
ts = np.linspace(0, 2 * np.pi, 81)
T = Truss(Q, ts)
uref = sY / E
u_true = T.forward(p_true)[0]
rng = np.random.default_rng(0)
um = u_true + 0.01 * np.abs(u_true).max() * rng.standard_normal(u_true.shape)
p0 = p_true * np.array([0.7, 1.2, 3.0])
hist = {}


def tracked(label):
    """Cost of q with a record of (forward solves so far, best cost so far)."""
    hist[label] = []

    def J(q):
        n0 = T.nsolve
        v = T.cost(p0 * np.exp(q), um, uref)
        if T.nsolve > n0:
            best = min(v, hist[label][-1][1]) if hist[label] else v
            hist[label].append((T.nsolve, best))
        return v
    return J


T.nsolve = 0
Jnm = tracked("Nelder--Mead")
minimize(Jnm, np.zeros(3), method="Nelder-Mead",
         options=dict(xatol=1e-8, fatol=1e-12, maxiter=4000, maxfev=4000))
T.nsolve = 0
Jb = tracked("BFGS + adjoint")
minimize(lambda q: (Jb(q), T.gradient_adjoint(p0 * np.exp(q), um, uref) * p0 * np.exp(q)),
         np.zeros(3), jac=True, method="BFGS", options=dict(gtol=1e-6))
T.nsolve = 0
Jl = tracked("Levenberg--Marquardt + DDM")


def resid(q):
    Jl(q)
    return ((T.forward(p0 * np.exp(q))[0] - um)[1:] / uref).ravel()


r = least_squares(resid, np.zeros(3), method="lm", xtol=1e-12, ftol=1e-14,
                  jac=lambda q: (T.ddm(p0 * np.exp(q))[1][1:] / uref).reshape(-1, 3)
                  * p0 * np.exp(q))
p_id = p0 * np.exp(r.x)
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9))
Q1 = np.array([Q(t)[0] for t in ts]) / sY
axs[0].plot(um[:, 0] / uref, Q1, "o", ms=2.2, color="0.45", label="measured (1% noise)")
axs[0].plot(T.forward(p0)[0][:, 0] / uref, Q1, color=orange, lw=1.1, ls="--",
            label="start $p_0$")
axs[0].plot(T.forward(p_id)[0][:, 0] / uref, Q1, color=blue, lw=1.3, label="identified")
axs[0].set_xlabel(r"$u_1/(\sigma_Y l/E)$")
axs[0].set_ylabel(r"$Q_1/N_0$")
axs[0].set_title("(a) response of the truss", fontsize=9)
axs[0].legend(fontsize=6.5)
for lab, c in [("Nelder--Mead", green), ("BFGS + adjoint", orange),
               ("Levenberg--Marquardt + DDM", blue)]:
    h = np.array(hist[lab])
    axs[1].semilogy(h[:, 0], h[:, 1], color=c, lw=1.3, label=lab)
    print(lab, "solves", int(h[-1, 0]), "J", h[-1, 1])
axs[1].axhline(T.cost(p_true, um, uref), color="0.5", lw=0.8, ls=":")
axs[1].text(700, 0.42, r"$\mathcal{J}(p_{\mathrm{true}})$: noise", fontsize=6.5, color="0.4")
axs[1].set_xscale("log")
axs[1].set_xlabel("forward solves")
axs[1].set_ylabel(r"best cost $\mathcal{J}$")
axs[1].set_title("(b) cost of the minimization", fontsize=9)
axs[1].legend(fontsize=6.5)
fig.tight_layout()
fig.savefig(OUT + "truss_id.pdf", bbox_inches="tight")

# ------------------------------------------------------------ indentation
Mb = Membrane(n=81)
pm = np.array([4.0, 1.0])
U = 0.2
u, pc, act = Mb.solve(pm, U)
K = Mb.K(pm)
w = np.zeros_like(u)
w[act] = 1.0
w[~act] = np.linalg.solve(K[np.ix_(~act, ~act)], -K[np.ix_(~act, act)] @ w[act])
fig, axs = plt.subplots(1, 3, figsize=(9.6, 2.7))
x = Mb.x
xs = np.linspace(-0.45, 0.45, 50)
axs[0].plot(x, -u, color=blue, lw=1.4, label=r"$-u$ (direct)")
axs[0].plot(xs, -(U - xs ** 2 / 1.0), color="0.3", lw=0.9, ls="--", label="punch")
axs[0].plot(x, -U * w, color=orange, lw=1.4, label=r"$-U\,w$ (adjoint)")
axs[0].plot(x[act], -u[act], "o", ms=2.5, color=red, label="contact zone")
axs[0].set_xlabel("$x$")
axs[0].set_title("(a) direct and adjoint fields", fontsize=9)
axs[0].legend(fontsize=6)
Us = np.linspace(0.0, 0.25, 51)
Fs, aw = [], []
for Uk in Us:
    uu, pp, aa = Mb.solve(pm, Uk)
    Fs.append(pp.sum())
    aw.append(np.ptp(x[aa]) / 2 if aa.any() else 0.0)
axs[1].plot(Us, Fs, color=blue, lw=1.4, label="force $F$")
ax2 = axs[1].twinx()
ax2.step(Us, aw, color=orange, lw=1.0, where="post")
ax2.set_ylabel("contact half-width", color=orange, fontsize=8)
axs[1].set_xlabel("depth $U$")
axs[1].set_ylabel("$F$", color=blue)
axs[1].set_title("(b) force--depth curve", fontsize=9)
ks = np.linspace(2.0, 6.0, 401)
for n, c in [(81, blue), (161, green)]:
    M = Membrane(n=n)
    g = np.array([M.dforce_adjoint([k, 1.0], 0.1)[0][0] for k in ks])
    axs[2].plot(ks, g, color=c, lw=1.2, label="$h=%.4f$" % M.h)
axs[2].set_xlabel("$k$")
axs[2].set_ylabel(r"$\partial F/\partial k$ (adjoint)")
axs[2].set_title(r"(c) kinks, $U=0.1$, $T=1$", fontsize=9)
axs[2].legend(fontsize=6.5)
fig.tight_layout()
fig.savefig(OUT + "indentation.pdf", bbox_inches="tight")

# ----------------------------------------------------------------- Norton
pn = np.array([200e3, 200.0, 500.0, 5.0])
tt, ee = relaxation_test(t_hold=1000.0)
s, ds = norton_relax(pn, tt, ee)
rng = np.random.default_rng(1)
sm = s + 0.01 * np.abs(s).max() * rng.standard_normal(s.shape)
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9))
axs[0].plot(tt[1:], sm[1:], "o", ms=1.8, color="0.5", label="data (1% noise)")
np.seterr(all="ignore")
resid = lambda q: (norton_relax(pn * np.exp(q), tt, ee, False) - sm)[1:] / 200.0
jac = lambda q: norton_relax(pn * np.exp(q), tt, ee)[1][1:] / 200.0 * pn * np.exp(q)
r = least_squares(resid, np.log([1.3, 0.5, 2.0, 0.6]), jac=jac, method="lm",
                  xtol=1e-12, ftol=1e-14)
pfit = pn * np.exp(r.x)
axs[0].plot(tt[1:], norton_relax(pfit, tt, ee, False)[1:], color=blue, lw=1.3,
            label="fit: $p/p_{\\mathrm{true}}=(%.2f, %.2f, %.2f, %.2f)$" % tuple(pfit / pn))
V4 = gauss_newton(ds[1:] / 200.0, scale=pn)[2]
other = pn * np.exp(0.4 * V4[:, 0])               # along the weakest direction
axs[0].plot(tt[1:], norton_relax(other, tt, ee, False)[1:], color=orange, lw=1.0, ls="--",
            label="$p/p_{\\mathrm{true}}=(%.2f, %.2f, %.2f, %.2f)$" % tuple(other / pn))
axs[0].set_xscale("log")
axs[0].set_xlabel("time (s)")
axs[0].set_ylabel(r"$\sigma$ (MPa)")
axs[0].set_title("(a) relaxation test", fontsize=9)
axs[0].legend(fontsize=6, loc="lower left")
Kg = np.linspace(250, 900, 61)
mg = np.linspace(3.0, 8.0, 61)
JJ = np.array([[0.5 * np.sum(((norton_relax([pn[0], pn[1], k, m], tt, ee, False) - s)
                              / 200.0) ** 2) for k in Kg] for m in mg])
cs = axs[1].contour(Kg, mg, np.log10(JJ + 1e-12), levels=np.arange(-4, 3.1, 0.5),
                    cmap="viridis", linewidths=0.8)
axs[1].plot(pn[2], pn[3], "*", color=red, ms=8)
G, lam, V, C = gauss_newton(ds[1:, 2:] / 200.0, scale=pn[2:])
v = V[:, 0]
tline = np.linspace(-0.6, 0.6, 20)
axs[1].plot(pn[2] * np.exp(tline * v[0]), pn[3] * np.exp(tline * v[1]), color=red, lw=0.8,
            ls="--")
axs[1].set_xlabel("$K$ (MPa s$^{1/m}$)")
axs[1].set_ylabel("$m$")
axs[1].set_title(r"(b) $\log_{10}\mathcal{J}$ at the true $E$, $\sigma_Y$", fontsize=9)
fig.tight_layout()
fig.savefig(OUT + "norton_id.pdf", bbox_inches="tight")
J_other = 0.5 * np.sum(((norton_relax(other, tt, ee, False) - s) / 200.0) ** 2)
print("Norton fit", np.round(pfit / pn, 3), "| other point", np.round(other / pn, 3), "rms gap",
      200 * np.sqrt(2 * J_other / (len(s) - 1)), "MPa")
