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
  storage.pdf      truss adjoint with checkpoints: peak number of stored states and
                   gradient error when the states are interpolated between checkpoints
Where in the book: Figures 7.3 (truss_id), 7.5 (indentation), 7.2 (norton_id) and
7.6 (storage); the data are those of Exercises 7.9-7.12.
Run from this directory: the module id_core is imported from the examples of ch7.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize, least_squares
sys.path.insert(0, "../../examples/ch7")
from id_core import (Truss, Membrane, norton_relax, relaxation_test, gauss_newton,
                     adjoint_with_storage)

blue, green, orange, red = "#1F5AC8", "#14963C", "#D9822B", "#B03030"
OUT = "../../../figures/ch7/"

# ------------------------------------------------------------------ truss
# truss of Section 7.1.2: E, sY, H in MPa, 1% noise, start p0 = (0.7, 1.2, 3.0) p_true
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
hist, current = {}, [None]
forward_plain = T.forward


def forward_recorded(p, *args, **kw):
    """Forward solve that also records (solves so far, best cost so far) for the
    method in progress; the count is T.nsolve, as in Exercise 7.9."""
    out = forward_plain(p, *args, **kw)
    v = 0.5 * np.sum((out[0] - um) ** 2) / uref ** 2      # J of this solve, (7.2)
    h = hist[current[0]]
    h.append((T.nsolve, min(v, h[-1][1]) if h else v))
    return out


T.forward = forward_recorded      # every solve (also inside DDM and adjoint) is recorded


def start(label):
    """Reset the solve counter and open the record of a method."""
    T.nsolve, current[0], hist[label] = 0, label, []


# the three minimizations of Section 7.3, with exactly the calls of Exercise 7.9,
# in q = log(p / p0)
J = lambda q: T.cost(p0 * np.exp(q), um, uref)


def J_and_grad(q):                            # one forward + one adjoint solve
    p = p0 * np.exp(q)
    return T.cost(p, um, uref), T.gradient_adjoint(p, um, uref) * p


resid = lambda q: ((T.forward(p0 * np.exp(q))[0] - um)[1:] / uref).ravel()
jac = lambda q: (T.ddm(p0 * np.exp(q))[1][1:] / uref).reshape(-1, 3) * p0 * np.exp(q)
start("Nelder--Mead")
minimize(J, np.zeros(3), method="Nelder-Mead",
         options=dict(xatol=1e-8, fatol=1e-12, maxiter=4000, maxfev=4000))
start("BFGS + adjoint")
minimize(J_and_grad, np.zeros(3), jac=True, method="BFGS", options=dict(gtol=1e-6))
start("Levenberg--Marquardt + DDM")
r = least_squares(resid, np.zeros(3), jac=jac, method="lm", xtol=1e-12, ftol=1e-14)
T.forward = forward_plain
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
# membrane of Section 7.5, (k_f, T_m) = (4, 1), depth U = 0.2; the adjoint field w
# is v_hat of (7.14) with the unit datum on the contact zone
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
# (b) force and contact half-width against the depth
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
# (c) adjoint dF/dk_f at U = 0.1 on two meshes: kinks, Section 7.5.3
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
# Norton filament, hold of 1000 s: E, sY (MPa), K (MPa s^(1/m)), m; 1% noise
pn = np.array([200e3, 200.0, 500.0, 5.0])
tt, ee = relaxation_test(t_hold=1000.0)
s, ds = norton_relax(pn, tt, ee)
# same noisy data as Exercise 7.11: the generator (seed 1) first draws the noise
# of the 10 s and 100 s holds, then that of the 1000 s hold shown here
rng = np.random.default_rng(1)
for t_short in (10.0, 100.0):
    rng.standard_normal(relaxation_test(t_hold=t_short)[0].shape)
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
# a parameter set along the weakest Gauss-Newton direction (log p + 0.4 v_1)
V4 = gauss_newton(ds[1:] / 200.0, scale=pn)[2]
other = pn * np.exp(0.4 * V4[:, 0])               # along the weakest direction
axs[0].plot(tt[1:], norton_relax(other, tt, ee, False)[1:], color=orange, lw=1.0, ls="--",
            label="$p/p_{\\mathrm{true}}=(%.2f, %.2f, %.2f, %.2f)$" % tuple(other / pn))
axs[0].set_xscale("log")
axs[0].set_xlabel("time (s)")
axs[0].set_ylabel(r"$\sigma$ (MPa)")
axs[0].set_title("(a) relaxation test", fontsize=9)
axs[0].legend(fontsize=6, loc="lower left")
# (b) cost on a (K, m) grid at the true E, sY, noise-free data, and the weakest
# direction of the 2 x 2 Gauss-Newton matrix of (K, m)
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

# ---------------------------------------------------------------- storage
# adjoint with checkpoints every c steps at p1 = (0.9, 1.2, 1.5) p_true, Section 7.6.4
T.nsolve = 0
p1 = p_true * np.array([0.9, 1.2, 1.5])
g_all = T.gradient_adjoint(p1, um, uref)
cs = np.array([1, 2, 3, 4, 5, 6, 8, 10, 13, 16, 20, 27, 40])
mem, err = [], []
for c in cs:
    n_ck = adjoint_with_storage(T, p1, um, uref, c, "recompute")[1]
    g_i = adjoint_with_storage(T, p1, um, uref, c, "interpolate")[0]
    mem.append(n_ck + (c - 1 if c > 1 else 0))
    err.append(np.abs(g_i - g_all).max() / np.abs(g_all).max())
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.7))
axs[0].plot(cs, mem, "o-", color=blue, ms=3, lw=1.2, label="checkpoints + one segment")
axs[0].axhline(81, color="0.5", ls=":", lw=0.9)
axs[0].text(14, 73, "all 81 states stored", fontsize=7, color="0.4")
axs[0].plot(cs, 2 * np.sqrt(80) * np.ones_like(cs), color=orange, lw=0.9, ls="--",
            label=r"$2\sqrt{N}$")
axs[0].set_xlabel("checkpoint spacing $c$ (steps)")
axs[0].set_ylabel("states held in memory")
axs[0].set_title("(a) recomputation: exact gradient", fontsize=9)
axs[0].legend(fontsize=6.5)
axs[1].loglog(cs[1:], np.array(err[1:]), "o-", color=red, ms=3, lw=1.2)
axs[1].set_xlabel("checkpoint spacing $c$ (steps)")
axs[1].set_ylabel("relative error of the gradient")
axs[1].set_title("(b) linear interpolation of the states", fontsize=9)
fig.tight_layout()
fig.savefig(OUT + "storage.pdf", bbox_inches="tight")
print("storage: memory", mem, "errors", np.round(err, 4))
