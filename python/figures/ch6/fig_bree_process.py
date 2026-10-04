"""
Generates the Bree figures of Chapter 6 (thin vessel, Maitournam Sec. 4.5):
  bree_diagram.pdf   asymptotic regimes computed cycle by cycle (perfect plasticity)
                     against Bree's boundaries, with the two cases (a), (b) used below
  bree_process.pdf   case X = 0.7, Y = 1.5, H = 0.02 E (slow shakedown): stress and
                     plastic strain through the wall, iterates of the direct cyclic
                     method and successive cycles
  bree_ratchet.pdf   same loads, H = 0 (ratchetting): direct cyclic iterates; the
                     mean plastic strain drifts, the residual stress converges
  bree_nonunique.pdf case X = 0.2, Y = 3 (alternating plasticity), H = 0.02 E:
                     two exact periodic states, cycle by cycle and direct cyclic
X = sigma_P / sigma_0, Y = sigma_T / sigma_0 (sigma_0 = sY = 280 MPa).
Where in the book: Figures 6.4 (diagram), 6.12 (process), 6.13 (ratchet) and 6.10
(nonunique); Sections 6.1.2, 6.5 and 6.6.
Run from this directory: the module cy_core is imported from the examples of ch6.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "../../examples/ch6")
from cy_core import (bree_vessel, incremental, global_local, period_map, fixed_point,
                     melan, cycle)

blue, green, orange = "#1F5AC8", "#14963C", "#D9822B"
s0, Ep = 280.0, 200e3 / 0.7                       # sY (MPa), plane modulus E/(1-nu)
eY = s0 / Ep
lam = lambda t: abs(np.sin(t))
ts = np.linspace(0, np.pi, 41)                     # one thermal cycle 0 -> 1 -> 0
tt = ts / np.pi
OUT = "../../../figures/ch6/"


def vessel(X, Y, Hr, nlay=100):
    """Bree vessel with sigma_P = X s0, sigma_T = Y s0, H = Hr E, nlay layers."""
    return bree_vessel(X * s0, Y * s0, lam, E=Ep, sY=s0, H=Hr * Ep, nlay=nlay)


# --------------------------------------------------------------- diagram
def regime(X, Y, ncyc=30):
    """Regime after ncyc cycles of 20 steps, 40 layers, H = 0 (Section 6.5.1):
    0 elastic, 1 elastic shakedown, 2 alternating plasticity, 3 ratchetting
    (drift of the mean plastic strain from one cycle to the next)."""
    S = vessel(X, Y, 0.0, nlay=40)
    z = np.zeros(S.nf)
    for _ in range(ncyc):
        zT, _, _, ep, _ = cycle(S, ts[::2], z)
        d, z = np.mean(zT - z), zT
    rng = np.max(ep.max(0) - ep.min(0))
    if abs(d) > 1e-6 * eY:
        return 3
    if rng > 1e-6 * eY:
        return 2
    return 1 if np.max(np.abs(z)) > 0 else 0


# 50 x 60 grid of load pairs; lines: Bree's boundaries X+Y=1, Y=2, X+Y/4=1, XY=1
Xs, Ys = np.linspace(0.01, 0.99, 50), np.linspace(0.02, 4.0, 60)
R = np.array([[regime(X, Y) for X in Xs] for Y in Ys])
fig, ax = plt.subplots(figsize=(4.4, 3.6))
cols = ["#ffffff", "#cfe0f6", "#d6efd9", "#f8dcc0"]
from matplotlib.colors import ListedColormap
ax.pcolormesh(Xs, Ys, R, cmap=ListedColormap(cols), shading="nearest", vmin=0, vmax=3)
x = np.linspace(0, 1, 200)
ax.plot(x, 1 - x, "k", lw=0.8)
xa = np.linspace(0.5, 1, 100)
ax.plot(xa, 4 * (1 - xa), "k", lw=0.8)
ax.plot([0, 0.5], [2, 2], "k", lw=0.8)
xb = np.linspace(0.25, 0.5, 100)
ax.plot(xb, 1 / xb, "k", lw=0.8)
for (X, Y, lab) in [(0.7, 1.5, "a"), (0.2, 3.0, "b")]:
    ax.plot(X, Y, "o", color="k", ms=4)
    ax.annotate(lab, (X, Y), xytext=(4, 3), textcoords="offset points", fontsize=8)
for (X, Y, txt) in [(0.25, 0.35, "E"), (0.35, 1.3, "S"), (0.12, 3.2, "P"),
                    (0.8, 2.2, "R"), (0.6, 3.4, "R")]:
    ax.text(X, Y, txt, fontsize=10, ha="center")
ax.set_xlim(0, 1)
ax.set_ylim(0, 4)
ax.set_xlabel(r"$X=\sigma_P/\sigma_0$")
ax.set_ylabel(r"$Y=\sigma_T/\sigma_0$")
fig.tight_layout()
fig.savefig(OUT + "bree_diagram.pdf", bbox_inches="tight")
print("diagram done")

# --------------------------------------------------- process, slow shakedown
# case (a) with H = 0.02 E: cycle by cycle (Picard on Pi) and direct cyclic method
# (Box 6.2, periodic closure), recording the iterates
S = vessel(0.7, 1.5, 0.02)
x = S.x
Pi = period_map(S, ts)
Zp, rp = fixed_point(Pi, np.zeros(S.nf), "picard", kmax=400, tol=1e-10 * eY)
dcm = global_local(S, ts, closure="periodic", kmax=600, tol=1e-9, record=True)
print("slow shakedown: cycles", len(rp), "DCM sweeps", dcm["iters"], dcm["err"][-1])
iT = len(ts) // 2                                  # t = T/2, maximum temperature
fig, axs = plt.subplots(2, 2, figsize=(7.0, 5.0), sharex=True)
bl, gr = plt.get_cmap("Blues"), plt.get_cmap("Greens")
ks = [0, 2, 9, 49, 199, len(dcm["rec_ep"]) - 1]
for j, k in enumerate(ks):
    c = bl(0.35 + 0.65 * j / (len(ks) - 1))
    axs[0, 0].plot(x, dcm["rec_sig"][k][iT] / s0, color=c, lw=1.1, label=f"$k={k + 1}$")
    axs[1, 0].plot(x, dcm["rec_ep"][k][0] / eY, color=c, lw=1.1)
cyc = [1, 2, 5, 20, 60, len(rp)]
z = np.zeros(S.nf)
c_ = 0
for j, nc in enumerate(cyc):
    while c_ < nc:
        zT, _, sig_c, ep_c, _ = cycle(S, ts, z)
        z0, z = z, zT
        c_ += 1
    c = gr(0.35 + 0.65 * j / (len(cyc) - 1))
    axs[0, 1].plot(x, sig_c[iT] / s0, color=c, lw=1.1, label=f"cycle {nc}")
    axs[1, 1].plot(x, ep_c[-1] / eY, color=c, lw=1.1)
axs[0, 0].set_title("direct cyclic: iterates $k$", fontsize=9)
axs[0, 1].set_title("cycle by cycle", fontsize=9)
axs[0, 0].set_ylabel(r"$\sigma(x,T/2)/\sigma_0$")
axs[1, 0].set_ylabel(r"$\varepsilon^p(x,0)/\varepsilon_Y$")
for a in axs[1]:
    a.set_xlabel(r"$x=(r-r_m)/e$")
axs[0, 0].legend(fontsize=6, ncol=2)
axs[0, 1].legend(fontsize=6, ncol=2)
fig.tight_layout()
fig.savefig(OUT + "bree_process.pdf", bbox_inches="tight")

# --------------------------------------------------------- ratchetting
# case (a) with H = 0: the direct cyclic iterates do not converge; the mean <ep>
# (kernel of Z) drifts while rho = -Z ep converges (Section 6.6.1)
S = vessel(0.7, 1.5, 0.0)
_, Z = melan(S)
dcm = global_local(S, ts, closure="periodic", kmax=60, tol=0.0, record=True)
E0 = np.array(dcm["rec_ep"])[:, 0, :]               # ep(x, t_0) after each sweep
mean = E0.mean(axis=1) / eY
rho = np.array([-Z @ e for e in E0]) / s0
drho = np.max(np.abs(np.diff(rho, axis=0)), axis=1)
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.7))
k = np.arange(1, len(mean) + 1)
axs[0].plot(k, mean, color=orange, lw=1.3, label=r"mean $\langle\varepsilon^p\rangle/\varepsilon_Y$")
ax2 = axs[0].twinx()
ax2.semilogy(k[1:], drho, color=blue, lw=1.3)
ax2.set_ylabel(r"$\max_x|\Delta\rho|/\sigma_0$", color=blue, fontsize=8)
axs[0].set_xlabel("iteration $k$")
axs[0].set_ylabel(r"$\langle\varepsilon^p(\cdot,0)\rangle/\varepsilon_Y$", color=orange)
axs[0].set_title("(a) drift of the mean, convergence of $\\rho$", fontsize=9)
for j, kk in enumerate([0, 2, 9, 29, len(rho) - 1]):
    axs[1].plot(S.x, rho[kk], color=bl(0.35 + 0.65 * j / 4), lw=1.1, label=f"$k={kk + 1}$")
axs[1].set_xlabel(r"$x$")
axs[1].set_ylabel(r"$\rho(x,0)/\sigma_0$")
axs[1].set_title("(b) residual stress through the wall", fontsize=9)
axs[1].legend(fontsize=6)
fig.tight_layout()
fig.savefig(OUT + "bree_ratchet.pdf", bbox_inches="tight")
print("ratchet: mean drift per iteration", np.diff(mean)[-5:], "last drho", drho[-1])

# ------------------------------------------------------- non-uniqueness
# case (b) with H = 0.02 E: two exact periodic states (Section 6.5.2), one from
# the virgin state cycle by cycle, one from the direct cyclic method
S = vessel(0.2, 3.0, 0.02)
_, Z = melan(S)
Pi = period_map(S, ts)
zP = fixed_point(Pi, np.zeros(S.nf), "picard", tol=1e-12)[0][-1]
zD = global_local(S, ts, closure="periodic", kmax=600, tol=1e-11)["ep"][0]
print("non-unique: |Pi zD - zD|/eY =", np.max(np.abs(Pi(zD) - zD)) / eY,
      " |zD - zP|/eY =", np.max(np.abs(zD - zP)) / eY)
_, _, sP, eP, _ = cycle(S, ts, zP)
_, _, sD, eD, _ = cycle(S, ts, zD)
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.7))
j = int(np.argmax(np.abs(Z @ (zP - zD))))           # point where the states differ most
for (sg, c, lab) in [(sP, green, "cycle by cycle"), (sD, blue, "direct cyclic")]:
    axs[0].plot(tt, sg[:, -1] / s0, color=c, lw=1.0, ls=":")
    axs[0].plot(tt, sg[:, j] / s0, color=c, lw=1.3, label=lab)
axs[0].set_xlabel("$t/T$")
axs[0].set_ylabel(r"$\sigma/\sigma_0$")
axs[0].set_title(f"(a) stress cycles at $x={S.x[j]:.2f}$ (dotted: outer skin)", fontsize=9)
axs[0].legend(fontsize=6)
axs[1].plot(S.x, -Z @ zP / s0, color=green, lw=1.3, label="cycle by cycle")
axs[1].plot(S.x, -Z @ zD / s0, color=blue, lw=1.3, label="direct cyclic")
axs[1].set_xlabel(r"$x$")
axs[1].set_ylabel(r"$\rho(x,0)/\sigma_0$")
axs[1].set_title("(b) residual stress at $t=0$", fontsize=9)
axs[1].legend(fontsize=6)
fig.tight_layout()
fig.savefig(OUT + "bree_nonunique.pdf", bbox_inches="tight")
