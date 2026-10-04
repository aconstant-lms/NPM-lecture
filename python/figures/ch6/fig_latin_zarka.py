"""
Generates the figures of Chapter 6, Sections 6.4-6.7:
  latin.pdf            truss, H = 0.05E, Q2 = N0, Q1 = 1.2 N0 sin t, window of two
                       periods with the initial condition: error indicator of the
                       whole-history initial-strain iteration (Box 6.2) and of LATIN
                       (conjugate directions) without and with relaxation
  bree_convergence.pdf Bree case (a) X = 0.7, Y = 1.5, H = 0.02E: error against the
                       iteration number for the period map (Picard = cycle by cycle,
                       Krasnoselskii-Mann, Anderson) and for the direct cyclic method
                       (plain and Anderson-accelerated)
  zarka.pdf            same case: Zarka's construction through the wall: admissible
                       band of the transformed parameter Y, Y after the first
                       half-cycle, Zarka's projection, and the limits of the
                       cycle-by-cycle and of the accelerated direct cyclic computations
Where in the book: Figure 6.9 (latin), Figure 6.11 (bree_convergence), Figure 6.14
(zarka); Boxes 6.2, 6.3, 6.5 and the period map of Section 6.5.2.
Run from this directory: the module cy_core is imported from the examples of ch6.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "../../examples/ch6")
from cy_core import (three_bar_truss, bree_vessel, incremental, global_local, latin,
                     period_map, fixed_point, dcm_sweep_map, melan, zarka)

blue, green, orange, red = "#1F5AC8", "#14963C", "#D9822B", "#B03030"
OUT = "../../../figures/ch6/"

# ------------------------------------------------------------------ LATIN
E, sY = 200e3, 200.0                               # modulus, yield stress (MPa)
# truss, third loading of Section 6.1.1, H = 0.05 E; window of two periods
S = three_bar_truss(E=E, sY=sY, H=0.05 * E,
                    Q=lambda t: np.array([1.2 * sY * np.sin(t), 1.0 * sY]))
ts = np.linspace(0, 4 * np.pi, 81)
ref = incremental(S, ts, np.zeros(3), tol=1e-12)[2]   # step-by-step reference
# Box 6.2 (initial condition) and Box 6.3 with h = 1/E or 4/E, mu = 0 or 0.3
runs = [("initial strain (Box 6.2)", global_local(S, ts, closure="initial", kmax=400, tol=1e-12),
         blue, "-"),
        (r"LATIN $hE=1$, $\mu=0$", latin(S, ts, 1 / E, mu=0.0, kmax=400, tol=1e-12), red, "--"),
        (r"LATIN $hE=1$, $\mu=0.3$", latin(S, ts, 1 / E, mu=0.3, kmax=400, tol=1e-12), red, "-"),
        (r"LATIN $hE=4$, $\mu=0.3$", latin(S, ts, 4 / E, mu=0.3, kmax=400, tol=1e-12), orange, "-")]
fig, ax = plt.subplots(figsize=(5.0, 3.2))
for lab, o, c, ls in runs:
    ax.semilogy(np.arange(1, len(o["err"]) + 1), o["err"], color=c, ls=ls, lw=1.3, label=lab)
    print(lab, "iterations", o["iters"], "indicator", o["err"][-1],
          "max|ep - ref|/eY", np.max(np.abs(o["ep"] - ref)) / (sY / E))
ax.set_xlabel("iteration $k$")
ax.set_ylabel("error indicator")
ax.set_ylim(1e-12, 2)
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(OUT + "latin.pdf", bbox_inches="tight")

# ------------------------------------------------------- Bree convergence
# Bree case (a): X = 0.7, Y = 1.5, H = 0.02 E (slow shakedown), 100 layers
s0, Ep = 280.0, 200e3 / 0.7      # s0 = sigma_Y of the vessel (MPa), plane modulus E/(1-nu)
eY = s0 / Ep
tb = np.linspace(0, np.pi, 41)                     # one thermal cycle, 40 steps
B = bree_vessel(0.7 * s0, 1.5 * s0, lambda t: abs(np.sin(t)), E=Ep, sY=s0, H=0.02 * Ep,
                nlay=100)
# iterations on the period map Pi (Picard, Krasnoselskii-Mann, Anderson(5)),
# then on the whole history (direct cyclic sweep, plain and Anderson(5))
Pi = period_map(B, tb)
z0 = np.zeros(B.nf)
_, rP = fixed_point(Pi, z0, "picard", kmax=400, tol=1e-10 * eY)
_, rK = fixed_point(Pi, z0, "km", theta=0.5, kmax=400, tol=1e-10 * eY)
_, rA = fixed_point(Pi, z0, "anderson", m=5, kmax=400, tol=1e-10 * eY)
M = dcm_sweep_map(B, tb)
h0 = np.zeros(len(tb) * B.nf)
_, dP = fixed_point(M, h0, "picard", kmax=400, tol=1e-10 * eY)
ZA, dA = fixed_point(M, h0, "anderson", m=5, kmax=400, tol=1e-10 * eY)
fig, ax = plt.subplots(figsize=(5.0, 3.2))
for r, lab, c, ls in [(rP, "period map: cycle by cycle", green, "-"),
                      (rK, r"period map: Krasnoselskii--Mann $\theta=0.5$", green, ":"),
                      (rA, "period map: Anderson(5)", orange, "-"),
                      (dP, "direct cyclic", blue, "-"),
                      (dA, "direct cyclic + Anderson(5)", blue, "--")]:
    ax.semilogy(np.arange(1, len(r) + 1), np.maximum(r / eY, 1e-14), color=c, ls=ls, lw=1.3,
                label=lab)
    print(lab, len(r), r[-1] / eY)
ax.set_xlabel("iteration (cycle or sweep)")
ax.set_ylabel(r"residual $/\varepsilon_Y$")
ax.set_ylim(1e-11, 10)
ax.legend(fontsize=6.5)
fig.tight_layout()
fig.savefig(OUT + "bree_convergence.pdf", bbox_inches="tight")

# ------------------------------------------------------------------ Zarka
# Box 6.5 on the same case; limits of the cycles (zP) and of DCM + Anderson (zD)
_, Z = melan(B)
zP = fixed_point(Pi, z0, "picard", kmax=400, tol=1e-12 * eY)[0][-1]
zD = ZA[-1].reshape(len(tb), B.nf)[0]
sel = np.array([B.solve_elastic(t, z0)[2] for t in tb])
ep1 = incremental(B, tb[:21], z0)[2][-1]          # end of the first half-cycle
Y1 = B.H * ep1 + Z @ ep1
ez, rho, Yz, ok = zarka(B, sel, Y1)
lo, hi = sel.max(0) - s0, sel.min(0) + s0
# admissible band of Y (Box 6.5, step 2) and Y = (H I + Z) ep of the two limits
Yp, Yd = B.H * zP + Z @ zP, B.H * zD + Z @ zD
print("Zarka: elastic shakedown possible everywhere:", ok.all(),
      "| max|ep_Zarka - ep_cycles|/eY =", np.max(np.abs(ez - zP)) / eY,
      "| max|ep_DCM - ep_cycles|/eY =", np.max(np.abs(zD - zP)) / eY)
x = B.x
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.8))
axs[0].fill_between(x, lo / s0, hi / s0, color="0.88", label="admissible band")
axs[0].plot(x, Y1 / s0, color="0.4", lw=1.0, ls="--", label=r"$Y_1$ (half-cycle)")
axs[0].plot(x, Yz / s0, color=red, lw=1.6, label="Zarka: projection")
axs[0].plot(x, Yp / s0, color=green, lw=1.0, ls=":", label="cycle by cycle")
axs[0].plot(x, Yd / s0, color=blue, lw=1.0, ls="-.", label="direct cyclic + Anderson")
axs[0].set_xlabel("$x$")
axs[0].set_ylabel(r"$Y/\sigma_Y$")
axs[0].set_title("(a) transformed parameter", fontsize=9)
axs[0].legend(fontsize=6)
axs[1].plot(x, ez / eY, color=red, lw=1.6, label="Zarka")
axs[1].plot(x, zP / eY, color=green, lw=1.0, ls=":", label="cycle by cycle")
axs[1].plot(x, zD / eY, color=blue, lw=1.0, ls="-.", label="direct cyclic + Anderson")
axs[1].set_xlabel("$x$")
axs[1].set_ylabel(r"$\varepsilon^p/\varepsilon_Y$")
axs[1].set_title("(b) shakedown plastic strain", fontsize=9)
axs[1].legend(fontsize=6)
fig.tight_layout()
fig.savefig(OUT + "zarka.pdf", bbox_inches="tight")
