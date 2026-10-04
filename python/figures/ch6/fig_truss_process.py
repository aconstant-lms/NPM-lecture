"""
Generates figures/ch6/truss_lattice.pdf and figures/ch6/truss_process.pdf
(Chapter 6). Three-bar truss, H = 0.05 E, constant Q2 = N0, alternating
Q1 = 1.2 N0 sin t.
truss_lattice: size of the plastic correction at each instant t_n and iteration k,
  (a) incremental initial-strain iterations (converged at each step, then next
  step), (b) iterations on the whole history (two periods, initial condition).
truss_process: (a) plastic strain of bar 1 over one period, iterates of the direct
  cyclic method; (b) the same quantity cycle after cycle (incremental);
  (c) convergence of the period: residual against the number of local
  evaluations (return maps) for cycle-by-cycle (Picard), Anderson and the direct
  cyclic method.
Where in the book: Figures 6.6 (truss_lattice) and 6.7 (truss_process), Section 6.3
("When it pays"); the lattice of computations (t_n, k) is that of Figure 6.5c-d.
Run from this directory: the module cy_core is imported from the examples of ch6.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
sys.path.insert(0, "../../examples/ch6")
from cy_core import three_bar_truss, incremental, global_local, period_map, fixed_point

blue, green, orange = "#1F5AC8", "#14963C", "#D9822B"
E, sY = 200e3, 200.0                               # modulus, yield stress (MPa)
eY = sY / E                                        # yield strain, unit of the plots
# third loading of Section 6.1.1 with kinematic hardening H = 0.05 E (N0 = sY)
S = three_bar_truss(E=E, sY=sY, H=0.05 * E,
                    Q=lambda t: np.array([1.2 * sY * np.sin(t), 1.0 * sY]))

# ------------------------------------------------------------ lattice figure
ts2 = np.linspace(0, 4 * np.pi, 81)                 # two periods, 40 steps each
_, _, ep_inc, iters, rec = incremental(S, ts2, np.zeros(3), "initial_strain",
                                       tol=1e-9, record=True)
# (a) incremental initial-strain iteration (Box 6.1), "for n, for k": the size of
# the plastic correction max_i |Delta ep_i| of each iteration k at each step n
kmax = 40
A = np.full((kmax, len(ts2) - 1), np.nan)
for n, steps in enumerate(rec):                      # correction of iteration k at step n
    d = np.r_[np.max(np.abs(steps[0] - ep_inc[n])),
              np.max(np.abs(np.diff(steps, axis=0)), axis=1)] if len(steps) > 1 else \
        np.array([np.max(np.abs(steps[0] - ep_inc[n]))])
    m = min(len(d), kmax)
    A[:m, n] = d[:m] / eY
# (b) whole-history iteration (Box 6.2, initial condition), "for k, for n":
# exactly kmax = 40 iterations (tol = 0), change of ep between iterations
out = global_local(S, ts2, closure="initial", kmax=kmax, tol=0.0, record=True)
R = np.array(out["rec_ep"])                          # (k, N+1, nf)
Bm = np.full((kmax, len(ts2) - 1), np.nan)
Bm[0] = np.max(np.abs(R[0, 1:]), axis=1) / eY
Bm[1:] = np.max(np.abs(np.diff(R[:, 1:], axis=0)), axis=2) / eY
print("incremental: iterations per step min/max/total", iters.min(), iters.max(), iters.sum())
print("whole history: sweeps", out["iters"], "final indicator", out["err"][-1])

fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
norm = LogNorm(1e-8, 1e0)
for ax, M, title in [(axs[0], A, "(a) incremental: for $n$, for $k$"),
                     (axs[1], Bm, "(b) whole history: for $k$, for $n$")]:
    Mm = np.where(M > 1e-8, M, np.where(np.isnan(M), np.nan, 1e-8))
    im = ax.imshow(Mm, origin="lower", aspect="auto", cmap="viridis_r", norm=norm,
                   extent=[0, 2, 0.5, kmax + 0.5], interpolation="nearest")
    ax.set_xlabel("time $t/T$")
    ax.set_title(title, fontsize=9)
axs[0].set_ylabel("iteration $k$")
cb = fig.colorbar(im, ax=axs, shrink=0.9, pad=0.02)
cb.set_label(r"$\max_i|\Delta\varepsilon^p_i|/\varepsilon_Y$", fontsize=8)
fig.savefig("../../../figures/ch6/truss_lattice.pdf", bbox_inches="tight")

# ------------------------------------------------------------ process figure
# one period of 40 steps; periodic state by three methods: Picard and Anderson on
# the period map Pi (Section 6.5.2), and the direct cyclic method (Box 6.2, periodic)
ts = np.linspace(0, 2 * np.pi, 41)
Pi = period_map(S, ts)
Zp, rp = fixed_point(Pi, np.zeros(3), "picard", kmax=300, tol=1e-9 * eY)
Za, ra = fixed_point(Pi, np.zeros(3), "anderson", m=3, kmax=300, tol=1e-9 * eY)
dcm = global_local(S, ts, closure="periodic", kmax=600, tol=1e-9, record=True)
print("cycles Picard", len(rp), "Anderson", len(ra), "DCM sweeps", dcm["iters"])
# cost: incremental Newton evaluations per cycle (sum of iterations)
cost_cycle = []
z = np.zeros(3)
for c in range(len(rp)):
    _, _, ep_c, it_c, _ = incremental(S, ts, z, "newton", tol=1e-10)
    cost_cycle.append(it_c.sum())
    z = ep_c[-1]
cost_cycle = np.cumsum(cost_cycle)
# errors on ep(0) with respect to the cycle-by-cycle limit zstar, in eY
zstar = Zp[-1]
Rd = np.array(dcm["rec_ep"])
res_dcm = np.max(np.abs(Rd[:, 0] - zstar), axis=1) / eY
res_pic = np.max(np.abs(Zp[:len(cost_cycle)] - zstar), axis=1) / eY

# (a) direct cyclic iterates, (b) cycles, (c) error against local evaluations
fig, axs = plt.subplots(1, 3, figsize=(7.6, 2.6))
tt = ts / (2 * np.pi)
cmap = plt.get_cmap("Blues")
ks = [0, 1, 4, 19, 99, len(Rd) - 1]
for j, k in enumerate(ks):
    axs[0].plot(tt, Rd[k, :, 0] / eY, color=cmap(0.35 + 0.65 * j / (len(ks) - 1)),
                lw=1.2, label=f"$k={k + 1}$")
axs[0].set_title("(a) direct cyclic, iterates", fontsize=9)
axs[0].legend(fontsize=6, ncol=2, loc="lower right")
z = np.zeros(3)
cyc = [1, 2, 5, 20, len(rp)]
gr = plt.get_cmap("Greens")
c = 0
for j, nc in enumerate(cyc):
    while c < nc:
        _, _, _, ep_c, _ = (lambda r: (None, None, None, r[2], None))(incremental(S, ts, z))
        z = ep_c[-1]
        c += 1
    axs[1].plot(tt, ep_c[:, 0] / eY, color=gr(0.35 + 0.65 * j / (len(cyc) - 1)),
                lw=1.2, label=f"cycle {nc}")
axs[1].set_title("(b) cycle by cycle", fontsize=9)
axs[1].legend(fontsize=6, loc="lower right")
for ax in axs[:2]:
    ax.set_xlabel("$t/T$")
axs[0].set_ylabel(r"$\varepsilon^p_1/\varepsilon_Y$")
N = len(ts) - 1                                  # one DCM sweep = N local evaluations
axs[2].semilogy(cost_cycle[:len(res_pic) - 1], np.maximum(res_pic[1:], 1e-12), color=green, lw=1.3,
                label="cycle by cycle")
axs[2].semilogy(cost_cycle[:len(ra)], np.maximum(ra / eY, 1e-12), color=orange, lw=1.3,
                label="Anderson on $\\Pi$")
axs[2].semilogy(N * np.arange(1, len(res_dcm) + 1), np.maximum(res_dcm, 1e-12), color=blue,
                lw=1.3, label="direct cyclic")
axs[2].set_xlabel("local evaluations")
axs[2].set_ylabel(r"error on $\varepsilon^p(0)$ $/\varepsilon_Y$")
axs[2].set_title("(c) convergence", fontsize=9)
axs[2].legend(fontsize=6)
fig.tight_layout()
fig.savefig("../../../figures/ch6/truss_process.pdf", bbox_inches="tight")
print("local evaluations to 1e-6: picard",
      cost_cycle[np.argmax(res_pic[1:] < 1e-6)], "dcm", N * (np.argmax(res_dcm < 1e-6) + 1))
