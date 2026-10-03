"""
Generates figures/ch1/ex5_convergence_loglog.pdf (Chapter 1, Exercise conv).
-u'' = sin(pi x) on (0,1), u(0) = u(1) = 0, solved with the matrix
(1/h) tridiag(-1,2,-1) and two loads:
  lumped     F_i = h f(x_i)        (the finite-difference scheme),
  consistent F_i = int f phi_i     (linear finite elements).
Left: maximal nodal error (O(h^2) for the lumped load; machine precision
for the consistent load, not plotted). Right: L2 error O(h^2)
(Aubin-Nitsche) and H1-seminorm error O(h) (Cea) of the finite element
solution.
Run: cd python/figures/ch1 && python3 fig5_convergence_loglog.py
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "../../examples/ch1")
import contextlib, io
with contextlib.redirect_stdout(io.StringIO()):
    from exo_conv import solve                       # same solver as the exercise

BLUE, GREEN, ORANGE, GRAY = "#1F5AC8", "#14963C", "#D85A30", "#666666"
ns = [3, 7, 15, 31, 63, 127]
lump = np.array([solve(n, "lumped") for n in ns])
cons = np.array([solve(n, "consistent") for n in ns])
h = lump[:, 0]

fig, (ax, bx) = plt.subplots(1, 2, figsize=(9, 3.4))
ax.loglog(h, lump[:, 1], "o-", color=BLUE, ms=4,
          label="lumped load (finite differences)")
ax.loglog(h, lump[0, 1] * (h / h[0])**2, "--", color=GRAY, lw=0.9,
          label="slope 2")
ax.set_xlabel("mesh size $h$"); ax.set_ylabel(r"$\max_i|u_i-u(x_i)|$")
ax.set_title("(a) nodal error", fontsize=10)
ax.text(0.97, 0.05, "consistent load: nodal error\n"
        rf"$\leq10^{{{int(np.ceil(np.log10(cons[:, 1].max())))}}}$"
        " (exact at the nodes)",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
        color=GREEN)
ax.legend(frameon=False, fontsize=8, loc="upper left")
ax.grid(True, which="both", alpha=0.3)

bx.loglog(h, cons[:, 3], "s-", color=ORANGE, ms=4,
          label=r"$|u-u_h|_{H^1}$ (C\'ea)".replace("\\'e", "é"))
bx.loglog(h, cons[:, 2], "o-", color=GREEN, ms=4,
          label=r"$\|u-u_h\|_{L^2}$ (Aubin–Nitsche)")
bx.loglog(h, cons[0, 3] * (h / h[0]), "--", color=GRAY, lw=0.9,
          label="slopes 1 and 2")
bx.loglog(h, cons[0, 2] * (h / h[0])**2, "--", color=GRAY, lw=0.9)
bx.set_xlabel("mesh size $h$")
bx.set_title("(b) finite elements, consistent load", fontsize=10)
bx.legend(frameon=False, fontsize=8, loc="lower right")
bx.grid(True, which="both", alpha=0.3)

fig.tight_layout()
fig.savefig("../../../figures/ch1/ex5_convergence_loglog.pdf")
for name, t in [("lumped", lump), ("consistent", cons)]:
    r = np.log(t[:-1, 1:] / t[1:, 1:]) / np.log(2)
    print(name, "rates (nodal, L2, H1):", np.round(r[-1], 2))
print("saved")
