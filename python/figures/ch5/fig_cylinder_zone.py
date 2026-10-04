"""
Generates figures/ch5/cylinder_zone.pdf (Chapter 5, exercise "The thick-walled
cylinder by finite elements"). Perfect J2 plasticity, a = 100, b = 200 mm,
plane strain: radius c of the plastic zone against the pressure p/p_L, finite
elements (ci_fem.solve, 200 elements, 20 steps per computation) against the
closed form p(c) = k ln(c/a) + k (1 - c^2/b^2)/2, k = 2 sY/sqrt(3).
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "../../examples/ch5")
from ci_core import Linear
from ci_fem import solve, p_of_c

blue, orange = "#1F5AC8", "#D9822B"
# Material (MPa) and radii (mm); k = 2 sY/sqrt3, limit pressure p_L = k ln(b/a)
E, nu, sY = 200e3, 0.3, 250.0
mu, kappa = E / (2 * (1 + nu)), E / (3 * (1 - 2 * nu))
a, b = 100.0, 200.0
k = 2 * sY / np.sqrt(3.0)
pL = k * np.log(b / a)
pe = p_of_c(a, a, b, k)                      # first yield at the bore

# Closed form p(c) of the Tresca-type solution (ci_fem.p_of_c)
c = np.linspace(a, b, 200)
fig, ax = plt.subplots(figsize=(5.0, 3.4))
ax.plot(p_of_c(c, a, b, k) / pL, c, "-", color=orange, lw=1.8, label="closed form")
# Finite elements: one computation per pressure, 20 steps, 200 elements;
# the front c is the outer node of the last element with alpha > 0
fr = np.array([0.56, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.98])
cfe = []
for f in fr:
    u, r, al, sig, its, _ = solve(a, b, f * pL, 20, 200, mu, kappa, Linear(sY, 0.0))
    pl = np.where(al > 1e-12)[0]
    cfe.append(r[pl[-1] + 1] if len(pl) else a)
    print(f"p/pL = {f:.2f}: c_FE = {cfe[-1]:.2f} mm")
ax.plot(fr, cfe, "o", color=blue, ms=5, label="finite elements (200 elements)")
ax.axvline(pe / pL, color="0.5", lw=0.8, ls=":")
ax.text(pe / pL + 0.01, 190, "first yield", fontsize=8, color="0.4")
ax.set_xlim(0.5, 1.0)
ax.set_ylim(a - 3, b + 3)
ax.set_xlabel(r"pressure $p/p_L$")
ax.set_ylabel(r"radius of the plastic zone $c$ (mm)")
ax.grid(alpha=0.25)
ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(0.12, 0.88))
fig.tight_layout()
fig.savefig("../../../figures/ch5/cylinder_zone.pdf")
print(f"p_e/p_L = {pe / pL:.4f}")
