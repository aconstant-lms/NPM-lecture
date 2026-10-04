"""
Generates figures/ch5/euler_filament.pdf (Chapter 5, Section 5.2).
Filament with combined hardening under the strain cycle 0 -> 4 eY -> -4 eY,
integrated with 3 steps per branch: backward Euler (return map) against
forward Euler (continuum tangent, no return), and the exact response.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "../../examples/ch5")
from ci_core import return_map_1d, forward_euler_1d

# MPa; K isotropic and H kinematic hardening moduli; strain path in units of eY
E, sY, K, H = 200e3, 250.0, 10e3, 10e3
eY = sY / E
path = [0.0, 4 * eY, -4 * eY]


# nstep steps per branch; scheme 'BE' = return map (Box 5.1),
# 'FE' = forward Euler with the continuum tangent. Returns eps/eY and sigma (MPa)
def run(nstep, scheme):
    eps_all, sig_all = [0.0], [0.0]
    ep = al = q = 0.0
    eps_prev = 0.0
    for a, b in zip(path[:-1], path[1:]):
        for eps in np.linspace(a, b, nstep + 1)[1:]:
            if scheme == "BE":
                s, ep, al, q, _ = return_map_1d(eps, ep, al, q, E, sY, K, H)
            else:
                s, ep, al, q, _ = forward_euler_1d(eps_prev, eps - eps_prev, ep, al, q,
                                                   E, sY, K, H)
            eps_prev = eps
            eps_all.append(eps)
            sig_all.append(s)
    return np.array(eps_all) / eY, np.array(sig_all)


blue, green, orange = "#1F5AC8", "#14963C", "#D9822B"
fig, ax = plt.subplots(figsize=(5.6, 3.6))
# Exact response: the return map is exact on monotone steps, so many small steps
# of backward Euler give the exact curve
e, s = run(2000, "BE")
ax.plot(e, s, color="0.6", lw=1.2, label="exact")
e, s = run(3, "BE")
ax.plot(e, s, "o", color=blue, ms=5, label="backward Euler (return map)")
e, s = run(3, "FE")
ax.plot(e, s, "s--", color=orange, ms=4, lw=1, label="forward Euler")
ax.axhline(0, color="k", lw=0.5)
ax.axvline(0, color="k", lw=0.5)
ax.set_xlabel(r"$\varepsilon/\varepsilon_Y$")
ax.set_ylabel(r"$\sigma$ [MPa]")
ax.grid(alpha=0.25)
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig("../../../figures/ch5/euler_filament.pdf")
print("FE stress after first branch: %.2f, exact %.2f MPa" % (run(3, "FE")[1][3], run(3, "BE")[1][3]))
# FE stress after first branch: 393.94, exact 318.18 MPa
