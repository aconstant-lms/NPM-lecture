"""
Generates figures/ch4/filament_cycles.pdf (Chapter 4, Section 4.2.3).
Elastoplastic filament with isotropic, kinematic and combined linear hardening
(Simo and Hughes, Section 1.3) under the strain cycle 0 -> e -> -e -> e.
The backward-Euler return map is exact for this model on monotone strain
increments, so small increments give the exact response.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

E, sY = 200e3, 250.0            # MPa
eY = sY / E
emax = 4 * eY


def response(K, H, path, nstep=400):
    """Stress along a piecewise linear strain path (exact return map)."""
    ep, q, al, eps_prev = 0.0, 0.0, 0.0, 0.0
    eps_all, sig_all = [0.0], [0.0]
    for a, b in zip(path[:-1], path[1:]):
        for eps in np.linspace(a, b, nstep)[1:]:
            s_tr = E * (eps - ep)
            f_tr = abs(s_tr - q) - (sY + K * al)
            if f_tr > 0:
                dg = f_tr / (E + K + H)
                n = np.sign(s_tr - q)
                ep += dg * n
                q += H * dg * n
                al += dg
            eps_all.append(eps)
            sig_all.append(E * (eps - ep))
    return np.array(eps_all), np.array(sig_all)


path = [0.0, emax, -emax, emax]
cases = [("isotropic", 20e3, 0.0), ("kinematic", 0.0, 20e3),
         ("combined", 10e3, 10e3)]
blue, green, orange = "#1F5AC8", "#14963C", "#D9822B"

fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3), sharey=True)
for ax, (name, K, H) in zip(axes, cases):
    e0, s0 = response(0.0, 0.0, path)
    ax.plot(e0 / eY, s0, color="0.7", lw=1.0)
    e, s = response(K, H, path)
    n1 = np.argmax(e >= emax - 1e-15)            # end of first loading
    ax.plot(e[:n1 + 1] / eY, s[:n1 + 1], color=blue, lw=2)
    ax.plot(e[n1:] / eY, s[n1:], color=orange, lw=2)
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_title(f"{name}  ($K$={K/1e3:.0f} GPa, $H$={H/1e3:.0f} GPa)",
                 fontsize=9)
    ax.set_xlabel(r"$\varepsilon/\varepsilon_Y$")
    ax.grid(alpha=0.25)
    print(name, "peaks:", np.round([s[n1], s[np.argmin(e)], s[-1]], 2))
axes[0].set_ylabel(r"$\sigma$ [MPa]")
fig.tight_layout()
fig.savefig("../../../figures/ch4/filament_cycles.pdf")
# isotropic peaks: [ 318.18 -442.15  543.58]
# kinematic peaks: [ 318.18 -318.18  318.18]
# combined peaks:  [ 318.18 -380.17  436.51]
