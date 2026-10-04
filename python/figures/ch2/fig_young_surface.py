"""Directional Young's modulus E(d) of tungsten, copper and zinc (Chapter 2).
Run: cd python/figures/ch2 && python3 fig_young_surface.py
"""
import numpy as np
from style_ch2 import plt, OUT
from matplotlib import cm, colors

def S_voigt(s11, s12, s13, s33, s44, s66):
    S = np.diag([s11, s11, s33, s44, s44, s66])
    S[0, 1] = S[1, 0] = s12
    S[0, 2] = S[2, 0] = S[1, 2] = S[2, 1] = s13
    return S

def young(S, d):
    """1/E(d) = (d x d) : S : (d x d), compliances in 1e-11 1/Pa -> E in GPa."""
    d1, d2, d3 = d
    s = np.array([d1**2, d2**2, d3**2, d2 * d3, d3 * d1, d1 * d2])
    return 100.0 / np.einsum("i...,ij,j...->...", s, S, s)

crystals = [  # Nye (1985), 1e-11 1/Pa
    ("tungsten (cubic)", S_voigt(0.257, -0.073, -0.073, 0.257, 0.660, 0.660)),
    ("copper (cubic)", S_voigt(1.49, -0.63, -0.63, 1.49, 1.33, 1.33)),
    ("zinc (hexagonal)", S_voigt(0.84, 0.11, -0.78, 2.87, 2.64, 2 * (0.84 - 0.11))),
]
th, ph = np.meshgrid(np.linspace(0, np.pi, 91), np.linspace(0, 2 * np.pi, 181))
d = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
Es = [young(S, d) for _, S in crystals]

fig = plt.figure(figsize=(7.0, 2.6))
for k, ((name, S), E) in enumerate(zip(crystals, Es)):
    ax = fig.add_subplot(1, 3, k + 1, projection="3d")
    x, y, z = E * d
    span = max(E.max() - E.min(), 1e-6 * E.max())   # each panel its own colour scale
    c = cm.viridis(0.15 + 0.8 * (E - E.min()) / span)
    ax.plot_surface(x, y, z, facecolors=c, rstride=2, cstride=2,
                    linewidth=0, antialiased=True, shade=True)
    R = 0.8 * E.max()
    ax.set_xlim(-R, R); ax.set_ylim(-R, R); ax.set_zlim(-R, R)
    ax.set_box_aspect((1, 1, 1)); ax.set_axis_off(); ax.view_init(22, 35)
    rng = (f"{E.min():.0f} GPa, isotropic" if span < 1e-3 * E.max()
           else f"{E.min():.0f}–{E.max():.0f} GPa")
    ax.set_title(f"{name}\n$E$: {rng}", pad=-4)
    print(f"{name:18s} Emin = {E.min():6.1f}  Emax = {E.max():6.1f} GPa")
fig.savefig(OUT + "young_surface.pdf", bbox_inches="tight")
# tungsten (cubic)   Emin =  389.1  Emax =  389.1 GPa
# copper (cubic)     Emin =   67.1  Emax =  192.0 GPa
# zinc (hexagonal)   Emin =   34.8  Emax =  124.1 GPa
