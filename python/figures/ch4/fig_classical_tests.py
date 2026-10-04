"""
Generates figures/ch4/classical_tests.pdf (Chapter 4, Section 4.3.2).
Theoretical curves of the von Mises and Tresca criteria in the coordinates
of the two classical experiments on yield of metals:
(a) tension-torsion of thin tubes (Taylor and Quinney 1931), in the plane
    (sigma/sigma_Y, tau/sigma_Y);
(b) Lode's diagram (Lode 1926): (sigma_1 - sigma_3)/sigma_Y against the
    stress parameter mu = (2 sigma_2 - sigma_1 - sigma_3)/(sigma_1 - sigma_3);
(c) Lode's strain-increment parameter nu against mu, for the Levy-Mises flow
    rule (nu = mu) and for the Tresca flow rule.
No experimental point is plotted: the measured values are not reproduced here
(see the text of Section 4.3.2 for what the measurements showed).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Figure 4.9. Shaded: the band between the two criteria. Stresses in units of sY.
blue, orange, grey = "#1F5AC8", "#D9822B", "0.6"
fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.05))

# (a) tension-torsion: sigma^2 + 3 tau^2 = 1 and sigma^2 + 4 tau^2 = 1
ax = axes[0]
s = np.linspace(0, 1, 200)
tm, tt = np.sqrt((1 - s**2) / 3), np.sqrt((1 - s**2) / 4)
ax.fill_between(s, tt, tm, color=grey, alpha=0.18, lw=0)
ax.plot(s, tm, color=blue, lw=2, label=r"von Mises $\sigma^2+3\tau^2=\sigma_Y^2$")
ax.plot(s, tt, color=orange, lw=2, label=r"Tresca $\sigma^2+4\tau^2=\sigma_Y^2$")
ax.set_xlabel(r"$\sigma/\sigma_Y$")
ax.set_ylabel(r"$\tau/\sigma_Y$")
ax.set_xlim(0, 1.12); ax.set_ylim(0, 0.75)
ax.text(0.03, 0.6, r"$1/\sqrt{3}$", color=blue, fontsize=8)
ax.text(0.03, 0.44, r"$1/2$", color=orange, fontsize=8)
ax.legend(loc="lower left", fontsize=7, frameon=False)
ax.grid(alpha=0.25)
ax.set_title("(a) tension–torsion (Taylor–Quinney)", fontsize=9)

# (b) Lode diagram: (s1-s3)/sY = 2/sqrt(3+mu^2) (von Mises), 1 (Tresca)
ax = axes[1]
mu = np.linspace(-1, 1, 201)
ax.plot(mu, 2 / np.sqrt(3 + mu**2), color=blue, lw=2, label="von Mises")
ax.plot(mu, np.ones_like(mu), color=orange, lw=2, label="Tresca")
ax.fill_between(mu, 1, 2 / np.sqrt(3 + mu**2), color=grey, alpha=0.18, lw=0)
ax.set_xlabel(r"$\mu=(2\sigma_2-\sigma_1-\sigma_3)/(\sigma_1-\sigma_3)$")
ax.set_ylabel(r"$(\sigma_1-\sigma_3)/\sigma_Y$")
ax.set_xlim(-1.05, 1.05); ax.set_ylim(0.95, 1.2)
ax.text(-1, 0.965, "tension", fontsize=7, ha="left")
ax.text(0, 0.965, "shear", fontsize=7, ha="center")
ax.text(1, 0.965, "compression", fontsize=7, ha="right")
ax.text(0, 2 / np.sqrt(3) + 0.006, r"$2/\sqrt{3}$", color=blue, fontsize=8,
        ha="center")
ax.legend(loc="upper right", fontsize=7, frameon=False)
ax.grid(alpha=0.25)
ax.set_title("(b) Lode's stress diagram", fontsize=9)

# (c) nu against mu: Levy-Mises nu = mu; Tresca nu = 0 on the faces,
#     nu in [-1,0] at mu=-1 and [0,1] at mu=1 (corners)
ax = axes[2]
ax.plot(mu, mu, color=blue, lw=2, label="L\u00e9vy\u2013Mises: $\\nu=\\mu$")
ax.plot([-1, -1, 1, 1], [-1, 0, 0, 1], color=orange, lw=2,
        label="Tresca (corners at $\\mu=\\pm1$)")
ax.set_xlabel(r"$\mu$")
ax.set_ylabel(r"$\nu$")
ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)
ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
ax.legend(loc="upper left", fontsize=7, frameon=False)
ax.grid(alpha=0.25)
ax.set_title("(c) Lode's flow diagram", fontsize=9)

fig.tight_layout()
fig.savefig("../../../figures/ch4/classical_tests.pdf")
