"""Appendix A figure: Gateaux vs Frechet differentiability.

Run from anywhere:   python3 scripts/appA/fig_gateaux_frechet.py
Output: figures/appA/gateaux_frechet.pdf
Notation follows the book: bold vectors (mathtext \\mathbf), matrices upright bold.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.bbox": "tight",
})
BLUE, ORANGE, GREEN, RED, GRAY = "#1f5ac8", "#d9822b", "#14963c", "#b22222", "#777777"

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
APA = os.path.join(ROOT, "figures", "appA")
os.makedirs(APA, exist_ok=True)

K = np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]], dtype=float)
f = np.ones(3)


def save(fig, path):
    fig.savefig(path)
    plt.close(fig)
    print("wrote", os.path.relpath(path, ROOT))


# ---------------------------------------------------------------- Gateaux vs Frechet
def gateaux():
    fig = plt.figure(figsize=(7.2, 3.0))
    n = 161
    a = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(a, a)
    with np.errstate(invalid="ignore", divide="ignore"):
        G1 = np.where((X == 0) & (Y == 0), 0.0, X ** 3 * Y / (X ** 4 + Y ** 2))
        G2 = np.where((X == 0) & (Y == 0), 0.0, X ** 2 * Y / (X ** 2 + Y ** 2))
    for k, (G, ttl) in enumerate([
        (G2, r"$g_1=x^2y/(x^2+y^2)$: directional, not linear"),
        (G1, "$g_2=x^3y/(x^4+y^2)$: G\u00e2teaux, not Fr\u00e9chet"),
    ]):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        ax.plot_surface(X, Y, G, cmap="coolwarm", linewidth=0, rstride=2, cstride=2, alpha=0.95)
        if k == 1:
            t = np.linspace(-1, 1, 200)
            ax.plot(t, t ** 2, t / 2, color="k", lw=1.5)
        ax.set_title(ttl, fontsize=8)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.view_init(elev=28, azim=-55)
    save(fig, os.path.join(APA, "gateaux_frechet.pdf"))


if __name__ == "__main__":
    gateaux()
