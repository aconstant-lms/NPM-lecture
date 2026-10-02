"""
Generates figures_notesC1/notes_coercivity.pdf
Notes C1, Exercise (positive definiteness): the same coercivity/boundedness
bounding-lines plot as ex3_coercivity.pdf, regenerated for the stand-alone
notes document with its own variable names (x, lambda_min, lambda_max).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

K = np.array([[2,-1,0],[-1,2,-1],[0,-1,2]], dtype=float)
eigvals = np.linalg.eigvalsh(K)
lam_min, lam_max = eigvals[0], eigvals[-1]

rng = np.random.default_rng(0)
n_samples = 300
u_samples = rng.normal(size=(n_samples, 3))
norm_sq = np.sum(u_samples**2, axis=1)
xtKx = np.einsum("ij,jk,ik->i", u_samples, K, u_samples)

assert np.all(xtKx >= lam_min*norm_sq - 1e-9)
assert np.all(xtKx <= lam_max*norm_sq + 1e-9)

fig, ax = plt.subplots(figsize=(5.2, 3.7))
xmax = norm_sq.max()*1.05
xs = np.linspace(0, xmax, 10)
ax.fill_between(xs, lam_min*xs, lam_max*xs, color="#378ADD", alpha=0.08)
ax.scatter(norm_sq, xtKx, s=14, alpha=0.6, color="#378ADD",
           label=r"$(\mathbf{u}^\top\mathbf{u},\ \mathbf{u}^\top\mathbf{K}\mathbf{u})$, random $\mathbf{u}$")
ax.plot(xs, lam_min*xs, "--", color="#0F6E56", lw=1.5,
        label=r"$\mathbf{u}^\top\mathbf{K}\mathbf{u} = \lambda_{\min}\, \mathbf{u}^\top\mathbf{u}$")
ax.plot(xs, lam_max*xs, "--", color="#D85A30", lw=1.5,
        label=r"$\mathbf{u}^\top\mathbf{K}\mathbf{u} = \lambda_{\max}\, \mathbf{u}^\top\mathbf{u}$")
ax.set_xlabel(r"$\mathbf{u}^\top\mathbf{u}$")
ax.set_ylabel(r"$\mathbf{u}^\top\mathbf{K}\mathbf{u}$")
ax.legend(frameon=False, fontsize=8, loc="upper left")
ax.grid(alpha=0.3)
ax.set_xlim(0, xmax)
ax.set_ylim(0, xtKx.max()*1.05)
fig.tight_layout()
fig.savefig("../../figures/ch1/notes_coercivity.pdf")
print("lambda_min =", lam_min, " lambda_max =", lam_max)
print("all points bounded correctly:", True)
