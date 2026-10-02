"""
Generates figures/ex3_coercivity.pdf
Exercise 3 (Section 4 of variational_formulations.tex): verifies BOTH the
coercivity lower bound alpha*||u||^2 and the boundedness upper bound
M*||u||^2 on 300 random vectors, plotted against ||u||^2 so both bounding
lines have simple slopes alpha and M. (Corrected version -- the original
plot only showed the coercivity line.)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

K = np.array([[2,-1,0],[-1,2,-1],[0,-1,2]], dtype=float)
eigvals = np.linalg.eigvalsh(K)
alpha, M = eigvals[0], eigvals[-1]

rng = np.random.default_rng(0)
n_samples = 300
u_samples = rng.normal(size=(n_samples, 3))
norm_sq = np.sum(u_samples**2, axis=1)                       # ||u||^2
a_uu = np.einsum("ij,jk,ik->i", u_samples, K, u_samples)      # a(u,u)

# sanity check both bounds numerically before plotting
assert np.all(a_uu >= alpha*norm_sq - 1e-9)
assert np.all(a_uu <= M*norm_sq + 1e-9)

fig, ax = plt.subplots(figsize=(5.4, 3.8))

xmax = norm_sq.max()*1.05
xs = np.linspace(0, xmax, 10)

ax.fill_between(xs, alpha*xs, M*xs, color="#378ADD", alpha=0.08,
                 label=r"admissible band $\alpha\|u\|^2 \leq a(u,u) \leq M\|u\|^2$")
ax.scatter(norm_sq, a_uu, s=14, alpha=0.6, color="#378ADD",
           label=r"$(\|u\|^2,\ a(u,u))$, random $u$")
ax.plot(xs, alpha*xs, "--", color="#0F6E56", lw=1.5,
        label=r"lower bound: $a(u,u)=\alpha\|u\|^2$ (coercivity)")
ax.plot(xs, M*xs, "--", color="#D85A30", lw=1.5,
        label=r"upper bound: $a(u,u)=M\|u\|^2$ (boundedness)")

ax.set_xlabel(r"$\|u\|^2$")
ax.set_ylabel(r"$a(u,u)=\mathbf{u}^\top\mathbf{K}\mathbf{u}$")
ax.legend(frameon=False, fontsize=7.5, loc="upper left")
ax.grid(alpha=0.3)
ax.set_xlim(0, xmax)
ax.set_ylim(0, a_uu.max()*1.05)
fig.tight_layout()
fig.savefig("../../figures/ch1/ex3_coercivity.pdf")
print("all points within [alpha*||u||^2, M*||u||^2]:",
      bool(np.all(a_uu >= alpha*norm_sq - 1e-9) and np.all(a_uu <= M*norm_sq + 1e-9)))
print("alpha =", alpha, " M =", M)
