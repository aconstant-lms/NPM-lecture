"""
Generates figures/ch1/resonance_example.pdf (Figure 1.10 of Chapter 1).
Linear resonance example of Section 1.6 ("What goes wrong without
coercivity", paragraph (a) A linear resonance): the solvability test for
-u''-pi^2 u = f, u(0) = u(1) = 0 (two right-hand sides), and the resulting
one-parameter family of solutions for the compatible case.
Run: cd python/figures/ch1 && python3 fig_resonance.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

x = np.linspace(0, 1, 400)
eig = np.sin(np.pi*x)          # the kernel / first eigenfunction

# Fredholm alternative: a solution exists iff int_0^1 f sin(pi x) dx = 0.
f1 = np.ones_like(x)                 # f = 1: incompatible
f2 = np.sin(2*np.pi*x)               # f = sin(2 pi x): compatible

prod1 = f1*eig
prod2 = f2*eig

fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))

# Left: the integrand f sin(pi x) of the test; net area 2/pi for f = 1, 0 for f2.
ax = axes[0]
ax.plot(x, prod1, color="#D85A30", label=r"$f=1$: $f(x)\sin(\pi x)$")
ax.fill_between(x, prod1, 0, color="#D85A30", alpha=0.18)
ax.plot(x, prod2, color="#0F6E56", label=r"$f=\sin(2\pi x)$: $f(x)\sin(\pi x)$")
ax.fill_between(x, prod2, 0, where=(prod2>=0), color="#0F6E56", alpha=0.25)
ax.fill_between(x, prod2, 0, where=(prod2<0), color="#0F6E56", alpha=0.10)
ax.axhline(0, color="gray", lw=0.8)
ax.set_xlabel("$x$")
ax.set_title("solvability test: " r"$\int_0^1 f(x)\sin(\pi x)\,dx$")
ax.text(0.5, 0.75, r"net area $=\frac{2}{\pi}\ne 0$" "\n" "(no solution)",
        color="#993C1D", fontsize=8, ha="center")
ax.text(0.5, -0.55, "positive and negative\nlobes cancel exactly:\nnet area $=0$\n(solutions exist)",
        color="#085041", fontsize=8, ha="center")
ax.legend(frameon=False, fontsize=8, loc="upper left")
ax.grid(alpha=0.3)

# Right: u_c = sin(2 pi x)/(3 pi^2) + c sin(pi x), a solution for every c.
ax = axes[1]
cs = [-0.15, -0.075, 0.0, 0.075, 0.15]
colors = ["#993C1D", "#D85A30", "#534AB7", "#7F77DD", "#26215C"]
for c, col in zip(cs, colors):
    u_c = np.sin(2*np.pi*x)/(3*np.pi**2) + c*np.sin(np.pi*x)
    ax.plot(x, u_c, color=col, lw=1.8, label=f"$c={c:+.3f}$")
ax.set_xlabel("$x$"); ax.set_ylabel("$u(x)$")
ax.set_title(r"every $u_c=\frac{\sin(2\pi x)}{3\pi^2}+c\sin(\pi x)$ solves it")
ax.legend(frameon=False, fontsize=7, loc="upper right")
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig("../../../figures/ch1/resonance_example.pdf")
print("saved")
