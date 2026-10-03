"""
Generates figures/ex8_9_10_profiles.pdf
Exercises 8-10 (Part II of variational_formulations.tex): three-panel
solution-profile comparison -- the non-unique pure-Neumann family, the
reaction-diffusion solution, and the linear-vs-nonlinear (p-Laplacian)
comparison.
"""
import numpy as np
from scipy.special import cbrt
from scipy.integrate import quad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

x = np.linspace(0, 1, 200)

# --- Exercise 8: pure Neumann problem, family u(x) = -x^2/2 + C ---
Cs = [-0.1, 0.0, 0.1]

# --- Exercise 9: -u'' + 2u = x, u(0)=u(1)=0 (evaluate the exact solution) ---
import sympy as sp
xs = sp.symbols('x')
u9 = sp.Function('u')
sol9 = sp.dsolve(sp.Eq(-u9(xs).diff(xs,2)+2*u9(xs), xs), u9(xs), ics={u9(0):0, u9(1):0}).rhs
u9_func = sp.lambdify(xs, sol9, "numpy")
y9 = u9_func(x)

# --- Exercise 10: nonlinear -( (u')^3 )' = 6 vs linear -u''=6, both u(0)=u(1)=0 ---
f10 = 6.0
up10 = lambda t: cbrt(3 - f10*t)          # u'(x) = (3-6x)^(1/3), from symmetry u'(1/2)=0
y10_nonlin = np.array([quad(up10, 0, xi)[0] for xi in x])
y10_lin = f10/2 * x*(1-x)                  # -u''=6, u(0)=u(1)=0 -> u = 3x(1-x)

print("nonlinear u(1) (should be ~0):", quad(up10, 0, 1)[0])

fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))

ax = axes[0]
colors = ["#7F77DD", "#534AB7", "#26215C"]
for C, c in zip(Cs, colors):
    ax.plot(x, -x**2/2 + C, color=c, label=f"$C={C:+.1f}$")
ax.set_title("pure Neumann:\nunique up to a constant")
ax.set_xlabel("$x$"); ax.set_ylabel("$u(x)$")
ax.legend(frameon=False, fontsize=8)
ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(x, y9, color="#0F6E56")
ax.set_title(r"$-u''+2u=x$" "\n" r"$u(0)=u(1)=0$")
ax.set_xlabel("$x$"); ax.set_ylabel("$u(x)$")
ax.grid(alpha=0.3)

ax = axes[2]
ax.plot(x, y10_lin, "--", color="#888780", label=r"linear ($p=2$)")
ax.plot(x, y10_nonlin, color="#D85A30", label=r"nonlinear ($p=4$)")
ax.set_title("linear vs.\n$p$-Laplacian, same $f=6$")
ax.set_xlabel("$x$"); ax.set_ylabel("$u(x)$")
ax.legend(frameon=False, fontsize=8)
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig("../../../figures/ch1/ex8_9_10_profiles.pdf")
print("saved")
