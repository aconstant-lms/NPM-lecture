"""
Generates figures/doublewell_bistability.pdf
Double-well potential example (remark after Section 4 of
variational_formulations.tex): the potential W(u) = (1/4)(u^2-1)^2 and the
gradient flow du/dt = u - u^3 from several perturbations of the unstable
state u=0, showing bistability.
"""
import numpy as np
from scipy.integrate import odeint
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Panel 1: the double-well potential ---
u = np.linspace(-1.6, 1.6, 400)
W = 0.25*(u**2 - 1)**2

# --- Panel 2: gradient flow du/dt = -W'(u) = u - u^3, from perturbations of the unstable state u=0 ---
def rhs(u, t):
    return u - u**3

t = np.linspace(0, 8, 400)
inits = [0.30, 0.10, 0.02, -0.02, -0.10, -0.30]
trajectories = {u0: odeint(rhs, u0, t).flatten() for u0 in inits}

fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))

ax = axes[0]
ax.plot(u, W, color="#0F6E56", lw=2)
for pt, lab in [(-1, "stable\nminimum"), (0, "unstable\nmaximum"), (1, "stable\nminimum")]:
    ax.plot(pt, 0.25*(pt**2-1)**2, "o", color="#D85A30", ms=6)
ax.annotate("unstable", xy=(0, 0.25), xytext=(0.15, 0.45),
            arrowprops=dict(arrowstyle="->", color="gray"), fontsize=8)
ax.annotate("well", xy=(-1, 0), xytext=(-1.5, 0.15),
            arrowprops=dict(arrowstyle="->", color="gray"), fontsize=8)
ax.annotate("well", xy=(1, 0), xytext=(1.15, 0.15),
            arrowprops=dict(arrowstyle="->", color="gray"), fontsize=8)
ax.set_xlabel("$u$"); ax.set_ylabel(r"$W(u)=\frac{1}{4}(u^2-1)^2$")
ax.set_title("the double-well potential")
ax.grid(alpha=0.3)

ax = axes[1]
colors = plt.cm.coolwarm(np.linspace(0, 1, len(inits)))
for (u0, traj), c in zip(trajectories.items(), colors):
    ax.plot(t, traj, color=c, label=f"$u(0)={u0:+.2f}$")
ax.axhline(1, color="gray", ls="--", lw=1)
ax.axhline(-1, color="gray", ls="--", lw=1)
ax.axhline(0, color="gray", ls=":", lw=1)
ax.set_xlabel("$t$"); ax.set_ylabel("$u(t)$")
ax.set_title(r"gradient flow $\dot u = u-u^3$: same equation," "\n" "opposite outcome depending on sign of perturbation")
ax.legend(frameon=False, fontsize=7, loc="center right")
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig("../../figures/ch1/doublewell_bistability.pdf")
print("saved. final values:", {u0: round(traj[-1],4) for u0,traj in trajectories.items()})
