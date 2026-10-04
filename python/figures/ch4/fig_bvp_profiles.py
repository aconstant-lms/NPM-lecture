"""
Generates figures/ch4/bvp_profiles.pdf (Chapter 4, exercises on
boundary-value problems). Stress profiles under load (solid) and residual
stresses after elastic unloading (dashed), from the closed-form solutions of
the exercises, elastic-perfectly plastic material:
(a) thick-walled cylinder b = 2a, Tresca-type criterion sigma_t - sigma_r = k,
    plastic front at c = 1.5a; elastic unloading by Lame's solution;
(b) torsion of a shaft of radius R, elastic core c = R/2 and limit state;
    unloading tau_res = tau - T r / J, J = pi R^4 / 2;
(c) bending of a rectangular beam, elastic core c = h/4 and plastic moment;
    eps = -kappa y, unloading sigma_res = sigma + M y / I, I = b h^3 / 12.
The script also prints the key values quoted in the text.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Figure 4.17, for Exercises 4.24 (cylinder), 4.25 (torsion) and 4.26 (bending).
# Residual stresses = stresses under load minus the elastic solution for the same
# load (elastic unloading).
blue, green, orange = "#1F5AC8", "#14963C", "#D9822B"
fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.3))

# (a) thick cylinder, units: r/a, stresses/k
a, b, c = 1.0, 2.0, 1.5
# pressure p(c)/k = ln(c/a) + (1 - c^2/b^2)/2 for a plastic front at r = c
p = np.log(c / a) + 0.5 * (1 - c**2 / b**2)            # pressure for front c
r = np.linspace(a, b, 400)
A = c**2 / (2 * b**2)
# plastic zone r <= c: sigma_t - sigma_r = k; elastic zone: Lame with A = c^2/(2 b^2)
sr = np.where(r <= c, -p + np.log(r / a), A * (1 - b**2 / r**2))
st = np.where(r <= c, -p + np.log(r / a) + 1, A * (1 + b**2 / r**2))
L = p * a**2 / (b**2 - a**2)                           # Lame, pressure p
sr_res = sr - L * (1 - b**2 / r**2)
st_res = st - L * (1 + b**2 / r**2)
ax = axes[0]
ax.axvspan(a, c, color=orange, alpha=0.08, lw=0)
ax.plot(r, st, color=blue, lw=2, label=r"$\sigma_\theta$")
ax.plot(r, sr, color=green, lw=2, label=r"$\sigma_r$")
ax.plot(r, st_res, color=blue, lw=1.5, ls="--")
ax.plot(r, sr_res, color=green, lw=1.5, ls="--")
ax.axhline(0, color="k", lw=0.5)
ax.text(1.03, -0.3, "plastic\nzone", fontsize=7, color=orange)
ax.set_xlabel(r"$r/a$"); ax.set_ylabel(r"stress$/k$")
ax.set_xlim(a, b)
ax.legend(loc="lower right", fontsize=7, frameon=False)
ax.grid(alpha=0.25)
ax.set_title(r"(a) thick cylinder, $b=2a$, $c=1.5a$", fontsize=9)
print(f"cylinder: p/k = {p:.4f}, p_Y/k = {0.5*(1-a**2/b**2):.4f}, "
      f"p_L/k = {np.log(b/a):.4f}")
print(f"  residual at r=a: sigma_t/k = {st_res[0]:.4f}, "
      f"(sigma_t-sigma_r)/k = {st_res[0]-sr_res[0]:.4f}")
print(f"  residual at r=b: sigma_t/k = {st_res[-1]:.4f}")

# (b) torsion, units: r/R, tau/k
rr = np.linspace(0, 1, 400)
ax = axes[1]
# elastic core r < c (tau = k r/c), plastic ring tau = k; torque T(c)
for cc, col, lab in ((0.5, blue, r"$c=R/2$"), (0.0, orange, "limit")):
    tau = np.where(rr < cc, rr / cc if cc > 0 else 1.0, 1.0)
    T = 2 * np.pi / 3 * (1 - cc**3 / 4)                # T/(k R^3)
    J = np.pi / 2
    tau_res = tau - T * rr / J
    ax.plot(rr, tau, color=col, lw=2, label=lab)
    ax.plot(rr, tau_res, color=col, lw=1.5, ls="--")
    print(f"torsion c={cc}: T/T_Y = {T/(np.pi/2):.4f}, "
          f"tau_res(R)/k = {tau_res[-1]:.4f}, max |tau_res|/k = "
          f"{np.abs(tau_res).max():.4f}, int tau_res r^2 dr = "
          f"{np.trapezoid(tau_res*rr**2, rr):.1e}")
ax.axhline(0, color="k", lw=0.5)
ax.set_xlabel(r"$r/R$"); ax.set_ylabel(r"$\tau/k$")
ax.set_xlim(0, 1); ax.set_ylim(-0.5, 1.15)
ax.legend(loc="lower left", fontsize=7, frameon=False)
ax.grid(alpha=0.25)
ax.set_title("(b) torsion of a shaft", fontsize=9)

# (c) bending, units: y/(h/2), sigma/sigma_Y; eps = -kappa y
y = np.linspace(-1, 1, 801)
ax = axes[2]
# elastic core |y| < c (linear stress), plastic layers sigma = -sign(y) sigma_Y
for cc, col, lab in ((0.5, blue, r"$c=h/4$"), (0.0, orange, r"$M_p$")):
    sig = np.where(np.abs(y) < cc, -y / cc if cc > 0 else -np.sign(y),
                   -np.sign(y))
    M = 1 - cc**2 / 3                                   # M/(sigma_Y b h^2/4)
    # sigma_res = sigma + M y / I, I = b h^3/12: M y / I = 1.5 M y (units)
    sig_res = sig + 3 * M * y / 2
    ax.plot(sig, y, color=col, lw=2, label=lab)
    ax.plot(sig_res, y, color=col, lw=1.5, ls="--")
    print(f"bending c={cc}: M/M_Y = {1.5*M:.4f}, sigma_res at y=-h/2 "
          f"(stretched fibre) = {sig_res[0]:.4f} sigma_Y")
ax.axvline(0, color="k", lw=0.5)
ax.set_xlabel(r"$\sigma/\sigma_Y$"); ax.set_ylabel(r"$y/(h/2)$")
ax.set_xlim(-1.25, 1.25); ax.set_ylim(-1, 1)
ax.legend(loc="upper right", fontsize=7, frameon=False)
ax.grid(alpha=0.25)
ax.set_title(r"(c) bending, $\varepsilon=-\kappa y$", fontsize=9)

fig.legend(handles=[plt.Line2D([], [], color="k", lw=2, label="under load"),
                    plt.Line2D([], [], color="k", lw=1.5, ls="--",
                               label="residual, after elastic unloading")],
           loc="lower center", ncol=2, fontsize=8, frameon=False)
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig("../../../figures/ch4/bvp_profiles.pdf")
