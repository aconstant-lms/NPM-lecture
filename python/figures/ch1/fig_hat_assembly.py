"""
Generates figures/ch1/hat_assembly.pdf (Figure 1.13; Chapter 1, Section 1.9.1
"Linear finite elements in one dimension").
(a) The hat functions of linear finite elements on (0,1), 4 elements.
(b) Element-by-element assembly: each element e adds its 2x2 matrix
    k^e = (1/h) [[1,-1],[-1,1]] to the rows and columns of its two nodes;
    deleting the rows and columns of the fixed end nodes leaves
    K = (1/h) tridiag(-1, 2, -1), the matrix of the spring chain.
Run: cd python/figures/ch1 && python3 fig_hat_assembly.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

BLUE, GREEN, ORANGE, GRAY = "#1F5AC8", "#14963C", "#D85A30", "#666666"
ECOL = ["#1F5AC8", "#14963C", "#D85A30", "#7F3FBF"]   # one colour per element
# Mesh: nel elements of size h on (0, 1), nodes x_0 = 0, ..., x_nel = 1.
nel = 4
h = 1.0 / nel
xn = np.linspace(0, 1, nel + 1)

# assembly, printed for the record
K = np.zeros((nel + 1, nel + 1))
# Element stiffness k^e_ij = int_e phi_i' phi_j' dx = (1/h) [[1, -1], [-1, 1]],
# added to the rows and columns of the nodes e and e+1.
ke = np.array([[1, -1], [-1, 1]]) / h
for e in range(nel):
    K[e:e + 2, e:e + 2] += ke
print("h*K (all nodes) =\n", (h * K).astype(int))
print("h*K (interior)  =\n", (h * K[1:-1, 1:-1]).astype(int))

fig, (ax, bx) = plt.subplots(1, 2, figsize=(10, 3.4),
                             gridspec_kw=dict(width_ratios=[1.15, 1]))

# (a) hat functions
x = np.linspace(0, 1, 801)
# phi_i(x) = max(0, 1 - |x - x_i| / h); slopes +-1/h.
for i in range(nel + 1):
    phi = np.clip(1 - np.abs(x - xn[i]) / h, 0, None)
    interior = 0 < i < nel
    ax.plot(x, phi, color=BLUE if interior else GRAY,
            lw=1.6 if interior else 0.9, ls="-" if interior else ":")
    ax.text(xn[i], 1.05, rf"$\varphi_{i}$", ha="center", va="bottom",
            color=BLUE if interior else GRAY, fontsize=9)
phi2 = np.clip(1 - np.abs(x - xn[2]) / h, 0, None)
ax.fill_between(x, phi2, color=BLUE, alpha=0.12, lw=0)
for e in range(nel):
    ax.plot([xn[e], xn[e + 1]], [-0.08, -0.08], color=ECOL[e], lw=4,
            solid_capstyle="butt")
    ax.text(xn[e] + h / 2, -0.2, f"element {e + 1}", color=ECOL[e],
            ha="center", va="top", fontsize=8)
ax.plot(xn, np.zeros_like(xn), "ko", ms=4)
ax.annotate(r"$\varphi_2'=1/h$", xy=(xn[2] - 0.2 * h, 0.8),
            xytext=(xn[1] + 0.12 * h, 0.9), fontsize=8, color=BLUE,
            ha="left", va="center",
            arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.8))
ax.annotate(r"$\varphi_2'=-1/h$", xy=(xn[2] + 0.2 * h, 0.8),
            xytext=(xn[3] - 0.12 * h, 0.9), fontsize=8, color=BLUE,
            ha="right", va="center",
            arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.8))
ax.set_xticks(xn)
ax.set_xticklabels([r"$x_0=0$", r"$x_1$", r"$x_2$", r"$x_3$", r"$x_4=1$"])
ax.set_yticks([0, 1])
ax.set_ylim(-0.38, 1.2)
ax.spines[["top", "right"]].set_visible(False)
ax.set_title("(a) hat functions; the end ones (dotted) are removed by "
             r"$u(0)=u(1)=0$", fontsize=9)

# (b) assembly of the global matrix
# Draw the 5x5 global matrix h K: one coloured square per element block; the
# diagonal entries of interior nodes receive two contributions ("1+1").
n = nel + 1
for e in range(nel):
    pad = 0.06 + 0.04 * (e % 2)
    bx.add_patch(Rectangle((e + pad, e + pad), 2 - 2 * pad, 2 - 2 * pad,
                           fc=ECOL[e], alpha=0.16, ec=ECOL[e], lw=1.2))
for i in range(n):
    for j in range(n):
        contrib = [e for e in range(nel) if e <= i <= e + 1 and e <= j <= e + 1]
        if not contrib:
            continue
        if i == j and len(contrib) == 2:
            txt = "1+1"
        else:
            txt = f"{int(round(h * K[i, j]))}"
        boundary = i in (0, n - 1) or j in (0, n - 1)
        bx.text(j + 0.5, i + 0.5, txt, ha="center", va="center",
                fontsize=11, color=GRAY if boundary else "black")
# Frame the interior block (boundary rows/columns, in grey, are removed).
bx.add_patch(Rectangle((1, 1), 3, 3, fc="none", ec="black", lw=1.8, zorder=5))
for k in (0, n - 1):
    bx.add_patch(Rectangle((0, k), n, 1, fc=GRAY, alpha=0.10, lw=0))
    bx.add_patch(Rectangle((k, 0), 1, n, fc=GRAY, alpha=0.10, lw=0))
for k in range(n + 1):
    bx.plot([0, n], [k, k], color=GRAY, lw=0.4)
    bx.plot([k, k], [0, n], color=GRAY, lw=0.4)
bx.set_xlim(-0.9, n + 0.1); bx.set_ylim(n + 0.1, -0.9)
bx.set_aspect("equal")
bx.set_xticks(np.arange(n) + 0.5); bx.set_yticks(np.arange(n) + 0.5)
bx.set_xticklabels(range(n)); bx.set_yticklabels(range(n))
bx.tick_params(length=0, labelsize=8)
bx.xaxis.tick_top()
for s in bx.spines.values():
    s.set_visible(False)
bx.text(n / 2, n + 0.55,
        r"$h\,\mathbf{K}$: the framed block is $\mathrm{tridiag}(-1,2,-1)$",
        ha="center", va="center", fontsize=9)
bx.set_title(r"(b) assembly of $\mathbf{k}^e=\frac{1}{h}\,[\,1\ \ {-1}\,;\ {-1}\ \ 1\,]$"
             ", one colour per element", fontsize=9, pad=16)

fig.tight_layout(w_pad=2)
fig.savefig("../../../figures/ch1/hat_assembly.pdf")
print("saved figures/ch1/hat_assembly.pdf")
