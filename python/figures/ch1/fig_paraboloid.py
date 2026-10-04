"""
Generates figures/ch1/paraboloid.pdf (Figure 1.3; Chapter 1, Section 1.2.4
"The minimum of the energy"): the quadratic energy
W(u) = (1/2) u^T K u - u^T f for the spring-chain stiffness matrix K,
sliced at the optimal u3, shown as a paraboloid with its minimum marked
at the solution of K u = f (in the code the unknowns are called x).
Run: cd python/figures/ch1 && python3 fig_paraboloid.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Same K, f as the spring-chain example (unit stiffnesses)
K = np.array([[2,-1,0],[-1,2,-1],[0,-1,2]], dtype=float)
f = np.array([1,1,1], dtype=float)
x_star = np.linalg.solve(K, f)

# Restrict to the (x1,x2) plane with x3 fixed at its optimal value, to plot a
# genuine paraboloid
x3_star = x_star[2]
x1 = np.linspace(x_star[0]-3, x_star[0]+3, 60)
x2 = np.linspace(x_star[1]-3, x_star[1]+3, 60)
X1, X2 = np.meshgrid(x1, x2)
# W on a 60 x 60 grid of (u1, u2) around the minimizer.
W = np.zeros_like(X1)
for i in range(X1.shape[0]):
    for j in range(X1.shape[1]):
        x = np.array([X1[i,j], X2[i,j], x3_star])
        W[i,j] = 0.5*x@K@x - x@f

fig = plt.figure(figsize=(6, 4.6))
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(X1, X2, W, cmap="Blues", alpha=0.85, linewidth=0, antialiased=True)
# Minimum value W(u*) = -1/2 f^T K^-1 f (checked in the printout).
Wmin = 0.5*x_star@K@x_star - x_star@f
ax.scatter([x_star[0]], [x_star[1]], [Wmin], color="#D85A30", s=70,
           edgecolor="white", linewidth=1.2, depthshade=False, zorder=10)
ax.text(x_star[0]+0.3, x_star[1]-0.3, Wmin - 1, r"$\mathbf{K}\mathbf{u}=\mathbf{f}$", color="#D85A30", fontsize=10)
ax.set_xlabel("$u_1$")
ax.set_ylabel("$u_2$")
ax.set_zlabel("$W(\mathbf{u})$")
ax.set_title(r"$W(\mathbf{u})=\frac{1}{2}\mathbf{u}^\top\mathbf{K}\mathbf{u} - \mathbf{u}^\top\mathbf{f}$: a paraboloid")
ax.view_init(elev=22, azim=-60)
fig.tight_layout()
fig.savefig("../../../figures/ch1/paraboloid.pdf")
print("x* =", x_star, " Wmin =", Wmin, " -0.5 f^T K^-1 f =", -0.5*f@np.linalg.solve(K,f))
