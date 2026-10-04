"""
Generates figures/ch1/ex1_gd_convergence.pdf (Figure 1.16 of Chapter 1).
Exercise 1.9 "Minimizing by gradient descent": convergence of gradient
descent on the 3-mass spring chain (Section 1.2.2), checked against the
theoretical contraction rate rho = max_i |1 - step*lambda_i(K)|.
Run: cd python/figures/ch1 && python3 fig1_gd_convergence.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Spring chain with unit stiffnesses: K = tridiag(-1, 2, -1), load f = (1, 1, 1),
# exact minimizer u* = K^-1 f.
K = np.array([[2,-1,0],[-1,2,-1],[0,-1,2]], dtype=float)
f = np.array([1,1,1], dtype=float)
u_star = np.linalg.solve(K, f)

# The error e_k = u_k - u* obeys e_{k+1} = (I - step K) e_k, so it contracts
# at the rate rho = largest |1 - step lambda_i| over the eigenvalues of K.
eigvals = np.linalg.eigvalsh(K)
alpha, M = eigvals[0], eigvals[-1]
step = 0.2
rho = max(abs(1 - step*ev) for ev in eigvals)   # theoretical contraction rate

# Gradient descent u_{k+1} = u_k - step grad W(u_k), grad W = K u - f, from u = 0.
u = np.zeros(3)
errors = [np.linalg.norm(u - u_star)]
n_iter = 60
for _ in range(n_iter):
    grad = K @ u - f
    u = u - step*grad
    errors.append(np.linalg.norm(u - u_star))

# Predicted error rho^k |u_0 - u*|, plotted against the observed one.
k = np.arange(n_iter+1)
theory = errors[0]*rho**k

fig, ax = plt.subplots(figsize=(5.2, 3.6))
ax.semilogy(k, errors, "o", ms=3.5, color="#1D9E75", label=r"$\|u_k-u^\star\|$ (gradient descent)")
ax.semilogy(k, theory, "--", color="#D85A30", label=fr"theory: $\rho^k\|u_0-u^\star\|$, $\rho={rho:.3f}$")
ax.set_xlabel("iteration $k$")
ax.set_ylabel(r"$\|u_k-u^\star\|$")
ax.legend(frameon=False, fontsize=9)
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
fig.savefig("../../../figures/ch1/ex1_gd_convergence.pdf")
print("alpha =", alpha, " M =", M, " rho =", rho)
print("final error:", errors[-1])
