"""
Generates figures/ex5_convergence_loglog.pdf
Exercise 5 (Sections 6, 10 of variational_formulations.tex): log-log
convergence plot for the discrete chain solving -u''=sin(pi x) against
an O(h^2) reference slope.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def solve_chain(n):
    h = 1.0/(n+1)
    x = np.linspace(h, 1-h, n)
    K = (2*np.eye(n) - np.eye(n,k=1) - np.eye(n,k=-1)) / h**2
    f = np.sin(np.pi*x)
    u = np.linalg.solve(K, f)
    u_exact = np.sin(np.pi*x)/np.pi**2
    return h, np.max(np.abs(u - u_exact))

ns = [3, 7, 15, 31, 63, 127]
hs, errs = zip(*(solve_chain(n) for n in ns))
hs, errs = np.array(hs), np.array(errs)

fig, ax = plt.subplots(figsize=(5.2, 3.6))
ax.loglog(hs, errs, "o-", color="#0F6E56", ms=5, label="observed max error")

# O(h^2) reference line through the first data point
C = errs[0] / hs[0]**2
ax.loglog(hs, C*hs**2, "--", color="#D85A30", label=r"reference slope $O(h^2)$")

ax.set_xlabel("mesh size $h$")
ax.set_ylabel(r"$\max_i |u_i - u(x_i)|$")
ax.legend(frameon=False, fontsize=9)
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
fig.savefig("../../figures/ch1/ex5_convergence_loglog.pdf")

rates = np.log(errs[:-1]/errs[1:]) / np.log(hs[:-1]/hs[1:])
print("h:", hs)
print("err:", errs)
print("observed rates:", rates)
