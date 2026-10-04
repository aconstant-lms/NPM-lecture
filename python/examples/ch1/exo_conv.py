"""Exercise conv (ch1) of the lecture notes "Nonlinear Problems in Mechanics".
Run: python3 python/examples/ch1/exo_conv.py
"""
# Exercise 1.13: solve -u'' = sin(pi x) on (0, 1), u(0) = u(1) = 0, with linear
# elements (Section 1.9.1) on n = 3, 7, ..., 63 interior nodes, and measure the
# errors for two load vectors: lumped (finite differences) and consistent (FEM).
import numpy as np
u_ex  = lambda x: np.sin(np.pi*x)/np.pi**2           # exact solution
du_ex = lambda x: np.cos(np.pi*x)/np.pi
g, w = np.polynomial.legendre.leggauss(5)            # Gauss points on (-1,1)

def solve(n, load):
    """Solve on n interior nodes with the "lumped" or "consistent" load.
    Returns h, the maximal nodal error, the L2 error |u - u_h|_L2 and the
    H1-seminorm error |u - u_h|_H1 = |u' - u_h'|_L2."""
    h = 1.0/(n+1); x = np.linspace(0, 1, n+2)        # nodes, with the two ends
    # Stiffness K_ij = int phi_i' phi_j' dx of the hat functions (the chain matrix).
    K = (2*np.eye(n) - np.eye(n,k=1) - np.eye(n,k=-1))/h   # (1/h) tridiag(-1,2,-1)
    if load == "lumped":       # F_i = h f(x_i): finite differences
        F = h*np.sin(np.pi*x[1:-1])
    else:                      # F_i = int f phi_i, exact (consistent load)
        F = 2*(1-np.cos(np.pi*h))/(np.pi**2*h)*np.sin(np.pi*x[1:-1])
    # Nodal values, with u = 0 at both ends (essential conditions).
    u = np.r_[0, np.linalg.solve(K, F), 0]
    # Errors integrated element by element with 5 Gauss points: on element e,
    # u_h = (1 - s) u_e + s u_{e+1} with s in [0, 1], and u_h' is constant.
    eL2 = eH1 = 0.0                                  # element-by-element quadrature
    for e in range(n+1):
        xq = x[e] + (g+1)*h/2; s = (xq-x[e])/h
        uh = (1-s)*u[e] + s*u[e+1]; duh = (u[e+1]-u[e])/h
        eL2 += h/2*np.sum(w*(u_ex(xq)-uh)**2)
        eH1 += h/2*np.sum(w*(du_ex(xq)-duh)**2)
    return h, np.max(np.abs(u-u_ex(x))), np.sqrt(eL2), np.sqrt(eH1)

# Table of errors for h = 1/4, ..., 1/64 (Figure 1.18). Cea's lemma bounds only
# the H1-seminorm (energy) error; the O(h^2) L2 rate is Aubin-Nitsche.
for load in ["lumped", "consistent"]:
    print(load, ":   h    max nodal err   L2 err    H1 seminorm err")
    for n in [3, 7, 15, 31, 63]:
        print("  %.4f  %.2e  %.2e  %.2e" % solve(n, load))
# lumped,     h=1/4:  5.37e-03  1.77e-03  5.18e-02
#             h=1/64: 2.03e-05  6.44e-06  3.19e-03   (nodal error / 4 when h / 2)
# consistent, h=1/4:  4.16e-17  3.98e-03  5.05e-02
#             h=1/64: 1.94e-15  1.58e-05  3.19e-03   (nodal error ~ 1e-16)
# both: L2 error / 4 (O(h^2)), H1 seminorm error / 2 (O(h)) when h / 2
