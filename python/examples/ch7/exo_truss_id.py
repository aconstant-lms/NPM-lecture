"""Exercise 7.x -- identification of (E, sY, H) on the three-bar truss:
gradients by finite differences, direct differentiation and adjoint;
Nelder-Mead, BFGS (adjoint gradient) and Levenberg-Marquardt (DDM sensitivities)."""
import numpy as np
from scipy.optimize import minimize, least_squares
from id_core import Truss, gauss_newton

# Exercise 7.9: the truss of Section 7.1.2. True parameters E = 200 GPa,
# sY = 200 MPa, H = 10 GPa (in MPa below); loads Q1 = 2.3 N0 sin t,
# Q2 = 0.8 N0 sin 2t with N0 = sY (unit section); observations u_1, u_2 at 80 steps.
E, sY, H = 200e3, 200.0, 10e3
p_true = np.array([E, sY, H])
Q = lambda t: np.array([2.3 * sY * np.sin(t), 0.8 * sY * np.sin(2 * t)])
ts = np.linspace(0, 2 * np.pi, 81)                 # one period, 80 steps
T = Truss(Q, ts)
uref = sY / E                                      # scale of the displacements, sY l / E
u_true = T.forward(p_true)[0]
# measurements: computed displacements + noise of 1% of their maximum
rng = np.random.default_rng(0)
um = u_true + 0.01 * np.abs(u_true).max() * rng.standard_normal(u_true.shape)

# --- 1. gradient at a wrong point, three ways (derivatives w.r.t. log p)
# Table 7.5: forward differences (Section 7.4.1) show truncation for large h and
# round-off for small h; DDM (Box 7.3) and adjoint (Box 7.4) agree to all digits.
# Multiplying by p gives p_j dJ/dp_j = dJ/d log p_j.
p1 = p_true * np.array([0.9, 1.2, 1.5])
print("J(p1) =", T.cost(p1, um, uref))
for h in [1e-2, 1e-4, 1e-6, 1e-8, 1e-10]:
    print("forward FD h=%.0e" % h, T.gradient_fd(p1, um, uref, h=h, central=False) * p1)
print("central FD h=1e-6 ", T.gradient_fd(p1, um, uref, h=1e-6) * p1)
print("DDM               ", T.gradient_ddm(p1, um, uref) * p1)
print("adjoint           ", T.gradient_adjoint(p1, um, uref) * p1)

# --- 2. identifiability: Gauss-Newton matrix in log-parameters at p_true
# S = du/dp / uref from the DDM (rows: 160 observations, columns: E, sY, H);
# box "The Gauss-Newton matrix decides what can be identified" (Section 7.2.2)
u, du = T.ddm(p_true)
G, lam, V, Corr = gauss_newton(du[1:].reshape(-1, 3) / uref, scale=p_true)
print("GN eigenvalues (log p):", lam)
print("weakest direction (E, sY, H):", V[:, 0])
print("correlations:\n", np.round(Corr, 3))
# under Q1 = N0 sin t the truss stays elastic: sY and H are invisible (Section 7.1.3)
Tel = Truss(lambda t: np.array([1.0 * sY * np.sin(t), 0.0]), ts)  # elastic loading
du_el = Tel.ddm(p_true)[1]
print("elastic loading, GN eigenvalues:",
      gauss_newton(du_el[1:].reshape(-1, 3) / uref, scale=p_true)[1])


# --- 3. identification in the variables q = log(p / p0)
# Section 7.3 ("The truss"), Figure 7.3: Nelder-Mead (Box 7.1), BFGS with the
# adjoint gradient, Levenberg-Marquardt (Box 7.2) with the DDM sensitivities.
def identify(p0):
    """Three minimizations from p0; prints p/p_true, J and the number of solves."""
    def report(label, q, extra=""):
        p, n = p0 * np.exp(q), T.nsolve
        print("  %-26s p/p_true = %s  J = %.6e  forward solves = %d %s" %
              (label, np.round(p / p_true, 4), T.cost(p, um, uref), n, extra))

    J = lambda q: T.cost(p0 * np.exp(q), um, uref)

    def J_and_grad(q):                       # one forward + one adjoint solve
        p = p0 * np.exp(q)
        return T.cost(p, um, uref), T.gradient_adjoint(p, um, uref) * p

    # residuals r = (u(p) - um) / uref (7.3) and their Jacobian S (7.4) in q
    resid = lambda q: ((T.forward(p0 * np.exp(q))[0] - um)[1:] / uref).ravel()
    jac = lambda q: (T.ddm(p0 * np.exp(q))[1][1:] / uref).reshape(-1, 3) * p0 * np.exp(q)

    T.nsolve = 0
    r = minimize(J, np.zeros(3), method="Nelder-Mead",
                 options=dict(xatol=1e-8, fatol=1e-12, maxiter=4000, maxfev=4000))
    report("Nelder-Mead", r.x)
    T.nsolve = 0
    r = minimize(J_and_grad, np.zeros(3), jac=True, method="BFGS", options=dict(gtol=1e-6))
    report("BFGS + adjoint", r.x, "and %d adjoint solves" % r.nfev)
    T.nsolve = 0
    r = least_squares(resid, np.zeros(3), jac=jac, method="lm", xtol=1e-12, ftol=1e-14)
    report("Levenberg-Marquardt + DDM", r.x, "and %d DDM sweeps" % r.njev)


# two starts: at the first the truss never yields, so dJ/dsY = dJ/dH = 0 and
# only E is adjusted; the second activates plasticity
for f0 in ([0.7, 1.5, 3.0], [0.7, 1.2, 3.0]):
    p0 = p_true * np.array(f0)
    plastic = any(st["plastic"].any() for st in T.forward(p0)[2][1:])
    print("start p0/p_true =", f0, " plastic at p0:", plastic)
    identify(p0)
