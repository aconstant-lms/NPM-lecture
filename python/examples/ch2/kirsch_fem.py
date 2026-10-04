"""Exercise "A plate with a hole: Kirsch against finite elements" (ch2).
Run: python3 python/examples/ch2/kirsch_fem.py
"""
# Exercise 2.21 (c): a quarter of a square plate of side 40a with a hole of radius a,
# remote tension sigma along e_1, plane stress, linear triangles (Section 2.7.4).
# The finite element stresses on the hole are compared with Kirsch's solution.
# Units: a = 1, E = 1, sigma = 1 (the stresses are printed as ratios to sigma).
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

def kirsch(r, th, a=1.0, s=1.0):
    """Stresses (s_rr, s_tt, s_rt) of an infinite plate, hole of radius a,
    remote tension s along x (Kirsch, 1898)."""
    # The formulas of the exercise, with q2 = a^2/r^2 and q4 = a^4/r^4.
    q2, q4, c, sn = a**2 / r**2, a**4 / r**4, np.cos(2 * th), np.sin(2 * th)
    srr = s / 2 * (1 - q2) + s / 2 * (1 - 4 * q2 + 3 * q4) * c
    stt = s / 2 * (1 + q2) - s / 2 * (1 + 3 * q4) * c
    srt = -s / 2 * (1 + 2 * q2 - 3 * q4) * sn
    return srr, stt, srt

def mesh(nt, a, L):
    """Quarter plate [0,L]^2 minus the disc r < a: nt cells along the hole,
    nodes (i radial, j angular) graded geometrically from the hole (nearly
    square cells), each quadrangle cut into two linear triangles.
    Returns the node coordinates X and the triangles T (three node numbers)."""
    # Cell size a pi/(2 nt) on the hole, multiplied by q from one ring to the next.
    q = 1 + np.pi / (2 * nt)                       # growth ratio of the cells
    nr = int(np.ceil(np.log(L / a) / np.log(q)))
    th = np.linspace(0, np.pi / 2, nt + 1)
    # Each ray at angle th joins a point of the hole to a point of the outer
    # boundary: the edge x = L for th <= pi/4, the edge y = L beyond.
    outer = np.where(th[:, None] <= np.pi / 4 + 1e-12,
                     np.c_[L + 0 * th, L * np.tan(np.minimum(th, np.pi / 4))],
                     np.c_[L / np.tan(np.maximum(th, np.pi / 4)), L + 0 * th])
    inner = a * np.c_[np.cos(th), np.sin(th)]
    # Geometric spacing s in [0, 1] along each ray, then node i on ray j.
    s = (q ** np.arange(nr + 1) - 1) / (q**nr - 1)
    X = (inner[None] + s[:, None, None] * (outer - inner)[None]).reshape(-1, 2)
    n = lambda i, j: i * (nt + 1) + j
    # Two triangles per quadrangle (i, j)-(i+1, j+1), counterclockwise.
    T = [[n(i, j), n(i + 1, j), n(i + 1, j + 1)] for i in range(nr) for j in range(nt)]
    T += [[n(i, j), n(i + 1, j + 1), n(i, j + 1)] for i in range(nr) for j in range(nt)]
    return X, np.array(T)

def solve(nt, a=1.0, L=20.0, E=1.0, nu=0.3, s=1.0):
    """Finite element solution with nt cells along the hole; a hole radius,
    L half side of the plate, E Young's modulus, nu Poisson's ratio, s remote
    tension. Returns the nodes, the nodal stresses (s_xx, s_yy, s_xy) and the
    number of free degrees of freedom."""
    X, T = mesh(nt, a, L)
    # Plane stress Hooke's law in Voigt form: sigma = D (eps_xx, eps_yy, 2 eps_xy).
    D = E / (1 - nu**2) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])
    rows, cols, vals, Bs, areas = [], [], [], [], []
    # Assembly. On each triangle the strain is constant, eps = B u_e, and the
    # element stiffness is k_e = A B^T D B (A = area of the triangle).
    for t in T:                                    # linear triangles (CST)
        (x1, y1), (x2, y2), (x3, y3) = X[t]
        A = 0.5 * ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
        # Gradients of the three shape functions: (b_k, c_k) / (2A).
        b, c = np.array([y2 - y3, y3 - y1, y1 - y2]), np.array([x3 - x2, x1 - x3, x2 - x1])
        B = np.zeros((3, 6))
        B[0, 0::2], B[1, 1::2], B[2, 0::2], B[2, 1::2] = b, c, c, b
        B /= 2 * A
        # Global degrees of freedom (u_x, u_y) of the three nodes.
        dof = np.ravel([[2 * k, 2 * k + 1] for k in t])
        rows += list(np.repeat(dof, 6)); cols += list(np.tile(dof, 6))
        vals += list((A * B.T @ D @ B).ravel()); Bs.append(B); areas.append(A)
    N = 2 * len(X)
    K = coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()
    # Nodal forces: each edge segment carries s times its length, half to each node.
    F = np.zeros(N)                                # traction s on the edge x = L
    edge = np.where(np.isclose(X[:, 0], L))[0]
    edge = edge[np.argsort(X[edge, 1])]
    for k1, k2 in zip(edge[:-1], edge[1:]):
        F[[2 * k1, 2 * k2]] += s * (X[k2, 1] - X[k1, 1]) / 2
    # Symmetry conditions (essential): the fixed dofs are removed from the system.
    fixed = np.r_[2 * np.where(np.isclose(X[:, 0], 0))[0] + 0,   # u_x = 0 on x = 0
                  2 * np.where(np.isclose(X[:, 1], 0))[0] + 1]   # u_y = 0 on y = 0
    free = np.setdiff1d(np.arange(N), fixed)
    U = np.zeros(N)
    U[free] = spsolve(K[free][:, free], F[free])
    # Stresses: constant per element, sigma = D B u_e, then averaged at the nodes
    # with the element areas as weights.
    sig = np.zeros((len(X), 3)); w = np.zeros(len(X))   # area-weighted nodal average
    for t, B, A in zip(T, Bs, areas):
        dof = np.ravel([[2 * k, 2 * k + 1] for k in t])
        sig[t] += A * (D @ B @ U[dof]); w[t] += A
    return X, sig / w[:, None], len(free)

if __name__ == "__main__":
    # Hoop stress on the hole: 3 sigma at (0, a) (concentration factor), -sigma at (a, 0).
    print("Kirsch: sigma_xx(0, a) = 3, sigma_yy(a, 0) = -1")
    for nt in [12, 24, 48, 96]:                    # plate 20a x 20a, a = 1
        X, sig, ndof = solve(nt)
        A = np.argmin(np.hypot(X[:, 0], X[:, 1] - 1))         # point (0, a)
        B = np.argmin(np.hypot(X[:, 0] - 1, X[:, 1]))         # point (a, 0)
        print(f"nt = {nt:2d}, {ndof:5d} dofs: sigma_xx(0, a) = {sig[A, 0]:.3f},"
              f" sigma_yy(a, 0) = {sig[B, 1]:.3f}")
# Kirsch: sigma_xx(0, a) = 3, sigma_yy(a, 0) = -1
# nt = 12,   624 dofs: sigma_xx(0, a) = 2.970, sigma_yy(a, 0) = -0.722
# nt = 24,  2352 dofs: sigma_xx(0, a) = 3.013, sigma_yy(a, 0) = -0.871
# nt = 48,  9120 dofs: sigma_xx(0, a) = 3.020, sigma_yy(a, 0) = -0.946
# nt = 96, 35712 dofs: sigma_xx(0, a) = 3.021, sigma_yy(a, 0) = -0.982
# 3.02, not 3: the plate is finite (side 40a); the value at (a, 0) converges slowly
