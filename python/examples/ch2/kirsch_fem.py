"""Exercise "A plate with a hole: Kirsch against finite elements" (ch2).
Run: python3 python/examples/ch2/kirsch_fem.py
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

def kirsch(r, th, a=1.0, s=1.0):
    """Stresses (s_rr, s_tt, s_rt) of an infinite plate, hole of radius a,
    remote tension s along x (Kirsch, 1898)."""
    q2, q4, c, sn = a**2 / r**2, a**4 / r**4, np.cos(2 * th), np.sin(2 * th)
    srr = s / 2 * (1 - q2) + s / 2 * (1 - 4 * q2 + 3 * q4) * c
    stt = s / 2 * (1 + q2) - s / 2 * (1 + 3 * q4) * c
    srt = -s / 2 * (1 + 2 * q2 - 3 * q4) * sn
    return srr, stt, srt

def mesh(nt, a, L):
    """Quarter plate [0,L]^2 minus the disc r < a: nt cells along the hole,
    nodes (i radial, j angular) graded geometrically from the hole (nearly
    square cells), each quadrangle cut into two linear triangles."""
    q = 1 + np.pi / (2 * nt)                       # growth ratio of the cells
    nr = int(np.ceil(np.log(L / a) / np.log(q)))
    th = np.linspace(0, np.pi / 2, nt + 1)
    outer = np.where(th[:, None] <= np.pi / 4 + 1e-12,
                     np.c_[L + 0 * th, L * np.tan(np.minimum(th, np.pi / 4))],
                     np.c_[L / np.tan(np.maximum(th, np.pi / 4)), L + 0 * th])
    inner = a * np.c_[np.cos(th), np.sin(th)]
    s = (q ** np.arange(nr + 1) - 1) / (q**nr - 1)
    X = (inner[None] + s[:, None, None] * (outer - inner)[None]).reshape(-1, 2)
    n = lambda i, j: i * (nt + 1) + j
    T = [[n(i, j), n(i + 1, j), n(i + 1, j + 1)] for i in range(nr) for j in range(nt)]
    T += [[n(i, j), n(i + 1, j + 1), n(i, j + 1)] for i in range(nr) for j in range(nt)]
    return X, np.array(T)

def solve(nt, a=1.0, L=20.0, E=1.0, nu=0.3, s=1.0):
    X, T = mesh(nt, a, L)
    D = E / (1 - nu**2) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])
    rows, cols, vals, Bs, areas = [], [], [], [], []
    for t in T:                                    # linear triangles (CST)
        (x1, y1), (x2, y2), (x3, y3) = X[t]
        A = 0.5 * ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
        b, c = np.array([y2 - y3, y3 - y1, y1 - y2]), np.array([x3 - x2, x1 - x3, x2 - x1])
        B = np.zeros((3, 6))
        B[0, 0::2], B[1, 1::2], B[2, 0::2], B[2, 1::2] = b, c, c, b
        B /= 2 * A
        dof = np.ravel([[2 * k, 2 * k + 1] for k in t])
        rows += list(np.repeat(dof, 6)); cols += list(np.tile(dof, 6))
        vals += list((A * B.T @ D @ B).ravel()); Bs.append(B); areas.append(A)
    N = 2 * len(X)
    K = coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()
    F = np.zeros(N)                                # traction s on the edge x = L
    edge = np.where(np.isclose(X[:, 0], L))[0]
    edge = edge[np.argsort(X[edge, 1])]
    for k1, k2 in zip(edge[:-1], edge[1:]):
        F[[2 * k1, 2 * k2]] += s * (X[k2, 1] - X[k1, 1]) / 2
    fixed = np.r_[2 * np.where(np.isclose(X[:, 0], 0))[0] + 0,   # u_x = 0 on x = 0
                  2 * np.where(np.isclose(X[:, 1], 0))[0] + 1]   # u_y = 0 on y = 0
    free = np.setdiff1d(np.arange(N), fixed)
    U = np.zeros(N)
    U[free] = spsolve(K[free][:, free], F[free])
    sig = np.zeros((len(X), 3)); w = np.zeros(len(X))   # area-weighted nodal average
    for t, B, A in zip(T, Bs, areas):
        dof = np.ravel([[2 * k, 2 * k + 1] for k in t])
        sig[t] += A * (D @ B @ U[dof]); w[t] += A
    return X, sig / w[:, None], len(free)

if __name__ == "__main__":
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
