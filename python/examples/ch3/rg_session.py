"""rg_session.py -- Session "Chapter 3, reciprocity gap": Python version.

Same blocks, same figures and same printed values as rg_session.dgibi
(Cast3M) and rg_session_fenics.py (FEniCS):

  BLOCK 0  parameters
  BLOCK 1  geometry and mesh (two lips: gmsh "Crack" plugin)
  BLOCK 2  model, stiffness matrix, centred coordinates          FIG 1
  BLOCK 3  procedures: reciprocity gap, measurement, opening
  BLOCK 4  experiment E1 (direct problem), opening on the crack FIG 2, 3
  BLOCK 5  measurement: Cauchy data on the boundary (+ noise)
  BLOCK 6  linear and quadratic adjoint fields: normal, line    FIG 4
  BLOCK 7  polynomial adjoint fields: moments, centre, length   FIG 5
  BLOCK 8  Fourier adjoint fields and the sine-sinh series      FIG 6, 7
  BLOCK 9  turning potential, matrix of the experiments        FIG 8, 9

Run:   python3 rg_session.py [eps] [outdir]
       eps    relative noise on the measured values (default 0)
       outdir folder for fig1.pdf ... fig9.pdf (default: current folder)
Needs: numpy, matplotlib, scikit-fem, gmsh (as replica.py).
"""

import sys
import os

import numpy as np
import gmsh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from skfem import MeshTri, Basis, ElementTriP2, BilinearForm
from skfem import condense, solve
from skfem.helpers import dot, grad


# ======================================================================
# BLOCK 0 - Parameters
# ======================================================================

ldom, hdom = 1.5, 2.0          # width and height of the rectangle
nelem = int(os.environ.get("RG_NELEM", "30"))   # elements per outer side
                                                # (nelem/2 on the crack)

A = np.array([0.5, 1.2])       # true crack tips (used to build the mesh
B = np.array([0.8, 1.35])      # and to compute the errors only)

eps = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0   # noise level
seed = 1                       # seed of the random noise

nxi = 8                        # Fourier fields: xi = 1, 2, ..., nxi
xiest = 2.0                    # xi used for the Fourier length estimate
nsine = 5                      # number of sine-sinh terms
nbeta = 12                     # turning potential: angles 0, 15, ..., 165

outdir = sys.argv[2] if len(sys.argv) > 2 else "."
os.makedirs(outdir, exist_ok=True)


def savefig(fig, k, title):
    """Every figure has its number in the title and a printed message."""
    fig.suptitle("FIG %d  %s" % (k, title), fontsize=10)
    fig.savefig(os.path.join(outdir, "fig%d.pdf" % k), bbox_inches="tight")
    plt.close(fig)
    print("  FIG %d : %s" % (k, title))


# ======================================================================
# BLOCK 1 - Geometry and mesh (as replica.py)
#   gmsh: the rectangle with the crack A-B embedded in it, element size
#   ldom/nelem, refined to lf/(nelem/2) at the crack. The "Crack" plugin
#   then duplicates the nodes of the crack (not the tips): the crack has
#   two lips, and u can jump.
# ======================================================================

lf = np.linalg.norm(B - A)                 # true crack length
tf = (B - A) / lf                          # true tangent
nf = np.array([-tf[1], tf[0]])             # true normal (-0.447, 0.894)

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
occ = gmsh.model.occ

rect = occ.addRectangle(0, 0, 0, ldom, hdom)
pa = occ.addPoint(A[0], A[1], 0, lf / (nelem // 2))
pb = occ.addPoint(B[0], B[1], 0, lf / (nelem // 2))
crack = occ.addLine(pa, pb)
occ.synchronize()
gmsh.model.mesh.embed(1, [crack], 2, rect)

gmsh.model.addPhysicalGroup(2, [rect], 1)
gmsh.model.addPhysicalGroup(1, [crack], 3)
gmsh.option.setNumber("Mesh.MeshSizeMax", ldom / nelem)
gmsh.model.mesh.setSize([(0, pa), (0, pb)], lf / (nelem // 2))
gmsh.model.mesh.generate(2)

gmsh.plugin.setNumber("Crack", "Dimension", 1)
gmsh.plugin.setNumber("Crack", "PhysicalGroup", 3)
gmsh.plugin.run("Crack")

# gmsh nodes and triangles -> scikit-fem mesh
tags, xyz, _ = gmsh.model.mesh.getNodes()
index = np.zeros(tags.max() + 1, int)
index[tags] = np.arange(len(tags))
_, enodes = gmsh.model.mesh.getElementsByType(2)
T = index[enodes.reshape(-1, 3)]
P = xyz.reshape(-1, 3)[:, :2]
gmsh.finalize()

used = np.unique(T)                        # drop nodes not in a triangle
renum = np.full(len(P), -1)
renum[used] = np.arange(len(used))
mesh = MeshTri(P[used].T.copy(), renum[T].T.copy())
ncell = mesh.t.shape[1]
print("BLOCK 1  mesh: %d triangles (P2 = TRI6 unknowns)" % ncell)


# ======================================================================
# BLOCK 2 - Model, stiffness matrix, centred coordinates
#   Conductivity K = 1, quadratic Lagrange elements (P2 = TRI6).
#   K is assembled WITHOUT boundary conditions: q = K u are then the
#   nodal fluxes (reactions), as KK * u in Cast3M.
# ======================================================================

basis = Basis(mesh, ElementTriP2())
K = BilinearForm(lambda u, v, w: dot(grad(u), grad(v))).assemble(basis)

xd = basis.doflocs.T                       # coordinates of the nodes
N = basis.N                                # number of nodes (unknowns)
x0 = np.array([0.5 * ldom, 0.5 * hdom])    # centre of the rectangle
xr, yr = xd[:, 0] - x0[0], xd[:, 1] - x0[1]

tol = 1e-9
bot = np.abs(xd[:, 1]) < tol
top = np.abs(xd[:, 1] - hdom) < tol
lef = np.abs(xd[:, 0]) < tol
rig = np.abs(xd[:, 0] - ldom) < tol
outer = bot | top | lef | rig              # outer boundary cdom
corner = (bot | top) & (lef | rig)

# nodes of the elements on the side of the crack towards nf
# (to tell the two lips apart)
cells = basis.element_dofs.T               # (ncell, 6) P2 nodes per cell
mid = mesh.p[:, mesh.t].mean(axis=1).T     # midpoints of the cells
upnode = np.zeros(N, bool)
upnode[cells[(mid - A) @ nf > 0].ravel()] = True

# triangulation of the vertices, for the colour plots
tri = mtri.Triangulation(*mesh.p, mesh.t.T)
vert = np.arange(mesh.p.shape[1])          # vertex nodes come first

ctrue = nf @ (A - x0)                      # true line n.X = c
print("BLOCK 2  %d nodes; true crack: n = (%.4f, %.4f), c = %.4f, "
      "length = %.4f" % (N, *nf, ctrue, lf))

fig, ax = plt.subplots(figsize=(3.6, 4.4))
ax.triplot(tri, lw=0.3, color="0.4")
ax.plot([0, ldom, ldom, 0, 0], [0, 0, hdom, hdom, 0], "r", lw=1.5)
ax.plot(*np.c_[A, B], "b", lw=2.5)
ax.set_aspect("equal")
savefig(fig, 1, "mesh, outer boundary (red), crack (blue)")


# ======================================================================
# BLOCK 3 - Procedures
# ======================================================================

def rgap(ub, qb, v):
    """Reciprocity gap from boundary nodal values:
         RG(v) = sum over the boundary nodes of  u (K v) - q v
       q = K u are the nodal fluxes (reactions), K v the nodal fluxes of v.
       It equals  int_Gamma [u] dn v  (book, eq. RG), exactly for the
       harmonic polynomials of degree <= 2."""
    Kv = K @ v
    return np.sum((ub * Kv - qb * v)[outer])


def measure(u, mu, mq, eps, rng):
    """Cauchy data on the boundary: u and q = K u at the boundary nodes.
       Multiplicative Gaussian noise of level eps on the MEASURED values
       only: u on the nodes mu, q on the nodes mq."""
    ub, qb = u.copy(), K @ u
    if eps > 0:
        ub[mu] *= 1 + eps * rng.standard_normal(mu.sum())
        qb[mq] *= 1 + eps * rng.standard_normal(mq.sum())
    return ub, qb


def opening(u):
    """Opening [u] = u(upper lip) - u(lower lip) along the crack, as a
       function of the abscissa from A; zero at the tips."""
    sc, ec = (xd - A) @ tf, (xd - A) @ nf
    on = np.flatnonzero((np.abs(ec) < tol) & (sc > tol) & (sc < lf - tol))
    pairs = {}
    for i in on:
        pairs.setdefault(round(sc[i], 9), []).append(i)
    S, J = [0.0], [0.0]
    for s in sorted(pairs):
        a, b = pairs[s] if upnode[pairs[s][0]] else pairs[s][::-1]
        S.append(s)
        J.append(u[a] - u[b])
    return np.r_[S, lf], np.r_[J, 0.0]


def dirichlet(mask, values):
    """Solve K u = 0 with u = values on the nodes of mask."""
    u = np.zeros(N)
    u[mask] = values[mask]
    return solve(*condense(K, np.zeros(N), x=u, D=np.flatnonzero(mask)))


# ======================================================================
# BLOCK 4 - Experiment E1 (direct problem)
#   Electrodes: u = 0 at the bottom, u = hdom at the top; zero flux on
#   the vertical sides. Far from the crack u = y: background field
#   E = (0, 1).
# ======================================================================

u1 = dirichlet(bot | top, xd[:, 1])
E1 = np.array([0.0, 1.0])

fig, ax = plt.subplots(figsize=(3.8, 4.4))
cs = ax.tricontourf(tri, u1[vert], 20, cmap="jet")
ax.tricontour(tri, u1[vert], 20, colors="k", linewidths=0.3)
ax.plot(*np.c_[A, B], "w", lw=2)
fig.colorbar(cs, ax=ax, shrink=0.8)
ax.set_aspect("equal")
savefig(fig, 2, "potential u1 (E1: electrodes bottom 0 / top hdom)")

# opening on the crack, and the opening of the same crack in an
# infinite body, 2 E_n sqrt(a^2 - s^2)  (Exercise "A crack in a uniform
# field", book Chapter 3)
S1, J1 = opening(u1)
m0true = np.trapezoid(J1, S1)
a0 = 0.5 * lf
Jinf = 2 * (E1 @ nf) * np.sqrt(np.maximum(a0**2 - (S1 - a0)**2, 0))
print("BLOCK 4  E1: int [u] on the lips = %.5f (infinite body %.5f)"
      % (m0true, np.pi * a0**2 * (E1 @ nf)))

fig, ax = plt.subplots(figsize=(4.6, 3.0))
ax.plot(S1 - a0, J1, "y-o", ms=3, lw=2, label="[u] read on the lips (FE)")
ax.plot(S1 - a0, Jinf, "k--", lw=1, label="infinite body")
ax.set_xlabel("S  (abscissa along the crack, from its centre)")
ax.set_ylabel("SAUT")
ax.legend(fontsize=8)
savefig(fig, 3, "opening [u1] on the crack")


# ======================================================================
# BLOCK 5 - Measurement: Cauchy data of E1 on the boundary
#   u is imposed on the electrodes and measured on the vertical sides
#   (not at the corners, where it is imposed); the flux is imposed (0)
#   on the vertical sides and measured on the electrodes.
# ======================================================================

rng = np.random.default_rng(seed)
ub, qb = measure(u1, (lef | rig) & ~corner, bot | top, eps, rng)
print("BLOCK 5  noise eps = %g;  RG(1) = %.1e (flux balance, 0)"
      % (eps, rgap(ub, qb, np.ones(N))))


# ======================================================================
# BLOCK 6 - Linear and quadratic adjoint fields: normal and line
#   RG(x) = m0 n_x, RG(y) = m0 n_y, with m0 = int [u] > 0  (Step 1)
#   RG((n.X)^2 - (t.X)^2) = 2 c m0                         (Step 2)
# ======================================================================

rx, ry = rgap(ub, qb, xr), rgap(ub, qb, yr)
m0 = np.hypot(rx, ry)
n = np.array([rx, ry]) / m0
t = np.array([-n[1], n[0]])
eta0, ss = n[0] * xr + n[1] * yr, t[0] * xr + t[1] * yr
c = rgap(ub, qb, eta0**2 - ss**2) / (2 * rgap(ub, qb, eta0))
eang = np.degrees(np.arccos(min(1.0, abs(n @ nf))))

print("BLOCK 6  RG(x) = %.5f, RG(y) = %.5f" % (rx, ry))
print("         normal n = (%.4f, %.4f), angle error %.4f deg"
      % (*n, eang))
print("         line   c = %.4f   (true %.4f, sign of n)" % (c, ctrue))

fig, ax = plt.subplots(figsize=(3.6, 4.4))
ax.plot([0, ldom, ldom, 0, 0], [0, 0, hdom, hdom, 0], "k", lw=1)
ax.plot(*np.c_[A, B], "b", lw=4, label="true crack")
L2 = np.c_[x0 + c * n - 2 * t, x0 + c * n + 2 * t]
ax.plot(*L2, "g", lw=1, label="line n.X = c")
ax.set_xlim(0, ldom); ax.set_ylim(0, hdom); ax.set_aspect("equal")


# ======================================================================
# BLOCK 7 - Polynomial adjoint fields: moments, centre, length
#   eta = n.X - c, s = t.X:
#     v = eta                -> m0 = int [u]
#     v = s eta              -> m1 = int [u] s       -> centre s0 = m1/m0
#     v = s'^2 eta - eta^3/3 -> m0 mu2 (s' = s - s0) -> a_mom = 2 sqrt(mu2)
#   dipole: m0 = pi a^2 E_n                          -> a_dip
# ======================================================================

eta = eta0 - c
m0 = rgap(ub, qb, eta)
s0 = rgap(ub, qb, ss * eta) / m0
sp = ss - s0
mu2 = rgap(ub, qb, sp**2 * eta - eta**3 / 3) / m0
amom = 2 * np.sqrt(mu2) if mu2 > 0 else 0.0
en = E1 @ n
adip = np.sqrt(abs(m0) / (np.pi * abs(en)))
xc = x0 + c * n + s0 * t                     # identified centre
tips = np.c_[xc - adip * t, xc + adip * t]
etip = min(max(np.linalg.norm(tips[:, 0] - A), np.linalg.norm(tips[:, 1] - B)),
           max(np.linalg.norm(tips[:, 1] - A), np.linalg.norm(tips[:, 0] - B)))

print("BLOCK 7  m0 = %.5f, s0 = %.4f, mu2 = %.6f, E_n = %.4f"
      % (m0, s0, mu2, en))
print("         length (2nd moment) = %.4f" % (2 * amom))
print("         length (dipole)     = %.4f, tip error %.4f   (true %.4f)"
      % (2 * adip, etip, lf))
print("         centre = (%.4f, %.4f)   (true (%.4f, %.4f))"
      % (*xc, *(0.5 * (A + B))))

ax.plot(*tips, "r", lw=2, label="reconstructed (dipole)")
ax.legend(fontsize=7, loc="lower right")
savefig(fig, 4, "true crack (blue), line (green), reconstructed (red)")

# FIG 5: the true opening against the two semi-ellipses of area m0
sl = (S1 + (A - x0) @ t) - s0                # lip abscissa from s0
if (B - A) @ t < 0:
    sl = ((A - x0) @ t - S1) - s0
fig, ax = plt.subplots(figsize=(4.6, 3.0))
ax.plot(sl, J1, "y", lw=3, label="true (FE)")
for a, col, lab in ((adip, "r", "dipole"), (amom, "b", "2nd moment")):
    if a > 0:
        s = np.linspace(-a, a, 101)
        ax.plot(s, 2 * m0 / (np.pi * a**2) * np.sqrt(a**2 - s**2), col,
                lw=1, label=lab)
ax.set_xlabel("S = t.X - s0")
ax.set_ylabel("SAUT")
ax.legend(fontsize=8)
savefig(fig, 5, "opening: true (yellow), dipole (red), 2nd moment (blue)")


# ======================================================================
# BLOCK 8 - Fourier adjoint fields and the sine-sinh series
#   (a) v = exp(xi eta) cos(xi s'), exp(xi eta) sin(xi s'), s' = s - s0:
#       RG / xi = int [u] cos(xi s'),  int [u] sin(xi s')  (Fourier
#       transform of the opening). For a semi-ellipse of half-length a:
#       C(xi) / m0 = 2 J1(xi a) / (xi a) = 1 - (xi a)^2/8 + (xi a)^4/192
#   (b) v_k = sin(l_k g) sinh(l_k eta) / l_k, l_k = k pi / L, g from the
#       point of the line at x = 0: [u](g) = sum (2/L) RG(v_k) sin(l_k g)
# ======================================================================

def j1ratio(x):
    """2 J1(x) / x by its power series (enough for x < 6)."""
    k = np.arange(12)
    from math import factorial
    return sum((-1)**j * (x / 2)**(2 * j) / (factorial(j) * factorial(j + 1))
               for j in k)


xis = np.arange(1, nxi + 1, dtype=float)
cfou, sfou, ctrue_f = [], [], []
for xi in xis:
    ex = np.exp(xi * eta)
    cfou.append(rgap(ub, qb, ex * np.cos(xi * sp)) / xi)
    sfou.append(rgap(ub, qb, ex * np.sin(xi * sp)) / xi)
    ctrue_f.append(np.trapezoid(J1 * np.cos(xi * sl), S1))
cfou, sfou, ctrue_f = map(np.array, (cfou, sfou, ctrue_f))

# length from one frequency: solve x^4/192 - x^2/8 + (1 - rho) = 0
rho = np.interp(xiest, xis, cfou) / m0
disc = 1 / 64 - (1 - rho) / 48
afou = 0.0                                   # 0 = no estimate
if 0 < rho < 1 and disc > 0:
    afou = np.sqrt(96 * (1 / 8 - np.sqrt(disc))) / xiest

print("BLOCK 8  Fourier fields, C(xi)/m0 (true opening in brackets):")
for xi, cf, sf, ct in zip(xis, cfou, sfou, ctrue_f):
    print("         xi = %2d : %8.4f  (%7.4f)   S(xi)/m0 = %8.4f"
          % (xi, cf / m0, ct / m0true, sf / m0))
print("         length (Fourier, xi = %g) = %.4f" % (xiest, 2 * afou))

fig, ax = plt.subplots(figsize=(4.6, 3.0))
xf = np.linspace(0, nxi, 200)
ax.plot(xf, j1ratio(xf * adip), "r", lw=1, label="semi-ellipse, a dipole")
ax.plot(xis, ctrue_f / m0true, "yo", ms=7, label="true opening (FE)")
ax.plot(xis, cfou / m0, "b+", ms=9, mew=2, label="RG / (xi m0)")
ax.plot(xis, sfou / m0, "gx", ms=6, label="sine part")
ax.set_ylim(-0.5, 1.5)                       # large xi leave the frame
ax.set_xlabel("XI")
ax.set_ylabel("C / m0")
ax.legend(fontsize=7)
savefig(fig, 6, "Fourier transform of the opening")

# (b) sine-sinh series along the identified line, t towards x > 0
nn, cc, tt = (n, c, t) if t[0] > 0 else (-n, -c, -t)
ga = -(x0[0] + cc * nn[0]) / tt[0]           # point of the line at x = 0
xga = x0 + cc * nn + ga * tt
gx = (xd - xga) @ tt
gy = (xd - xga) @ nn
LL = np.sqrt(2) * max(ldom, hdom)
lg = np.linspace(0, LL, 401)
rn, total, sums = [], np.zeros_like(lg), []
for k in range(1, nsine + 1):
    lk = k * np.pi / LL
    rn.append(rgap(ub, qb, np.sin(lk * gx) * np.sinh(lk * gy) / lk))
    total = total + (2 / LL) * rn[-1] * np.sin(lk * lg)
    sums.append(total.copy())
sgn = np.sign(rn[0])
print("BLOCK 8  sine-sinh coefficients RG_k:", np.round(rn, 4))

gA, gB = (A - xga) @ tt, (B - xga) @ tt
fig, ax = plt.subplots(figsize=(4.6, 3.0))
ax.plot(gA + (gB - gA) * S1 / lf, J1, "y", lw=3, label="true [u1]")
for k, col in ((1, "c"), (3, "b"), (5, "r")):
    if k <= nsine:
        ax.plot(lg, sgn * sums[k - 1], col, lw=1, label="n = %d" % k)
ax.set_xlabel("G  (distance along the line from x = 0)")
ax.set_ylabel("SAUT")
ax.legend(fontsize=7)
savefig(fig, 7, "sine-sinh series: n = 1, 3, 5 and true opening")


# ======================================================================
# BLOCK 9 - Turning potential and the matrix of the experiments
#   u = d.X on the whole boundary, d = (cos b, sin b): the background
#   field is d. Only the flux is measured. By linearity
#   u_b = cos b ux + sin b uy: two solves give all the angles.
#   Row k of R: (RG(x), RG(y)) of the experiment b_k = m0_k n.
# ======================================================================

ux = dirichlet(outer, xr)
uy = dirichlet(outer, yr)
none = np.zeros(N, bool)

betas = np.arange(nbeta) * 180.0 / nbeta
R = np.zeros((nbeta, 2))
data = []
for k, b in enumerate(betas):
    cb, sb = np.cos(np.radians(b)), np.sin(np.radians(b))
    ubk, qbk = measure(cb * ux + sb * uy, none, outer, eps, rng)
    data.append((ubk, qbk))
    R[k] = rgap(ubk, qbk, xr), rgap(ubk, qbk, yr)

# rank one: R = m n^T. The normal is the first right singular vector.
U, sv, Vt = np.linalg.svd(R)
nR = Vt[0] * np.sign(Vt[0] @ n)
msig = R @ nR                                # signals m0 of each angle
w = msig / np.linalg.norm(msig)              # optimal weights
dvir = np.array([w @ np.cos(np.radians(betas)), w @ np.sin(np.radians(betas))])
# virtual experiment and its length (dipole)
ubv = sum(wk * d[0] for wk, d in zip(w, data))
qbv = sum(wk * d[1] for wk, d in zip(w, data))
tR = np.array([-nR[1], nR[0]])
etv = nR[0] * xr + nR[1] * yr
ssv = tR[0] * xr + tR[1] * yr
cv = rgap(ubv, qbv, etv**2 - ssv**2) / (2 * rgap(ubv, qbv, etv))
m0v = rgap(ubv, qbv, etv - cv)
adv = np.sqrt(abs(m0v) / (np.pi * abs(dvir @ nR)))

kmin = np.argmin(np.abs(msig))
print("BLOCK 9  singular values of R: %.5f, %.2e  (rank one)" % tuple(sv))
print("         normal from R: (%.4f, %.4f), angle error %.4f deg"
      % (*nR, np.degrees(np.arccos(min(1, abs(nR @ nf))))))
print("         |m0| min %.5f at %g deg, max %.5f at %g deg"
      % (abs(msig[kmin]), betas[kmin], abs(msig).max(),
         betas[np.argmax(abs(msig))]))
print("         virtual experiment: d = (%.4f, %.4f), signal %.5f"
      % (*dvir, np.linalg.norm(msig)))
print("         c = %.4f, length (dipole) = %.4f" % (cv, 2 * adv))

alpha = np.degrees(np.arctan2(tf[1], tf[0]))
bf = np.linspace(0, 180, 181)
fig, ax = plt.subplots(figsize=(4.6, 3.0))
ax.plot(bf, np.pi * a0**2 * np.abs(np.sin(np.radians(bf - alpha))), "k--",
        lw=1, label="pi a^2 |sin(b - alpha)|")
ax.plot(betas, np.abs(msig), "ko-", ms=4, label="|m0| from R")
ax.set_xlabel("BETA (degrees)")
ax.set_ylabel("M0")
ax.legend(fontsize=7)
savefig(fig, 8, "signal |m0| of the turning potential")

fig, ax = plt.subplots(figsize=(4.6, 3.0))
ax.plot(betas, R[:, 0], "bo-", ms=4, label="RG(x)")
ax.plot(betas, R[:, 1], "rs-", ms=4, label="RG(y)")
ax.plot(betas, R[:, 1] / nR[1] * nR[0], "k:", lw=1, label="RG(y) n_x / n_y")
ax.axhline(0, color="0.6", lw=0.5)
ax.set_xlabel("BETA (degrees)")
ax.set_ylabel("RG")
ax.legend(fontsize=7)
savefig(fig, 9, "the two columns of R are proportional")
