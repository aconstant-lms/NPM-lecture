"""rg_reference.py -- Python twin of crack_rg_extension.dgibi (Chapter 3):
the cracked rectangle with quadratic triangles, the two experiments E1, E2,
the reciprocity gap from boundary nodal values, and the identification
(normal, line, moments, dipole, sine-sinh series, rotation of the loads).
It gives the reference values printed in the gibiane file and in the companion.
Run: python3 rg_reference.py [eps] [nsamples]   (needs numpy, scikit-fem, triangle)
"""
import sys

import numpy as np
import triangle
from skfem import MeshTri, Basis, ElementTriP2, BilinearForm, condense, solve
from skfem.helpers import dot, grad

# --- data of the gibiane file: rectangle ldom x hdom, crack A-B, divisions
import os
ldom, hdom = 1.5, 2.0
nelem = int(os.environ.get("RG_NELEM", "30"))     # divisions of the outer sides
A, B = np.array([0.5, 1.2]), np.array([0.8, 1.35])
X0 = np.array([0.5 * ldom, 0.5 * hdom])             # centre of the coordinates
lf = np.linalg.norm(B - A)
tf = (B - A) / lf
nf = np.array([-tf[1], tf[0]])                      # true normal (-0.447, 0.894)


def line(P, Q, n):
    """n + 1 equally spaced points from P to Q (as DROITE n P Q)."""
    s = np.linspace(0, 1, n + 1)[:, None]
    return P + s * (Q - P)


def mesh_polygon(chain):
    """Free triangulation of the polygon given as a list of point arrays (each
    from one corner to the next), with no point added on its boundary (as SURF)."""
    pts = np.vstack([c[:-1] for c in chain])
    n = len(pts)
    seg = np.column_stack([np.arange(n), (np.arange(n) + 1) % n])
    h = lf / (nelem // 2)                           # finest boundary spacing
    T = triangle.triangulate(dict(vertices=pts, segments=seg),
                             "pq30Ya%.6f" % (0.433 * (2.2 * h) ** 2))
    return T["vertices"], T["triangles"]


def cracked_mesh():
    p1, p2, p3, p4 = map(np.array, ([0, 0], [ldom, 0], [ldom, hdom], [0, hdom]))
    d12, d23 = line(p1, p2, nelem), line(p2, p3, nelem)
    d34, d41 = line(p3, p4, nelem), line(p4, p1, nelem)
    d1a, db3 = line(p1, A, nelem // 2), line(B, p3, nelem // 2)
    lip = line(A, B, nelem // 2)
    # lower part: p1 p2 p3 B A ; upper part: p1 A B p3 p4
    Pm, Tm = mesh_polygon([d12, d23, db3[::-1], lip[::-1], d1a[::-1]])
    Pp, Tp = mesh_polygon([d1a, lip, db3, d34, d41])
    # merge: nodes of the upper part coinciding with lower-part nodes are shared,
    # except the interior nodes of the crack, which are duplicated (two lips)
    on_crack = lambda P: (np.abs((P - A) @ nf) < 1e-9) & ((P - A) @ tf > 1e-9) \
        & ((P - A) @ tf < lf - 1e-9)
    P = list(Pm)
    idx = np.empty(len(Pp), int)
    for i, x in enumerate(Pp):
        j = np.flatnonzero(np.linalg.norm(Pm - x, axis=1) < 1e-9)
        if len(j) and not on_crack(x[None])[0]:
            idx[i] = j[0]
        else:
            idx[i] = len(P)
            P.append(x)
    P = np.array(P)
    T = np.vstack([Tm, idx[Tp]])
    upper = np.r_[np.zeros(len(Tm), bool), np.ones(len(Tp), bool)]
    return MeshTri(P.T, T.T), upper


mesh, upper = cracked_mesh()
basis = Basis(mesh, ElementTriP2())
K = BilinearForm(lambda u, v, w: dot(grad(u), grad(v))).assemble(basis)
xd = basis.doflocs.T                                 # coordinates of the dofs
xr = xd - X0                                         # centred coordinates
outer = (np.abs(xd[:, 0]) < 1e-9) | (np.abs(xd[:, 0] - ldom) < 1e-9) \
    | (np.abs(xd[:, 1]) < 1e-9) | (np.abs(xd[:, 1] - hdom) < 1e-9)
bot, top = np.abs(xd[:, 1]) < 1e-9, np.abs(xd[:, 1] - hdom) < 1e-9
lef, rig = np.abs(xd[:, 0]) < 1e-9, np.abs(xd[:, 0] - ldom) < 1e-9
hor, ver = bot | top, lef | rig
corner = hor & ver


def dirichlet(mask, values):
    """Solve K u = 0 with u = values on the dofs of mask; returns u and the nodal
    fluxes q = K u (the reactions, nonzero only on the Dirichlet dofs)."""
    u = np.zeros(basis.N)
    u[mask] = values[mask]
    u = solve(*condense(K, np.zeros(basis.N), x=u, D=np.flatnonzero(mask)))
    return u, K @ u


# E1: u = 0 at the bottom, u = hdom at the top -> E = (0, 1)
# E2: u = 0 on the left,   u = ldom on the right -> E = (1, 0)
u1, q1 = dirichlet(bot | top, xd[:, 1])
u2, q2 = dirichlet(lef | rig, xd[:, 0])


def rgap(ub, qb, v):
    """The gibiane RG: sum over the outer boundary of q v - u (K v); equals
    -int_Gamma [u] dn v (minus the RG of the book), exactly for v of degree <= 2."""
    Kv = K @ v
    return np.sum((qb * v - ub * Kv)[outer])


def noisy(u, q, mu, mq, eps, rng):
    """Multiplicative Gaussian noise of relative level eps on the measured values
    only: u on mu, q on mq (as the procedure mesbr)."""
    ub, qb = u.copy(), q.copy()
    if eps > 0:
        ub[mu] *= 1 + eps * rng.standard_normal(mu.sum())
        qb[mq] *= 1 + eps * rng.standard_normal(mq.sum())
    return ub, qb


def idfis(d1, d2, E1, E2, nexp=2, comb=True):
    """Procedure idfis: normal, line c, centre s0, lengths (moments, dipole)."""
    (ub1, qb1), (ub2, qb2) = d1, d2
    r1v = np.array([rgap(ub1, qb1, xr[:, 0]), rgap(ub1, qb1, xr[:, 1])])
    r2v = np.array([rgap(ub2, qb2, xr[:, 0]), rgap(ub2, qb2, xr[:, 1])]) if nexp > 1 \
        else np.zeros(2)
    n0 = r1v / np.linalg.norm(r1v) if np.linalg.norm(r1v) >= np.linalg.norm(r2v) \
        else r2v / np.linalg.norm(r2v)
    s = np.array([r1v @ n0, r2v @ n0])
    if comb:
        w = s / np.linalg.norm(s)
    else:
        w = np.zeros(2)
        k = np.argmax(np.abs(s))
        w[k] = np.sign(s[k])
    ub, qb = w[0] * ub1 + w[1] * ub2, w[0] * qb1 + w[1] * qb2
    r = w[0] * r1v + w[1] * r2v
    n = r / np.linalg.norm(r)
    en = w[0] * (np.array(E1) @ n) + w[1] * (np.array(E2) @ n)
    t = np.array([-n[1], n[0]])
    eta0, ss = xr @ n, xr @ t
    c = rgap(ub, qb, eta0 ** 2 - ss ** 2) / (2 * rgap(ub, qb, eta0))
    eta = eta0 - c
    m0 = rgap(ub, qb, eta)
    s0 = rgap(ub, qb, ss * eta) / m0
    sp = ss - s0
    mom2 = rgap(ub, qb, sp ** 2 * eta - eta ** 3 / 3) / m0
    amom = 2 * np.sqrt(mom2) if mom2 > 0 else 0.0
    adip = np.sqrt(abs(m0) / (np.pi * abs(en)))
    return dict(n=n, t=t, c=c, s0=s0, amom=amom, adip=adip, m0=m0, en=en, w=w)


def erfis(r, a):
    """Angle error (degrees) and tip error, as the procedure erfis."""
    eang = np.degrees(np.arccos(min(1.0, abs(r["n"] @ nf))))
    xm = X0 + r["c"] * r["n"] + (r["s0"] - a) * r["t"]
    xp = X0 + r["c"] * r["n"] + (r["s0"] + a) * r["t"]
    e1 = max(np.linalg.norm(xm - A), np.linalg.norm(xp - B))
    e2 = max(np.linalg.norm(xp - A), np.linalg.norm(xm - B))
    return eang, min(e1, e2)


def measure(eps, rng):
    """Cauchy data of E1 and E2 with noise: E1 measures u on the vertical sides
    (without the corners, where u is imposed) and q at the bottom and top; E2
    the other way round (procedure mesbr of the gibiane file)."""
    return (noisy(u1, q1, ver & ~corner, hor, eps, rng),
            noisy(u2, q2, hor & ~corner, ver, eps, rng))


def report(eps=0.0, seed=0):
    rng = np.random.default_rng(seed)
    d1, d2 = measure(eps, rng)
    out = []
    for nexp in (1, 2):
        r = idfis(d1, d2, (0, 1), (1, 0), nexp)
        out.append((r, erfis(r, r["amom"]), erfis(r, r["adip"])))
    return out


def sine_series(ub, qb, n, c, nn=7):
    """Coefficients RG_n of the sine-sinh fields along the identified line, as in
    the gibiane file: t oriented towards x > 0, abscissa g from the point of the
    line at x = 0, LL = sqrt(2) max(ldom, hdom); returns g, RG_n, partial sums."""
    t = np.array([-n[1], n[0]])
    if t[0] < 0:
        n, c, t = -n, -c, -t
    ga = -(X0[0] + c * n[0]) / t[0]
    xga = X0 + c * n + ga * t
    gx, gy = (xd - xga) @ t, (xd - xga) @ n
    LL = 1.4142 * max(ldom, hdom)
    lg = np.linspace(0, LL, 401)
    rn, sums = [], []
    total = np.zeros_like(lg)
    for k in range(1, nn + 1):
        lk = k * np.pi / LL
        r = rgap(ub, qb, np.sin(lk * gx) * np.sinh(lk * gy) / lk)
        rn.append(r)
        total = total + (2 / LL) * r * np.sin(lk * lg)
        sums.append(total.copy())
    return lg, np.array(rn), np.sign(rn[0]) * np.array(sums), xga


def rotation(eps=0.0, seed=0, betas=np.arange(0, 181, 5)):
    """Potential u = d.x on the whole boundary, d = (cos b, sin b); only the flux
    is measured (noisy). Single experiment and pair (b, b + 90). Rows: beta,
    |m0|, angle and tip errors of the single experiment, then of the pair."""
    rng = np.random.default_rng(seed)
    ux, qx = dirichlet(outer, xr[:, 0])
    uy, qy = dirichlet(outer, xr[:, 1])
    rows = []
    for b in betas:
        cb, sb = np.cos(np.radians(b)), np.sin(np.radians(b))
        da = noisy(cb * ux + sb * uy, cb * qx + sb * qy, outer & False, outer, eps, rng)
        do = noisy(cb * uy - sb * ux, cb * qy - sb * qx, outer & False, outer, eps, rng)
        r1 = idfis(da, da, (cb, sb), (cb, sb), 1)
        r2 = idfis(da, do, (cb, sb), (-sb, cb), 2)
        rows.append((b, abs(r1["m0"]), *erfis(r1, r1["adip"]), *erfis(r2, r2["adip"])))
    return np.array(rows)


def noise_study(eps, nsamples=200, seed=0):
    """Lengths by the second moment and by the dipole over nsamples noisy copies of
    the data: mean, standard deviation and number of failures (mu2 <= 0), for
    E1 alone and for the combination E1 + E2."""
    out = {}
    for nexp in (1, 2):
        rng = np.random.default_rng(seed)
        lm, ld, fail, ang = [], [], 0, []
        for _ in range(nsamples):
            d1, d2 = measure(eps, rng)
            r = idfis(d1, d2, (0, 1), (1, 0), nexp)
            ld.append(2 * r["adip"])
            ang.append(erfis(r, r["adip"])[0])
            if r["amom"] > 0:
                lm.append(2 * r["amom"])
            else:
                fail += 1
        out[nexp] = dict(dip=(np.mean(ld), np.std(ld)),
                         mom=(np.mean(lm), np.std(lm)) if lm else (np.nan, np.nan),
                         fail=fail, ang=np.mean(ang))
    return out


def true_openings():
    """Openings [u] = u(upper lip) - u(lower lip) of E1 and E2 along the crack,
    and their moments by quadrature (the quantities the gaps should give)."""
    sc, ec = (xd - A) @ tf, (xd - A) @ nf
    on = np.flatnonzero((np.abs(ec) < 1e-9) & (sc > 1e-9) & (sc < lf - 1e-9))
    updofs = set(basis.element_dofs[:, upper].ravel())
    pos = {}
    for i in on:
        pos.setdefault(round(sc[i], 9), []).append(i)
    S, J1, J2 = [0.0], [0.0], [0.0]
    for k in sorted(pos):
        a, b = pos[k] if pos[k][0] in updofs else pos[k][::-1]
        S.append(k), J1.append(u1[a] - u1[b]), J2.append(u2[a] - u2[b])
    S, J1, J2 = np.r_[S, lf], np.r_[J1, 0.0], np.r_[J2, 0.0]

    def moments(J):
        m0 = np.trapezoid(J, S)
        s0 = np.trapezoid(J * S, S) / m0
        return m0, 4 * np.sqrt(np.trapezoid(J * (S - s0) ** 2, S) / m0)
    return S, J1, J2, moments(J1), moments(J2)


if __name__ == "__main__":
    eps = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
    print("mesh: %d triangles, %d P2 dofs" % (mesh.t.shape[1], basis.N))
    print("true crack: n = (%.4f, %.4f), c = %.4f, length = %.4f"
          % (*nf, nf @ (A - X0), lf))
    print("flux balance RG(1), E1: %.1e" % rgap(u1, q1, np.ones(basis.N)))
    S, J1, J2, mo1, mo2 = true_openings()
    print("true openings: m0 = %.5f (E1), %.5f (E2); second-moment lengths "
          "%.4f, %.4f; [u1] / (2 E_n sqrt(a^2 - s^2)) at the centre %.3f"
          % (mo1[0], mo2[0], mo1[1], mo2[1],
             np.interp(lf / 2, S, J1) / (2 * (nf @ [0, 1]) * lf / 2)))
    for (r, (ea1, ta1), (ea2, ta2)), name in zip(report(eps), ("E1", "E1 + E2")):
        print("== %s (eps = %g)" % (name, eps))
        print("  n = (%.4f, %.4f), angle error %.3f deg, c = %.4f"
              % (*r["n"], ea1, r["c"]))
        print("  m0 = %.5f, E_n = %.4f, s0 = %.4f, weights %s"
              % (r["m0"], r["en"], r["s0"], np.round(r["w"], 4)))
        print("  length (moments) = %.4f, tip error %.4f" % (2 * r["amom"], ta1))
        print("  length (dipole)  = %.4f, tip error %.4f" % (2 * r["adip"], ta2))
    r = idfis((u1, q1), (u2, q2), (0, 1), (1, 0), 1)
    lg, rn, sums, xga = sine_series(u1, q1, r["n"], r["c"])
    print("sine-sinh coefficients RG_1..7 (E1, exact data):", np.round(rn, 4))
    rot = rotation(0.0)
    k = np.argmin(rot[:, 1])
    print("rotation, exact data: |m0| = %.4f (0 deg), min %.4f at %d deg, max %.4f"
          % (rot[0, 1], rot[k, 1], rot[k, 0], rot[:, 1].max()))
    rot = rotation(1e-3, seed=2)
    print("rotation, noise 1e-3: largest angle error single %.2f deg (at %d), "
          "pair %.2f deg; largest tip error single %.3f, pair %.3f"
          % (rot[:, 2].max(), rot[np.argmax(rot[:, 2]), 0], rot[:, 4].max(),
             rot[:, 3].max(), rot[:, 5].max()))
    for e in (1e-3, 1e-2):
        o = noise_study(e)
        for k, name in ((1, "E1"), (2, "E1 + E2")):
            d = o[k]
            print("noise %g, %-7s: dipole %.4f +- %.4f, moments %.4f +- %.4f "
                  "(failures %d/200), mean angle error %.2f deg"
                  % (e, name, *d["dip"], *d["mom"], d["fail"], d["ang"]))
