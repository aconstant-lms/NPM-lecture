"""Thick-walled cylinder under internal pressure (Chapter 5, Exercise "Consistent
against continuum tangent"): axisymmetric plane strain, J2 plasticity with Voce
hardening, radial return at the quadrature points and global Newton method.

STATUS: UNTESTED. Written for dolfinx 0.8/0.9 (FEniCSx); it has not been run.
Reference values from python/examples/ch5/exo_tangent.py (80 elements, 10 steps):
    consistent tangent: 47 Newton iterations, u(a) = 3.4526 mm
    continuum tangent:  72 Newton iterations

Run (serial):  cd fenics && python3 cylinder_plasticity.py
The return map is the one of python/examples/ch5/ci_core.py (Box 5.2), applied
point by point; FEniCS assembles the residual and the tangent stiffness.
"""
import sys
import numpy as np
import ufl
import basix.ufl
from mpi4py import MPI
from dolfinx import fem, mesh

sys.path.insert(0, "../python/examples/ch5")
from ci_core import radial_return, C_elastic, C_algorithmic, C_continuum, Voce  # noqa: E402

# ---------------------------------------------------------------- data
E, nu = 200e3, 0.3                       # MPa
mu, kappa = E / (2 * (1 + nu)), E / (3 * (1 - 2 * nu))
hard = Voce(250.0, 600.0, 30.0)          # s0, s_inf (MPa), delta
a, b, p_max, n_steps, n_el = 100.0, 200.0, 320.0, 10, 80
TANGENT = "alg"                          # "alg" (consistent) or "cont" (continuum)

# ------------------------------------------------------ mesh and spaces
domain = mesh.create_interval(MPI.COMM_WORLD, n_el, [a, b])
V = fem.functionspace(domain, ("Lagrange", 1))
deg_q = 1                                 # one Gauss point per element
cell = domain.basix_cell()
Qv = fem.functionspace(domain, basix.ufl.quadrature_element(cell, value_shape=(4,), degree=deg_q))
Qt = fem.functionspace(domain, basix.ufl.quadrature_element(cell, value_shape=(4, 4), degree=deg_q))
dx = ufl.Measure("dx", domain=domain,
                 metadata={"quadrature_degree": deg_q, "quadrature_scheme": "default"})

r = ufl.SpatialCoordinate(domain)[0]


def eps(w):                              # [e_rr, e_tt, e_zz, e_rz], plane strain
    return ufl.as_vector([w.dx(0), w / r, 0.0, 0.0])


u, du, v = fem.Function(V), ufl.TrialFunction(V), ufl.TestFunction(V)
sig_q, C_q, eps_q = fem.Function(Qv), fem.Function(Qt), fem.Function(Qv)

# pressure on the bore r = a: facet tag 1
facets = mesh.locate_entities_boundary(domain, 0, lambda x: np.isclose(x[0], a))
tags = mesh.meshtags(domain, 0, facets, np.full(len(facets), 1, dtype=np.int32))
ds = ufl.Measure("ds", domain=domain, subdomain_data=tags)
p = fem.Constant(domain, 0.0)

# the factor 2 pi of the volume element cancels; r dx is the axisymmetric measure
F = r * ufl.dot(sig_q, eps(v)) * dx - p * r * v * ds(1)
J = r * ufl.dot(ufl.dot(C_q, eps(du)), eps(v)) * dx
F_form, J_form = fem.form(F), fem.form(J)

ipts = Qv.element.interpolation_points
ipts = ipts() if callable(ipts) else ipts     # method in 0.8, property in 0.9
eps_expr = fem.Expression(eps(u), ipts)

# internal variables at the quadrature points (ci_core uses 6 Voigt components)
nq = len(eps_q.x.array) // 4
ep_n, beta_n, al_n = np.zeros((nq, 6)), np.zeros((nq, 6)), np.zeros(nq)
ep, beta, al = ep_n.copy(), beta_n.copy(), al_n.copy()
Cel = C_elastic(mu, kappa)


def local_stage():
    """Radial return at every quadrature point, from the converged z_n."""
    eps_q.interpolate(eps_expr)
    e4 = eps_q.x.array.reshape(nq, 4)
    S, C = sig_q.x.array.reshape(nq, 4), C_q.x.array.reshape(nq, 4, 4)
    for i in range(nq):
        e6 = np.r_[e4[i], 0.0, 0.0]
        s6, ep[i], al[i], beta[i], dg, n, nx, _ = radial_return(
            e6, ep_n[i], al_n[i], beta_n[i], mu, kappa, hard)
        if dg == 0.0:
            Ct = Cel
        elif TANGENT == "alg":
            Ct = C_algorithmic(dg, nx, n, mu, kappa, hard.slope(al[i]))
        else:
            Ct = C_continuum(n, mu, kappa, hard.slope(al[i]))
        S[i] = s6[:4]
        C[i] = Ct[:4, :4]


total = 0
for step in range(n_steps):
    p.value = p_max * (step + 1) / n_steps
    for it in range(40):
        local_stage()
        R = -fem.assemble_vector(F_form).array          # R = F_ext - F_int
        rn = np.linalg.norm(R)
        if it == 0:
            r0 = max(rn, 1.0)
        if rn < 1e-8 * max(1.0, p.value * a):
            break
        K = fem.assemble_matrix(J_form).to_dense()
        u.x.array[:] += np.linalg.solve(K, R)
        u.x.scatter_forward()
    total += it + 1
    ep_n[:], beta_n[:], al_n[:] = ep, beta, al           # accept the step
    print(f"step {step + 1:2d}: p = {p.value:6.1f} MPa, Newton iterations {it + 1}")
x = V.tabulate_dof_coordinates()[:, 0]
print(f"total iterations {total}, u(a) = {u.x.array[np.argmin(abs(x - a))]:.6f} mm")
