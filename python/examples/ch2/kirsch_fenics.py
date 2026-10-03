"""Plate with a hole (Kirsch problem) in FEniCSx (dolfinx), ch2.

Provided as a starting point for the exercise "A plate with a hole: Kirsch
against finite elements"; NOT run when the notes were prepared (dolfinx was
not available). Written for dolfinx >= 0.8 with the gmsh Python API.
Run (in a FEniCSx environment): python3 kirsch_fenics.py

Quarter of a square plate [0, L]^2 with a hole of radius a, plane stress,
remote tension s along x applied on the edge x = L, symmetry conditions
u_x = 0 on x = 0 and u_y = 0 on y = 0. Quadratic triangles (P2).
Compare sigma_xx(0, a) with 3 s and the hoop stress on the hole with
s (1 - 2 cos 2 theta), as in kirsch_fem.py.
"""
import numpy as np
import gmsh
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, mesh, io
from dolfinx.fem.petsc import LinearProblem

a, L, s = 1.0, 20.0, 1.0
E, nu = 1.0, 0.3
h_hole, h_far = 0.03, 1.5                  # element sizes at the hole and far away

# ---- geometry and mesh (gmsh) -------------------------------------------
gmsh.initialize()
gmsh.model.add("kirsch")
occ = gmsh.model.occ
plate = occ.addRectangle(0, 0, 0, L, L)
hole = occ.addDisk(0, 0, 0, a, a)
occ.cut([(2, plate)], [(2, hole)])
occ.synchronize()
surf = gmsh.model.getEntities(2)[0][1]
gmsh.model.addPhysicalGroup(2, [surf], 1)
dist = gmsh.model.mesh.field.add("Distance")
arcs = [c for _, c in gmsh.model.getEntities(1)     # the curve inside the box [0, a]^2
        if max(gmsh.model.getBoundingBox(1, c)[3:5]) < a + 1e-6]
gmsh.model.mesh.field.setNumbers(dist, "CurvesList", arcs)
thr = gmsh.model.mesh.field.add("Threshold")
gmsh.model.mesh.field.setNumber(thr, "InField", dist)
gmsh.model.mesh.field.setNumber(thr, "SizeMin", h_hole)
gmsh.model.mesh.field.setNumber(thr, "SizeMax", h_far)
gmsh.model.mesh.field.setNumber(thr, "DistMin", 0.0)
gmsh.model.mesh.field.setNumber(thr, "DistMax", 0.6 * L)
gmsh.model.mesh.field.setAsBackgroundMesh(thr)
gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
gmsh.model.mesh.generate(2)
domain, _, _ = io.gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
gmsh.finalize()

# ---- plane-stress elasticity ---------------------------------------------
V = fem.functionspace(domain, ("Lagrange", 2, (2,)))
mu = E / (2 * (1 + nu))
lmbda = E * nu / (1 - nu**2)               # plane-stress Lame constant

def eps(v):
    return ufl.sym(ufl.grad(v))

def sigma(v):
    return lmbda * ufl.tr(eps(v)) * ufl.Identity(2) + 2 * mu * eps(v)

fdim = domain.topology.dim - 1
left = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[0], 0))
bottom = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[1], 0))
right = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[0], L))
bcs = [fem.dirichletbc(PETSc.ScalarType(0), fem.locate_dofs_topological(V.sub(0), fdim, left), V.sub(0)),
       fem.dirichletbc(PETSc.ScalarType(0), fem.locate_dofs_topological(V.sub(1), fdim, bottom), V.sub(1))]
tags = mesh.meshtags(domain, fdim, np.sort(right), np.full(len(right), 1, dtype=np.int32))
ds = ufl.Measure("ds", domain=domain, subdomain_data=tags)

u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
t = fem.Constant(domain, PETSc.ScalarType((s, 0.0)))
problem = LinearProblem(ufl.inner(sigma(u), eps(v)) * ufl.dx, ufl.dot(t, v) * ds(1),
                        bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()

# ---- stresses: projection on discontinuous P1, then point evaluation -------
W = fem.functionspace(domain, ("DG", 1, (2, 2)))
sig_h = fem.Function(W)
sig_h.interpolate(fem.Expression(sigma(uh), W.element.interpolation_points()))

def evaluate(f, pts):
    from dolfinx import geometry
    tree = geometry.bb_tree(domain, domain.topology.dim)
    p3 = np.c_[pts, np.zeros(len(pts))]
    cand = geometry.compute_collisions_points(tree, p3)
    cells = geometry.compute_colliding_cells(domain, cand, p3)
    return np.array([f.eval(p, cells.links(i)[:1]) for i, p in enumerate(p3)])

th = np.linspace(0, np.pi / 2, 19)
pts = 1.0001 * a * np.c_[np.cos(th), np.sin(th)]
S = evaluate(sig_h, pts).reshape(-1, 2, 2)
c, sn = np.cos(th), np.sin(th)
stt = S[:, 0, 0] * sn**2 + S[:, 1, 1] * c**2 - 2 * S[:, 0, 1] * sn * c
for ti, fe in zip(np.degrees(th), stt):
    print(f"theta = {ti:5.1f}  FEM {fe:6.3f}  Kirsch {1 - 2 * np.cos(2 * np.radians(ti)):6.3f}")
