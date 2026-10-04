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

# Data (dimensionless, as in Exercise 2.21): hole radius a, half side L of the
# quarter plate (plate of side 40a), remote tension s.
a, L, s = 1.0, 20.0, 1.0
# Young's modulus E (scales the displacement only) and Poisson's ratio nu.
E, nu = 1.0, 0.3
h_hole, h_far = 0.03, 1.5                  # element sizes at the hole and far away

# ---- geometry and mesh (gmsh) -------------------------------------------
gmsh.initialize()
gmsh.model.add("kirsch")
occ = gmsh.model.occ
# The quarter plate: square [0, L]^2 minus the disc of radius a.
plate = occ.addRectangle(0, 0, 0, L, L)
hole = occ.addDisk(0, 0, 0, a, a)
occ.cut([(2, plate)], [(2, hole)])
occ.synchronize()
surf = gmsh.model.getEntities(2)[0][1]
gmsh.model.addPhysicalGroup(2, [surf], 1)
# Mesh grading: element size h_hole on the hole, growing with the distance d to
# the hole up to h_far at d = 0.6 L (Threshold field on a Distance field).
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
# Hand the gmsh mesh to dolfinx (rank 0 builds it, 2D geometry).
domain, _, _ = io.gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
gmsh.finalize()

# ---- plane-stress elasticity ---------------------------------------------
# Displacement space: continuous quadratic triangles (P2), vector valued.
V = fem.functionspace(domain, ("Lagrange", 2, (2,)))
# Plane stress: sigma = lambda* tr(eps) I + 2 mu eps (in-plane components), with the
# shear modulus mu and the reduced Lame constant lambda* = E nu / (1 - nu^2).
mu = E / (2 * (1 + nu))
lmbda = E * nu / (1 - nu**2)               # plane-stress Lame constant

def eps(v):
    """Small strain tensor eps(v) = sym(grad v)."""
    return ufl.sym(ufl.grad(v))

def sigma(v):
    """Plane-stress Hooke law sigma = lambda* tr(eps) I + 2 mu eps."""
    return lmbda * ufl.tr(eps(v)) * ufl.Identity(2) + 2 * mu * eps(v)

# Boundary edges (dimension 1): symmetry lines x = 0 and y = 0, loaded edge x = L.
fdim = domain.topology.dim - 1
left = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[0], 0))
bottom = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[1], 0))
right = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[0], L))
# Essential symmetry conditions: u_x = 0 on x = 0 and u_y = 0 on y = 0.
bcs = [fem.dirichletbc(PETSc.ScalarType(0), fem.locate_dofs_topological(V.sub(0), fdim, left), V.sub(0)),
       fem.dirichletbc(PETSc.ScalarType(0), fem.locate_dofs_topological(V.sub(1), fdim, bottom), V.sub(1))]
# Tag the edge x = L with 1, so that ds(1) integrates over this edge only.
tags = mesh.meshtags(domain, fdim, np.sort(right), np.full(len(right), 1, dtype=np.int32))
ds = ufl.Measure("ds", domain=domain, subdomain_data=tags)

# Weak form (virtual work, Section 2.5): find u with
#   int sigma(u) : eps(v) dx = int t . v ds  for all admissible v,
# traction t = s e_x on x = L; direct LU solve.
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
t = fem.Constant(domain, PETSc.ScalarType((s, 0.0)))
problem = LinearProblem(ufl.inner(sigma(u), eps(v)) * ufl.dx, ufl.dot(t, v) * ds(1),
                        bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()

# ---- stresses: projection on discontinuous P1, then point evaluation -------
# sigma(u_h) is piecewise linear for P2 displacements: its interpolation in the
# discontinuous P1 tensor space is exact.
W = fem.functionspace(domain, ("DG", 1, (2, 2)))
sig_h = fem.Function(W)
sig_h.interpolate(fem.Expression(sigma(uh), W.element.interpolation_points()))

def evaluate(f, pts):
    """Values of the function f at the 2D points pts (one cell per point,
    found with a bounding-box tree)."""
    from dolfinx import geometry
    tree = geometry.bb_tree(domain, domain.topology.dim)
    p3 = np.c_[pts, np.zeros(len(pts))]
    cand = geometry.compute_collisions_points(tree, p3)
    cells = geometry.compute_colliding_cells(domain, cand, p3)
    return np.array([f.eval(p, cells.links(i)[:1]) for i, p in enumerate(p3)])

# Points on the hole every 5 degrees (just inside the material, r = 1.0001 a).
th = np.linspace(0, np.pi / 2, 19)
pts = 1.0001 * a * np.c_[np.cos(th), np.sin(th)]
S = evaluate(sig_h, pts).reshape(-1, 2, 2)
c, sn = np.cos(th), np.sin(th)
# Hoop stress sigma_tt = e_t . sigma e_t with e_t = (-sin th, cos th); Kirsch on
# the hole: sigma_tt = s (1 - 2 cos 2 th), from -s at th = 0 to 3 s at th = 90 deg.
stt = S[:, 0, 0] * sn**2 + S[:, 1, 1] * c**2 - 2 * S[:, 0, 1] * sn * c
for ti, fe in zip(np.degrees(th), stt):
    print(f"theta = {ti:5.1f}  FEM {fe:6.3f}  Kirsch {1 - 2 * np.cos(2 * np.radians(ti)):6.3f}")
