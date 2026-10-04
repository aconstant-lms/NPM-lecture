"""Exercise 7.x -- reverse-mode automatic differentiation (JAX) of the truss
identification: the gradient of the cost equals the discrete adjoint."""
import numpy as np
import jax
import jax.numpy as jnp
from id_core import Truss

# Exercise 7.13 (a): the cost of the truss of Exercise 7.9 written with jax.numpy,
# differentiated in reverse mode (Section 7.7.2, "Automatic differentiation").
# Measurements here are the noise-free displacements at p_true; moduli in MPa.
jax.config.update("jax_enable_x64", True)
E, sY, H = 200e3, 200.0, 10e3
p_true = np.array([E, sY, H])
Q = lambda t: np.array([2.3 * sY * np.sin(t), 0.8 * sY * np.sin(2 * t)])
ts = np.linspace(0, 2 * np.pi, 81)
T = Truss(Q, ts)
uref = sY / E
um = T.forward(p_true)[0]
B, w = jnp.array(T.B), jnp.array(T.w)
F = jnp.array([Q(t) for t in ts])


def return_map(eps, z, p):                     # Box 5.1, K = 0, written with jnp
    """Returns sigma, the new plastic strain and the tangent modulus of each bar."""
    E, sY, H = p
    xi = E * (eps - z) - H * z
    dg = jnp.maximum(jnp.abs(xi) - sY, 0.0) / (E + H)
    zn = z + jnp.sign(xi) * dg
    return E * (eps - zn), zn, jnp.where(dg > 0, E * H / (E + H), E)


def cost(p, newton_its=8):
    """J(p) of (7.3) with a fixed number of Newton iterations per step, so that
    jax.grad differentiates exactly the iterations that are run."""
    u, z, J = jnp.zeros(2), jnp.zeros(3), 0.0
    for n in range(1, len(ts)):
        for _ in range(newton_its):            # fixed number of Newton iterations
            sig, zn, Et = return_map(B @ u, z, p)
            K = B.T @ ((w * Et)[:, None] * B)
            u = u + jnp.linalg.solve(K, F[n] - B.T @ (w * sig))
        z = return_map(B @ u, z, p)[1]         # plastic strain at the end of the step
        J = J + 0.5 * jnp.sum((u - um[n]) ** 2) / uref ** 2
    return J


# gradient at p1 = (0.9, 1.2, 1.5) p_true: reverse mode against Box 7.4, printed
# as p dJ/dp = dJ/d log p; then with 1, 2 and 4 Newton iterations per step
p1 = jnp.array(p_true * np.array([0.9, 1.2, 1.5]))
g_ad = jax.grad(cost)(p1)
g_adj = T.gradient_adjoint(np.array(p1), um, uref)
print("J            ", float(cost(p1)), T.cost(np.array(p1), um, uref))
print("jax.grad     ", np.array(g_ad) * np.array(p1))
print("adjoint      ", g_adj * np.array(p1))
print("relative gap ", np.abs(np.array(g_ad) - g_adj).max() / np.abs(g_adj).max())
for its in (1, 2, 4):
    print("%d Newton its: relative gap %.1e" % (its, np.abs(np.array(jax.grad(cost)(p1, its)) - g_adj).max()
                                                / np.abs(g_adj).max()))
