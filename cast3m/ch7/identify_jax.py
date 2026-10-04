"""identify_jax.py -- companion exercise C7.6: a Cast3M computation inside a JAX
program. jax.custom_vjp declares the derivative of the black box (here central
differences of Cast3M runs); JAX differentiates everything around it."""
import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import minimize

import castem_run as cr

# The cost is written in JAX: log parameters, scaling, a smooth penalty. JAX
# cannot see inside Cast3M; pure_callback calls it, custom_vjp supplies the
# vector-Jacobian product v^T du/dlog p from cr.jacobian_fd (6 runs). Replacing
# the finite differences by a Cast3M direct differentiation (Box 7.3) or adjoint
# (Box 7.4) would change one function and nothing else.
jax.config.update("jax_enable_x64", True)
um = jnp.array(cr.measurements())
shape = jax.ShapeDtypeStruct((81, 2), jnp.float64)


def _u(logp):
    return np.asarray(cr.displacements(np.exp(np.asarray(logp))), np.float64)


def _jac(logp):
    return np.asarray(cr.jacobian_fd(np.exp(np.asarray(logp))), np.float64)


@jax.custom_vjp
def castem_u(logp):                           # u(t_n) of node O for log p
    return jax.pure_callback(_u, shape, logp)


def castem_u_fwd(logp):
    return castem_u(logp), logp


def castem_u_bwd(logp, v):                    # v^T (du / dlog p)
    J = jax.pure_callback(_jac, jax.ShapeDtypeStruct((81, 2, 3), jnp.float64), logp)
    return (jnp.einsum("nc,ncj->j", v, J),)


castem_u.defvjp(castem_u_fwd, castem_u_bwd)


def cost(q, p0, prior=0.0):
    """J of Exercise 7.9 in q = log(p/p0), plus an optional Tikhonov term."""
    u = castem_u(jnp.log(p0) + q)
    return 0.5 * jnp.sum((u - um) ** 2) / cr.UREF ** 2 + 0.5 * prior * jnp.sum(q ** 2)


value_and_grad = jax.value_and_grad(cost)

# (a) Taylor test (Box 7.5) of the gradient at p1 = (0.9, 1.2, 1.5) p_true
p1 = cr.P_TRUE * np.array([0.9, 1.2, 1.5])
q0, dq = jnp.zeros(3), jnp.array([1.0, -0.5, 0.3])
J0, g0 = value_and_grad(q0, p1)
print("J(p1) = %.4f, dJ/dlog p = %s" % (J0, np.round(np.array(g0), 6)))
print("   h       e0(h)      e1(h)")
for h in (1e-1, 3e-2, 1e-2, 3e-3, 1e-3):
    Jh = cost(q0 + h * dq, p1)
    print("%7.0e  %.3e  %.3e" % (h, abs(Jh - J0), abs(Jh - J0 - h * jnp.dot(g0, dq))))

# (b) BFGS on the JAX cost and gradient, from p0 = (0.7, 1.2, 3) p_true
p0 = cr.P_TRUE * np.array([0.7, 1.2, 3.0])
cr.nruns = 0
res = minimize(lambda q: tuple(np.asarray(a) for a in value_and_grad(jnp.array(q), p0)),
               np.zeros(3), jac=True, method="BFGS", options={"gtol": 1e-6})
print("BFGS: p/p_true = %s, J = %.4f, %d iterations, %d Cast3M runs"
      % (np.round(p0 * np.exp(res.x) / cr.P_TRUE, 4), res.fun, res.nit, cr.nruns))
# Book (adjoint gradient): BFGS reaches (1.0022, 0.9992, 1.0036), J = 0.2911,
# with 28 direct solves and 28 adjoints; Exercise 7.9: dJ/dlog p at p1 =
# (179.805452, 305.777548, 5.967107).
