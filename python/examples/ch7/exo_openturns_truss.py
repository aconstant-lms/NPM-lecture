"""Exercise 7.x -- the truss identification driven by OpenTURNS: the model is
wrapped as a PythonFunction with the DDM gradient, and calibrated by nonlinear
least squares (Levenberg-Marquardt of CMinpack)."""
import numpy as np
import openturns as ot
from id_core import Truss

# Exercise 7.13 (b): the truss of Exercise 7.9 (same data and noise), moduli in MPa.
# OpenTURNS is the driver; the direct solver and the derivative layer are ours.
E, sY, H = 200e3, 200.0, 10e3
p_true = np.array([E, sY, H])
Q = lambda t: np.array([2.3 * sY * np.sin(t), 0.8 * sY * np.sin(2 * t)])
ts = np.linspace(0, 2 * np.pi, 81)
T = Truss(Q, ts)
uref = sY / E
u_true = T.forward(p_true)[0]
rng = np.random.default_rng(0)
um = u_true + 0.01 * np.abs(u_true).max() * rng.standard_normal(u_true.shape)
p0 = p_true * np.array([0.7, 1.2, 3.0])           # start, q = log(p / p0) = 0

cache = {}                     # DDM sensitivities at the last parameter point


def sensitivities(q):                             # one DDM sweep per parameter point
    if tuple(q) not in cache:
        cache.clear()
        cache[tuple(q)] = T.ddm(p0 * np.exp(np.array(q)))[1]
    return cache[tuple(q)]


def model(x):                  # x = (n, q1, q2, q3) -> displacement at instant n
    n, q = int(round(x[0])), np.array(x[1:])
    return list(T.forward(p0 * np.exp(q))[0][n] / uref)   # forward solve is cached


def gradient(x):               # OpenTURNS convention: (input dim) x (output dim)
    n, q = int(round(x[0])), list(x[1:])
    G = np.zeros((4, 2))
    G[1:, :] = (sensitivities(q)[n] / uref * p0 * np.exp(q)).T
    return G.tolist()


# observation map: input (n, q), output (u_1, u_2)(t_n) / uref; the parameters q
# are frozen in a ParametricFunction so that the calibration only sees n
f = ot.PythonFunction(4, 2, model, gradient=gradient)
g = ot.ParametricFunction(f, [1, 2, 3], [0.0, 0.0, 0.0])   # parameters q, input n
x_obs = ot.Sample([[n] for n in range(1, len(ts))])
y_obs = ot.Sample((um[1:] / uref).tolist())

# nonlinear least squares from q = 0, i.e. p = p0; without bootstrap the posterior
# is a Gaussian approximation at the minimum, to compare with Section 7.2.2
T.nsolve = 0
calib = ot.NonLinearLeastSquaresCalibration(g, x_obs, y_obs, [0.0, 0.0, 0.0])
calib.setBootstrapSize(0)                         # no bootstrap: Gaussian posterior
calib.setOptimizationAlgorithm(ot.CMinpack())     # Levenberg-Marquardt
calib.run()
res = calib.getResult()
q = np.array(res.getParameterMAP())
print("OpenTURNS version", ot.__version__)
print("p/p_true =", np.round(p0 * np.exp(q) / p_true, 4), " forward solves:", T.nsolve)
print("posterior standard deviation of log p:",
      np.round(np.array(res.getParameterPosterior().getStandardDeviation()), 4))
