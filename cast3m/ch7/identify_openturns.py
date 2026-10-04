"""identify_openturns.py -- companion exercise C7.5: the truss identification with
OpenTURNS as the driver and Cast3M as the direct solver (Exercise 7.13 (b) with
the Python model replaced by Cast3M runs, gradient by central differences)."""
import numpy as np
import openturns as ot

import castem_run as cr

# Same layout as python/examples/ch7/exo_openturns_truss.py: the observation map
# (n, q) -> u(t_n) / u_ref, with q = log(p / p0) frozen as the parameters of a
# ParametricFunction; every new q costs one Cast3M run (cached), every gradient
# six (central differences in q, run in parallel by castem_run.jacobian_fd).
um = cr.measurements()
p0 = cr.P_TRUE * np.array([0.7, 1.2, 3.0])        # start, q = 0


def model(x):                  # x = (n, q1, q2, q3) -> displacement at instant n
    n, q = int(round(x[0])), np.array(x[1:])
    return list(cr.displacements(p0 * np.exp(q))[n] / cr.UREF)


def gradient(x):               # OpenTURNS convention: (input dim) x (output dim)
    n, q = int(round(x[0])), np.array(x[1:])
    G = np.zeros((4, 2))
    G[1:, :] = (cr.jacobian_fd(p0 * np.exp(q))[n] / cr.UREF).T
    return G.tolist()


f = ot.PythonFunction(4, 2, model, gradient=gradient)
g = ot.ParametricFunction(f, [1, 2, 3], [0.0, 0.0, 0.0])
x_obs = ot.Sample([[n] for n in range(1, 81)])
y_obs = ot.Sample((um[1:] / cr.UREF).tolist())

cr.nruns = 0
calib = ot.NonLinearLeastSquaresCalibration(g, x_obs, y_obs, [0.0, 0.0, 0.0])
calib.setBootstrapSize(0)                         # Gaussian posterior at the MAP
calib.setOptimizationAlgorithm(ot.CMinpack())     # Levenberg-Marquardt
calib.run()
res = calib.getResult()
q = np.array(res.getParameterMAP())
print("OpenTURNS", ot.__version__)
print("p/p_true =", np.round(p0 * np.exp(q) / cr.P_TRUE, 4), " Cast3M runs:", cr.nruns)
print("posterior standard deviation of log p:",
      np.round(np.array(res.getParameterPosterior().getStandardDeviation()), 4))
# Book (exact derivatives): p/p_true = (1.0022, 0.9992, 1.0036) in 14 solves,
# standard deviations (0.0059, 0.0009, 0.0079).
