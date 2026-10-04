"""Exercise "Stretching of a sheet in plane strain" (ch4) of the lecture notes.
Run: python3 python/examples/ch4/exo_sheet.py
"""
# Homogeneous sheet, perfect von Mises plasticity: strain-driven computation with
# the radial return of Chapter 5 (Box 5.2, K = H = 0), sigma_yy = 0 and
# eps_zz = 0, compared with the closed form eps_xx(theta) of part (d).
import numpy as np
from scipy.optimize import brentq
E, nu, sY = 200e3, 0.3, 250.0                 # MPa, -, MPa
mu, kap = E / (2 * (1 + nu)), E / (3 * (1 - 2 * nu))   # shear and bulk moduli (MPa)
I = np.eye(3)

def stress(eps, ep):                      # J2 perfect plasticity, radial return
    """Radial return from the plastic strain ep (3x3) for the total strain eps.
    Returns sigma (MPa) and the updated plastic strain."""
    e = eps - ep
    # Elastic predictor: trial deviator s_tr = 2 mu dev(eps - eps^p)
    s_tr = 2 * mu * (e - np.trace(e) / 3 * I)
    # Yield function f = ||s_tr|| - sqrt(2/3) sY (radius of the von Mises cylinder)
    f = np.linalg.norm(s_tr) - np.sqrt(2 / 3) * sY
    # Plastic corrector: Delta eps^p = Delta gamma n, Delta gamma = f / (2 mu)
    dep = f / (2 * mu) * s_tr / np.linalg.norm(s_tr) if f > 0 else 0 * I
    return kap * np.trace(e) * I + s_tr - 2 * mu * dep, ep + dep

# Closed form of part (d): sigma_xx = R sin(theta + pi/6), sigma_zz = R sin(theta - pi/6)
R = 2 * sY / np.sqrt(3)
# theta0: angle of the elastic line sigma_zz = nu sigma_xx,
# tan(theta0) = (1 + nu) / (sqrt3 (1 - nu))
th0 = np.arctan((1 + nu) / (np.sqrt(3) * (1 - nu)))
s0 = R * np.sin(th0 + np.pi / 6)                       # sigma_xx at first yield
lt = lambda t: np.log(np.tan(t / 2 + np.pi / 3))       # the log term of eps_xx(theta)
print("first yield: sigma_xx = %.3f = sY/sqrt(1-nu+nu^2) = %.3f"
      % (s0, sY / np.sqrt(1 - nu + nu**2)))
ep = np.zeros((3, 3))
# 40000 small steps up to eps_xx = 0.004; at each step eps_yy is the root of sigma_yy = 0
for exx in np.linspace(0, 0.004, 40001)[1:]:  # strain control, sigma_yy = 0, eps_zz = 0
    g = lambda y: stress(np.diag([exx, y, 0.0]), ep)[0][1, 1]
    sig, ep = stress(np.diag([exx, brentq(g, -0.05, 0.05, xtol=1e-15), 0.0]), ep)
    if np.isclose(exx * 1e4 % 5, 0) and exx > 0.0015:  # closed form at eps_xx(theta)
        # theta of the computed stress, then eps_xx(theta) from the formula of part (d)
        th = np.arctan2(np.sqrt(3) * (sig[0, 0] + sig[2, 2]), 3 * (sig[0, 0] - sig[2, 2]))
        exx_cf = s0 * (1 - nu**2) / E + R / E * (0.75 * (lt(th) - lt(th0))
                 + (1 - 2 * nu) * (np.sin(th - np.pi / 6) - np.sin(th0 - np.pi / 6)))
        print("eps_xx %.4f  sigma_xx %.3f  sigma_zz %.3f  closed form eps_xx %.6f"
              % (exx, sig[0, 0], sig[2, 2], exx_cf))
# Limit of part (c), reached only asymptotically: the plane-strain point theta = pi/3
print("limit stresses 2 sY/sqrt3 = %.3f, sY/sqrt3 = %.3f" % (R, R / 2))
# first yield: sigma_xx = 281.272 = sY/sqrt(1-nu+nu^2) = 281.272
# eps_xx 0.0020  sigma_xx 286.484  sigma_zz 112.499  closed form eps_xx 0.002000
# eps_xx 0.0025  sigma_xx 287.765  sigma_zz 124.049  closed form eps_xx 0.002500
# eps_xx 0.0030  sigma_xx 288.303  sigma_zz 131.470  closed form eps_xx 0.003000
# eps_xx 0.0040  sigma_xx 288.615  sigma_zz 139.197  closed form eps_xx 0.004000
# limit stresses 2 sY/sqrt3 = 288.675, sY/sqrt3 = 144.338
