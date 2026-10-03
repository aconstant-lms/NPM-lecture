"""Exercise "From compliances to stiffnesses" (ch2) of the lecture notes.
Run: python3 python/examples/ch2/elastic_constants.py
"""
import numpy as np

def voigt(a11, a12, a13, a33, a44, a66, a14=0.0, shear=1.0):
    """Symmetric 6x6 Voigt matrix with the 3-axis as main axis (cubic,
    tetragonal 4/mmm, hexagonal, trigonal 32 or 3m). The trigonal term a56 is
    2 a14 for compliances (shear = 2) and a14 for stiffnesses (shear = 1)."""
    A = np.diag([a11, a11, a33, a44, a44, a66])
    A[0, 1], A[0, 2], A[1, 2] = a12, a13, a13
    A[0, 3], A[1, 3], A[4, 5] = a14, -a14, shear * a14
    return np.triu(A) + np.triu(A, 1).T

def S_matrix(cls, s11, s12, s44, s33=None, s13=None, s66=None, s14=0.0):
    if cls == "cubic":                     # s33 = s11, s13 = s12, s66 = s44
        s33, s13, s66 = s11, s12, s44
    elif cls in ("hexagonal", "trigonal"): # transverse isotropy in the plane 12
        s66 = 2 * (s11 - s12)
    return voigt(s11, s12, s13, s33, s44, s66, s14, shear=2.0)

data = [  # Nye (1985), compliances in 1e-11 1/Pa
    ("NaCl", "cubic", dict(s11=2.21, s12=-0.45, s44=7.83)),
    ("aluminium", "cubic", dict(s11=1.59, s12=-0.58, s44=3.52)),
    ("copper", "cubic", dict(s11=1.49, s12=-0.63, s44=1.33)),
    ("nickel", "cubic", dict(s11=0.799, s12=-0.312, s44=0.844)),
    ("tungsten", "cubic", dict(s11=0.257, s12=-0.073, s44=0.660)),
    ("tin", "tetragonal", dict(s11=1.85, s12=-0.99, s44=5.70, s33=1.18, s13=-0.25, s66=13.5)),
    ("ADP", "tetragonal", dict(s11=1.8, s12=0.7, s44=11.3, s33=4.3, s13=-1.1, s66=16.2)),
    ("zinc", "hexagonal", dict(s11=0.84, s12=0.11, s44=2.64, s33=2.87, s13=-0.78)),
    ("quartz", "trigonal", dict(s11=1.27, s12=-0.17, s44=2.01, s33=0.97, s13=-0.15, s14=-0.43)),
    ("tourmaline", "trigonal", dict(s11=0.40, s12=-0.10, s44=1.51, s33=0.63, s13=-0.016, s14=-0.058)),
]
print("GPa          c11  c12  c13  c33  c44  c66  c14")
for name, cls, s in data:
    C = 100 * np.linalg.inv(S_matrix(cls, **s))   # 1/(1e-11 1/Pa) = 100 GPa
    assert np.linalg.eigvalsh(C).min() > 0         # stability
    ij = [(0, 0), (0, 1), (0, 2), (2, 2), (3, 3), (5, 5), (0, 3)]
    print(f"{name:11s}", " ".join(f"{C[i, j]:4.0f}" for i, j in ij))

# cadmium (hexagonal): measured adiabatic stiffnesses at 300 K, in GPa
c11, c33, c44, c13, c66 = 114.5, 50.85, 19.85, 39.9, 37.5
c12 = c11 - 2 * c66
S = 100 * np.linalg.inv(voigt(c11, c12, c13, c33, c44, c66))
print("cadmium c12 =", c12, "GPa; s11 s12 s13 s33 s44 =",
      np.round([S[0, 0], S[0, 1], S[0, 2], S[2, 2], S[3, 3]], 2), "1e-11/Pa")
# GPa          c11  c12  c13  c33  c44  c66  c14
# NaCl          51   13   13   51   13   13    0
# aluminium    108   62   62  108   28   28    0
# copper       176  129  129  176   75   75    0
# nickel       250  160  160  250  118  118    0
# tungsten     502  199  199  502  152  152    0
# tin           84   49   28   97   18    7    0
# ADP           71  -20   13   30    9    6    0
# zinc         164   27   52   63   38   68    0
# quartz        87    8   15  108   57   40   17
# tourmaline   268   66    8  159   67  101    8
# cadmium c12 = 39.5 GPa; s11 s12 s13 s33 s44 = [ 1.21 -0.12 -0.86  3.31  5.04] 1e-11/Pa
