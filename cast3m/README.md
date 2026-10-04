# Cast3M input files

Cast3M versions of exercises of the notes, sorted by chapter. They are
the files printed in the companion document `cast3m_companion/`
(exercises numbered C<chapter>.<n>); Chapter 3 (reciprocity gap) is
treated separately.

| file | book exercise | companion | status |
|---|---|---|---|
| `ch1/springs_multipliers.dgibi` | Ch. 1, spring chain, constraint, reciprocity | C1.1 | untested |
| `ch2/kirsch_elements.dgibi` | Ch. 2, Kirsch (d): T3/T6/Q4/Q8, 3 meshes, balance of reactions | C2.1 | untested |
| `ch4/sheet_plane_strain.dgibi` | Ch. 4, stretching of a sheet (MEC563 file, SI units, sY = 200 MPa) | C4.1 | author's file, run in the course |
| `ch4/sheet_vm_250.dgibi` | Ch. 4, same exercise with the data of `exo_sheet.py` | C4.1 | untested |
| `ch4/filament_cyclic.dgibi` | Ch. 4, cyclic loading, isotropic/kinematic hardening (bar element) | C4.2 | untested |
| `ch4/cylinder_residual.dgibi` | Ch. 4, thick cylinder: loading, unloading, residual stresses | C4.3 | untested |
| `ch5/return_map_point.dgibi` | Ch. 5, radial return (Box 5.2) **programmed on Gauss-point fields**; shear and non-proportional path | C5.1 | untested |
| `ch5/cylinder_radial_return.dgibi` | Ch. 5, thick cylinder: the same return map + equilibrium iterations in gibiane, against PASAPAS | C5.2 | untested |
| `ch5/cylinder_pressure.dgibi` | Ch. 5, thick cylinder with Voce hardening, PASAPAS | C5.2 | untested |
| `ch6/bree_tube.dgibi` | Ch. 6, Bree's problem on an axisymmetric tube, cycle by cycle | C6.1 | untested |
| `ch7/truss.template` + drivers | Ch. 7, identification of the truss with Cast3M as direct solver | C7.3-C7.6 | drivers tested with the stand-in, template untested |

The Kirsch file `python/examples/ch2/kirsch_cast3m.dgibi` (also untested)
stays where the Python README points to it.

## Running

    castem25 ch5/return_map_point.dgibi        (or the local Cast3M command)

Each file prints its results next to the reference values of the Python
scripts (`mess` lines), so a run is checked by reading its output. The files
end with `opti donn 5` (hand over to the keyboard) and draw with `dess`/`trac`,
as the course files do; for an unattended run remove `opti donn 5` and add
`opti trac psc` at the top (the drawings go to a PostScript file).

## The drivers of Chapter 7 (`ch7/`)

`truss.template` is a complete Cast3M file in which `__E__`, `__SY__`,
`__H__` stand for the parameters. The drivers fill it, run Cast3M in a
directory of its own (`runs/...`, one per parameter point: Cast3M writes its
`fort.*` files where it runs), check the output for `ERREUR`, and read the CSV
file `truss_u.csv` written by the procedure `EXPORTCSV` of the Cast3M
introduction.

| file | driver | needs |
|---|---|---|
| `sweep_sy.sh` | shell: `sed` + Cast3M + `awk`, the cost J(sY) | bash, awk |
| `identify_scipy.py` | `scipy.optimize.least_squares` (Levenberg-Marquardt), Jacobian by central differences (6 parallel runs) | numpy, scipy |
| `identify_openturns.py` | OpenTURNS `NonLinearLeastSquaresCalibration` + CMinpack, posterior | openturns |
| `identify_jax.py` | JAX: Cast3M inside `jax.pure_callback` + `jax.custom_vjp`; Taylor test, BFGS | jax, scipy |
| `castem_run.py` | the common runner (template, run, cache, counter, central differences) | |
| `make_measurements.py` | writes `truss_measurements.csv` (book data, seed 0) | the notes' `id_core` |
| `tools/castem_stub.py` | stands in for Cast3M: same file in, same CSV out, computed with `id_core` | |

The command is taken from the environment variable `CASTEM` (default
`castem25`). Without Cast3M, the drivers can be checked against the stand-in:

    cd cast3m/ch7
    export CASTEM="python3 $PWD/tools/castem_stub.py"
    ./sweep_sy.sh
    python3 identify_scipy.py
    python3 identify_openturns.py
    python3 identify_jax.py

With the stand-in they reproduce the book (Exercises 7.9 and 7.13):
p/p_true = (1.0022, 0.9992, 1.0036), J = 0.2911 (SciPy: 79 runs; OpenTURNS: 78
runs, posterior standard deviations (0.0059, 0.0009, 0.0079); JAX + BFGS: 210
runs); from (0.7, 1.5, 3) p_true SciPy stops at E/E_true = 0.4424, J = 522.5.
With Cast3M the numbers should agree to the PASAPAS tolerance.

## Constructions to check first

The files were written against the operator notices and the Cast3M
introduction, without a Cast3M installation. These are the constructions most
likely to need a fix; each one is used in the files listed.

1. Arithmetic between two `MCHAML` with the same single component `SCAL`
   (`*`, `/`, `+`, `-`), and `+`/`-` between two stress fields
   (`ch5/*`).
2. `EXCO` on an `MCHAML` renaming a component in both directions,
   `exco x 'SMXX' 'SCAL'` and `exco y 'SCAL' 'SMXX'`, and `ET` joining the
   components back into one field (`ch5/*`).
3. The component name of `VMIS` (assumed `SCAL`), `MASQ` and `EXP` applied to
   an `MCHAML` (`ch5/*`).
4. `CHAN 'CHAM' (coor 1 dom) mo 'STRESSES'` for the radius of the Gauss points
   (`ch4/cylinder_residual`, `ch5/cylinder_radial_return`).
5. The name `EPSE` of the cumulated plastic strain in the internal variables
   (`ch4/*`, `ch5/cylinder_radial_return`).
6. Bar elements (`BARR`) with `PLASTIQUE ISOTROPE` and `PLASTIQUE CINEMATIQUE`,
   and the meaning of `H` in the kinematic model: the first turning-point stress
   of `ch4/filament_cyclic` decides it (257.389 MPa if Cast3M's H is the
   uniaxial plastic modulus of the notes, 254.95 if that modulus is 2H/3,
   261.00 if it is 3H/2; the later turning points are symmetric in any case).
7. The `PASAPAS` entries `'PRECISION'` (`ch7`) and `'TEMPERATURES'` with a
   `CHAR 'T'` loading (`ch6`).
8. `SIN` applied to a `LISTREEL` (angles in degrees) (`ch6`, `ch7`).
9. The residual `ff - (bsig mo s) - (aa * uu)` of the hand-written equilibrium
   loop, taken from the Cast3M introduction (Chapter "Elastoplasticity")
   (`ch5/cylinder_radial_return`).
10. The type of the stress field rebuilt by `smul` and `devia` from components
    (`EXCO` + `ET`): if `-` with the trial stress or a later `VMIS` refuses it,
    give it back its type with `chan 'TYPE' ... 'CONTRAINTES'` (`ch5/*`).
11. `DROI n p q 'DINI' d1 'DFIN' d2` with a positive `n`: the count is imposed
    (needed by `DALL` on opposite sides); check with `trac dom` that the sizes
    are graded (`ch2/kirsch_elements`, as in the older `kirsch_cast3m.dgibi`).
12. The gibiane reading of numbers written by the drivers as `2.000000000000e+05`
    (`ch7`), and a `PRECISION` of 1e-10 in `PASAPAS`: if it subdivides or fails,
    use 1e-8 and a relative step of 1e-3 in `castem_run.jacobian_fd`.

A review of the files by a second reader (logic, procedure signatures, loop
names, 72-column lines, left-to-right arithmetic) found and fixed a variable
named `et`, which hides the operator `ET`, and lists of instants built with
`prog ... pas ...` that may miss their last value by rounding; they are now
built as a number times `prog 0. pas 1. n`.

Files marked untested are printed in the companion document with a red note;
they go into the book only once they have been run.
