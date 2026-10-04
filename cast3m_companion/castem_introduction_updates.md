# Ideas for the Cast3M introduction (aconstant-lms/Castem_Introduction)

Notes from writing the Cast3M companion of the lecture notes. Nothing here has
been applied to the introduction; each item says where it would go.

## Elastoplasticity (`ch_plasticity.tex`)

1. **A second hand-written algorithm, with the return map in gibiane.** The
   current section programs the equilibrium loop and lets `ecou` integrate the
   law. Add a section where the radial return itself is written on the
   Gauss-point fields: `sigm` for the predictor, `vmis` for the norm, `masq`
   for the plastic points, a scalar Newton iteration done at all points at
   once, the update along the deviator built with `exco`/`et`. This is the
   procedure `rrj2` of `cast3m/ch5/return_map_point.dgibi` (Box C5.1 of the
   companion). It makes the "programming" section really program the
   constitutive law, and it is the bridge to MFront-style thinking.
2. **Name the iteration matrix.** The constant-stiffness loop is the
   initial-strain iteration (Box 6.1 of the notes), a modified Newton method:
   say so, and add the exercise "count the iterations against `pasapas` with
   its tangent option" (the existing exercise "Constant versus tangent
   stiffness" can carry the numbers of the cylinder, companion C5.2).
3. **Start every iteration from the converged state.** The current loop
   updates `sig . n` and `alpha . n` in place at each equilibrium iteration,
   and calls `ecou` with the *iteration* increment `deps`. For an
   associated, path-independent step this is what `ecou` expects, but the
   notes insist (Box 5.3) that the return map always starts from z_n with the
   *total* increment since t_n; worth a sentence, or a switch to that form.
4. **Calibrate the conventions with closed forms.** A short table "what
   Cast3M calls H, TRAC, EPSE" with the one-element tests of the companion
   (filament cycle C4.2: the reverse yield stress tells whether `H` is the
   modulus of q = H eps^p).

## Operations on fields (`ch_elasticity.tex` or a new short section)

5. An `optable` of the arithmetic on `MCHAML`: `+ - * /` between fields with
   the same components, products with a number, `exco` to rename a component
   (`'SMXX'` <-> `'SCAL'`), `et` to join components, `masq` for
   characteristic functions, `maxi`/`abs` for norms, `exp`, `**`. The
   companion uses all of them; they are the vocabulary of any algorithm written
   on fields. The list "Constructions to check first" (`cast3m/README.md`)
   is exactly what such a table should settle against the notices.

## Pre- and post-processing (`ch_python.tex`)

6. **The runner as a function.** The parametric pattern (template, run,
   read) becomes reusable as `castem_run.displacements(p)`: one directory per
   run, check for `ERREUR`, read the CSV of `EXPORTCSV`, a cache on the
   parameters, central differences run in parallel (`ThreadPoolExecutor`).
   See `cast3m/ch7/castem_run.py`.
7. **Drivers.** Add short sections, each 20 lines, with the truss template:
   - SciPy `least_squares` with a finite-difference Jacobian;
   - OpenTURNS: `PythonFunction` + `ParametricFunction` +
     `NonLinearLeastSquaresCalibration` (and `coupling_tools` as the
     OpenTURNS-native way to fill templates);
   - JAX: `jax.pure_callback` + `jax.custom_vjp`, the derivative of a black box
     declared by hand, checked by a Taylor test.
8. **A stand-in for testing.** `tools/castem_stub.py` reads the same
   `calcul.dgibi` and writes the same CSV from a Python model: the drivers can
   be developed and tested where Cast3M is not installed (laptops, CI). Worth
   recommending in general.
9. **A pitfall worth an exercise.** In a first version of the truss template
   the loads were proportional to the parameter `sY`: the computed
   displacements are then invariant when (E, sY, H) are scaled together, and
   the identification slides along the ray. Keep loads and data fixed.

## Input and output (`ch_importexport.tex`)

10. **Procedures shared between files.** `EXPORTCSV` is copied into the
    template, and the companion repeats `smul`, `devia`, `hard`, `rrj2` in
    two files. A `procedur/` directory next to the files (section "util")
    would avoid the copies: say how a `.procedur` file must look (header line,
    name) so that `CASTEM_PROCEDUR` finds it.
11. `exte[rieur]` is described in one paragraph: an example calling a Python
    post-processing at the end of a run would complete it.

## Shared files

12. The companion copies the `castemcom`/`castemlines` boxes from the
    "Cast3M-specific" part of `preamble/style.tex`. If both documents keep
    them, move them to a small shared file (e.g. `preamble/castem_boxes.tex`)
    copied verbatim like the shared part of `style.tex`.
13. The gibiane `listings` language of the companion
    (`companion_preamble.tex`: comments in column 1, case-insensitive
    keywords) could serve the introduction for its full-file listings.

## Small things noticed

14. `sheet_plane_strain.dgibi` (MEC563) is in SI units while its header says
    mm/MPa; it stretches along y while the exercise of the notes stretches
    along x; its comment speaks of kinematic hardening for a perfectly plastic
    model. Harmless, but confusing when the file is printed.
15. Gibiane lines are read up to 72 columns: a remark in "Editing gibiane
    files" would save a debugging session (long `mess` lines).
