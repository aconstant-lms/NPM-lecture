# Working on the lecture notes "Nonlinear Problems in Mechanics"

Author: Andrei Constantinescu (LMS, CNRS & Ecole Polytechnique). LaTeX book, see README.md for the layout.

## Conventions (apply to every edit)
- Notation lives in `preamble/notation.tex`; use its macros, never ad-hoc bold:
  vectors and 2nd-order tensors bold italic (`\vu`, `\vn`, `\sig`, `\eps`, `\vect{}`, `\tens{}`),
  4th-order tensors blackboard bold (`\bbC`, `\bbS`, `\bbI`, `\tensf{}`), matrices bold upright (`\mat{K}`).
- Style is the RG style of Chapter 3 (`preamble/style.tex`): short declarative sentences, sans-serif headings,
  key statements in `gbox`, exercises with `\exercise{Title}\label{exo:...}` + `solution` environment.
- Every chapter (and appendix) follows the frame: `\framepart{Outline}` (objectives + plan), numbered
  sections, `\framepart{Summary}` (a `gbox*` of bold-led items), `\framepart{Exercises}` (sub-parts with
  `\framesub{...}`), and the `chapterbib` which prints the `References` frame part. Frame parts are
  unnumbered, in small caps: refer to them with `\pageref`/`\nameref`, never `\ref`.
- Index concepts with `\index{concept}` or `\index{concept!subconcept}` on the line after a section label,
  `\paragraph`, `gbox` title or exercise label (never inside captions or math). Reuse existing top-level
  entries (see the printed index) instead of creating near-duplicates; use `sortkey@Display` for accents/math.
- In running text, inline integrals and fractions are written with `\displaystyle` (`$\displaystyle \int...$`,
  `$\displaystyle \frac...$`); `\tfrac` stays small. Not in tables, captions or headings.
- Algorithms: in the chapter text, Simo & Hughes "Box" style: a `gbox` titled `Box <ch>.<n>  <title>`,
  numbered steps (`enumerate`, bold labels), each a short sentence followed by its formulas; cite as Box~... .
  In exercise parts, CMAME-style pseudocode (`algorithm` + `algpseudocode`, Input/Output, numbered lines).
- Tonti diagrams: kinematic boxes `kin` (blue), static boxes `stat` (green), labels `kinlab`/`statlab`
  under the boxes; the data (essential conditions, balance and natural conditions) go in the upper row.
- Exercises start with `\framesub{Short questions}` (exam-style, short solutions), then `\framesub{Problems}`
  or thematic sub-parts.
- Constraints are added to Lagrangians with the sign `-\lambda\,(\mat{P}\vu-\vuD)`, so that the multiplier is
  the reaction (positive); keep this convention in every chapter.
- New symbols go into the notation list `front/notation_list.tex` (after the table of contents) as well as
  `preamble/notation.tex`. Appendices in order: `app0_prerequisites` (A), `appA_function_spaces` (B),
  `appB_variational_derivatives` (C), `appC_integration_by_parts` (D), `appD_iterative_methods` (E); always cite them by label.
- Each chapter ends with its own `chapterbib`; bibitem keys must be unique across the whole book.
- Labels are prefixed per chapter: ch1 `sec:var-*`, ch2 `sec:el-*`/`sec:diff`, ch3 `*:rg-*`, ch4 `*:pl-*`, ch5 `*:pn-*`, ch6 `*:cy-*`, appendices `app:*`.
- Author comments in the sources look like `%% AC: ...`; process them when asked, then delete them.
- Red `\draftnote{...}` marks open questions for the author; leave them unless resolved.

## Workflow
- One conversation per chapter; touch other files only when needed (cross-references, notation).
- Build: `pdflatex -interaction=nonstopmode main.tex` three times. Before committing, the log must have
  no undefined references, no multiply-defined labels and no overfull boxes.
- Python lives in `python/` (see python/README.md): figure scripts in `python/figures/<chapter>/`
  (`cd python/figures/ch1 && python3 <script>.py`, outputs into figures/ch1), exercise code in
  `python/examples/<chapter>/`, printed in exercise solutions with `\lstinputlisting[firstline=4]{...}`
  after the results stated in words. Never paste code inline in the .tex: edit the file, run it, quote its
  output in comments. The book itself does not mention the repository.
- Commit with a clear message per round of corrections (e.g. "ch2: fix Voigt table, add exercise on ...")
  and push to `main`. The commit message becomes the release note, so make it readable for the author.
- Every push to `main` triggers `.github/workflows/build-pdf.yml`, which compiles the book and publishes
  `NPM_lecture_notes.pdf` on the release "latest": https://github.com/aconstant-lms/npm-lecture/releases/latest
  After pushing, check the run succeeded (`gh run list` / Actions tab) and give the author that link;
  attach the PDF in the conversation only if asked.
