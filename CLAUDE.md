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
- Each chapter ends with its own `chapterbib`; bibitem keys must be unique across the whole book.
- Labels are prefixed per chapter: ch1 `sec:var-*`, ch2 `sec:el-*`/`sec:diff`, ch3 `*:rg-*`, appendices `app:*`.
- Author comments in the sources look like `%% AC: ...`; process them when asked, then delete them.
- Red `\draftnote{...}` marks open questions for the author; leave them unless resolved.

## Workflow
- One conversation per chapter; touch other files only when needed (cross-references, notation).
- Build: `pdflatex -interaction=nonstopmode main.tex` three times. Before committing, the log must have
  no undefined references, no multiply-defined labels and no overfull boxes.
- Python lives in `python/` (see python/README.md): figure scripts in `python/figures/<chapter>/`
  (`cd python/figures/ch1 && python3 <script>.py`, outputs into figures/ch1), exercise code in
  `python/examples/<chapter>/`, included in the book with `\lstinputlisting[firstline=4]{...}`.
  Never paste code inline in the .tex: create or edit the file, run it, quote its output in comments.
- Commit with a clear message per round of corrections (e.g. "ch2: fix Voigt table, add exercise on ...")
  and push to `main`. The commit message becomes the release note, so make it readable for the author.
- Every push to `main` triggers `.github/workflows/build-pdf.yml`, which compiles the book and publishes
  `NPM_lecture_notes.pdf` on the release "latest": https://github.com/aconstant-lms/npm-lecture/releases/latest
  After pushing, check the run succeeded (`gh run list` / Actions tab) and give the author that link;
  attach the PDF in the conversation only if asked.
