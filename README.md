# Nonlinear Problems in Mechanics: lecture notes (first draft)

**Latest PDF:** [releases/latest](https://github.com/aconstant-lms/npm-lecture/releases/latest) — rebuilt automatically on every push to `main`.

## Build
    pdflatex main && pdflatex main && pdflatex main
No BibTeX: every chapter carries its own `chapterbib` list (keys must be unique book-wide).
The index is produced by makeindex, run automatically by `imakeidx` (style `preamble/index.ist`).
`\drafttrue` / `\draftfalse` in `main.tex` shows/hides the red notes to the author.
Uncomment `\includeonly{...}` in `main.tex` to compile a single chapter.

## Layout
    main.tex                      book skeleton (frontmatter, 3 chapters, 3 appendices)
    preamble/style.tex            RG-style typography: sans-serif headings, gray boxes (gbox),
                                  \exercise{...} + solution environment, chapterbib, TikZ styles
    preamble/notation.tex         notation for the whole book (see below)
    front/titlepage.tex           title page (placeholder: swap in the Cast3M intro title page)
    front/overview.tex            Course overview (one page; slot left for "How to use this book")
    chapters/ch1_variational.tex  Ch. 1  Why variational formulations matter  (VF v4 + notes C1 + slides)
    chapters/ch2_elliptic.tex     Ch. 2  Elliptic problems: diffusion and elasticity (notes C1 §3 + slides)
    chapters/ch3_rg_crack.tex     Ch. 3  Identification of a planar crack by the reciprocity gap (rg_crack_theory)
    appendices/appA_...           App. A Function spaces, Poincare inequality, Cea's lemma
    appendices/appB_...           App. B Variational derivatives (Gateaux, Frechet, functional derivative, E-L)
    appendices/appC_...           App. C Integration by parts: the Gauss-Ostrogradsky family
    figures/ch1, ch2, appA        figures (ch2 = slide figures)
    python/figures/...            figure scripts (run from their own directory)
    python/examples/...           exercise code, printed in the book with \lstinputlisting

## Notation (preamble/notation.tex)
    vectors, 2nd-order tensors   bold italic    \vect{u} \tens{\sigma}  shortcuts \vu \vn \sig \eps \tk
    4th-order tensors            blackboard     \tensf{C}               shortcuts \bbC \bbS \bbI
    matrices (discrete)          bold upright   \mat{K} \mat{B}
    operators                    \dive \tr \dev \sym \argmin, \T (transpose), \dd (measure), \jump{}
