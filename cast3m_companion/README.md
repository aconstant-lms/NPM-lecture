# Cast3M companion to the lecture notes (draft)

A separate document, in the style of the notes, with Cast3M exercises for
Chapters 1, 2 and 4-7 (Chapter 3, reciprocity gap, is treated separately).
It cites the sections, boxes and exercises of the book by their numbers and
prints the files of `../cast3m/`. Exercises are numbered C<chapter>.<n>.

    cast3m_companion.tex      main file (book class, ../preamble/style + notation)
    companion_preamble.tex    gibiane listings, the castemcom/castemlines boxes of
                              the Cast3M introduction, exercise numbers C<ch>.<n>
    cc_intro.tex              using this companion: three uses of Cast3M, list of
                              the exercises, constructions to check first
    cc_ch1.tex ... cc_ch7.tex one chapter per chapter of the notes: a table
                              "Chapter n of the notes and Cast3M" (done and to do),
                              the gibiane constructions, summary, exercises
    castem_introduction_updates.md   ideas for the Cast3M introduction
                              (aconstant-lms/Castem_Introduction)

## Build

The book must be compiled first: its `.aux` files give the numbers cited here
(package `xr-hyper`, prefix `B-`, e.g. `\ref{B-exo:pn-fem}`).

    cd ..  && pdflatex main && pdflatex main
    cd cast3m_companion && latexmk -pdf cast3m_companion

`.latexmkrc` puts the repository root on `TEXINPUTS`, so the paths are written
as in the book (`preamble/style`, `cast3m/ch5/...`). The CI builds the book,
then this document, and attaches `NPM_cast3m_companion.pdf` to the release
"latest" next to the notes.

## Merging into the book later

Each chapter `cc_chN.tex` has the frame of a chapter of the notes. To merge,
move its table and sections into the chapter N of the notes (as a section
"With Cast3M"), its exercises into the exercises of chapter N, drop the prefix
`B-` of the references, and replace `\untested{...}` once the file has run.
