# Building the thesis locally

The thesis source is intentionally kept local and is ignored by Git. The local
MiKTeX Portable environment is also stored under `.tmp/miktex/`, which is ignored.

## Build command

From the repository root, run:

```powershell
.\scripts\build_thesis.ps1
```

The script configures MiKTeX Portable and the Perl runtime bundled with Git, then
runs `latexmk` with `pdflatex` and `biber` as needed. The generated PDF is written to:

```text
.tmp/latex-build/main.pdf
```

To use a different output directory:

```powershell
.\scripts\build_thesis.ps1 -OutputDirectory ".tmp\another-build"
```

## Environment layout

- `.venv/`: Python environment for the PBO package and tests.
- `.tmp/miktex/portable/`: local MiKTeX Portable installation and package repository.
- `.tmp/latex-build/`: generated PDF and auxiliary LaTeX files.
- `thesis/`: local LaTeX source, deliberately not tracked in Git.

Do not commit generated PDFs, auxiliary files, package caches, or credentials.
