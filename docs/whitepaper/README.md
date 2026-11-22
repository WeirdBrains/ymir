# Ymir white paper (LaTeX source)

Editable source of the Ymir white paper. Author: m3talsmith. Copyright 2026
[Weird Brains LLC](https://weirdbrains.com), BSD-3-Clause.

## Build

```sh
latexmk -pdf ymir-whitepaper.tex   # or: make
```

Without `latexmk`: `make manual` (pdflatex, bibtex, pdflatex twice).
Needs TikZ, pgfplots, booktabs, xltabular, tcolorbox and listings (all in TeX Live).
The PDF and build files are ignored by `.gitignore`; do not commit them.

## Layout

- `ymir-whitepaper.tex`: preamble, title page and the list of sections.
- `sections/`: one file per section, in reading order.
- `figures/`: TikZ diagrams and pgfplots charts, one file each; `styles.tex` holds shared styles and colors.
- `references.bib`: sources, printed in full in the Sources section (`\nocite{*}`).

## Conventions

- `\code{...}` sets inline code; underscores need no escaping.
- `\status{...}` and `\planned` keep the built, planned and proposal labels.
- `\WB` prints "Weird Brains LLC" linked to https://weirdbrains.com.
- Figures marked illustrative or proposal in their captions are not measurements or built features.
