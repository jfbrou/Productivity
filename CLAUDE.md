# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Academic research project analyzing Canada's post-2000 productivity growth slowdown. Decomposes aggregate TFP (total-factor productivity) growth into within-industry productivity, structural change (Baumol effects), and factor reallocation using an industry-level accounting framework (1961–2019). Includes U.S. comparison. Based on Hulten's theorem and the Baqaee-Farhi framework.

**Author:** Jean-Félix Brouillette (HEC Montréal)

## Repository Structure

- **`Programs/`** — Python analysis scripts. `productivity.py` is the main pipeline (~1600 lines); `io_can.py` processes input-output tables; `fake.py` generates simulated panel data.
- **`Latex/`** — LaTeX source files. `draft.tex` is the main research paper; `note.tex` is a French-language guide for a CPP report; `references.bib` is the shared bibliography.
- **`Data/`** — Raw data (Statistics Canada IO tables, HDF5 databases). Gitignored.
- **`Figures/`** — Generated PNG charts (referenced by LaTeX as `../Figures/`).
- **`Tables/`** — Generated LaTeX table fragments (included via `\input` in draft.tex).
- **`Literature/`** — Reference PDFs. Gitignored.

## Key Conventions

### Python
- Python 3.12 with a local `.venv`. Key packages: `pandas`, `numpy`, `matplotlib`, `scipy.sparse`, `stats_can`, `xlrd`.
- Environment variables loaded from `.env` for path configuration.
- NAICS code mappings maintained as dictionaries; industry names standardized with NAICS codes in brackets.
- Time series computed as log differences; data rescaled to base year (1961=100); rolling averages use 2-year windows.
- Matplotlib figures use Palatino serif font.

### LaTeX
- Custom HEC Montréal blue (`#002855`) color palette with a 7-color scheme for figures.
- Custom macros: `\partialof`, `\deltaof`, `\innerprod`, `\resizeeq`.
- Bibliography: `natbib` with AER style.
- `note.tex` is written in French (babel package).
- Compiled PDFs are tracked in git (unlike most LaTeX projects).

### Git
- Overleaf integration exists (historical merge from `overleaf-2025-12-06-1821`).
- `.gitignore` excludes `Data/`, `Graveyard/`, `Literature/`, notebooks, and HDF5 files. LaTeX `.tex`, `.bib`, and `.pdf` files are tracked.
