# Review of `note.pdf` and Its Generating Code

This note reviews the current state of `Latex/note.pdf`, `Latex/note.tex`, and the code that generates its figures and tables, mainly `Programs/note.py`.

The main conclusion is:

- The core Canadian decomposition is implemented correctly and is numerically self-consistent.
- The main problems are reproducibility, the international comparison methodology, and several mismatches between the prose/captions and the generated outputs.

## Scope and verification

Files reviewed:

- `Latex/note.tex`
- `Latex/note.pdf`
- `Programs/note.py`
- `Programs/productivity.py`
- `Tables/note_decomposition.tex`
- `Tables/note_lp_levels.tex`
- `Tables/note_international.tex`
- `Tables/tfp_decomposition.tex`
- Generated figures used in the note

Checks performed:

- PDF text extraction and visual inspection of the rendered pages
- Comparison of LaTeX prose against generated tables and figures
- Reproduction attempt of `Programs/note.py`
- Numerical inspection of the domestic decomposition logic
- Benchmark comparison against `Programs/productivity.py`
- Inspection of the international-comparison construction and coverage

## Executive summary

High-priority issues:

1. `Programs/note.py` is not reproducible from a normal repo-root invocation because it derives key paths from `os.getcwd()`.
2. The international decomposition in Table 3 is not fully aligned with the stated objective: it mixes a 15-sector decomposition with `TOT_IND` aggregate labor-productivity growth, and it effectively uses 2002--2019 rather than 2000--2019 for the amplified decomposition.
3. Figure 2 is plotted incorrectly as `100 * (1 + cumulative log growth)` even though the label says `1961=100`; it should exponentiate cumulative log growth.

Medium-priority issues:

4. Section 5 and the caption of Figure 4/5 describe a level contribution over 2000--2019, but the code actually plots the change in contributions after 2000 relative to before 2000.
5. Several hard-coded numbers in the prose have drifted from the generated outputs.

## Findings

### 1. `note.py` is not reproducible from the repo root

Severity: High

Relevant code:

- `Programs/note.py:234`
- `Programs/note.py:245-247`

Problem:

- The script instantiates `StatsCan()` with its default `data_folder`.
- It also sets:

```python
BASE_DIR = Path(os.getcwd()).parent
FIG_DIR = BASE_DIR / 'Figures'
TAB_DIR = BASE_DIR / 'Tables'
```

- This only works if the script is launched from `Programs/`.
- If the user runs:

```bash
.venv/bin/python Programs/note.py
```

from the repository root, the script looks for `stats_can.h5` in the wrong place and resolves output/data directories one level above the repository.

Observed behavior:

- Running from the repo root caused `stats_can` to look for:

```text
/Users/jfbrou/Library/CloudStorage/Dropbox/GitHub/Productivity/stats_can.h5
```

even though the local cache is at:

```text
Programs/stats_can.h5
```

- Because the cache was not found, the script attempted a network download.

Why this matters:

- The script is not reliably reproducible unless the user already knows the intended working directory.
- In a restricted environment, this turns into a hard failure.

Proposed fix:

- Replace `os.getcwd()`-based path logic with file-relative paths:

```python
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
FIG_DIR = ROOT_DIR / 'Figures'
TAB_DIR = ROOT_DIR / 'Tables'
DATA_DIR = ROOT_DIR / 'Data'
sc = StatsCan(data_folder=SCRIPT_DIR)
```

- This makes the script runnable from either the repo root or `Programs/`.

### 2. The international decomposition does not match the stated objective cleanly

Severity: High

Relevant code:

- `Programs/note.py:1087-1133`
- `Latex/note.tex:293`
- `Tables/note_international.tex`

Problem A: effective period is 2002--2019, not 2000--2019

- The code constructs `active` observations with `year > EU_START`, so 2001 is the first candidate year.
- It then computes:

```python
alpha_by_year = ...
alpha_bar_by_year = alpha_by_year.rolling(2).mean()
yr = yr.join(alpha_bar_by_year.rename('alpha_bar'))
yr = yr.dropna(subset=['alpha_bar'])
```

- Because `alpha_bar` is formed after 2000 has already been excluded, the first non-missing `alpha_bar` is for 2002.
- As a result, the amplified within/Baumol series and the residual capital component in Table 3 are averaged over 2002--2019, not 2000--2019.

Problem B: sector coverage is mixed

- Within/Baumol are computed from a 15-sector subset:

```python
SECTIONS = ['A', 'B', 'C', 'D-E', 'F', 'G', 'H', 'I', 'J', 'K-L', 'M', 'N', 'Q', 'R', 'S']
```

- Aggregate labor-productivity growth is then taken from:

```python
agg_lp = ga[(ga['geo_code'] == country) &
            (ga['nace_r2_code'] == 'TOT_IND') &
            (ga['year'].isin(yr.index))]
lp_ann = agg_lp['LP1_G'].mean()
```

- `TOT_IND` is not the same object as the 15-sector subset. In the cached EU-KLEMS data, the 15 included sectors sum to only about 85--89% of `TOT_IND` value added for several countries inspected.
- That means the capital term in Table 3 is partly a residual for omitted sectors, not just the intended `K/Y` contribution within the stated 15-sector system.

Why this matters:

- The table and the prose describe a clean 15-sector decomposition for 2000--2019.
- The implemented calculation is a hybrid of:
  - a 15-sector within/Baumol calculation,
  - an 18-year amplified window beginning in 2002,
  - and a `TOT_IND` aggregate LP growth measure with broader coverage.

Proposed fix:

- Build `alpha_bar` using the full 2000--2019 sector panel so that 2001 is retained correctly.
- Compute labor-productivity growth from the same 15-sector system instead of borrowing `TOT_IND`.
- If needed, load the EU-KLEMS labor-input variables necessary to build sector-consistent `Y/L`.
- If keeping the current method, revise the note and table captions to say exactly what is being decomposed.

### 3. Figure 2 is plotted with the wrong transformation

Severity: High

Relevant code:

- `Programs/note.py:745-749`
- `Latex/note.tex:240`

Current code:

```python
ax.plot(decomp_full['year'], 100 * (decomp_full['total'].cumsum() + 1), ...)
ax.plot(decomp_full['year'], 100 * (decomp_full['within'].cumsum() + 1), ...)
```

Problem:

- `decomp_full['total']` and `decomp_full['within']` are annual log changes.
- If the y-axis is labeled as an index with `1961=100`, cumulative log growth must be exponentiated:

```python
100 * np.exp(cumsum)
```

- The current implementation uses a linear approximation.

Why this matters:

- The chart understates the level differences in the later years.
- The label and the plotted quantity do not match.

Proposed fix:

```python
ax.plot(..., 100 * np.exp(decomp_full['total'].cumsum()), ...)
ax.plot(..., 100 * np.exp(decomp_full['within'].cumsum()), ...)
```

### 4. Section 5 and the capital-industry figure are described inconsistently

Severity: Medium

Relevant code and text:

- `Programs/note.py:914-924`
- `Latex/note.tex:269-278`
- `Figures/note_capital_industry.png`

Problem:

- The code computes:

```python
k_change = 100 * (ind_k_post - ind_k_pre)
```

- So the figure is the change in industry capital contributions after 2000 relative to before 2000.
- But the section opener and caption say it is a decomposition of capital growth for `2000--2019`.
- The paragraph then switches back and describes the figure correctly as a variation.

Additional inconsistency:

- The paragraph says only oil/gas and transport increased their contribution.
- In the current computed results, 10 industries have positive changes.

Why this matters:

- The section mixes two different interpretations:
  - level contribution during 2000--2019,
  - change in contribution relative to the pre-2000 period.

Proposed fix:

- Make the section and caption match the code explicitly:
  - e.g. "Variation de la contribution sectorielle à la croissance du capital agrégé (après 2000 moins avant 2000)".
- Rewrite the paragraph from the current ranking rather than from an earlier draft.

### 5. Some prose numbers no longer match the generated outputs

Severity: Medium

Relevant text:

- `Latex/note.tex:250`
- `Latex/note.tex:305`

Examples:

- The text says the electronics industry fell by `-7` percentage points.
- The current code produces roughly `-5.63` pp for that slowdown.

- The policy section says:

```text
La PTF intra-sectorielle ... est passée de 1.1 % par année (1961--1980) à 0.1 % (2000--2019).
```

- The current raw within-TFP values are about:
  - `1.11%` for 1961--1980
  - `0.12%` for 2000--2019

- Elsewhere in the note, the labor-productivity contribution of within-industry TFP is reported as:
  - `1.79%`
  - `0.20%`

Why this matters:

- The document alternates between raw TFP growth and amplified labor-productivity contributions.
- Some rounded values appear to come from an older run.

Proposed fix:

- Decide explicitly whether each sentence refers to:
  - raw Hulten within-industry TFP, or
  - the amplified contribution to labor-productivity growth.
- Refresh all quantitative prose from the current generated outputs after regenerating the tables and figures.

## Additional methodological note

Relevant code:

- `Programs/note.py:1202-1211`
- `Tables/note_lp_levels.tex`

Observation:

- The level decomposition uses the US 2019 capital share for all countries:

```python
alpha_us = 1 - us_row['labsh']
p19['ky_comp'] = 100 * p19['ky_rel'] ** (alpha_us / (1 - alpha_us))
```

- This can be defensible as a Klenow-Rodriguez-Clare style normalization, but it should be stated explicitly in the note.

Recommendation:

- Clarify in the table note or text that the level decomposition uses a common US benchmark capital share rather than country-specific `\bar\alpha`.

## What appears to be working correctly

The core Canadian decomposition looks sound.

Evidence:

- The domestic script passed identity checks for:
  - labor productivity decomposition,
  - Hulten within/Baumol decomposition,
  - amplified three-term decomposition.
- The within/Baumol components match the benchmark values in `Tables/tfp_decomposition.tex` to rounding.

Current domestic summary from `Programs/note.py`:

- `d ln(Y/L)`: `2.01`, `1.32`, `0.52`
- Within-industry LP contribution: `1.79`, `1.26`, `0.20`
- Baumol LP contribution: `-0.13`, `-0.27`, `-0.33`
- Capital contribution: `0.35`, `0.33`, `0.65`

This supports the main substantive claim of the note:

- the post-2000 labor-productivity slowdown is primarily a within-industry productivity problem, not a collapse in capital deepening.

## Recommended repair order

1. Fix path handling and `StatsCan` cache resolution in `Programs/note.py`.
2. Fix Figure 2 to exponentiate cumulative log growth.
3. Repair the international decomposition so period coverage and sector coverage are internally consistent.
4. Regenerate all note tables and figures.
5. Refresh the prose and captions in `Latex/note.tex` from the regenerated outputs.

## Bottom line

The domestic accounting exercise is in good shape.

The main risks for `note.pdf` are not arithmetic mistakes in the Canadian decomposition, but:

- a fragile script entry point,
- an international table that does not cleanly implement the stated objective,
- and several narrative/caption statements that no longer match the computed outputs.
