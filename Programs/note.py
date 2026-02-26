"""
Standalone script for the CPP report on Canada's labor productivity growth.

This script implements a two-step accounting decomposition of aggregate
labor productivity growth using industry-level data from Statistics Canada
(Table 36-10-0217-01), covering 39 NAICS industries over 1961-2019.

METHODOLOGY
-----------
Step 1: Labor Productivity Decomposition
    Starting from an aggregate CRS production function Y = A*F(K,L),
    one can derive the exact identity (see appendix in note.tex):

        d ln(Y/L) = [1/(1-a)] d ln A + [a/(1-a)] d ln(K/Y)

    where a = alpha_bar is the Tornqvist-averaged aggregate capital share.
    The first term is the TFP contribution (amplified by 1/(1-a) > 1), and
    the second is the capital-deepening contribution.

    Aggregate growth rates are constructed via Tornqvist indices:
    - d ln A = sum_i S_bar_i * d ln A_i    (Hulten's theorem)
    - d ln K = sum_i omega_bar^K_i * d ln K_i   (Divisia capital index)
    - d ln L = sum_i omega_bar^L_i * d ln L_i   (Divisia labor index)
    - d ln Y = d ln A + a * d ln K + (1-a) * d ln L  (production identity)

    The Tornqvist weights (S_bar, omega_bar) are 2-year rolling averages
    of current-period shares, following Diewert (1976).

Step 2: Aggregate TFP Decomposition
    Hulten aggregate TFP growth is decomposed via add-and-subtract:

        d ln A = sum_i S_{i,t0} d ln A_i  +  sum_i (S_bar_i - S_{i,t0}) d ln A_i
                 ========================     ====================================
                      Within-industry              Composition (Baumol)

    "Within" measures TFP growth at constant economic structure (base-period
    shares). "Baumol" captures the effect of structural change: if the economy
    shifts toward sectors with low TFP growth (because their relative prices
    rise), this term is negative and drags down aggregate TFP.

    Base-period shares S_{i,t0} are reset at each subperiod start:
    t0 = 1961 for 1961-1980, t0 = 1980 for 1980-2000, t0 = 2000 for 2000-2019.

DATA SOURCE
-----------
Statistics Canada Table 36-10-0217-01:
    Multifactor productivity, value-added, gross domestic product and
    related variables, by North American Industry Classification System (NAICS).

    Variables used:
    - Multifactor productivity based on value-added (index, TFP_i)
    - Labour input (index, L_i)
    - Capital input (index, K_i)
    - Gross domestic product, GDP (nominal dollars, VA_i for shares)
    - Labour compensation (nominal dollars, for labor cost shares)
    - Capital cost (nominal dollars, for capital cost shares)

OUTPUTS
-------
- Figures/note_labor_productivity.png   (Figure 1: LP decomposition)
- Figures/note_tfp_decomposition.png    (Figure 2: TFP with/without Baumol)
- Figures/note_baumol_scatter.png       (Figure 3: VA share vs TFP growth)
- Tables/note_decomposition.tex         (Summary table, LaTeX)

REFERENCES
----------
- Hulten (1978), "Growth Accounting with Intermediate Inputs", RES
- Diewert (1976), "Exact and Superlative Index Numbers", J. Econometrics
- Baumol (1967), "Macroeconomics of Unbalanced Growth", AER

Author: Jean-Felix Brouillette (HEC Montreal)
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
from stats_can import StatsCan

########################################################################
# Helper functions                                                      #
########################################################################

def setup_figure():
    """Create a figure with the project's standard formatting."""
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return fig, ax


def finalize_figure(fig, ax, filepath):
    """Add source attribution, save to disk, and close."""
    ax.text(1, 1.01, 'Source: Statistics Canada', fontsize=10,
            color='k', ha='right', va='bottom', transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(filepath, transparent=True, dpi=300)
    plt.close()


def annualize(series_or_df, col, start, end):
    """Compute annualized growth rate (%) over a subperiod.

    Sums the column for years (start, end] and divides by the number
    of years. The start year itself is excluded because growth rates
    are defined as t vs t-1, so the first year of growth in a subperiod
    starting at t0 is t0+1.
    """
    mask = (series_or_df['year'] > start) & (series_or_df['year'] <= end)
    return 100 * series_or_df.loc[mask, col].sum() / (end - start)


def annualize_decomp(decomp_df, col, start, end):
    """Annualize a column from a decomposition DataFrame.

    The decomposition DataFrames have a zero-row at the start year
    (since cumulative growth starts at zero). This row does not affect
    the sum, so we sum all rows and divide by the number of years.
    """
    return 100 * decomp_df[col].sum() / (end - start)


def compute_tfp_decomposition(df_ind, start, end, base_col):
    """Compute within/Baumol TFP decomposition for a subperiod.

    Parameters
    ----------
    df_ind : DataFrame
        Industry-level panel with columns 's_bar', 'tfp_growth', and base_col.
    start, end : int
        Start and end years of the subperiod.
    base_col : str
        Column name for the base-period VA shares (e.g., 's_1961').

    Returns
    -------
    DataFrame with columns: year, within, baumol, total.
        'within' = sum_i S_{i,t0} * d ln A_i  (base-period weights)
        'baumol' = sum_i (S_bar_i - S_{i,t0}) * d ln A_i  (weight changes)
        'total'  = within + baumol = Hulten aggregate TFP (= d ln A)
    """
    result = pd.DataFrame({'year': range(start, end + 1)})

    d = df_ind.copy()
    d['within_term'] = d[base_col] * d['tfp_growth']
    d['baumol_term'] = (d['s_bar'] - d[base_col]) * d['tfp_growth']

    agg_terms = d.groupby('year').agg(
        within=('within_term', 'sum'),
        baumol=('baumol_term', 'sum')
    ).reset_index()

    result = pd.merge(result, agg_terms, on='year', how='left')
    # Set start year to zero: this is the baseline for cumulative sums.
    # Growth in year t is measured from t-1 to t, so the first actual
    # growth observation for a subperiod starting at t0 is at t0+1.
    result.loc[result['year'] == start, ['within', 'baumol']] = 0.0
    result['total'] = result['within'] + result['baumol']

    return result


########################################################################
# Configuration                                                         #
########################################################################

sc = StatsCan()

rc('font', **{'family': 'serif', 'serif': ['Palatino']})
rc('text', usetex=True)

PALETTE = ['#002855', '#26d07c', '#ff585d', '#f3d03e',
           '#0072ce', '#eb6fbd', '#00aec7', '#888b8d']

BASE_DIR = Path(os.getcwd()).parent
FIG_DIR = BASE_DIR / 'Figures'
TAB_DIR = BASE_DIR / 'Tables'

########################################################################
# Constants: industry classification and variable selection             #
########################################################################

# Maps StatsCan industry names to short NAICS codes (39 industries).
# These are the leaf-level industries in Table 36-10-0217-01 after
# dropping aggregate sectors and sub-industries listed in DROP_LIST.
INDUSTRY_TO_NAICS = {
    'Accommodation and food services [72]': '72',
    'Administrative and support, waste management and remediation services [56]': '56',
    'Arts, entertainment and recreation [71]': '71',
    'Beverage and tobacco product manufacturing [312]': '312',
    'Chemical manufacturing [325]': '325',
    'Clothing, Leather and allied product manufacturing': '315-316',
    'Computer and electronic product manufacturing [334]': '334',
    'Construction [23]': '23',
    'Crop and animal production': '111-112',
    'Electrical equipment, appliance and component manufacturing [335]': '335',
    'Fabricated metal product manufacturing [332]': '332',
    'Finance, insurance, real estate and renting and leasing': '52-53',
    'Fishing, hunting and trapping [114]': '114',
    'Food manufacturing [311]': '311',
    'Forestry and logging [113]': '113',
    'Furniture and related product manufacturing [337]': '337',
    'Health care and social assistance (except hospitals)': '62',
    'Information and cultural industries [51]': '51',
    'Machinery manufacturing [333]': '333',
    'Mining (except oil and gas) [212]': '212',
    'Miscellaneous manufacturing [339]': '339',
    'Non-metallic mineral product manufacturing [327]': '327',
    'Oil and gas extraction [211]': '211',
    'Other services (except public administration) [81]': '81',
    'Paper manufacturing [322]': '322',
    'Petroleum and coal products manufacturing [324]': '324',
    'Plastics and rubber products manufacturing [326]': '326',
    'Primary metal manufacturing [331]': '331',
    'Printing and related support activities [323]': '323',
    'Professional, scientific and technical services [54]': '54',
    'Retail trade [44-45]': '44-45',
    'Support activities for agriculture and forestry [115]': '115',
    'Support activities for mining and oil and gas extraction [213]': '213',
    'Textile and textile product mills': '313-314',
    'Transportation and warehousing [48-49]': '48-49',
    'Transportation equipment manufacturing [336]': '336',
    'Utilities [221]': '221',
    'Wholesale trade [41]': '41',
    'Wood product manufacturing [321]': '321'
}

# Sectors to drop: aggregate industries and sub-industries that overlap
# with the 38 leaf-level industries above.
DROP_LIST = [
    'Agriculture, forestry, fishing and hunting [11]',
    'Mining and oil and gas extraction [21]',
    'Electric power generation, transmission and distribution [2211]',
    'Natural gas distribution, water and other systems',
    'Manufacturing [31-33]',
    'Air, rail, water and scenic and sightseeing transportation and support activities for transportation',
    'Truck transportation [484]',
    'Transit and ground passenger transportation [485]',
    'Pipeline transportation [486]',
    'Postal service and couriers and messengers',
    'Warehousing and storage [493]',
    'Motion picture and sound recording industries [512]',
    'Broadcasting, telecommunications, publishing industries and other information services',
    'Administrative and support services [561]',
    'Waste management and remediation services [562]',
    'Educational services (except universities)',
    'Repair and maintenance [811]',
    'Religious, grant-making, civic, and professional and similar organizations [813]',
    'Personal and laundry services and private households'
]

# Variables to keep from Table 36-10-0217-01.
RELEVANT_VARS = [
    'Multifactor productivity based on value-added',  # TFP index
    'Labour input',                                    # Labor input index
    'Capital input',                                   # Capital input index
    'Gross domestic product (GDP)',                     # Nominal VA ($)
    'Labour compensation',                             # Nominal labor cost ($)
    'Capital cost'                                     # Nominal capital cost ($)
]

########################################################################
# 1. Fetch, clean, and validate the data                                #
########################################################################

print('Fetching data from Statistics Canada Table 36-10-0217-01...')
df = sc.table_to_df('36-10-0217-01')

# Filter to the 38 leaf-level industries
df = df[~df['North American Industry Classification System (NAICS)'].isin(DROP_LIST)]

# Keep only the 6 relevant variables
df = df[df['Multifactor productivity and related variables'].isin(RELEVANT_VARS)]

# Reshape from long to wide: one row per (industry, year)
df = df.pivot_table(
    index=['North American Industry Classification System (NAICS)', 'REF_DATE'],
    columns='Multifactor productivity and related variables',
    values='VALUE'
).reset_index().rename_axis(None, axis=1)

df = df.rename(columns={
    'North American Industry Classification System (NAICS)': 'industry',
    'REF_DATE': 'date',
    'Multifactor productivity based on value-added': 'tfp',
    'Labour input': 'labor',
    'Capital input': 'capital',
    'Gross domestic product (GDP)': 'va',
    'Labour compensation': 'labor_cost',
    'Capital cost': 'capital_cost'
})

df['year'] = df['date'].dt.year
df = df.drop(columns=['date'])
df = df[df['year'] < 2020]
df['naics'] = df['industry'].map(INDUSTRY_TO_NAICS)
df = df.sort_values(['naics', 'year']).reset_index(drop=True)

# --- Data validation ---
n_industries = df['naics'].nunique()
n_years = df['year'].nunique()
assert n_industries == 39, f'Expected 39 industries, got {n_industries}'
assert df['year'].min() == 1961, f'Data starts at {df["year"].min()}, expected 1961'
assert df['year'].max() == 2019, f'Data ends at {df["year"].max()}, expected 2019'
assert len(df) == n_industries * n_years, (
    f'Unbalanced panel: {len(df)} rows for {n_industries}x{n_years}')
assert df[['tfp', 'capital', 'labor', 'va', 'labor_cost', 'capital_cost']].notna().all().all(), (
    'Missing values in core variables')
print(f'Panel: {n_industries} industries x {n_years} years = {len(df)} observations')

########################################################################
# 2. Industry-level growth rates and Tornqvist weights                  #
########################################################################

# --- Growth rates (log-differences within each industry) ---
# d ln X_{it} = ln(X_{it}) - ln(X_{i,t-1})
# These are NaN for 1961 (no previous year) and defined for 1962-2019.
for var, col in [('tfp', 'tfp_growth'), ('capital', 'capital_growth'), ('labor', 'labor_growth')]:
    df[col] = df.groupby('naics')[var].transform(lambda x: np.log(x).diff())

# --- Nominal VA shares ---
# s_{it} = VA_{it} / sum_j VA_{jt}  (current-period share)
# s_bar_{it} = (s_{it} + s_{i,t-1}) / 2  (Tornqvist average)
df['va_agg'] = df.groupby('year')['va'].transform('sum')
df['s'] = df['va'] / df['va_agg']
df['s_bar'] = df.groupby('naics')['s'].transform(lambda x: x.rolling(2).mean())

# --- Capital cost shares (for Divisia capital aggregation) ---
# omega^K_{it} = capital_cost_{it} / sum_j capital_cost_{jt}
df['capital_cost_agg'] = df.groupby('year')['capital_cost'].transform('sum')
df['omega_k'] = df['capital_cost'] / df['capital_cost_agg']
df['omega_k_bar'] = df.groupby('naics')['omega_k'].transform(lambda x: x.rolling(2).mean())

# --- Labor cost shares (for Divisia labor aggregation) ---
# omega^L_{it} = labor_cost_{it} / sum_j labor_cost_{jt}
df['labor_cost_agg'] = df.groupby('year')['labor_cost'].transform('sum')
df['omega_l'] = df['labor_cost'] / df['labor_cost_agg']
df['omega_l_bar'] = df.groupby('naics')['omega_l'].transform(lambda x: x.rolling(2).mean())

# --- Base-period VA shares for subperiod decompositions ---
# For 1961: use the 1962 Tornqvist average (first non-NaN value).
# For 1980, 2000: use the Tornqvist average at those years.
for base_year, label in [(1962, 's_1961'), (1980, 's_1980'), (2000, 's_2000')]:
    base_shares = df.loc[df['year'] == base_year, ['naics', 's_bar']].rename(columns={'s_bar': label})
    df = pd.merge(df, base_shares, on='naics', how='left')

# --- Validate Tornqvist weights ---
# Shares should sum to ~1 across industries for each year (1962+).
weight_sums = df[df['year'] >= 1962].groupby('year').agg(
    s_bar_sum=('s_bar', 'sum'),
    omega_k_sum=('omega_k_bar', 'sum'),
    omega_l_sum=('omega_l_bar', 'sum')
)
for col, label in [('s_bar_sum', 'VA shares'), ('omega_k_sum', 'Capital weights'),
                   ('omega_l_sum', 'Labor weights')]:
    deviation = (weight_sums[col] - 1).abs().max()
    print(f'{label} sum to 1 check — max deviation: {deviation:.6f}')
    assert deviation < 0.01, f'{label} do not sum to ~1'

# --- Cross-check: cumulative growth = log ratio of index levels ---
# For each industry, sum of d ln A_i from 1962-2019 should equal
# ln(TFP_{i,2019} / TFP_{i,1961}).
industry_check = df.groupby('naics').apply(
    lambda g: pd.Series({
        'cum_growth': g['tfp_growth'].sum(),
        'log_ratio': np.log(g['tfp'].iloc[-1] / g['tfp'].iloc[0])
    }),
    include_groups=False
)
max_disc = (industry_check['cum_growth'] - industry_check['log_ratio']).abs().max()
print(f'Industry TFP growth consistency — max discrepancy: {max_disc:.2e}')
assert max_disc < 1e-10, 'Industry-level growth rates inconsistent with index levels'

########################################################################
# 3. Step 1: Aggregate labor productivity decomposition                 #
########################################################################

# --- Aggregate capital share ---
# alpha_t = sum_i capital_cost_{it} / sum_i VA_{it}
# alpha_bar_t = (alpha_t + alpha_{t-1}) / 2  (Tornqvist average)
agg = df.groupby('year').agg(
    capital_cost_total=('capital_cost', 'sum'),
    va_total=('va', 'sum')
).reset_index()
agg['alpha'] = agg['capital_cost_total'] / agg['va_total']
agg['alpha_bar'] = agg['alpha'].rolling(2).mean()

# Sanity check: capital share in a reasonable range
print(f'Aggregate capital share range: [{agg["alpha"].min():.3f}, {agg["alpha"].max():.3f}]')
assert 0.25 < agg['alpha'].min() and agg['alpha'].max() < 0.55, (
    'Capital share outside [0.25, 0.55]')

# --- Aggregate growth rates via Tornqvist indices ---
# Compute the weighted contributions at the industry level, then sum.
df['dlnA_term'] = df['s_bar'] * df['tfp_growth']        # S_bar_i * d ln A_i
df['dlnK_term'] = df['omega_k_bar'] * df['capital_growth']  # omega^K_i * d ln K_i
df['dlnL_term'] = df['omega_l_bar'] * df['labor_growth']    # omega^L_i * d ln L_i

# Aggregate by year (dropping 1961 where all terms are NaN)
yearly = df.dropna(subset=['dlnA_term']).groupby('year').agg(
    dlnA=('dlnA_term', 'sum'),  # Hulten aggregate TFP growth
    dlnK=('dlnK_term', 'sum'),  # Divisia aggregate capital growth
    dlnL=('dlnL_term', 'sum')   # Divisia aggregate labor growth
).reset_index()

yearly = pd.merge(yearly, agg[['year', 'alpha_bar']], on='year')

# --- Production identity ---
# d ln Y = d ln A + alpha_bar * d ln K + (1 - alpha_bar) * d ln L
# This constructs aggregate output growth from the CRS production function.
yearly['dlnY'] = (yearly['dlnA']
                  + yearly['alpha_bar'] * yearly['dlnK']
                  + (1 - yearly['alpha_bar']) * yearly['dlnL'])

# --- Labor productivity and capital-output ratio growth ---
yearly['dlnYL'] = yearly['dlnY'] - yearly['dlnL']   # d ln(Y/L)
yearly['dlnKY'] = yearly['dlnK'] - yearly['dlnY']   # d ln(K/Y)

# --- Decomposition ---
# d ln(Y/L) = [1/(1-alpha_bar)] * d ln A + [alpha_bar/(1-alpha_bar)] * d ln(K/Y)
yearly['tfp_contrib'] = yearly['dlnA'] / (1 - yearly['alpha_bar'])
yearly['ky_contrib'] = yearly['alpha_bar'] / (1 - yearly['alpha_bar']) * yearly['dlnKY']

# --- Identity check (exact by construction) ---
residual_1 = (yearly['tfp_contrib'] + yearly['ky_contrib'] - yearly['dlnYL']).abs().max()
print(f'\nStep 1 identity — max residual: {residual_1:.2e}')
assert residual_1 < 1e-10, f'Step 1 identity fails: residual = {residual_1}'

# Prepend a 1961 baseline row (all zeros) for cumulative plots.
# Growth rates are undefined for 1961; the cumsum starts at 0.
row_1961 = pd.DataFrame({
    'year': [1961], 'dlnA': [0.0], 'dlnK': [0.0], 'dlnL': [0.0],
    'alpha_bar': [np.nan], 'dlnY': [0.0], 'dlnYL': [0.0], 'dlnKY': [0.0],
    'tfp_contrib': [0.0], 'ky_contrib': [0.0]
})
yearly = pd.concat([row_1961, yearly], ignore_index=True)

########################################################################
# 4. Step 2: Aggregate TFP decomposition (within vs Baumol)            #
########################################################################

decomp_full = compute_tfp_decomposition(df, 1961, 2019, 's_1961')
decomp_61_80 = compute_tfp_decomposition(df, 1961, 1980, 's_1961')
decomp_80_00 = compute_tfp_decomposition(df, 1980, 2000, 's_1980')
decomp_00_19 = compute_tfp_decomposition(df, 2000, 2019, 's_2000')

# --- Identity check: within + baumol = Hulten aggregate TFP ---
# For years after the start year, decomp['total'] should equal yearly['dlnA'].
for name, decomp, start in [('1961-2019', decomp_full, 1961),
                             ('1961-1980', decomp_61_80, 1961),
                             ('1980-2000', decomp_80_00, 1980),
                             ('2000-2019', decomp_00_19, 2000)]:
    merged = pd.merge(decomp[decomp['year'] > start], yearly[['year', 'dlnA']], on='year')
    residual = (merged['total'] - merged['dlnA']).abs().max()
    print(f'Step 2 identity ({name}) — max residual: {residual:.2e}')
    assert residual < 1e-10, f'Step 2 identity fails for {name}'

########################################################################
# 5. Annualized growth rates and summary table                         #
########################################################################

PERIODS = [(1961, 2019), (1961, 1980), (1980, 2000), (2000, 2019)]
DECOMPS = [decomp_full, decomp_61_80, decomp_80_00, decomp_00_19]

# Panel A: Labor productivity decomposition
lp      = [annualize(yearly, 'dlnYL', s, e) for s, e in PERIODS]
tfp_c   = [annualize(yearly, 'tfp_contrib', s, e) for s, e in PERIODS]
ky_c    = [annualize(yearly, 'ky_contrib', s, e) for s, e in PERIODS]

# Panel B: TFP decomposition
tfp_agg = [annualize_decomp(d, 'total', s, e) for d, (s, e) in zip(DECOMPS, PERIODS)]
within  = [annualize_decomp(d, 'within', s, e) for d, (s, e) in zip(DECOMPS, PERIODS)]
baumol  = [annualize_decomp(d, 'baumol', s, e) for d, (s, e) in zip(DECOMPS, PERIODS)]

# Print summary
print('\n--- Annualized growth rates (%) ---')
header = f'{"":30s} {"1961-2019":>10s} {"1961-1980":>10s} {"1980-2000":>10s} {"2000-2019":>10s}'
print(header)
print('-' * len(header))
print(f'{"d ln(Y/L)":30s} {lp[0]:10.2f} {lp[1]:10.2f} {lp[2]:10.2f} {lp[3]:10.2f}')
print(f'{"  TFP contribution":30s} {tfp_c[0]:10.2f} {tfp_c[1]:10.2f} {tfp_c[2]:10.2f} {tfp_c[3]:10.2f}')
print(f'{"  K/Y contribution":30s} {ky_c[0]:10.2f} {ky_c[1]:10.2f} {ky_c[2]:10.2f} {ky_c[3]:10.2f}')
print(f'{"d ln A (Hulten)":30s} {tfp_agg[0]:10.2f} {tfp_agg[1]:10.2f} {tfp_agg[2]:10.2f} {tfp_agg[3]:10.2f}')
print(f'{"  Within-industry":30s} {within[0]:10.2f} {within[1]:10.2f} {within[2]:10.2f} {within[3]:10.2f}')
print(f'{"  Composition (Baumol)":30s} {baumol[0]:10.2f} {baumol[1]:10.2f} {baumol[2]:10.2f} {baumol[3]:10.2f}')

########################################################################
# 6. Benchmarking against productivity.py (Baqaee-Farhi decomposition) #
########################################################################

# The existing productivity.py decomposes TFP using the Baqaee-Farhi
# framework, which adds capital and labor reallocation terms:
#   Total(BF) = Within + Baumol + Capital_realloc + Labor_realloc
#
# Our Hulten TFP = Within + Baumol (no reallocation terms).
# The Within and Baumol components should match exactly between the two
# codes, since they use the same data, weights, and base-period shares.
#
# Reference values from Tables/tfp_decomposition.tex (generated by
# productivity.py):
BENCHMARK = {
    'within': [0.78, 1.11, 0.76, 0.12],
    'baumol': [-0.27, -0.08, -0.16, -0.20],
    'total_bf': [0.50, 0.99, 0.60, -0.09],  # includes reallocation
}

print('\n--- Benchmark against productivity.py ---')
print(f'{"":25s} {"note.py":>8s} {"prod.py":>8s} {"diff":>8s}')
all_pass = True
for i, (s, e) in enumerate(PERIODS):
    label = f'{s}-{e}'
    # Within-industry TFP
    diff_w = within[i] - BENCHMARK['within'][i]
    match_w = abs(diff_w) < 0.015  # tolerance for rounding to 2 decimals
    print(f'  Within {label:12s}  {within[i]:8.2f} {BENCHMARK["within"][i]:8.2f} {diff_w:+8.4f} {"OK" if match_w else "MISMATCH"}')
    all_pass = all_pass and match_w
    # Baumol
    diff_b = baumol[i] - BENCHMARK['baumol'][i]
    match_b = abs(diff_b) < 0.015
    print(f'  Baumol {label:12s}  {baumol[i]:8.2f} {BENCHMARK["baumol"][i]:8.2f} {diff_b:+8.4f} {"OK" if match_b else "MISMATCH"}')
    all_pass = all_pass and match_b

# Hulten TFP vs Baqaee-Farhi TFP (should differ by reallocation terms)
print('\n  Hulten TFP vs Baqaee-Farhi TFP (difference = reallocation terms):')
for i, (s, e) in enumerate(PERIODS):
    diff_r = tfp_agg[i] - BENCHMARK['total_bf'][i]
    print(f'    {s}-{e}: Hulten={tfp_agg[i]:.2f}%, BF={BENCHMARK["total_bf"][i]:.2f}%, '
          f'reallocation gap={diff_r:+.2f}%')

assert all_pass, 'Benchmark comparison failed — Within/Baumol values do not match productivity.py'
print('\nAll benchmark checks passed.')

# --- Additional cross-validations ---
# Conesa & Pujolas (2019) report TFP growth of 0.16% per year for 2002-2014.
# Our Hulten TFP should be close (they use a different methodology).
mask_cp = (yearly['year'] > 2002) & (yearly['year'] <= 2014)
dlnA_cp = 100 * yearly.loc[mask_cp, 'dlnA'].sum() / (2014 - 2002)
print(f'\nAggregate TFP growth 2002-2014: {dlnA_cp:.2f}%')
print(f'  Conesa & Pujolas (2019) report: 0.16%')
print(f'  Difference likely due to methodology (they use aggregate data, we use Hulten)')

# Panel A identity: TFP contribution + K/Y contribution = d ln(Y/L)
for i, (s, e) in enumerate(PERIODS):
    residual = abs(tfp_c[i] + ky_c[i] - lp[i])
    assert residual < 1e-10, f'Panel A identity fails for {s}-{e}'
print('Panel A identity check (TFP + K/Y = LP): passed for all periods')

# Panels A-B consistency: annualized d ln A should match between panels.
# Panel A derives d ln A from yearly['dlnA']; Panel B from decomp['total'].
for i, (s, e) in enumerate(PERIODS):
    dlnA_from_A = annualize(yearly, 'dlnA', s, e)
    residual = abs(dlnA_from_A - tfp_agg[i])
    assert residual < 1e-10, f'Panel A-B consistency fails for {s}-{e}'
print('Panel A-B consistency check (d ln A same in both): passed for all periods')

########################################################################
# 7. Figure 1: Labor productivity decomposition                        #
########################################################################

fig, ax = setup_figure()

# Labor productivity index: 100 * (1 + cumulative log growth).
# This is an approximation to 100 * exp(cumsum), valid for moderate growth.
ax.plot(yearly['year'], 100 * (yearly['dlnYL'].cumsum() + 1),
        label='Labor productivity', color=PALETTE[0], linewidth=2)
# Counterfactual: what if only TFP contributed (K/Y constant)?
ax.plot(yearly['year'], 100 * (yearly['tfp_contrib'].cumsum() + 1),
        label='Without capital deepening', color=PALETTE[1], linewidth=2)

ax.set_xlim(1961, 2019)
ax.set_xticks(range(1965, 2015 + 1, 5))
ax.set_xticklabels(range(1965, 2015 + 1, 5), fontsize=14)
ax.set_ylabel('Labor productivity (1961=100)', fontsize=14, rotation=0, ha='left')
ax.yaxis.set_label_coords(0, 1.01)
ax.grid(True, which='major', axis='y', color='gray', linestyle=':', linewidth=0.5)
ax.legend(frameon=False, fontsize=14)

finalize_figure(fig, ax, FIG_DIR / 'note_labor_productivity.png')

########################################################################
# 8. Figure 2: TFP decomposition (Baumol effect)                       #
########################################################################

fig, ax = setup_figure()

ax.plot(decomp_full['year'], 100 * (decomp_full['total'].cumsum() + 1),
        label='Total', color=PALETTE[0], linewidth=2)
# Counterfactual: TFP if economic structure stayed at 1961 shares.
ax.plot(decomp_full['year'], 100 * (decomp_full['within'].cumsum() + 1),
        label='Without Baumol', color=PALETTE[1], linewidth=2)

ax.set_xlim(1961, 2019)
ax.set_xticks(range(1965, 2015 + 1, 5))
ax.set_xticklabels(range(1965, 2015 + 1, 5), fontsize=14)
ax.set_ylim(100, 150)
ax.set_yticks(range(100, 150 + 1, 5))
ax.set_yticklabels(range(100, 150 + 1, 5), fontsize=14)
ax.set_ylabel('Aggregate TFP (1961=100)', fontsize=14, rotation=0, ha='left')
ax.yaxis.set_label_coords(0, 1.01)
ax.grid(True, which='major', axis='y', color='gray', linestyle=':', linewidth=0.5)
ax.legend(frameon=False, fontsize=14)

finalize_figure(fig, ax, FIG_DIR / 'note_tfp_decomposition.png')

########################################################################
# 9. Figure 3: Baumol scatter (VA share change vs TFP growth)          #
########################################################################

# For each industry, compute:
# x = cumulative TFP growth from 1961 to 2019 (sum of d ln A_i, in %)
# y = change in nominal VA share (s_{2019} - s_{1961}, in percentage points)
# The expected negative correlation is the Baumol effect: high-TFP-growth
# industries see falling relative prices and declining nominal VA shares.
s_1961 = df[df['year'] == 1961].set_index('naics')['s']
s_2019 = df[df['year'] == 2019].set_index('naics')['s']
cum_tfp_by_industry = df[df['year'] > 1961].groupby('naics')['tfp_growth'].sum()

scatter_data = pd.DataFrame({
    'cum_tfp': 100 * cum_tfp_by_industry,        # approximate %
    'delta_s': 100 * (s_2019 - s_1961)            # percentage points
}).dropna()

fig, ax = setup_figure()

ax.scatter(scatter_data['cum_tfp'], scatter_data['delta_s'],
           color=PALETTE[0], s=40, zorder=3, edgecolors='white', linewidth=0.5)

# OLS regression line to visualize the Baumol correlation
z = np.polyfit(scatter_data['cum_tfp'], scatter_data['delta_s'], 1)
x_fit = np.linspace(scatter_data['cum_tfp'].min() - 5,
                    scatter_data['cum_tfp'].max() + 5, 100)
ax.plot(x_fit, z[0] * x_fit + z[1], '--', color=PALETTE[2], linewidth=1.5, alpha=0.8)

ax.axhline(0, color='gray', linewidth=0.5, linestyle='-')
ax.axvline(0, color='gray', linewidth=0.5, linestyle='-')

ax.set_xlabel(r'Cumulative TFP growth, 1961--2019 (\%)', fontsize=14)
ax.set_ylabel(r'$\Delta$ VA share (p.p.)', fontsize=14, rotation=0, ha='left')
ax.yaxis.set_label_coords(0, 1.01)
ax.tick_params(axis='both', labelsize=14)

finalize_figure(fig, ax, FIG_DIR / 'note_baumol_scatter.png')

########################################################################
# 10. LaTeX table: Summary decomposition                                #
########################################################################

def fmt(v):
    """Format a growth rate for the LaTeX table (e.g., '1.28\\%')."""
    return f'{v:.2f}\\%'

with open(TAB_DIR / 'note_decomposition.tex', 'w') as f:
    lines = [
        r'\begin{table}[H]',
        r'\centering',
        r'\begin{threeparttable}',
        r"\caption{D\'ecomposition de la croissance de la productivit\'e du travail}",
        r'\begin{tabular}{lcccc}',
        r'\hline',
        r'\hline',
        r'& 1961--2019 & 1961--1980 & 1980--2000 & 2000--2019 \\',
        r'\hline',
        r"\multicolumn{5}{l}{\textit{Panel A\,: Productivit\'e du travail}} \\[0.3em]",
        r'$\Delta \ln(Y/L)$ & ' + ' & '.join(fmt(v) for v in lp) + r' \\',
        r'\quad Contribution PTF & ' + ' & '.join(fmt(v) for v in tfp_c) + r' \\',
        r'\quad Contribution $K/Y$ & ' + ' & '.join(fmt(v) for v in ky_c) + r' \\[0.3em]',
        r'\hline',
        r"\multicolumn{5}{l}{\textit{Panel B\,: PTF agr\'eg\'ee}} \\[0.3em]",
        r'$\Delta \ln A$ & ' + ' & '.join(fmt(v) for v in tfp_agg) + r' \\',
        r'\quad Intra-industries & ' + ' & '.join(fmt(v) for v in within) + r' \\',
        r'\quad Composition (Baumol) & ' + ' & '.join(fmt(v) for v in baumol) + r' \\',
        r'\hline',
        r'\hline',
        r'\end{tabular}',
        r'\begin{tablenotes}[flushleft]',
        r'\footnotesize',
        r"\item \textit{Note}\,: Ce tableau pr\'esente la d\'ecomposition de la croissance "
        r"annuelle moyenne de la productivit\'e du travail et de la PTF agr\'eg\'ee pour "
        r"diff\'erentes sous-p\'eriodes. Source\,: Statistique Canada, tableau 36-10-0217-01.",
        r'\end{tablenotes}',
        r'\label{tab:note_decomposition}',
        r'\end{threeparttable}',
        r'\end{table}'
    ]
    f.write('\n'.join(lines))

print('\nDone. Figures saved to Figures/, table saved to Tables/.')
