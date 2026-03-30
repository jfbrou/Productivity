"""
Standalone script for the CPP report on Canada's labor productivity growth.

This script implements a three-term decomposition of aggregate labor
productivity growth using industry-level data from Statistics Canada
(Table 36-10-0217-01), covering 39 NAICS industries over 1961-2019.

For the international comparison, the script uses Penn World Table 11.0
for OECD countries.

METHODOLOGY
-----------
The decomposition combines a labor productivity identity with Hulten's
theorem to express d ln(Y/L) as three additive terms:

    d ln(Y/L) = [1/(1-a)] * Within  +  [1/(1-a)] * Baumol  +  [a/(1-a)] * d ln(K/Y)
                 ──────────────────     ──────────────────      ─────────────────────
                   Within-industry       Reallocation              Capital deepening
                        TFP               (Baumol)

This is obtained by substituting the Hulten TFP decomposition
(d ln A = Within + Baumol) into the labor productivity identity
(d ln(Y/L) = [1/(1-a)] d ln A + [a/(1-a)] d ln(K/Y)).

The script also computes a simple international comparison using
Penn World Table 11.0 for OECD countries.

DATA SOURCES
------------
1. Statistics Canada Table 36-10-0217-01 (Canadian industry-level data)
2. Penn World Table 11.0 (international comparison)

OUTPUTS
-------
- Figures/note_labor_productivity.png   (Figure 1: 3-term LP decomposition)
- Figures/note_tfp_decomposition.png    (Figure 2: TFP with/without Baumol)
- Figures/note_baumol_scatter.png       (Figure 3: VA share vs TFP growth)
- Figures/note_tfp_slowdown.png         (Figure 4: Industry TFP slowdown)
- Tables/note_decomposition.tex         (Table 1: Canadian decomposition)
- Tables/note_decomposition_kl.tex      (Diagnostic: K/L version of the decomposition)
- Figures/note_oecd_rankings.png        (Figure 5: Original OECD LP rank change)
- Tables/note_lp_levels.tex             (Table 2: Original OECD LP levels in 2019)
- Tables/note_international.tex         (Table 3: Original OECD LP growth decomposition, 2000-2019)

Author: Jean-Felix Brouillette (HEC Montreal)
"""

import re
import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rc
from stats_can import StatsCan

########################################################################
# Helper functions                                                      #
########################################################################

def setup_figure():
    """Create a figure with the project's standard formatting."""
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return fig, ax


def finalize_figure(fig, ax, filepath):
    """Add source attribution, save to disk, and close."""
    ax.text(1, 1.01, 'Source : Statistique Canada', fontsize=8,
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


def amplify_decomp(decomp, yearly_df, start):
    """Amplify within/baumol by 1/(1-alpha_bar) for LP decomposition.

    Converts TFP-level within and Baumol terms to LP-level terms by
    applying the amplification factor 1/(1-alpha_bar).
    """
    merged = pd.merge(decomp, yearly_df[['year', 'alpha_bar']], on='year', how='left')
    merged['within_lp'] = merged['within'] / (1 - merged['alpha_bar'])
    merged['baumol_lp'] = merged['baumol'] / (1 - merged['alpha_bar'])
    # Start year: within=baumol=0, alpha_bar may be NaN -> set to 0
    merged.loc[merged['year'] == start, ['within_lp', 'baumol_lp']] = 0.0
    return merged


def industry_within_lp_contrib(df_ind, yearly_df, start, end, base_col):
    """Annualized industry contributions to within-industry LP growth.

    Uses the same base-period shares and aggregate amplification factor as the
    main three-term decomposition, then annualizes each industry's direct
    within-industry contribution over the requested subperiod.
    """
    active = df_ind.loc[(df_ind['year'] > start) & (df_ind['year'] <= end),
                        ['year', 'industry', base_col, 'tfp_growth']].copy()
    active = pd.merge(active, yearly_df[['year', 'alpha_bar']], on='year', how='left')
    active['within_lp_term'] = active[base_col] * active['tfp_growth'] / (1 - active['alpha_bar'])
    return 100 * active.groupby('industry')['within_lp_term'].sum() / (end - start)


def aggregate_to_nace(df):
    """Aggregate 39-industry panel to 15 NACE sectors via Törnqvist indices.

    Used only for the international comparison (Table 2), where Canadian
    data must match the international NACE-style aggregation used in the project.

    Dollar variables (va, capital_cost, labor_cost) are summed.
    Growth rates are Törnqvist-aggregated using within-sector weights:
      d ln A_N = Σ_{i∈N} (s̄_i / s̄_N) * d ln A_i
      d ln K_N = Σ_{i∈N} (ω̄^K_i / ω̄^K_N) * d ln K_i
      d ln L_N = Σ_{i∈N} (ω̄^L_i / ω̄^L_N) * d ln L_i

    The nesting property ensures Σ_N s̄_N * d ln A_N = Σ_i s̄_i * d ln A_i.
    """
    df = df.copy()
    df['nace'] = df['naics'].map(NAICS_TO_NACE)

    # Sum dollar variables within each (nace, year) — all years incl. 1961
    dollars = df.groupby(['nace', 'year']).agg(
        va=('va', 'sum'),
        capital_cost=('capital_cost', 'sum'),
        labor_cost=('labor_cost', 'sum'),
    ).reset_index()

    # Within-sector sums of Törnqvist weights for growth rate aggregation
    df['s_bar_N'] = df.groupby(['nace', 'year'])['s_bar'].transform('sum')
    df['omk_N'] = df.groupby(['nace', 'year'])['omega_k_bar'].transform('sum')
    df['oml_N'] = df.groupby(['nace', 'year'])['omega_l_bar'].transform('sum')

    # Weighted growth rates
    df['tfp_w'] = (df['s_bar'] / df['s_bar_N']) * df['tfp_growth']
    df['cap_w'] = (df['omega_k_bar'] / df['omk_N']) * df['capital_growth']
    df['lab_w'] = (df['omega_l_bar'] / df['oml_N']) * df['labor_growth']

    # Sum with min_count=1 so 1961 (all-NaN) stays NaN instead of 0
    growth = (df.groupby(['nace', 'year'])[['tfp_w', 'cap_w', 'lab_w']]
              .sum(min_count=1).reset_index())
    growth = growth.rename(columns={
        'tfp_w': 'tfp_growth', 'cap_w': 'capital_growth', 'lab_w': 'labor_growth'})

    result = pd.merge(dollars, growth, on=['nace', 'year'])

    # Recompute VA shares
    result['va_agg'] = result.groupby('year')['va'].transform('sum')
    result['s'] = result['va'] / result['va_agg']
    result['s_bar'] = result.groupby('nace')['s'].transform(
        lambda x: x.rolling(2).mean())

    # Recompute capital cost shares
    result['capital_cost_agg'] = result.groupby('year')['capital_cost'].transform('sum')
    result['omega_k'] = result['capital_cost'] / result['capital_cost_agg']
    result['omega_k_bar'] = result.groupby('nace')['omega_k'].transform(
        lambda x: x.rolling(2).mean())

    # Recompute labor cost shares
    result['labor_cost_agg'] = result.groupby('year')['labor_cost'].transform('sum')
    result['omega_l'] = result['labor_cost'] / result['labor_cost_agg']
    result['omega_l_bar'] = result.groupby('nace')['omega_l'].transform(
        lambda x: x.rolling(2).mean())

    # Base-period VA shares for subperiod decompositions
    for base_year, label in [(1962, 's_1961'), (1980, 's_1980'), (2000, 's_2000')]:
        base = result.loc[result['year'] == base_year,
                          ['nace', 's_bar']].rename(columns={'s_bar': label})
        result = pd.merge(result, base, on='nace', how='left')

    # Rename nace→naics for downstream compatibility
    result = result.rename(columns={'nace': 'naics'})
    result = result.sort_values(['naics', 'year']).reset_index(drop=True)

    return result


########################################################################
# Configuration                                                         #
########################################################################

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
FIG_DIR = ROOT_DIR / 'Figures'
TAB_DIR = ROOT_DIR / 'Tables'

sc = StatsCan(data_folder=SCRIPT_DIR)

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Fira Sans']})
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage[sfdefault,light]{FiraSans}'
                          r'\usepackage[T1]{fontenc}'
                          r'\usepackage[utf8]{inputenc}')

PALETTE = ['#002855', '#26d07c', '#ff585d', '#f3d03e',
           '#0072ce', '#eb6fbd', '#00aec7', '#888b8d']

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

# NAICS → NACE concordance for aggregation to 15 sectors (used for Table 2).
# O (Public admin) and P (Education) are absent from Canadian business-sector data.
NAICS_TO_NACE = {
    '111-112': 'A', '113': 'A', '114': 'A', '115': 'A',
    '211': 'B', '212': 'B', '213': 'B',
    '311': 'C', '312': 'C', '313-314': 'C', '315-316': 'C',
    '321': 'C', '322': 'C', '323': 'C', '324': 'C', '325': 'C',
    '326': 'C', '327': 'C', '331': 'C', '332': 'C', '333': 'C',
    '334': 'C', '335': 'C', '336': 'C', '337': 'C', '339': 'C',
    '221': 'D-E',
    '23': 'F',
    '41': 'G', '44-45': 'G',
    '48-49': 'H',
    '72': 'I',
    '51': 'J',
    '52-53': 'K-L',
    '54': 'M',
    '56': 'N',
    '62': 'Q',
    '71': 'R',
    '81': 'S',
}

# Sectors to drop: aggregate industries and sub-industries that overlap
# with the 39 leaf-level industries above.
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

# Filter to the 39 leaf-level industries
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

# --- Industry capital shares (for industry-level K/Y diagnostics) ---
# alpha_{it} = capital_cost_{it} / VA_{it}
# alpha_bar_{it} = (alpha_{it} + alpha_{i,t-1}) / 2
df['alpha_industry'] = df['capital_cost'] / df['va']
df['alpha_industry_bar'] = df.groupby('naics')['alpha_industry'].transform(
    lambda x: x.rolling(2).mean())

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
yearly['dlnKL'] = yearly['dlnK'] - yearly['dlnL']   # d ln(K/L)

# --- Decomposition ---
# d ln(Y/L) = [1/(1-alpha_bar)] * d ln A + [alpha_bar/(1-alpha_bar)] * d ln(K/Y)
yearly['tfp_contrib'] = yearly['dlnA'] / (1 - yearly['alpha_bar'])
yearly['ky_contrib'] = yearly['alpha_bar'] / (1 - yearly['alpha_bar']) * yearly['dlnKY']

# Alternative K/L decomposition:
# d ln(Y/L) = d ln A + alpha_bar * d ln(K/L)
# This is exact but, unlike the K/Y version, it attributes part of the
# capital deepening induced by TFP to the capital term itself.
yearly['tfp_kl'] = yearly['dlnA']
yearly['kl_contrib'] = yearly['alpha_bar'] * yearly['dlnKL']

# --- Identity check (exact by construction) ---
residual_1 = (yearly['tfp_contrib'] + yearly['ky_contrib'] - yearly['dlnYL']).abs().max()
print(f'\nStep 1 identity — max residual: {residual_1:.2e}')
assert residual_1 < 1e-10, f'Step 1 identity fails: residual = {residual_1}'
residual_1_kl = (yearly['tfp_kl'] + yearly['kl_contrib'] - yearly['dlnYL']).abs().max()
print(f'Step 1 identity (K/L version) — max residual: {residual_1_kl:.2e}')
assert residual_1_kl < 1e-10, f'Step 1 K/L identity fails: residual = {residual_1_kl}'

# Prepend a 1961 baseline row (all zeros) for cumulative plots.
# Growth rates are undefined for 1961; the cumsum starts at 0.
row_1961 = pd.DataFrame({
    'year': [1961], 'dlnA': [0.0], 'dlnK': [0.0], 'dlnL': [0.0],
    'alpha_bar': [np.nan], 'dlnY': [0.0], 'dlnYL': [0.0],
    'dlnKY': [0.0], 'dlnKL': [0.0],
    'tfp_contrib': [0.0], 'ky_contrib': [0.0],
    'tfp_kl': [0.0], 'kl_contrib': [0.0]
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

# --- Amplify within/baumol for 3-term LP decomposition ---
# d ln(Y/L) = [1/(1-a)]*within + [1/(1-a)]*baumol + [a/(1-a)]*d ln(K/Y)
decomp_full_lp = amplify_decomp(decomp_full, yearly, 1961)
decomp_61_80_lp = amplify_decomp(decomp_61_80, yearly, 1961)
decomp_80_00_lp = amplify_decomp(decomp_80_00, yearly, 1980)
decomp_00_19_lp = amplify_decomp(decomp_00_19, yearly, 2000)

# Identity check: within_lp + baumol_lp = tfp_contrib (year level)
for name, dlp, start in [('1961-2019', decomp_full_lp, 1961),
                          ('1961-1980', decomp_61_80_lp, 1961),
                          ('1980-2000', decomp_80_00_lp, 1980),
                          ('2000-2019', decomp_00_19_lp, 2000)]:
    merged = pd.merge(dlp[dlp['year'] > start], yearly[['year', 'tfp_contrib']], on='year')
    residual = (merged['within_lp'] + merged['baumol_lp'] - merged['tfp_contrib']).abs().max()
    print(f'3-term amplification ({name}) — max residual: {residual:.2e}')
    assert residual < 1e-10, f'3-term amplification fails for {name}'

########################################################################
# 5. Annualized growth rates and summary table                         #
########################################################################

PERIODS = [(1961, 2019), (1961, 1980), (1980, 2000), (2000, 2019)]
DECOMPS = [decomp_full, decomp_61_80, decomp_80_00, decomp_00_19]

# Panel A: Labor productivity decomposition
lp      = [annualize(yearly, 'dlnYL', s, e) for s, e in PERIODS]
tfp_c   = [annualize(yearly, 'tfp_contrib', s, e) for s, e in PERIODS]
ky_c    = [annualize(yearly, 'ky_contrib', s, e) for s, e in PERIODS]
tfp_kl_c = [annualize(yearly, 'tfp_kl', s, e) for s, e in PERIODS]
kl_c     = [annualize(yearly, 'kl_contrib', s, e) for s, e in PERIODS]

# Panel B: TFP decomposition
tfp_agg = [annualize_decomp(d, 'total', s, e) for d, (s, e) in zip(DECOMPS, PERIODS)]
within  = [annualize_decomp(d, 'within', s, e) for d, (s, e) in zip(DECOMPS, PERIODS)]
baumol  = [annualize_decomp(d, 'baumol', s, e) for d, (s, e) in zip(DECOMPS, PERIODS)]

# 3-term LP decomposition
DECOMPS_LP = [decomp_full_lp, decomp_61_80_lp, decomp_80_00_lp, decomp_00_19_lp]
within_lp = [annualize_decomp(d, 'within_lp', s, e) for d, (s, e) in zip(DECOMPS_LP, PERIODS)]
baumol_lp = [annualize_decomp(d, 'baumol_lp', s, e) for d, (s, e) in zip(DECOMPS_LP, PERIODS)]

# Print summary
print('\n--- Annualized growth rates (%) ---')
header = f'{"":30s} {"1961-2019":>10s} {"1961-1980":>10s} {"1980-2000":>10s} {"2000-2019":>10s}'
print(header)
print('-' * len(header))
print(f'{"d ln(Y/L)":30s} {lp[0]:10.2f} {lp[1]:10.2f} {lp[2]:10.2f} {lp[3]:10.2f}')
print(f'{"  Within-industry TFP":30s} {within_lp[0]:10.2f} {within_lp[1]:10.2f} {within_lp[2]:10.2f} {within_lp[3]:10.2f}')
print(f'{"  Reallocation (Baumol)":30s} {baumol_lp[0]:10.2f} {baumol_lp[1]:10.2f} {baumol_lp[2]:10.2f} {baumol_lp[3]:10.2f}')
print(f'{"  K/Y contribution":30s} {ky_c[0]:10.2f} {ky_c[1]:10.2f} {ky_c[2]:10.2f} {ky_c[3]:10.2f}')
print(f'{"d ln A (Hulten)":30s} {tfp_agg[0]:10.2f} {tfp_agg[1]:10.2f} {tfp_agg[2]:10.2f} {tfp_agg[3]:10.2f}')
print(f'{"  Within-industry":30s} {within[0]:10.2f} {within[1]:10.2f} {within[2]:10.2f} {within[3]:10.2f}')
print(f'{"  Composition (Baumol)":30s} {baumol[0]:10.2f} {baumol[1]:10.2f} {baumol[2]:10.2f} {baumol[3]:10.2f}')
print('\n--- Alternative K/L decomposition (%) ---')
print(header)
print('-' * len(header))
print(f'{"d ln(Y/L)":30s} {lp[0]:10.2f} {lp[1]:10.2f} {lp[2]:10.2f} {lp[3]:10.2f}')
print(f'{"  Within-industry TFP":30s} {within[0]:10.2f} {within[1]:10.2f} {within[2]:10.2f} {within[3]:10.2f}')
print(f'{"  Reallocation (Baumol)":30s} {baumol[0]:10.2f} {baumol[1]:10.2f} {baumol[2]:10.2f} {baumol[3]:10.2f}')
print(f'{"  K/L contribution":30s} {kl_c[0]:10.2f} {kl_c[1]:10.2f} {kl_c[2]:10.2f} {kl_c[3]:10.2f}')

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

# 3-term identity: within_lp + baumol_lp + ky_c = d ln(Y/L)
for i, (s, e) in enumerate(PERIODS):
    residual = abs(within_lp[i] + baumol_lp[i] + ky_c[i] - lp[i])
    assert residual < 1e-10, f'3-term LP identity fails for {s}-{e}'
print('3-term LP identity check (Within + Baumol + K/Y = LP): passed for all periods')

# K/L identity: within + baumol + alpha*dln(K/L) = d ln(Y/L)
for i, (s, e) in enumerate(PERIODS):
    residual = abs(within[i] + baumol[i] + kl_c[i] - lp[i])
    assert residual < 1e-10, f'K/L identity fails for {s}-{e}'
print('K/L identity check (Within + Baumol + K/L = LP): passed for all periods')

# Panels A-B consistency: annualized d ln A should match between panels.
# Panel A derives d ln A from yearly['dlnA']; Panel B from decomp['total'].
for i, (s, e) in enumerate(PERIODS):
    dlnA_from_A = annualize(yearly, 'dlnA', s, e)
    residual = abs(dlnA_from_A - tfp_agg[i])
    assert residual < 1e-10, f'Panel A-B consistency fails for {s}-{e}'
print('Panel A-B consistency check (d ln A same in both): passed for all periods')

########################################################################
# 7. Figure 1: Labor productivity decomposition (stacked bars)         #
########################################################################

fig, ax = setup_figure()

# Sub-period annualized growth rates (indices 1,2,3 = the three sub-periods).
period_labels = ['1961--1980', '1980--2000', '2000--2019']
within_vals = np.array([within_lp[1], within_lp[2], within_lp[3]])
baumol_vals = np.array([baumol_lp[1], baumol_lp[2], baumol_lp[3]])
ky_vals     = np.array([ky_c[1], ky_c[2], ky_c[3]])

x = np.arange(len(period_labels))
width = 0.5

# Stack: positives above zero, negatives below zero.
components = [
    (within_vals, PALETTE[0], 'PTF intra-industries'),
    (baumol_vals, PALETTE[4], r'R\'eallocation (Baumol)'),
    (ky_vals, PALETTE[1], 'Approfondissement du capital'),
]

for i in range(len(x)):
    pos_bottom = 0
    neg_bottom = 0
    for vals, color, label in components:
        v = vals[i]
        if v >= 0:
            ax.bar(x[i], v, width, bottom=pos_bottom, color=color,
                   label=label if i == 0 else '')
            ax.text(x[i], pos_bottom + v / 2, f'{v:.2f}',
                    ha='center', va='center', fontsize=10, fontweight='bold',
                    color='white')
            pos_bottom += v
        else:
            ax.bar(x[i], v, width, bottom=neg_bottom, color=color,
                   label=label if i == 0 else '')
            ax.text(x[i], neg_bottom + v / 2, f'{v:.2f}',
                    ha='center', va='center', fontsize=10, fontweight='bold',
                    color='white')
            neg_bottom += v

# Total LP growth above each bar
for i in range(len(x)):
    total = within_vals[i] + baumol_vals[i] + ky_vals[i]
    top = sum(v for v in [within_vals[i], baumol_vals[i], ky_vals[i]] if v > 0)
    ax.text(x[i], top + 0.04, f'{total:.2f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold',
            color='k')

ax.axhline(0, color='k', linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(period_labels, fontsize=11)
yticks = ax.get_yticks()
ax.set_yticks(yticks)
ax.set_yticklabels([f'{tick:g}' for tick in yticks], fontsize=11)
ax.set_ylabel(r'Croissance annualisée (\%)', fontsize=11, rotation=0, ha='left')
ax.yaxis.set_label_coords(0, 1.02)
ax.grid(True, which='major', axis='y', color='gray', linestyle=':', linewidth=0.5)
ax.legend(frameon=False, fontsize=10, loc='upper right')

finalize_figure(fig, ax, FIG_DIR / 'note_labor_productivity.png')

########################################################################
# 8. Figure 2: TFP decomposition (Baumol effect)                       #
########################################################################

fig, ax = setup_figure()

tfp_total_index = 100 * np.exp(decomp_full['total'].cumsum())
tfp_within_index = 100 * np.exp(decomp_full['within'].cumsum())

ax.plot(decomp_full['year'], tfp_total_index,
        label='Total', color=PALETTE[0], linewidth=2)
# Counterfactual: TFP if economic structure stayed at 1961 shares.
ax.plot(decomp_full['year'], tfp_within_index,
        label='Sans effet Baumol', color=PALETTE[1], linewidth=2)

ax.set_xlim(1961, 2019)
ax.set_xticks(range(1965, 2015 + 1, 5))
ax.set_xticklabels(range(1965, 2015 + 1, 5), fontsize=11)
ymin = 5 * np.floor((min(tfp_total_index.min(), tfp_within_index.min()) - 1) / 5)
ymax = 5 * np.ceil((max(tfp_total_index.max(), tfp_within_index.max()) + 1) / 5)
ystep = 10 if (ymax - ymin) > 60 else 5
yticks = np.arange(ymin, ymax + ystep, ystep)
ax.set_ylim(ymin, ymax)
ax.set_yticks(yticks)
ax.set_yticklabels([f'{int(t)}' for t in yticks], fontsize=11)
ax.set_ylabel('PTF agrégée (1961=100)', fontsize=11, rotation=0, ha='left')
ax.yaxis.set_label_coords(0, 1.02)
ax.grid(True, which='major', axis='y', color='gray', linestyle=':', linewidth=0.5)
ax.legend(frameon=False, fontsize=10)

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
           color=PALETTE[0], s=50, alpha=0.7, zorder=3,
           edgecolors='white', linewidth=0.5)

# OLS regression line spanning the full x-axis domain
z = np.polyfit(scatter_data['cum_tfp'], scatter_data['delta_s'], 1)

# Draw regression line after autoscaling so it spans the full domain
xlim = ax.get_xlim()
x_fit = np.array(xlim)
ax.plot(x_fit, z[0] * x_fit + z[1], '-', color=PALETTE[1], linewidth=1.5, zorder=2)
ax.set_xlim(xlim)

ax.set_xlabel(r'Croissance cumulée de la PTF, 1961--2019 (\%)', fontsize=11)
ax.xaxis.set_label_coords(0.5, -0.1)
ax.set_ylabel(r'$\Delta$ part de la VA (p.p.)', fontsize=11, rotation=0, ha='left')
ax.yaxis.set_label_coords(0, 1.02)
ax.grid(True, which='major', axis='both', color='gray', linestyle=':', linewidth=0.5)
ax.tick_params(axis='both', labelsize=11)

finalize_figure(fig, ax, FIG_DIR / 'note_baumol_scatter.png')

########################################################################
# 10. Figure 4: Industry-level TFP slowdown (horizontal bars)          #
########################################################################

# For each industry, compute annualized TFP growth before and after 2000,
# then show the change (post minus pre) as a horizontal bar chart.
ind_pre = (df[(df['year'] > 1961) & (df['year'] <= 2000)]
           .groupby('industry')['tfp_growth']
           .sum() / (2000 - 1961))
ind_post = (df[(df['year'] > 2000) & (df['year'] <= 2019)]
            .groupby('industry')['tfp_growth']
            .sum() / (2019 - 2000))
slowdown = 100 * (ind_post - ind_pre)
slowdown = slowdown.sort_values(ascending=True)

# Strip NAICS codes and abbreviate long industry names for cleaner labels
ABBREV = {
    'Computer and electronic product manufacturing': 'Computer & electronics mfg.',
    'Petroleum and coal products manufacturing': 'Petroleum & coal products mfg.',
    'Beverage and tobacco product manufacturing': 'Beverage & tobacco mfg.',
    'Mining (except oil and gas)': 'Mining (excl. oil & gas)',
    'Miscellaneous manufacturing': 'Misc. manufacturing',
    'Transportation equipment manufacturing': 'Transportation equip. mfg.',
    'Chemical manufacturing': 'Chemical mfg.',
    'Electrical equipment, appliance and component manufacturing': 'Electrical equip. & appliance mfg.',
    'Transportation and warehousing': 'Transportation & warehousing',
    'Plastics and rubber products manufacturing': 'Plastics & rubber mfg.',
    'Furniture and related product manufacturing': 'Furniture & related mfg.',
    'Non-metallic mineral product manufacturing': 'Non-metallic mineral mfg.',
    'Fabricated metal product manufacturing': 'Fabricated metal mfg.',
    'Primary metal manufacturing': 'Primary metal mfg.',
    'Clothing, Leather and allied product manufacturing': 'Clothing & leather mfg.',
    'Information and cultural industries': 'Information & cultural ind.',
    'Textile and textile product mills': 'Textile mills',
    'Food manufacturing': 'Food mfg.',
    'Support activities for agriculture and forestry': 'Agric. & forestry support',
    'Machinery manufacturing': 'Machinery mfg.',
    'Health care and social assistance (except hospitals)': 'Health care & social assist.',
    'Professional, scientific and technical services': 'Prof., scientific & tech. services',
    'Wood product manufacturing': 'Wood product mfg.',
    'Support activities for mining and oil and gas extraction': 'Mining & oil support',
    'Paper manufacturing': 'Paper mfg.',
    'Other services (except public administration)': 'Other services (excl. public admin.)',
    'Printing and related support activities': 'Printing & related',
    'Arts, entertainment and recreation': 'Arts, entertainment & recreation',
    'Crop and animal production': 'Crop & animal production',
    'Administrative and support, waste management and remediation services': 'Admin., waste mgmt. & remediation',
    'Accommodation and food services': 'Accommodation & food services',
    'Fishing, hunting and trapping': 'Fishing, hunting & trapping',
    'Finance, insurance, real estate and renting and leasing': 'Finance, insurance & real estate',
    'Forestry and logging': 'Forestry & logging',
    'Oil and gas extraction': 'Oil & gas extraction',
    'Wholesale trade': 'Wholesale trade',
    'Retail trade': 'Retail trade',
    'Construction': 'Construction',
    'Utilities': 'Utilities',
}
# Escape '&' for LaTeX rendering (usetex=True)
clean_labels = [ABBREV.get(re.sub(r'\s*\[.*?\]', '', name),
                           re.sub(r'\s*\[.*?\]', '', name)).replace('&', r'\&')
                for name in slowdown.index]

# Color: coral for slowdowns, green for accelerations
colors = [PALETTE[2] if v < 0 else PALETTE[1] for v in slowdown.values]

fig, ax = plt.subplots(figsize=(8, 9))
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)

ax.barh(range(len(slowdown)), slowdown.values, color=colors, height=0.7)

ax.set_yticks(range(len(slowdown)))
ax.set_yticklabels(clean_labels, fontsize=9)
ax.invert_yaxis()
ax.axvline(0, color='k', linewidth=0.8)

# Axis formatting
ax.set_xlabel(r'Variation de la croissance annualisée de la PTF (p.p.)',
              fontsize=11, ha='center')
ax.xaxis.set_label_coords(0.5, -0.04)
ax.tick_params(axis='x', labelsize=11)

# Spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)

# Gridlines: vertical only
ax.grid(True, which='major', axis='x', color='gray', linestyle=':', linewidth=0.5)

# Explicit x-limits and ticks with a small margin around the data
data_min = slowdown.min()
data_max = slowdown.max()
tick_step = 1 if (data_max - data_min) <= 10 else 2
xmin = tick_step * np.floor((data_min - 0.4) / tick_step)
xmax = tick_step * np.ceil((data_max + 0.4) / tick_step)
xticks = np.arange(xmin, xmax + 0.001, tick_step)
ax.set_xlim(xmin, xmax)
ax.set_xticks(xticks)
ax.set_xticklabels([f'{tick:g}' for tick in xticks], fontsize=11)

finalize_figure(fig, ax, FIG_DIR / 'note_tfp_slowdown.png')
print('Figure 4: Industry TFP slowdown saved.')
print('  Industries with the sharpest TFP slowdowns:')
for name, value in slowdown.head(8).items():
    clean_name = re.sub(r'\s*\[.*?\]', '', name)
    print(f'    {clean_name}: {value:+.2f} pp')
print('  Industries with TFP acceleration:')
for name, value in slowdown.tail(5).sort_values(ascending=False).items():
    clean_name = re.sub(r'\s*\[.*?\]', '', name)
    print(f'    {clean_name}: {value:+.2f} pp')

########################################################################
# 12. LaTeX table: Summary decomposition                                #
########################################################################

def fmt(v):
    """Format a growth rate for the LaTeX table (e.g., '1.28')."""
    return f'{v:.2f}'

def fmt_bold(v):
    """Format a growth rate with bold navy for the 2000-2019 column."""
    return r'\textbf{' + f'{v:.2f}' + '}'

def row(label, vals, bold_last=True):
    """Build a table row with optional bold last column."""
    cells = [fmt(v) for v in vals[:3]]
    cells.append(fmt_bold(vals[3]) if bold_last else fmt(vals[3]))
    return label + ' & ' + ' & '.join(cells) + r' \\'

with open(TAB_DIR / 'note_decomposition.tex', 'w') as f:
    lines = [
        r'\begin{table}[H]',
        r'\centering',
        r'\sffamily',
        r'\renewcommand{\arraystretch}{1.3}',
        r'\begin{threeparttable}',
        r"\caption{D\'ecomposition de la croissance annuelle moyenne de la productivit\'e du travail (\%)}",
        r'\label{tab:note_decomposition}',
        r'\begin{tabular}{l*{4}{c}}',
        r'\toprule',
        r'& 1961--2019 & 1961--1980 & 1980--2000 & \textbf{2000--2019} \\',
        r'\midrule',
        row(r'$\Delta \ln(Y/L)$', lp),
        row(r'\quad Terme PTF intra-ind.', within_lp),
        row(r"\quad Terme r\'ealloc. (Baumol)", baumol_lp),
        row(r'\quad Terme capital', ky_c),
        r'\bottomrule',
        r'\end{tabular}',
        r'\begin{tablenotes}\footnotesize',
        r"\item \textit{Note}\,: Croissance annuelle moyenne en points de "
        r"pourcentage. Les trois composantes s'additionnent exactement "
        r"\`a $\Delta \ln(Y/L)$. Les contributions de la PTF intra-industries "
        r"et de la r\'eallocation sont amplifi\'ees par le facteur $1/(1-\bar\alpha) > 1$, qui refl\`ete "
        r"l'effet indirect de la PTF sur l'accumulation du capital (voir annexe~A). "
        r"Source\,: Statistique Canada, tableau 36-10-0217-01.",
        r'\end{tablenotes}',
        r'\end{threeparttable}',
        r'\end{table}'
    ]
    f.write('\n'.join(lines))

with open(TAB_DIR / 'note_decomposition_kl.tex', 'w') as f:
    lines = [
        r'\begin{table}[H]',
        r'\centering',
        r'\sffamily',
        r'\renewcommand{\arraystretch}{1.3}',
        r'\begin{threeparttable}',
        r"\caption{D\'ecomposition alternative de la croissance annuelle moyenne de la productivit\'e du travail avec $K/L$ (\%)}",
        r'\label{tab:note_decomposition_kl}',
        r'\begin{tabular}{l*{4}{c}}',
        r'\toprule',
        r'& 1961--2019 & 1961--1980 & 1980--2000 & \textbf{2000--2019} \\',
        r'\midrule',
        row(r'$\Delta \ln(Y/L)$', lp),
        row(r'\quad Terme PTF intra-ind.', within),
        row(r"\quad Terme r\'ealloc. (Baumol)", baumol),
        row(r'\quad Terme capital ($K/L$)', kl_c),
        r'\bottomrule',
        r'\end{tabular}',
        r'\begin{tablenotes}\footnotesize',
        r"\item \textit{Note}\,: D\'ecomposition alternative avec $K/L$, "
        r"pour laquelle $\Delta \ln(Y/L) = \Delta \ln A + \bar\alpha\,\Delta \ln(K/L)$. "
        r"Contrairement \`a la version principale avec $K/Y$, cette variante attribue "
        r"une partie de l'accumulation de capital induite par la PTF \`a la composante capital. "
        r"Les trois composantes s'additionnent exactement \`a $\Delta \ln(Y/L)$. "
        r"Source\,: Statistique Canada, tableau 36-10-0217-01.",
        r'\end{tablenotes}',
        r'\end{threeparttable}',
        r'\end{table}'
    ]
    f.write('\n'.join(lines))

########################################################################
# 13. PWT 11.0 international comparison                                 #
########################################################################

print('\n--- PWT 11.0 OECD Comparison ---')

PWT_VERSION = '11.0'
PWT_URL = 'https://dataverse.nl/api/access/datafile/554030'
PWT_START, PWT_END = 2000, 2019
PWT_LEVEL_YEAR = 2019
pwt_path = ROOT_DIR / 'Data' / 'pwt110.dta'

# Original OECD members (1961), used to keep the comparison historically consistent.
OECD_CODES = [
    'AUT', 'BEL', 'CAN', 'DNK', 'FRA', 'DEU', 'GRC', 'ISL', 'IRL', 'ITA',
    'LUX', 'NLD', 'NOR', 'PRT', 'ESP', 'SWE', 'CHE', 'TUR', 'GBR', 'USA',
]

COUNTRY_FR = {
    'AUT': 'Autriche', 'BEL': 'Belgique', 'CAN': r'\textbf{Canada}',
    'CHE': 'Suisse', 'DEU': 'Allemagne', 'DNK': 'Danemark',
    'ESP': 'Espagne', 'FRA': 'France',
    'GBR': 'Royaume-Uni', 'GRC': r"Gr\`ece", 'HUN': 'Hongrie',
    'IRL': 'Irlande', 'ISL': 'Islande', 'ITA': 'Italie',
    'LUX': 'Luxembourg', 'NLD': 'Pays-Bas', 'NOR': r'Norv\`ege',
    'PRT': 'Portugal', 'SWE': r'Su\`ede',
    'TUR': 'Turquie', 'USA': r"\'Etats-Unis",
}

if not pwt_path.exists():
    print(f'Downloading Penn World Table {PWT_VERSION}...')
    urllib.request.urlretrieve(PWT_URL, pwt_path)
    print(f'  Saved ({pwt_path.stat().st_size / 1e6:.1f} MB)')
else:
    print(f'Using cached PWT {PWT_VERSION}: {pwt_path}')

pwt = pd.read_stata(pwt_path)
pwt_oecd = pwt[pwt['countrycode'].isin(OECD_CODES)].copy()

# --- Table 2: 2019 level comparison ---
p19 = pwt_oecd[pwt_oecd['year'] == PWT_LEVEL_YEAR][
    ['countrycode', 'country', 'cgdpo', 'emp', 'avh', 'ck', 'labsh']
].copy()
p19 = p19.dropna(subset=['cgdpo', 'emp', 'avh', 'ck', 'labsh'])
p19 = p19[(p19[['cgdpo', 'emp', 'avh', 'ck']] > 0).all(axis=1)].copy()

# For cross-country levels at a single date, use cgdpo rather than rgdpo.
# PWT's ck is a cross-sectional current-PPP measure of capital services
# (normalized to USA = 1), which is usable for relative levels but not
# for time-series growth accounting.
p19['yl'] = p19['cgdpo'] / (p19['emp'] * p19['avh'])
p19['ky'] = p19['ck'] / p19['cgdpo']
p19['kl'] = p19['ck'] / (p19['emp'] * p19['avh'])
us_row = p19[p19['countrycode'] == 'USA'].iloc[0]
alpha_us = 1 - us_row['labsh']

p19['yl_rel'] = 100 * p19['yl'] / us_row['yl']
p19['ky_rel'] = p19['ky'] / us_row['ky']
p19['kl_rel'] = 100 * p19['kl'] / us_row['kl']
p19['ky_comp'] = 100 * p19['ky_rel'] ** (alpha_us / (1 - alpha_us))
p19['a_comp'] = p19['yl_rel'] * 100 / p19['ky_comp']
p19['level_rank'] = p19['yl_rel'].rank(ascending=False, method='min').astype(int)

intl_levels = p19.sort_values('yl_rel', ascending=False).reset_index(drop=True)

print(f'PWT level data available for {len(intl_levels)}/{len(OECD_CODES)} OECD countries')
print(f'  US alpha = {alpha_us:.3f}, amplification = {1/(1-alpha_us):.3f}')
ca_lev = intl_levels[intl_levels['countrycode'] == 'CAN'].iloc[0]
print(f'  Canada: Y/L={ca_lev["yl_rel"]:.0f}, '
      f'prod. resid.={ca_lev["a_comp"]:.0f}, '
      f'K/Y={ca_lev["ky_comp"]:.0f}, K/L={ca_lev["kl_rel"]:.0f}, '
      f'rank={int(ca_lev["level_rank"])}/{len(intl_levels)}')

# --- Figure 5: Rank change in labor productivity levels, 2000 vs 2019 ---
rank_years = [2000, 2019]
rank_frames = []
for year in rank_years:
    py = pwt_oecd[pwt_oecd['year'] == year][['countrycode', 'country', 'cgdpo', 'emp', 'avh']].copy()
    py = py.dropna(subset=['cgdpo', 'emp', 'avh'])
    py = py[(py[['cgdpo', 'emp', 'avh']] > 0).all(axis=1)].copy()
    py['yl'] = py['cgdpo'] / (py['emp'] * py['avh'])
    py['rank'] = py['yl'].rank(ascending=False, method='min').astype(int)
    py = py[['countrycode', 'rank']].rename(columns={'rank': f'rank_{year}'})
    rank_frames.append(py)

rank_compare = rank_frames[0].merge(rank_frames[1], on='countrycode', how='inner')
rank_compare = rank_compare.sort_values('rank_2019').reset_index(drop=True)

fig, ax = plt.subplots(figsize=(8, 7.5))
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)

for _, r in rank_compare.iterrows():
    code = r['countrycode']
    x = [0, 1]
    y = [r['rank_2000'], r['rank_2019']]
    if code == 'CAN':
        color = PALETTE[0]
        lw = 2.8
        alpha = 1.0
        zorder = 3
    else:
        color = '0.75'
        lw = 1.1
        alpha = 0.9
        zorder = 1
    ax.plot(x, y, color=color, linewidth=lw, alpha=alpha, zorder=zorder)
    ax.scatter(x, y, color=color, s=28 if code == 'CAN' else 16, alpha=alpha, zorder=zorder)

can_rank = rank_compare[rank_compare['countrycode'] == 'CAN'].iloc[0]
ax.text(-0.08, can_rank['rank_2000'], 'Canada', color=PALETTE[0],
        fontsize=11, fontweight='bold', ha='right', va='center')
ax.text(1.08, can_rank['rank_2019'], 'Canada', color=PALETTE[0],
        fontsize=11, fontweight='bold', ha='left', va='center')
ax.text(-0.08, can_rank['rank_2000'] - 0.55, f'{int(can_rank["rank_2000"])}e rang',
        color=PALETTE[0], fontsize=9, ha='right', va='center')
ax.text(1.08, can_rank['rank_2019'] - 0.55, f'{int(can_rank["rank_2019"])}e rang',
        color=PALETTE[0], fontsize=9, ha='left', va='center')

ax.text(0.00, 0.75, '2000', fontsize=12, fontweight='bold', ha='center', va='bottom')
ax.text(0.94, 0.75, '2019', fontsize=12, fontweight='bold', ha='center', va='bottom')
ax.set_xlim(-0.35, 1.35)
ax.set_ylim(len(rank_compare) + 0.5, 0.5)
ax.set_xticks([])
ax.set_yticks(range(1, len(rank_compare) + 1))
ax.set_yticklabels([f'{i}' for i in range(1, len(rank_compare) + 1)], fontsize=9)
ax.set_ylabel('Rang de productivité du travail', fontsize=11)
ax.grid(True, axis='y', color='0.88', linestyle=':', linewidth=0.6)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)

finalize_figure(fig, ax, FIG_DIR / 'note_oecd_rankings.png')
print('Figure 5: Original OECD labor-productivity rankings saved.')
print(f'  Canada rank: {int(can_rank["rank_2000"])} in 2000 -> {int(can_rank["rank_2019"])} in 2019')

# --- Table 3: LP growth decomposition, 2000-2019 ---
# For time-series growth accounting, use rkna rather than ck. rkna is the
# PWT time-series measure of capital services at constant national prices.
required_years = set(range(PWT_START, PWT_END + 1))
intl_growth_rows = []
for code in OECD_CODES:
    g = pwt_oecd[(pwt_oecd['countrycode'] == code) &
                 (pwt_oecd['year'] >= PWT_START) &
                 (pwt_oecd['year'] <= PWT_END)][
        ['countrycode', 'country', 'year', 'rgdpo', 'emp', 'avh', 'rkna']
    ].copy()
    if set(g['year']) != required_years:
        continue
    g = g.dropna(subset=['rgdpo', 'emp', 'avh', 'rkna']).sort_values('year')
    if len(g) != len(required_years):
        continue
    if (g[['rgdpo', 'emp', 'avh', 'rkna']] <= 0).any().any():
        continue

    g['yl'] = g['rgdpo'] / (g['emp'] * g['avh'])
    g['ky'] = g['rkna'] / g['rgdpo']
    lp_ann = 100 * (np.log(g['yl'].iloc[-1]) - np.log(g['yl'].iloc[0])) / (PWT_END - PWT_START)
    ky_growth = 100 * (np.log(g['ky'].iloc[-1]) - np.log(g['ky'].iloc[0])) / (PWT_END - PWT_START)
    ky_ann = 100 * (alpha_us / (1 - alpha_us)) * (
        np.log(g['ky'].iloc[-1]) - np.log(g['ky'].iloc[0])
    ) / (PWT_END - PWT_START)
    prod_ann = lp_ann - ky_ann

    intl_growth_rows.append({
        'countrycode': code,
        'country': g['country'].iloc[0],
        'lp_growth': lp_ann,
        'prod_term': prod_ann,
        'ky_term': ky_ann,
        'ky_growth': ky_growth,
    })

intl_growth = pd.DataFrame(intl_growth_rows).sort_values('lp_growth', ascending=False).reset_index(drop=True)
intl_growth['lp_rank'] = np.arange(1, len(intl_growth) + 1)

print('\n--- PWT LP Growth Decomposition ---')
print(f'PWT growth data available for {len(intl_growth)}/{len(OECD_CODES)} OECD countries')
if not intl_growth.empty and (intl_growth['countrycode'] == 'CAN').any():
    ca_growth = intl_growth[intl_growth['countrycode'] == 'CAN'].iloc[0]
    print(f'  Canada: LP={ca_growth["lp_growth"]:+.2f}, '
          f'prod. resid.={ca_growth["prod_term"]:+.2f}, '
          f'K/Y term={ca_growth["ky_term"]:+.2f} '
          f'({int(ca_growth["lp_rank"])}/{len(intl_growth)})')

########################################################################
# 14. LaTeX tables: International comparison                            #
########################################################################

with open(TAB_DIR / 'note_lp_levels.tex', 'w') as f:
    lines = [
        r'\begin{table}[H]',
        r'\centering',
        r'\sffamily',
        r'\footnotesize',
        r'\renewcommand{\arraystretch}{1.06}',
        r'\begin{threeparttable}',
        r"\caption{Niveau de la productivit\'e du travail en 2019, membres originels de l'OCDE (\'Etats-Unis = 100)}",
        r'\label{tab:note_lp_levels}',
        r'\begin{tabular}{l *{4}{r}}',
        r'\toprule',
        r"& $Y/L$ & Prod.\ r\'esid. & $K/Y$ & $K/L$ \\",
        r'\midrule',
    ]
    for _, r in intl_levels.iterrows():
        code = r['countrycode']
        name = COUNTRY_FR.get(code, r['country']).replace('&', r'\&')
        if code == 'CAN':
            lines.append(
                f'{name} & \\textbf{{{r["yl_rel"]:.0f}}} '
                f'& \\textbf{{{r["a_comp"]:.0f}}} '
                f'& \\textbf{{{r["ky_comp"]:.0f}}} '
                f'& \\textbf{{{r["kl_rel"]:.0f}}} \\\\'
            )
        else:
            lines.append(
                f'{name} & {r["yl_rel"]:.0f} & {r["a_comp"]:.0f} '
                f'& {r["ky_comp"]:.0f} & {r["kl_rel"]:.0f} \\\\'
            )
    lines += [
        r'\bottomrule',
        r'\end{tabular}',
        r'\begin{tablenotes}\footnotesize',
        r"\item \textit{Note}\,: $Y/L$ = PIB par heure travaill\'ee, calcul\'e comme "
        r"$cgdpo/(emp \times avh)$ en 2019. "
        r"$K/Y$ = composante du ratio services du capital/production, "
        r"$(K/Y)^{\bar\alpha/(1-\bar\alpha)}$, calcul\'ee \`a partir de $ck$ "
        r"(services du capital aux PPA courantes, normalis\'es \`a "
        r"\'Etats-Unis $=1$) avec la part du capital am\'ericaine en 2019. "
        r"La productivit\'e r\'esiduelle = $Y/L$ divis\'e par $K/Y$. "
        r"Elle est donc conditionnelle \`a cette mesure des services du capital. "
        r"$K/L$ = services du capital par heure travaill\'ee, rapport\'es au niveau am\'ericain. "
        r"Toutes les colonnes sont index\'ees aux \'Etats-Unis ($=100$). "
        r"\'Echantillon\,: 20 membres originels de l'OCDE. "
        r"Source\,: Penn World Table 11.0.",
        r'\end{tablenotes}',
        r'\end{threeparttable}',
        r'\end{table}'
    ]
    f.write('\n'.join(lines))

print(f'Table 2 (OECD levels) saved to {TAB_DIR / "note_lp_levels.tex"}')

with open(TAB_DIR / 'note_international.tex', 'w') as f:
    lines = [
        r'\begin{table}[H]',
        r'\centering',
        r'\sffamily',
        r'\footnotesize',
        r'\renewcommand{\arraystretch}{1.06}',
        r'\begin{threeparttable}',
        r"\caption{D\'ecomposition de la croissance annuelle moyenne de la productivit\'e du travail, membres originels de l'OCDE, 2000--2019 (\%)}",
        r'\label{tab:note_international}',
        r'\begin{tabular}{l *{4}{r}}',
        r'\toprule',
        r"& $\Delta \ln(Y/L)$ & Terme prod.\ r\'esid. & Terme $K/Y$ & Rang \\",
        r'\midrule',
    ]
    for _, r in intl_growth.iterrows():
        code = r['countrycode']
        name = COUNTRY_FR.get(code, r['country']).replace('&', r'\&')
        if code == 'CAN':
            lines.append(
                f'{name} & \\textbf{{{r["lp_growth"]:.2f}}} '
                f'& \\textbf{{{r["prod_term"]:.2f}}} '
                f'& \\textbf{{{r["ky_term"]:.2f}}} '
                f'& \\textbf{{{int(r["lp_rank"])}}} \\\\'
            )
        else:
            lines.append(
                f'{name} & {r["lp_growth"]:.2f} & {r["prod_term"]:.2f} '
                f'& {r["ky_term"]:.2f} & {int(r["lp_rank"])} \\\\'
            )
    lines += [
        r'\bottomrule',
        r'\end{tabular}',
        r'\begin{tablenotes}\footnotesize',
        r"\item \textit{Note}\,: Croissance annuelle moyenne en points de pourcentage, 2000--2019. "
        r"Pour chaque pays, la d\'ecomposition suit l'identit\'e agr\'eg\'ee "
        r"$\Delta \ln(Y/L) = \text{Terme prod.\ r\'esid.} + \frac{\bar\alpha}{1-\bar\alpha}\Delta \ln(K/Y)$, "
        r"en utilisant la m\^eme part de capital de r\'ef\'erence que dans le tableau~\ref{tab:note_lp_levels}, "
        r"soit la part du capital am\'ericaine en 2019. Le terme de productivit\'e est obtenu "
        r"comme r\'esidu afin d'assurer l'identit\'e. $Y/L$ est calcul\'e \`a partir de "
        r"$rgdpo/(emp \times avh)$ et le terme $K/Y$ \`a partir de $rkna/rgdpo$, o\`u $rkna$ "
        r"mesure les services du capital en prix nationaux constants. Le classement "
        r"et le terme r\'esiduel restent sensibles au concept de PIB retenu dans la PWT; "
        r"ce tableau doit donc \^etre lu comme un exercice de contexte plut\^ot que "
        r"comme une d\'ecomposition structurelle d\'efinitive. "
        r"\'Echantillon\,: 20 membres originels de l'OCDE. "
        r"Source\,: Penn World Table 11.0.",
        r'\end{tablenotes}',
        r'\end{threeparttable}',
        r'\end{table}'
    ]
    f.write('\n'.join(lines))

print(f'Table 3 (OECD LP growth decomposition) saved to {TAB_DIR / "note_international.tex"}')
print('\nDone. Figures saved to Figures/, tables saved to Tables/.')
