"""
Diagnostic script: check whether the industry-level production identity holds.

    d ln VA_i = d ln A_i + alpha_bar_i * d ln K_i + (1 - alpha_bar_i) * d ln L_i

where alpha_bar_i is the Tornqvist average of
      alpha_i = capital_cost_i / (capital_cost_i + labor_cost_i).

Uses the same data source and industry filtering as note.py.
"""

import numpy as np
import pandas as pd
from stats_can import StatsCan

sc = StatsCan()

# ─────────────────────────────────────────────────────────────────────
# Industry filtering (copied from note.py)
# ─────────────────────────────────────────────────────────────────────

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

# ─────────────────────────────────────────────────────────────────────
# Step 1: Fetch the raw table and list ALL available variables
# ─────────────────────────────────────────────────────────────────────

print("=" * 72)
print("STEP 1: Fetch Table 36-10-0217-01 and list all variables")
print("=" * 72)

df_raw = sc.table_to_df('36-10-0217-01')

var_col = 'Multifactor productivity and related variables'
all_vars = sorted(df_raw[var_col].unique())

print(f"\nAll {len(all_vars)} unique values of '{var_col}':\n")
for i, v in enumerate(all_vars, 1):
    print(f"  {i:2d}. {v}")

# ─────────────────────────────────────────────────────────────────────
# Step 2: Build industry-level panel with ALL key variables
# ─────────────────────────────────────────────────────────────────────

print("\n" + "=" * 72)
print("STEP 2: Build industry-level panel")
print("=" * 72)

VARS_TO_FETCH = [
    'Multifactor productivity based on value-added',   # TFP index (A_i)
    'Labour input',                                     # L index
    'Capital input',                                    # K index
    'Capital cost',                                     # nominal capital cost
    'Labour compensation',                              # nominal labor cost
    'Gross domestic product (GDP)',                      # NOMINAL VA
    'Real gross domestic product (GDP)',                 # REAL VA (chain-weighted)
    'Combined labour and capital inputs',               # Combined K+L index (VA framework)
    'Labour productivity based on value-added',         # LP = real VA / L
]

print(f"\nFetching: {VARS_TO_FETCH}")

df = df_raw.copy()
df = df[~df['North American Industry Classification System (NAICS)'].isin(DROP_LIST)]
df = df[df[var_col].isin(VARS_TO_FETCH)]

# Pivot to wide: one row per (industry, year)
df = df.pivot_table(
    index=['North American Industry Classification System (NAICS)', 'REF_DATE'],
    columns=var_col,
    values='VALUE'
).reset_index().rename_axis(None, axis=1)

df = df.rename(columns={
    'North American Industry Classification System (NAICS)': 'industry',
    'REF_DATE': 'date',
    'Multifactor productivity based on value-added': 'tfp',
    'Labour input': 'labor',
    'Capital input': 'capital',
    'Gross domestic product (GDP)': 'va_nominal',
    'Real gross domestic product (GDP)': 'va_real',
    'Labour compensation': 'labor_cost',
    'Capital cost': 'capital_cost',
    'Combined labour and capital inputs': 'combined_kl',
    'Labour productivity based on value-added': 'lp_va',
})

df['year'] = df['date'].dt.year
df = df.drop(columns=['date'])
df = df[df['year'] < 2020]
df['naics'] = df['industry'].map(INDUSTRY_TO_NAICS)
df = df.dropna(subset=['naics'])
df = df.sort_values(['naics', 'year']).reset_index(drop=True)

n_ind = df['naics'].nunique()
n_yr = df['year'].nunique()
print(f"\nPanel: {n_ind} industries x {n_yr} years = {len(df)} obs")
print(f"Years: {df['year'].min()} to {df['year'].max()}")

# Check for missing data in each column
print("\nMissing values per column:")
for col in ['tfp', 'capital', 'labor', 'va_nominal', 'va_real',
            'capital_cost', 'labor_cost', 'combined_kl', 'lp_va']:
    if col in df.columns:
        nmiss = df[col].isna().sum()
        print(f"  {col:20s}: {nmiss} missing ({nmiss/len(df)*100:.1f}%)")
    else:
        print(f"  {col:20s}: COLUMN NOT FOUND")

# ─────────────────────────────────────────────────────────────────────
# Step 3: Compute growth rates
# ─────────────────────────────────────────────────────────────────────

print("\n" + "=" * 72)
print("STEP 3: Compute growth rates and factor shares")
print("=" * 72)

# Log-differences for index variables
for var, col in [('tfp', 'dlnA'), ('capital', 'dlnK'), ('labor', 'dlnL'),
                 ('va_real', 'dlnVA_real'), ('combined_kl', 'dlnCI'),
                 ('lp_va', 'dlnLP')]:
    if var in df.columns:
        df[col] = df.groupby('naics')[var].transform(lambda x: np.log(x).diff())

# Nominal VA growth
df['dlnVA_nom'] = df.groupby('naics')['va_nominal'].transform(lambda x: np.log(x).diff())

# Industry-level capital share
df['alpha'] = df['capital_cost'] / (df['capital_cost'] + df['labor_cost'])

# Tornqvist average: alpha_bar_i = (alpha_{i,t} + alpha_{i,t-1}) / 2
df['alpha_bar'] = df.groupby('naics')['alpha'].transform(lambda x: x.rolling(2).mean())

# Predicted real VA growth from production identity
df['predicted_dlnVA'] = (df['dlnA']
                         + df['alpha_bar'] * df['dlnK']
                         + (1 - df['alpha_bar']) * df['dlnL'])

print("\nSample (first 3 industries, years 1962-1965):")
sample = df[(df['year'].between(1962, 1965))].head(12)
print(sample[['naics', 'year', 'dlnA', 'dlnK', 'dlnL', 'alpha_bar',
              'predicted_dlnVA', 'dlnVA_real', 'dlnCI']].to_string(index=False))

# ─────────────────────────────────────────────────────────────────────
# Step 4: Identity checks
# ─────────────────────────────────────────────────────────────────────

# Drop 1961 (growth rates undefined)
dc = df[df['year'] >= 1962].copy()

print("\n" + "=" * 72)
print("STEP 4: Identity checks")
print("=" * 72)

# ────────────────────────────────────────────────────────────────
# CHECK A: d ln VA_real vs predicted (the MAIN check)
# ────────────────────────────────────────────────────────────────
if 'dlnVA_real' in dc.columns and dc['dlnVA_real'].notna().any():
    dc['resid_real'] = dc['dlnVA_real'] - dc['predicted_dlnVA']

    print("\n--- CHECK A: REAL VA growth vs production identity ---")
    print("  d ln VA_real vs d ln A + alpha_bar * d ln K + (1-alpha_bar) * d ln L")
    print(f"  Max abs residual:  {dc['resid_real'].abs().max():.6e}")
    print(f"  Mean abs residual: {dc['resid_real'].abs().mean():.6e}")
    print(f"  Std of residual:   {dc['resid_real'].std():.6e}")

    if dc['resid_real'].abs().max() < 1e-8:
        print("  ==> IDENTITY HOLDS EXACTLY (machine precision)")
    elif dc['resid_real'].abs().max() < 0.001:
        print("  ==> Identity holds approximately (< 0.1% residual)")
    else:
        print("  ==> Identity does NOT hold exactly.")

    # Top 10 residuals
    top10 = dc.nlargest(10, 'resid_real', keep='first')[
        ['naics', 'year', 'dlnVA_real', 'predicted_dlnVA', 'resid_real', 'dlnA', 'dlnK', 'dlnL', 'alpha_bar']
    ]
    print("\n  Top 10 largest positive residuals (real VA grew FASTER than predicted):")
    with pd.option_context('display.float_format', '{:.6f}'.format, 'display.width', 150):
        print(top10.to_string(index=False))

    bot10 = dc.nsmallest(10, 'resid_real', keep='first')[
        ['naics', 'year', 'dlnVA_real', 'predicted_dlnVA', 'resid_real', 'dlnA', 'dlnK', 'dlnL', 'alpha_bar']
    ]
    print("\n  Top 10 largest negative residuals (real VA grew SLOWER than predicted):")
    with pd.option_context('display.float_format', '{:.6f}'.format, 'display.width', 150):
        print(bot10.to_string(index=False))

    # Per-industry summary
    by_ind_real = dc.groupby('naics')['resid_real'].agg(['mean', 'std', 'min', 'max'])
    print("\n  Per-industry residual summary (real VA - predicted):")
    print(f"  {'NAICS':>10s} {'Mean':>10s} {'Std':>10s} {'Min':>10s} {'Max':>10s}")
    for naics, row in by_ind_real.iterrows():
        flag = " ***" if abs(row['max']) > 0.01 or abs(row['min']) > 0.01 else ""
        print(f"  {naics:>10s} {row['mean']:10.6f} {row['std']:10.6f} {row['min']:10.6f} {row['max']:10.6f}{flag}")
else:
    print("\n*** REAL VA column not available or all NaN. Cannot do CHECK A. ***")

# ────────────────────────────────────────────────────────────────
# CHECK B: Combined K+L inputs vs alpha_bar * d ln K + (1-alpha_bar) * d ln L
# ────────────────────────────────────────────────────────────────
if 'dlnCI' in dc.columns and dc['dlnCI'].notna().any():
    dc['predicted_CI'] = dc['alpha_bar'] * dc['dlnK'] + (1 - dc['alpha_bar']) * dc['dlnL']
    dc['resid_CI'] = dc['dlnCI'] - dc['predicted_CI']

    print("\n--- CHECK B: Combined K+L inputs vs Tornqvist aggregate ---")
    print("  d ln(Combined KL) vs alpha_bar * d ln K + (1-alpha_bar) * d ln L")
    print(f"  Max abs residual:  {dc['resid_CI'].abs().max():.6e}")
    print(f"  Mean abs residual: {dc['resid_CI'].abs().mean():.6e}")

    if dc['resid_CI'].abs().max() < 1e-8:
        print("  ==> Combined inputs IS the Tornqvist aggregate of K and L (exact).")
    else:
        print("  ==> Combined inputs != Tornqvist aggregate. Possible reasons:")
        print("      - StatsCan uses different weights (e.g., Fisher vs Tornqvist)")
        print("      - Weights use total cost (capital+labor+intermediate) not just K+L")

    # Top 10 residuals
    dc['abs_resid_CI'] = dc['resid_CI'].abs()
    top10_ci = dc.nlargest(10, 'abs_resid_CI')[
        ['naics', 'year', 'dlnCI', 'predicted_CI', 'resid_CI']
    ]
    print("\n  Top 10 absolute residuals:")
    with pd.option_context('display.float_format', '{:.6f}'.format, 'display.width', 120):
        print(top10_ci.to_string(index=False))
else:
    print("\n*** Combined K+L inputs column not available. Skipping CHECK B. ***")

# ────────────────────────────────────────────────────────────────
# CHECK C: d ln VA_real vs d ln A + d ln(Combined KL)
# ────────────────────────────────────────────────────────────────
if 'dlnVA_real' in dc.columns and 'dlnCI' in dc.columns:
    dc['dlnVA_from_TFP_CI'] = dc['dlnA'] + dc['dlnCI']
    dc['resid_TFP_CI'] = dc['dlnVA_real'] - dc['dlnVA_from_TFP_CI']

    print("\n--- CHECK C: Real VA = TFP * Combined inputs ---")
    print("  d ln VA_real vs d ln A + d ln(Combined KL)")
    print(f"  Max abs residual:  {dc['resid_TFP_CI'].abs().max():.6e}")
    print(f"  Mean abs residual: {dc['resid_TFP_CI'].abs().mean():.6e}")

    if dc['resid_TFP_CI'].abs().max() < 1e-8:
        print("  ==> Real VA = TFP * Combined inputs EXACTLY.")
        print("      This means TFP is computed as: A_i = VA_real_i / f(K_i, L_i)")
    else:
        print("  ==> Real VA != TFP * Combined inputs exactly.")

    # Top 10 residuals
    dc['abs_resid_TFP_CI'] = dc['resid_TFP_CI'].abs()
    top10_tc = dc.nlargest(10, 'abs_resid_TFP_CI')[
        ['naics', 'year', 'dlnVA_real', 'dlnVA_from_TFP_CI', 'resid_TFP_CI']
    ]
    print("\n  Top 10 absolute residuals:")
    with pd.option_context('display.float_format', '{:.6f}'.format, 'display.width', 120):
        print(top10_tc.to_string(index=False))

# ────────────────────────────────────────────────────────────────
# CHECK D: Level consistency of real VA index
#   VA_real is an index (2017=100?). If identity holds, then
#   cumulative d ln VA_real = ln(VA_real_2019 / VA_real_1961)
#   should equal cumulative predicted_dlnVA
# ────────────────────────────────────────────────────────────────
if 'dlnVA_real' in dc.columns:
    print("\n--- CHECK D: Cumulative growth consistency ---")
    cum = dc.groupby('naics').agg(
        cum_dlnVA_real=('dlnVA_real', 'sum'),
        cum_predicted=('predicted_dlnVA', 'sum'),
    ).reset_index()
    cum['diff'] = cum['cum_dlnVA_real'] - cum['cum_predicted']
    print(f"  Max abs difference in cumulative growth (1962-2019): {cum['diff'].abs().max():.6e}")
    print(f"\n  Per-industry cumulative comparison:")
    print(f"  {'NAICS':>10s} {'cum(dlnVA)':>12s} {'cum(pred)':>12s} {'diff':>12s}")
    for _, row in cum.iterrows():
        flag = " ***" if abs(row['diff']) > 0.01 else ""
        print(f"  {row['naics']:>10s} {row['cum_dlnVA_real']:12.6f} {row['cum_predicted']:12.6f} {row['diff']:12.6f}{flag}")

# ────────────────────────────────────────────────────────────────
# CHECK E: Nominal VA vs real VA (implied deflator)
# ────────────────────────────────────────────────────────────────
if 'dlnVA_real' in dc.columns:
    dc['implied_deflator'] = dc['dlnVA_nom'] - dc['dlnVA_real']

    print("\n--- CHECK E: Implied industry VA deflators ---")
    print("  d ln P_VA_i = d ln VA_nominal_i - d ln VA_real_i")
    defl = dc.groupby('naics')['implied_deflator'].mean() * 100
    print(f"\n  {'NAICS':>10s}  {'Avg annual deflator (%)':>10s}")
    for naics, val in defl.items():
        print(f"  {naics:>10s}  {val:10.2f}%")
    print(f"\n  Overall average: {defl.mean():.2f}%/year")

# ────────────────────────────────────────────────────────────────
# CHECK F: Industry capital shares
# ────────────────────────────────────────────────────────────────
print("\n--- CHECK F: Industry capital shares alpha_i ---")
alpha_stats = dc.groupby('naics')['alpha'].agg(['mean', 'min', 'max'])
print(f"\n  {'NAICS':>10s} {'Mean':>8s} {'Min':>8s} {'Max':>8s}")
for naics, row in alpha_stats.iterrows():
    flag = " ***" if row['max'] > 0.9 or row['min'] < 0.05 else ""
    print(f"  {naics:>10s} {row['mean']:8.3f} {row['min']:8.3f} {row['max']:8.3f}{flag}")

# ────────────────────────────────────────────────────────────────
# CHECK G: LP identity: d ln LP = d ln VA_real - d ln L
# ────────────────────────────────────────────────────────────────
if 'dlnLP' in dc.columns and 'dlnVA_real' in dc.columns:
    dc['resid_LP'] = dc['dlnLP'] - (dc['dlnVA_real'] - dc['dlnL'])
    print("\n--- CHECK G: Labor productivity identity ---")
    print("  d ln LP vs d ln VA_real - d ln L")
    print(f"  Max abs residual: {dc['resid_LP'].abs().max():.6e}")
    if dc['resid_LP'].abs().max() < 1e-8:
        print("  ==> LP = VA_real / L holds exactly.")
    else:
        print("  ==> LP != VA_real / L exactly.")
        dc['abs_resid_LP'] = dc['resid_LP'].abs()
        top_lp = dc.nlargest(5, 'abs_resid_LP')[
            ['naics', 'year', 'dlnLP', 'dlnVA_real', 'dlnL', 'resid_LP']
        ]
        print("\n  Top 5 absolute residuals:")
        with pd.option_context('display.float_format', '{:.6f}'.format):
            print(top_lp.to_string(index=False))

# ─────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────

print("\n" + "=" * 72)
print("OVERALL SUMMARY")
print("=" * 72)

print("""
Variables available in Table 36-10-0217-01:
  - Multifactor productivity based on value-added (TFP index, A_i)
  - Capital input (index, K_i)
  - Labour input (index, L_i)
  - Capital cost (dollars)
  - Labour compensation (dollars)
  - Gross domestic product (GDP) = NOMINAL value added
  - Real gross domestic product (GDP) = REAL value added (chain-weighted index)
  - Combined labour and capital inputs (index)
  - Labour productivity based on value-added (index)

The production identity being tested:
  d ln VA_i(real) = d ln A_i + alpha_bar_i * d ln K_i + (1 - alpha_bar_i) * d ln L_i
""")

if 'resid_real' in dc.columns:
    max_r = dc['resid_real'].abs().max()
    if max_r < 1e-8:
        print("MAIN RESULT: The identity holds EXACTLY (to machine precision).")
        print(f"  Max |residual| = {max_r:.2e}")
    elif max_r < 0.001:
        print("MAIN RESULT: The identity holds approximately.")
        print(f"  Max |residual| = {max_r:.6f}")
    else:
        print("MAIN RESULT: The identity does NOT hold exactly.")
        print(f"  Max |residual| = {max_r:.6f}")
        mean_r = dc['resid_real'].abs().mean()
        print(f"  Mean |residual| = {mean_r:.6f}")
        print("\n  This suggests that StatsCan's TFP is NOT computed as the simple")
        print("  Solow residual from the production identity with Tornqvist weights.")
        print("  Possible sources of discrepancy:")
        print("    1. StatsCan uses Fisher (not Tornqvist) for combined inputs")
        print("    2. Weights use total cost shares (including intermediates) not just K+L")
        print("    3. Real VA is chain-weighted (Fisher ideal) not Tornqvist")
        print("    4. Labour composition adjustments affect the identity")

print("\nDiagnostic complete.")
