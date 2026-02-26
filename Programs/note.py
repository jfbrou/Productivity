"""
Standalone script for the CPP report note.

Fetches industry-level data from Statistics Canada Table 36-10-0217-01,
computes a two-step decomposition of aggregate labor productivity growth,
and generates figures and a summary table.

Step 1: d ln(Y/L) = [1/(1-α)] d ln A + [α/(1-α)] d ln(K/Y)
Step 2: d ln A = Within + Baumol

Author: Jean-Félix Brouillette (HEC Montréal)
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
from stats_can import StatsCan

########################################################################
# Configuration                                                         #
########################################################################

sc = StatsCan()

rc('font', **{'family': 'serif', 'serif': ['Palatino']})
rc('text', usetex=True)

palette = ['#002855', '#26d07c', '#ff585d', '#f3d03e', '#0072ce', '#eb6fbd', '#00aec7', '#888b8d']

base_dir = Path(os.getcwd()).parent
fig_dir = base_dir / 'Figures'
tab_dir = base_dir / 'Tables'

########################################################################
# Industry-to-NAICS mapping                                            #
########################################################################

industry_to_naics = {
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

########################################################################
# Fetch and clean the data from Table 36-10-0217-01                    #
########################################################################

df = sc.table_to_df('36-10-0217-01')

drop_list = [
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
df = df[~df['North American Industry Classification System (NAICS)'].isin(drop_list)]

relevant_vars = [
    'Multifactor productivity based on value-added',
    'Labour input',
    'Capital input',
    'Gross domestic product (GDP)',
    'Labour compensation',
    'Capital cost'
]
df = df[df['Multifactor productivity and related variables'].isin(relevant_vars)]

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
df['naics'] = df['industry'].map(industry_to_naics)
df = df.sort_values(['naics', 'year']).reset_index(drop=True)

########################################################################
# Industry-level growth rates and Törnqvist weights                    #
########################################################################

# Log-differences of TFP, capital, and labor indices
df['tfp_growth'] = df.groupby('naics')['tfp'].transform(lambda x: np.log(x).diff())
df['capital_growth'] = df.groupby('naics')['capital'].transform(lambda x: np.log(x).diff())
df['labor_growth'] = df.groupby('naics')['labor'].transform(lambda x: np.log(x).diff())

# Nominal VA shares (Törnqvist-averaged)
df['va_agg'] = df.groupby('year')['va'].transform('sum')
df['s'] = df['va'] / df['va_agg']
df['s_bar'] = df.groupby('naics')['s'].transform(lambda x: x.rolling(2).mean())

# Capital cost shares (Törnqvist-averaged)
df['capital_cost_agg'] = df.groupby('year')['capital_cost'].transform('sum')
df['omega_k'] = df['capital_cost'] / df['capital_cost_agg']
df['omega_k_bar'] = df.groupby('naics')['omega_k'].transform(lambda x: x.rolling(2).mean())

# Labor cost shares (Törnqvist-averaged)
df['labor_cost_agg'] = df.groupby('year')['labor_cost'].transform('sum')
df['omega_l'] = df['labor_cost'] / df['labor_cost_agg']
df['omega_l_bar'] = df.groupby('naics')['omega_l'].transform(lambda x: x.rolling(2).mean())

# Base-period VA shares for subperiod decompositions
df = pd.merge(df, df.loc[df['year'] == 1962, ['naics', 's_bar']].rename(columns={'s_bar': 's_1961'}), on='naics', how='left')
df = pd.merge(df, df.loc[df['year'] == 1980, ['naics', 's_bar']].rename(columns={'s_bar': 's_1980'}), on='naics', how='left')
df = pd.merge(df, df.loc[df['year'] == 2000, ['naics', 's_bar']].rename(columns={'s_bar': 's_2000'}), on='naics', how='left')

########################################################################
# Step 1: Aggregate labor productivity decomposition                    #
########################################################################

# Aggregate capital share
agg = df.groupby('year').agg(
    capital_cost_total=('capital_cost', 'sum'),
    va_total=('va', 'sum')
).reset_index()
agg['alpha'] = agg['capital_cost_total'] / agg['va_total']
agg['alpha_bar'] = agg['alpha'].rolling(2).mean()

# Aggregate growth rates via Törnqvist
df['dlnA_term'] = df['s_bar'] * df['tfp_growth']
df['dlnK_term'] = df['omega_k_bar'] * df['capital_growth']
df['dlnL_term'] = df['omega_l_bar'] * df['labor_growth']

yearly = df.dropna(subset=['dlnA_term']).groupby('year').agg(
    dlnA=('dlnA_term', 'sum'),
    dlnK=('dlnK_term', 'sum'),
    dlnL=('dlnL_term', 'sum')
).reset_index()

yearly = pd.merge(yearly, agg[['year', 'alpha_bar']], on='year')

# Production identity: d ln Y = d ln A + α d ln K + (1-α) d ln L
yearly['dlnY'] = yearly['dlnA'] + yearly['alpha_bar'] * yearly['dlnK'] + (1 - yearly['alpha_bar']) * yearly['dlnL']

# Labor productivity and capital-output ratio growth
yearly['dlnYL'] = yearly['dlnY'] - yearly['dlnL']
yearly['dlnKY'] = yearly['dlnK'] - yearly['dlnY']

# Decomposition: d ln(Y/L) = [1/(1-α)] d ln A + [α/(1-α)] d ln(K/Y)
yearly['tfp_contrib'] = yearly['dlnA'] / (1 - yearly['alpha_bar'])
yearly['ky_contrib'] = yearly['alpha_bar'] / (1 - yearly['alpha_bar']) * yearly['dlnKY']

# Verify Step 1 identity
residual_1 = (yearly['tfp_contrib'] + yearly['ky_contrib'] - yearly['dlnYL']).abs().max()
print(f'Step 1 identity check — max residual: {residual_1:.2e}')
assert residual_1 < 1e-10, f'Step 1 identity fails with residual {residual_1}'

# Prepend 1961 baseline row with zeros for cumulative plots
row_1961 = pd.DataFrame({
    'year': [1961], 'dlnA': [0.0], 'dlnK': [0.0], 'dlnL': [0.0],
    'alpha_bar': [np.nan], 'dlnY': [0.0], 'dlnYL': [0.0], 'dlnKY': [0.0],
    'tfp_contrib': [0.0], 'ky_contrib': [0.0]
})
yearly = pd.concat([row_1961, yearly], ignore_index=True)

########################################################################
# Step 2: Aggregate TFP decomposition (within vs Baumol)               #
########################################################################

def compute_tfp_decomposition(df_ind, start, end, base_col):
    """Compute within/Baumol TFP decomposition for a subperiod."""
    result = pd.DataFrame({'year': range(start, end + 1)})

    d = df_ind.copy()
    d['within_term'] = d[base_col] * d['tfp_growth']
    d['baumol_term'] = (d['s_bar'] - d[base_col]) * d['tfp_growth']

    agg_terms = d.groupby('year').agg(
        within=('within_term', 'sum'),
        baumol=('baumol_term', 'sum')
    ).reset_index()

    result = pd.merge(result, agg_terms, on='year', how='left')
    result.loc[result['year'] == start, ['within', 'baumol']] = 0.0
    result['total'] = result['within'] + result['baumol']

    return result

decomp_full = compute_tfp_decomposition(df, 1961, 2019, 's_1961')
decomp_61_80 = compute_tfp_decomposition(df, 1961, 1980, 's_1961')
decomp_80_00 = compute_tfp_decomposition(df, 1980, 2000, 's_1980')
decomp_00_19 = compute_tfp_decomposition(df, 2000, 2019, 's_2000')

# Verify Step 2 identity: within + baumol = Hulten aggregate TFP
for name, decomp, start in [('1961-2019', decomp_full, 1961),
                             ('1961-1980', decomp_61_80, 1961),
                             ('1980-2000', decomp_80_00, 1980),
                             ('2000-2019', decomp_00_19, 2000)]:
    merged = pd.merge(decomp[decomp['year'] > start], yearly[['year', 'dlnA']], on='year')
    residual = (merged['total'] - merged['dlnA']).abs().max()
    print(f'Step 2 identity check ({name}) — max residual: {residual:.2e}')
    assert residual < 1e-10, f'Step 2 identity fails for {name}'

########################################################################
# Annualized growth rates for the table                                 #
########################################################################

def ann(yearly_df, col, start, end):
    """Annualized growth rate (%) for a period from the yearly DataFrame."""
    mask = (yearly_df['year'] > start) & (yearly_df['year'] <= end)
    return 100 * yearly_df.loc[mask, col].sum() / (end - start)

def ann_decomp(decomp_df, col, start, end):
    """Annualized growth rate (%) from a decomposition DataFrame."""
    return 100 * decomp_df[col].sum() / (end - start)

periods = [(1961, 2019), (1961, 1980), (1980, 2000), (2000, 2019)]
decomps = [decomp_full, decomp_61_80, decomp_80_00, decomp_00_19]

# Panel A: Labor productivity
lp = [ann(yearly, 'dlnYL', s, e) for s, e in periods]
tfp_c = [ann(yearly, 'tfp_contrib', s, e) for s, e in periods]
ky_c = [ann(yearly, 'ky_contrib', s, e) for s, e in periods]

# Panel B: TFP decomposition
tfp_total = [ann_decomp(d, 'total', s, e) for d, (s, e) in zip(decomps, periods)]
within = [ann_decomp(d, 'within', s, e) for d, (s, e) in zip(decomps, periods)]
baumol = [ann_decomp(d, 'baumol', s, e) for d, (s, e) in zip(decomps, periods)]

# Print summary
print('\n--- Annualized growth rates (%) ---')
print(f'{"":30s} {"1961-2019":>10s} {"1961-1980":>10s} {"1980-2000":>10s} {"2000-2019":>10s}')
print(f'{"d ln(Y/L)":30s} {lp[0]:10.2f} {lp[1]:10.2f} {lp[2]:10.2f} {lp[3]:10.2f}')
print(f'{"  TFP contribution":30s} {tfp_c[0]:10.2f} {tfp_c[1]:10.2f} {tfp_c[2]:10.2f} {tfp_c[3]:10.2f}')
print(f'{"  K/Y contribution":30s} {ky_c[0]:10.2f} {ky_c[1]:10.2f} {ky_c[2]:10.2f} {ky_c[3]:10.2f}')
print(f'{"d ln A":30s} {tfp_total[0]:10.2f} {tfp_total[1]:10.2f} {tfp_total[2]:10.2f} {tfp_total[3]:10.2f}')
print(f'{"  Within":30s} {within[0]:10.2f} {within[1]:10.2f} {within[2]:10.2f} {within[3]:10.2f}')
print(f'{"  Baumol":30s} {baumol[0]:10.2f} {baumol[1]:10.2f} {baumol[2]:10.2f} {baumol[3]:10.2f}')

########################################################################
# Figure 1: Labor productivity decomposition                           #
########################################################################

fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)

ax.plot(yearly['year'], 100 * (yearly['dlnYL'].cumsum() + 1),
        label='Labor productivity', color=palette[0], linewidth=2)
ax.plot(yearly['year'], 100 * (yearly['tfp_contrib'].cumsum() + 1),
        label='Without capital deepening', color=palette[1], linewidth=2)

ax.set_xlim(1961, 2019)
ax.set_xticks(range(1965, 2015 + 1, 5))
ax.set_xticklabels(range(1965, 2015 + 1, 5), fontsize=14)
ax.set_ylabel('Labor productivity (1961=100)', fontsize=14, rotation=0, ha='left')
ax.yaxis.set_label_coords(0, 1.01)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, which='major', axis='y', color='gray', linestyle=':', linewidth=0.5)
ax.legend(frameon=False, fontsize=14)
ax.text(1, 1.01, 'Source: Statistics Canada', fontsize=10, color='k', ha='right', va='bottom', transform=ax.transAxes)

fig.tight_layout()
fig.savefig(fig_dir / 'note_labor_productivity.png', transparent=True, dpi=300)
plt.close()

########################################################################
# Figure 2: TFP decomposition (Baumol effect)                          #
########################################################################

fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)

ax.plot(decomp_full['year'], 100 * (decomp_full['total'].cumsum() + 1),
        label='Total', color=palette[0], linewidth=2)
ax.plot(decomp_full['year'], 100 * (decomp_full['within'].cumsum() + 1),
        label='Without Baumol', color=palette[1], linewidth=2)

ax.set_xlim(1961, 2019)
ax.set_xticks(range(1965, 2015 + 1, 5))
ax.set_xticklabels(range(1965, 2015 + 1, 5), fontsize=14)
ax.set_ylim(100, 150)
ax.set_yticks(range(100, 150 + 1, 5))
ax.set_yticklabels(range(100, 150 + 1, 5), fontsize=14)
ax.set_ylabel('Aggregate TFP (1961=100)', fontsize=14, rotation=0, ha='left')
ax.yaxis.set_label_coords(0, 1.01)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, which='major', axis='y', color='gray', linestyle=':', linewidth=0.5)
ax.legend(frameon=False, fontsize=14)
ax.text(1, 1.01, 'Source: Statistics Canada', fontsize=10, color='k', ha='right', va='bottom', transform=ax.transAxes)

fig.tight_layout()
fig.savefig(fig_dir / 'note_tfp_decomposition.png', transparent=True, dpi=300)
plt.close()

########################################################################
# Figure 3: Baumol scatter (VA share change vs TFP growth)             #
########################################################################

# Compute industry-level scatter data
s_1961 = df[df['year'] == 1961].set_index('naics')['s']
s_2019 = df[df['year'] == 2019].set_index('naics')['s']
cum_tfp = df[df['year'] > 1961].groupby('naics')['tfp_growth'].sum()

scatter = pd.DataFrame({
    'cum_tfp': 100 * cum_tfp,
    'delta_s': 100 * (s_2019 - s_1961)
}).dropna()

fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)

ax.scatter(scatter['cum_tfp'], scatter['delta_s'],
           color=palette[0], s=40, zorder=3, edgecolors='white', linewidth=0.5)

# Regression line
z = np.polyfit(scatter['cum_tfp'], scatter['delta_s'], 1)
x_range = np.linspace(scatter['cum_tfp'].min() - 5, scatter['cum_tfp'].max() + 5, 100)
ax.plot(x_range, z[0] * x_range + z[1], '--', color=palette[2], linewidth=1.5, alpha=0.8)

ax.axhline(0, color='gray', linewidth=0.5, linestyle='-')
ax.axvline(0, color='gray', linewidth=0.5, linestyle='-')

ax.set_xlabel(r'Cumulative TFP growth, 1961--2019 (\%)', fontsize=14)
ax.set_ylabel(r'$\Delta$ VA share (p.p.)', fontsize=14, rotation=0, ha='left')
ax.yaxis.set_label_coords(0, 1.01)
ax.tick_params(axis='both', labelsize=14)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.text(1, 1.01, 'Source: Statistics Canada', fontsize=10, color='k', ha='right', va='bottom', transform=ax.transAxes)

fig.tight_layout()
fig.savefig(fig_dir / 'note_baumol_scatter.png', transparent=True, dpi=300)
plt.close()

########################################################################
# Table: Summary decomposition                                         #
########################################################################

def fmt(v):
    return f'{v:.2f}\\%'

table = open(tab_dir / 'note_decomposition.tex', 'w')
lines = [
    r'\begin{table}[H]',
    r'\centering',
    r'\begin{threeparttable}',
    r'\caption{D\'ecomposition de la croissance de la productivit\'e du travail}',
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
    r'$\Delta \ln A$ & ' + ' & '.join(fmt(v) for v in tfp_total) + r' \\',
    r'\quad Intra-industries & ' + ' & '.join(fmt(v) for v in within) + r' \\',
    r'\quad Composition (Baumol) & ' + ' & '.join(fmt(v) for v in baumol) + r' \\',
    r'\hline',
    r'\hline',
    r'\end{tabular}',
    r'\begin{tablenotes}[flushleft]',
    r'\footnotesize',
    r"\item \textit{Note}\,: Ce tableau pr\'esente la d\'ecomposition de la croissance annuelle moyenne de la productivit\'e du travail et de la PTF agr\'eg\'ee pour diff\'erentes sous-p\'eriodes. Source\,: Statistique Canada, tableau 36-10-0217-01.",
    r'\end{tablenotes}',
    r'\label{tab:note_decomposition}',
    r'\end{threeparttable}',
    r'\end{table}'
]
table.write('\n'.join(lines))
table.close()

print('\nDone. Figures saved to Figures/, table saved to Tables/.')
