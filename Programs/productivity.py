import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
from stats_can import StatsCan
from datetime import datetime
from scipy import sparse

# Initialize the StatsCan API
sc = StatsCan()

# Set the figures' font
rc('font', **{'family': 'serif', 'serif': ['Palatino']})
rc('text', usetex=True)

# Define my color palette
palette = ['#002855', '#26d07c', '#ff585d', '#f3d03e', '#0072ce', '#eb6fbd', '#00aec7', '#888b8d']

########################################################################
# Prepare the Table 36-10-0217-01 Statistics Canada data               # 
########################################################################

# Retrieve the data from Table 36-10-0217-01
df = sc.table_to_df('36-10-0217-01')

# Define a NAICS code dictionary
naics_map = {
    'Accommodation and food services [72]': '72',
    'Administrative and support, waste management and remediation services [56]':  '56',
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

# Drop several sectors and industries
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

# Keep the relevant variables
relevant_vars = [
    'Labour input',
    'Capital input',
    'Gross domestic product (GDP)', 
    'Real gross domestic product (GDP)', 
    'Labour compensation',
    'Hours worked',
    'Capital cost'
]
df = df[df['Multifactor productivity and related variables'].isin(relevant_vars)]

# Reshape the DataFrame
df = df.pivot_table(index=['North American Industry Classification System (NAICS)', 'REF_DATE'], 
                    columns='Multifactor productivity and related variables', 
                    values='VALUE').reset_index().rename_axis(None, axis=1)

# Rename the columns
df = df.rename(columns={
    'North American Industry Classification System (NAICS)': 'industry',
    'REF_DATE': 'date',
    'Labour input': 'labor',
    'Capital input': 'capital',
    'Gross domestic product (GDP)': 'va',
    'Real gross domestic product (GDP)': 'real_va',
    'Labour compensation': 'labor_cost',
    'Hours worked': 'hours',
    'Capital cost': 'capital_cost'
})

# Recode the date column to year
df['year'] = df['date'].dt.year
df = df.drop(columns=['date'])
df = df[df['year'] < 2020]

# Map the industry to the NAICS code
df['code'] = df['industry'].map(naics_map)

# Rescale the variables to 1961=100
df['real_va'] = df['real_va'] / df.loc[df['year'] == 1961, 'real_va'].values[0] * 100
df['capital'] = df['capital'] / df.loc[df['year'] == 1961, 'capital'].values[0] * 100
df['labor'] = df['labor'] / df.loc[df['year'] == 1961, 'labor'].values[0] * 100

# Calculate TFP with and without the capital adjustment for every industry
df['tfp'] = [np.nan] * df.shape[0]
df['tfp_adj'] = [np.nan] * df.shape[0]
df.loc[df['year'] == 1961, 'tfp'] = 100
df.loc[df['year'] == 1961, 'tfp_adj'] = 100
for i in df['industry'].unique():
    for year in range(1962, 2019 + 1, 1):
        alpha_now = df.loc[(df['year'] == year) & (df['industry'] == i), 'capital_cost'].iloc[0] / (df.loc[(df['year'] == year) & (df['industry'] == i), 'capital_cost'].iloc[0] + df.loc[(df['year'] == year) & (df['industry'] == i), 'labor_cost'].iloc[0])
        alpha_prev = df.loc[(df['year'] == year - 1) & (df['industry'] == i), 'capital_cost'].iloc[0] / (df.loc[(df['year'] == year - 1) & (df['industry'] == i), 'capital_cost'].iloc[0] + df.loc[(df['year'] == year - 1) & (df['industry'] == i), 'labor_cost'].iloc[0])
        alpha = 0.5 * (alpha_now + alpha_prev)
        df.loc[(df['year'] == year) & (df['industry'] == i), 'tfp'] = df.loc[(df['year'] == year - 1) & (df['industry'] == i), 'tfp'].iloc[0] * np.exp(np.log(df.loc[(df['year'] == year) & (df['industry'] == i), 'real_va'].iloc[0] / df.loc[(df['year'] == year - 1) & (df['industry'] == i), 'real_va'].iloc[0]) - alpha * np.log(df.loc[(df['year'] == year) & (df['industry'] == i), 'capital'].iloc[0] / df.loc[(df['year'] == year - 1) & (df['industry'] == i), 'capital'].iloc[0]) - (1 - alpha) * np.log(df.loc[(df['year'] == year) & (df['industry'] == i), 'labor'].iloc[0] / df.loc[(df['year'] == year - 1) & (df['industry'] == i), 'labor'].iloc[0]))
        df.loc[(df['year'] == year) & (df['industry'] == i), 'tfp_adj'] = df.loc[(df['year'] == year - 1) & (df['industry'] == i), 'tfp_adj'].iloc[0] * np.exp(np.log(df.loc[(df['year'] == year) & (df['industry'] == i), 'real_va'].iloc[0] / df.loc[(df['year'] == year - 1) & (df['industry'] == i), 'real_va'].iloc[0]) - (alpha / (1 - alpha)) * np.log((df.loc[(df['year'] == year) & (df['industry'] == i), 'capital'].iloc[0] / df.loc[(df['year'] == year) & (df['industry'] == i), 'real_va'].iloc[0]) / (df.loc[(df['year'] == year - 1) & (df['industry'] == i), 'capital'].iloc[0] / df.loc[(df['year'] == year - 1) & (df['industry'] == i), 'real_va'].iloc[0])) - np.log(df.loc[(df['year'] == year) & (df['industry'] == i), 'labor'].iloc[0] / df.loc[(df['year'] == year - 1) & (df['industry'] == i), 'labor'].iloc[0]))

# Calculate the share of value-added of each industry within year
df['va_agg'] = df.groupby('year')['va'].transform('sum')
df['b'] = df['va'] / df['va_agg']
df['b'] = df.groupby('industry')['b'].transform(lambda x: x.rolling(2).mean())
df = df.drop(columns=['va_agg'])

# Calculate the share of value-added of each industry for years 1961, 1980, and 2000
df = pd.merge(df, df.loc[df['year'] == 1962, ['industry', 'b']].rename(columns={'b': 'b_1961'}), on='industry', how='left')
df = pd.merge(df, df.loc[df['year'] == 1980, ['industry', 'b']].rename(columns={'b': 'b_1980'}), on='industry', how='left')
df = pd.merge(df, df.loc[df['year'] == 2000, ['industry', 'b']].rename(columns={'b': 'b_2000'}), on='industry', how='left')

# Calculate the log difference of TFP, capital, and labor within each industry
df['tfp_growth'] = df.groupby('industry')['tfp'].transform(lambda x: np.log(x).diff())
df['tfp_adj_growth'] = df.groupby('industry')['tfp'].transform(lambda x: np.log(x).diff())

# Calculate the industry-level output elasticities of capital and labor
df['alpha_k'] = df['capital_cost'] / (df['capital_cost'] + df['labor_cost'])
df['alpha_k'] = df.groupby('industry')['alpha_k'].transform(lambda x: x.rolling(2).mean())
df['alpha_l'] = df['labor_cost'] / (df['capital_cost'] + df['labor_cost'])
df['alpha_l'] = df.groupby('industry')['alpha_l'].transform(lambda x: x.rolling(2).mean())

# Calculate the share of total labor and capital costs of each industry within year
df['capital_cost_agg'] = df.groupby('year')['capital_cost'].transform('sum')
df['omega_k'] = df['capital_cost'] / df['capital_cost_agg']
df['omega_k'] = df.groupby('industry')['omega_k'].transform(lambda x: x.rolling(2).mean())
df['labor_cost_agg'] = df.groupby('year')['labor_cost'].transform('sum')
df['omega_l'] = df['labor_cost'] / df['labor_cost_agg']
df['omega_l'] = df.groupby('industry')['omega_l'].transform(lambda x: x.rolling(2).mean())
df = df.drop(columns=['capital_cost_agg', 'labor_cost_agg'])

# Calculate the productivity and Baumol terms between 1961 and 2019
df_1961_2019 = pd.DataFrame({'year': range(1961, 2019 + 1)})
df['within'] = df['b_1961'] * df['tfp_growth']
df_1961_2019 = pd.merge(df_1961_2019, df.groupby('year', as_index=False).agg({'within': 'sum'}).rename(columns={'within': 'productivity'}), on='year', how='left')
df['between'] = (df['b'] - df['b_1961']) * df['tfp_growth']
df_1961_2019 = pd.merge(df_1961_2019, df.groupby('year', as_index=False).agg({'between': 'sum'}).rename(columns={'between': 'baumol'}), on='year', how='left')
df = df.drop(columns=['within', 'between'])

# Calculate the productivity and Baumol terms between 1961 and 1980
df_1961_1980 = pd.DataFrame({'year': range(1961, 1980 + 1)})
df['within'] = df['b_1961'] * df['tfp_growth']
df_1961_1980 = pd.merge(df_1961_1980, df.groupby('year', as_index=False).agg({'within': 'sum'}).rename(columns={'within': 'productivity'}), on='year', how='left')
df['between'] = (df['b'] - df['b_1961']) * df['tfp_growth']
df_1961_1980 = pd.merge(df_1961_1980, df.groupby('year', as_index=False).agg({'between': 'sum'}).rename(columns={'between': 'baumol'}), on='year', how='left')

# Calculate the productivity and Baumol terms between 1980 and 2019
df_1980_2019 = pd.DataFrame({'year': range(1980, 2019 + 1)})
df['within'] = df['b_1980'] * df['tfp_growth']
df_1980_2019 = pd.merge(df_1980_2019, df.groupby('year', as_index=False).agg({'within': 'sum'}).rename(columns={'within': 'productivity'}), on='year', how='left')
df['between'] = (df['b'] - df['b_1980']) * df['tfp_growth']
df_1980_2019 = pd.merge(df_1980_2019, df.groupby('year', as_index=False).agg({'between': 'sum'}).rename(columns={'between': 'baumol'}), on='year', how='left')
df_1980_2019.loc[df_1980_2019['year'] == 1980, 'productivity'] = 0
df_1980_2019.loc[df_1980_2019['year'] == 1980, 'baumol'] = 0

########################################################################
# Prepare the Table 36-10-0001-01 Statistics Canada data               # 
########################################################################

# Retrieve the data from Table 36-10-0001-01
df_io = sc.table_to_df('36-10-0001-01')

# Restrict on basic prices
df_io = df_io[df_io['Valuation'] == 'Basic price']

# Drop total supply and use, and codes ["BS551113", "BS610000"]
df_io = df_io[df_io['Supply'] != 'Total supply']
df_io = df_io[df_io['Use'] != 'Total use']

# Keep the relevant columns
df_io = df_io[['REF_DATE', 'Supply', 'Use', 'VALUE']].rename(columns={'REF_DATE': 'date', 'Supply': 'supply', 'Use': 'use', 'VALUE': 'value'})

# Recode the date column to year
df_io['year'] = df_io['date'].dt.year
df_io = df_io.drop(columns=['date'])
df_io = df_io[df_io['year'] < 2020]

# Identify the "supply" and "use" codes
df_io['supply_code'] = df_io['supply'].str[-9:-1]
df_io['use_code'] = df_io['use'].str[-9:-1]

# Only keep the supply and use codes that start with "BS" (business sector)
df_io = df_io[df_io['supply_code'].str.startswith('BS')]
df_io = df_io[df_io['use_code'].str.startswith('BS')]

# Drop the supply and use codes "BS551113" and "BS610000"
df_io = df_io[~df_io['supply_code'].isin(['BS551113', 'BS610000'])]
df_io = df_io[~df_io['use_code'].isin(['BS551113', 'BS610000'])]

# Define aggregated codes at the 3-digit NAICS level
df_io['supply_code_agg'] = df_io['supply_code'].str[2:5]
df_io['use_code_agg'] = df_io['use_code'].str[2:5]

# Aggregate the data frame at the 3-digit NAICS level
df_io = df_io.groupby(['supply_code_agg', 'use_code_agg', 'year'], as_index=False).agg({'value': 'sum'})

# Define an aggregation grouping
group_list = {
    '111': '111-112',
    '112': '111-112',
    '23A': '23',
    '23B': '23',
    '23C': '23',
    '23D': '23',
    '23E': '23',
    '31A': '313-314',
    '31B': '315-316',
    '411': '41',
    '412': '41',
    '413': '41',
    '414': '41',
    '415': '41',
    '416': '41',
    '417': '41',
    '418': '41',
    '419': '41',
    '441': '44-45',
    '442': '44-45',
    '443': '44-45',
    '444': '44-45',
    '445': '44-45',
    '446': '44-45',
    '447': '44-45',
    '448': '44-45',
    '451': '44-45',
    '452': '44-45',
    '453': '44-45',
    '454': '44-45',
    '481': '48-49',
    '482': '48-49',
    '483': '48-49',
    '484': '48-49',
    '485': '48-49',
    '486': '48-49',
    '488': '48-49',
    '48A': '48-49',
    '491': '48-49',
    '492': '48-49',
    '493': '48-49',
    '511': '51',
    '512': '51',
    '515': '51',
    '517': '51',
    '518': '51',
    '519': '51',
    '521': '52-53',
    '522': '52-53',
    '524': '52-53',
    '52A': '52-53',
    '531': '52-53',
    '532': '52-53',
    '533': '52-53',
    '541': '54',
    '561': '56',
    '562': '56',
    '621': '62',
    '623': '62',
    '624': '62',
    '713': '71',
    '71A': '71',
    '721': '72',
    '722': '72',
    '811': '81',
    '812': '81',
    '813': '81'
}

# Map the aggregation grouping
df_io.loc[df_io['supply_code_agg'].isin(group_list.keys()), 'supply_code_agg'] = df_io.loc[df_io['supply_code_agg'].isin(group_list.keys()), 'supply_code_agg'].map(group_list)
df_io.loc[df_io['use_code_agg'].isin(group_list.keys()), 'use_code_agg'] = df_io.loc[df_io['use_code_agg'].isin(group_list.keys()), 'use_code_agg'].map(group_list)

# Aggregate the data frame at the coarser 3-digit NAICS level
df_io = df_io.groupby(['supply_code_agg', 'use_code_agg', 'year'], as_index=False).agg({'value': 'sum'})

# Create a DataFrame with all possible combinations of codes
all_codes = list(set(df_io['supply_code_agg'].unique()) | set(df_io['use_code_agg'].unique())) + ['capital', 'labor']
df_io_all = pd.DataFrame([(supply, use, year) for supply in all_codes for use in all_codes for year in range(2013, 2019 + 1)], columns=['supply_code_agg', 'use_code_agg', 'year'])
df_io = pd.merge(df_io_all, df_io, on=['supply_code_agg', 'use_code_agg', 'year'], how='left')

# Include the capital and labor costs
df_capital = df[['capital_cost', 'code', 'year']].rename(columns={'code': 'use_code_agg'})
df_capital['supply_code_agg'] = 'capital'
df_capital['capital_cost'] = df_capital['capital_cost'] * 1000
df_io = pd.merge(df_io, df_capital, on=['use_code_agg', 'supply_code_agg', 'year'], how='left')
df_io.loc[(df_io['supply_code_agg'] == 'capital') & ~df_io['use_code_agg'].isin(['capital', 'labor']), 'value'] = df_io.loc[(df_io['supply_code_agg'] == 'capital') & ~df_io['use_code_agg'].isin(['capital', 'labor']), 'capital_cost']
df_io = df_io.drop(columns=['capital_cost'])
df_labor = df[['labor_cost', 'code', 'year']].rename(columns={'code': 'use_code_agg'})
df_labor['supply_code_agg'] = 'labor'
df_labor['labor_cost'] = df_labor['labor_cost'] * 1000
df_io = pd.merge(df_io, df_labor, on=['supply_code_agg', 'use_code_agg', 'year'], how='left')
df_io.loc[(df_io['supply_code_agg'] == 'labor') & ~df_io['use_code_agg'].isin(['capital', 'labor']), 'value'] = df_io.loc[(df_io['supply_code_agg'] == 'labor') & ~df_io['use_code_agg'].isin(['capital', 'labor']), 'labor_cost']
df_io = df_io.drop(columns=['labor_cost'])

# Fill in the missing values with 0
df_io.loc[df_io['value'].isna(), 'value'] = 0

# Calculate the cost share of each industry
df_io['cost_share'] = df_io.groupby(['year', 'use_code_agg'])['value'].transform(lambda x: x / x.sum())
df_io.loc[df_io['cost_share'].isna(), 'cost_share'] = 0

# Sort the data frame by year, use_code_agg, and supply_code_agg
df_io = df_io.sort_values(by=['year', 'use_code_agg', 'supply_code_agg'])

# Create the cost-based IO matrices for each year
df_lambda = pd.DataFrame({'year': df['year'].unique()})
df_lambda['lambda_k'] = np.nan
df_lambda['lambda_l'] = np.nan
for year in df_io['year'].unique():
    df_io_year = df_io[df_io['year'] == year]
    io_matrix = df_io_year.pivot(index='use_code_agg', columns='supply_code_agg', values='cost_share').values
    io_matrix = sparse.csr_matrix(io_matrix)
    b = df.loc[df['year'] == year, ['code', 'va']].sort_values(by=['code'])['va'].values
    b = b / b.sum()
    b = np.append(b, [0, 0])
    lambda_tilde = np.matmul(b.transpose(), np.linalg.inv(np.eye(io_matrix.shape[0]) - io_matrix))
    df_lambda.loc[df_lambda['year'] == year, 'lambda_k'] = lambda_tilde[0, -2]
    df_lambda.loc[df_lambda['year'] == year, 'lambda_l'] = lambda_tilde[0, -1]

########################################################################
# Plot the TFP decomposition                                           # 
########################################################################

# Initialize the figure
fig, ax = plt.subplots(figsize=(8, 5))

# Set the background color of the figure to transparent
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)

# Plot the data
ax.plot(df_1961_2019['year'], 100 * (df_1961_2019['productivity'].cumsum() + df_1961_2019['baumol'].cumsum() + 1), label='Total', color=palette[0], linewidth=2)
ax.plot(df_1961_2019['year'], 100 * (df_1961_2019['productivity'].cumsum() + 1), label='Without Baumol', color=palette[1], linewidth=2)

# Set the horizontal axis
ax.set_xlim(1961, 2020)
ax.set_xticks(range(1965, 2020 + 1, 5))
ax.set_xticklabels(range(1965, 2020 + 1, 5), fontsize=12)

# Set the vertical axis
ax.set_ylim(100, 150)
ax.set_yticks(range(100, 150 + 1, 5))
ax.set_yticklabels(range(100, 150 + 1, 5), fontsize=12)
ax.set_ylabel('Aggregate TFP (1961=100)', fontsize=12, rotation=0, ha='left')
ax.yaxis.set_label_coords(0, 1.01)

# Remove the top and right axes
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, which='major', axis='y', color='gray', linestyle=':', linewidth=0.5)

# Set the legend
ax.legend(frameon=False, fontsize=12)

# Add a note about the data source
ax.text(1, 1.01, 'Source: Statistics Canada', fontsize=8, color='k', ha='right', va='bottom', transform=ax.transAxes)

# Save and close the figure
fig.tight_layout()
fig.savefig(os.path.join(Path(os.getcwd()).parent, 'Figures', 'baumol.png'), transparent=True, dpi=300)
plt.close()

########################################################################
# Tabulate the TFP growth decomposition for 1961-1980 and 1980-2019    # 
########################################################################

# Calculate the different terms for the two periods
productivity_1980_2019= 100 * df_1980_2019['productivity'].cumsum().iloc[-1] / (2019 - 1980)
baumol_1980_2019 = 100 * df_1980_2019['baumol'].cumsum().iloc[-1] / (2019 - 1980)
total_1980_2019 = productivity_1980_2019 + baumol_1980_2019
productivity_1961_1980 = 100 * df_1961_1980['productivity'].cumsum().iloc[-1] / (1980 - 1961)
baumol_1961_1980 = 100 * df_1961_1980['baumol'].cumsum().iloc[-1] / (1980 - 1961)
total_1961_1980 = productivity_1961_1980 + baumol_1961_1980
productivity_1961_2019 = 100 * df_1961_2019['productivity'].cumsum().iloc[-1] / (2019 - 1961)
baumol_1961_2019 = 100 * df_1961_2019['baumol'].cumsum().iloc[-1] / (2019 - 1961)
total_1961_2019 = productivity_1961_2019 + baumol_1961_2019

# Write a table with the TFP growth decomposition
table = open(os.path.join(Path(os.getcwd()).parent, 'Tables', 'tfp_growth.tex'), 'w')
lines = [r'\begin{table}[h]',
         r'\centering',
         r'\begin{threeparttable}',
         r'\caption{TFP Growth Decomposition}',
         r'\begin{tabular}{lcccc}',
         r'\hline',
         r'\hline',
         r'& & 1961-2019 & 1961-1980 & 1980-2019 \\',
         r'\hline',
         r'Productivity & & ' + '{:.2f}'.format(productivity_1961_2019) + r'\% & ' \
                     + '{:.2f}'.format(productivity_1961_1980) + r'\% & ' \
                     + '{:.2f}'.format(productivity_1980_2019) + r'\% \\',
         r'Baumol & & ' + '{:.2f}'.format(baumol_1961_2019) + r'\% & ' \
                     + '{:.2f}'.format(baumol_1961_1980) + r'\% & ' \
                     + '{:.2f}'.format(baumol_1980_2019) + r'\% \\',
         r'Capital & & & & \\',
         r'Labor & & & & \\',
         r'\hline',
         r'Total & & ' + '{:.2f}'.format(total_1961_2019) + r'\% & ' \
                     + '{:.2f}'.format(total_1961_1980) + r'\% & ' \
                     + '{:.2f}'.format(total_1980_2019) + r'\% \\',
         r'\hline',
         r'\hline',
         r'\end{tabular}',
         r'\begin{tablenotes}[flushleft]',
         r'\footnotesize',
         r'\item Note:',
         r'\end{tablenotes}',
         r'\label{tab:tfp_growth}',
         r'\end{threeparttable}',
         r'\end{table}']
table.write('\n'.join(lines))
table.close()

########################################################################
# Plot TFP growth against growth in value-added across industries      # 
########################################################################

# Only keep the relevant years and columns
df_baumol = df.loc[(df['year'] == 1962) | (df['year'] == 2019), ['year', 'industry', 'code', 'tfp', 'va']]

# Calculate the growth rates
df_baumol.loc[:, ['tfp', 'va']] = df_baumol.groupby('code', as_index=False)[['tfp', 'va']].transform(lambda x: np.log(x).diff() / (2019 - 1961))
df_baumol = df_baumol.dropna(subset=['tfp', 'va'])

# Initialize the figure
fig, ax = plt.subplots(figsize=(8, 6))

# Set the background color of the figure to transparent
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)

# Plot the data
ax.scatter(df_baumol['tfp'], df_baumol['va'], color=palette[1], edgecolor='k', linewidths=0.75, s=75)

# Plot the OLS regression line
slope, intercept = np.polyfit(df_baumol['tfp'], df_baumol['va'], 1)
x = np.linspace(-0.02, 0.03, 100)
y = slope * x + intercept
ax.plot(x, y, color=palette[0], linestyle='dotted')

# Set the horizontal axis
ax.set_xlim(-0.02, 0.03)
ax.set_xticks(np.arange(-0.02, 0.03 + 0.001, 0.01))
ax.set_xticklabels([str(x) + r'\%' for x in range(-2, 3 + 1, 1)], fontsize=12)
ax.set_xlabel('Annual TFP growth', fontsize=12)

# Set the vertical axis
ax.set_ylim(0, 0.12)
ax.set_yticks(np.arange(0, 0.12 + 0.01, 0.02))
ax.set_yticklabels([str(x) + r'\%' for x in range(0, 12 + 1, 2)], fontsize=12)
ax.set_ylabel('Annual GDP growth', fontsize=12, rotation=0, ha='left')
ax.yaxis.set_label_coords(0, 1.01)

# Remove the top and right axes
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, which='major', axis='y', color='gray', linestyle=':', linewidth=0.5)

# Identify the oil and gas extraction industry
position_211 = (df_baumol.loc[df_baumol['code'] == '211', 'tfp'].values[0], df_baumol.loc[df_baumol['code'] == '211', 'va'].values[0])
ax.text(position_211[0] + 0.003, position_211[1] - 0.02, 'Oil and gas extraction', fontsize=10, color='k', ha='center', va='center')
ax.annotate('', xy=(position_211[0] + 0.003, position_211[1] - 0.0175), xytext=(position_211[0], position_211[1] - 0.001), arrowprops=dict(arrowstyle='->', color='k', lw=1))

# Identify the computer and electronic product manufacturing industry
position_334 = (df_baumol.loc[df_baumol['code'] == '334', 'tfp'].values[0], df_baumol.loc[df_baumol['code'] == '334', 'va'].values[0])
ax.text(position_334[0] - 0.005, position_334[1] + 0.03, 'Computer and electronic\nproduct manufacturing', fontsize=10, color='k', ha='center', va='center')
ax.annotate('', xy=(position_334[0] - 0.005, position_334[1] + 0.025), xytext=(position_334[0], position_334[1] + 0.001), arrowprops=dict(arrowstyle='->', color='k', lw=1))

# Identify the arts, entertainment and recreation industry
position_71 = (df_baumol.loc[df_baumol['code'] == '71', 'tfp'].values[0], df_baumol.loc[df_baumol['code'] == '71', 'va'].values[0])
ax.text(position_71[0] + 0.0065, position_71[1] + 0.02, 'Arts, entertainment\nand recreation', fontsize=10, color='k', ha='center', va='center')
ax.annotate('', xy=(position_71[0] + 0.0065, position_71[1] + 0.015), xytext=(position_71[0] + 0.00025, position_71[1]), arrowprops=dict(arrowstyle='->', color='k', lw=1))

# Identify the wood product manufacturing industry
position_321 = (df_baumol.loc[df_baumol['code'] == '321', 'tfp'].values[0], df_baumol.loc[df_baumol['code'] == '321', 'va'].values[0])
ax.text(position_321[0] + 0.005, position_321[1] - 0.03, 'Wood product\nmanufacturing', fontsize=10, color='k', ha='center', va='center')
ax.annotate('', xy=(position_321[0] + 0.005, position_321[1] - 0.025), xytext=(position_321[0], position_321[1] - 0.001), arrowprops=dict(arrowstyle='->', color='k', lw=1))

# Add a note about the data source
ax.text(1, 1.01, 'Source: Statistics Canada', fontsize=8, color='k', ha='right', va='bottom', transform=ax.transAxes)

# Save and close the figure
fig.tight_layout()
fig.savefig(os.path.join(Path(os.getcwd()).parent, 'Figures', 'va_tfp_growth.png'), transparent=True, dpi=300)
plt.close()

########################################################################
# Plot TFP growth against growth in prices across industries           # 
########################################################################

# Only keep the relevant years and columns
df_baumol = df.loc[(df['year'] == 1962) | (df['year'] == 2019), ['year', 'industry', 'code', 'tfp', 'va', 'real_va']]

# Calculate the price in each industry
df_baumol['price'] = df_baumol['va'] / df_baumol['real_va']

# Calculate the growth rates
df_baumol.loc[:, ['tfp', 'price']] = df_baumol.groupby('code', as_index=False)[['tfp', 'price']].transform(lambda x: np.log(x).diff() / (2019 - 1961))
df_baumol = df_baumol.dropna(subset=['tfp', 'price'])

# Initialize the figure
fig, ax = plt.subplots(figsize=(8, 6))

# Set the background color of the figure to transparent
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)

# Plot the data
ax.scatter(df_baumol['tfp'], df_baumol['price'], color=palette[1], edgecolor='k', linewidths=0.75, s=75)

# Plot the OLS regression line
slope, intercept = np.polyfit(df_baumol['tfp'], df_baumol['price'], 1)
x = np.linspace(-0.02, 0.03, 100)
y = slope * x + intercept
ax.plot(x, y, color=palette[0], linestyle='dotted')

# Set the horizontal axis
ax.set_xlim(-0.02, 0.03)
ax.set_xticks(np.arange(-0.02, 0.03 + 0.001, 0.01))
ax.set_xticklabels([str(x) + r'\%' for x in range(-2, 3 + 1, 1)], fontsize=12)
ax.set_xlabel('Annual TFP growth', fontsize=12)

# Set the vertical axis
ax.set_ylim(0, 0.06)
ax.set_yticks(np.arange(0, 0.06 + 0.01, 0.01))
ax.set_yticklabels([str(x) + r'\%' for x in range(0, 6 + 1, 1)], fontsize=12)
ax.set_ylabel('Annual price growth', fontsize=12, rotation=0, ha='left')
ax.yaxis.set_label_coords(0, 1.01)

# Remove the top and right axes
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, which='major', axis='y', color='gray', linestyle=':', linewidth=0.5)

# Identify the oil and gas extraction industry
position_211 = (df_baumol.loc[df_baumol['code'] == '211', 'tfp'].values[0], df_baumol.loc[df_baumol['code'] == '211', 'price'].values[0])
ax.text(position_211[0] + 0.0035, position_211[1] - 0.02, 'Oil and gas extraction', fontsize=10, color='k', ha='center', va='center')
ax.annotate('', xy=(position_211[0] + 0.003, position_211[1] - 0.0175), xytext=(position_211[0], position_211[1] - 0.0005), arrowprops=dict(arrowstyle='->', color='k', lw=1))

# Identify the computer and electronic product manufacturing industry
position_334 = (df_baumol.loc[df_baumol['code'] == '334', 'tfp'].values[0], df_baumol.loc[df_baumol['code'] == '334', 'price'].values[0])
ax.text(position_334[0] - 0.0125, position_334[1], 'Computer and electronic\nproduct manufacturing', fontsize=10, color='k', ha='center', va='center')
ax.annotate('', xy=(position_334[0] - 0.007, position_334[1]), xytext=(position_334[0] - 0.00025, position_334[1]), arrowprops=dict(arrowstyle='->', color='k', lw=1))

# Identify the arts, entertainment and recreation industry
position_71 = (df_baumol.loc[df_baumol['code'] == '71', 'tfp'].values[0], df_baumol.loc[df_baumol['code'] == '71', 'price'].values[0])
ax.text(position_71[0] + 0.01, position_71[1] + 0.0075, 'Arts, entertainment\nand recreation', fontsize=10, color='k', ha='center', va='center')
ax.annotate('', xy=(position_71[0] + 0.01, position_71[1] + 0.005), xytext=(position_71[0] + 0.00025, position_71[1]), arrowprops=dict(arrowstyle='->', color='k', lw=1))

# Identify the wood product manufacturing industry
position_321 = (df_baumol.loc[df_baumol['code'] == '321', 'tfp'].values[0], df_baumol.loc[df_baumol['code'] == '321', 'price'].values[0])
ax.text(position_321[0] + 0.005, position_321[1] + 0.01, 'Wood product\nmanufacturing', fontsize=10, color='k', ha='center', va='center')
ax.annotate('', xy=(position_321[0] + 0.005, position_321[1] + 0.0075), xytext=(position_321[0], position_321[1] + 0.0005), arrowprops=dict(arrowstyle='->', color='k', lw=1))

# Add a note about the data source
ax.text(1, 1.01, 'Source: Statistics Canada', fontsize=8, color='k', ha='right', va='bottom', transform=ax.transAxes)

# Save and close the figure
fig.tight_layout()
fig.savefig(os.path.join(Path(os.getcwd()).parent, 'Figures', 'price_tfp_growth.png'), transparent=True, dpi=300)
plt.close()

########################################################################
# Plot real GDP growth against growth in prices across industries      # 
########################################################################

# Only keep the relevant years and columns
df_baumol = df.loc[(df['year'] == 1962) | (df['year'] == 2019), ['year', 'industry', 'code', 'tfp', 'va', 'real_va']]

# Calculate the price in each industry
df_baumol['price'] = df_baumol['va'] / df_baumol['real_va']

# Calculate the growth rates
df_baumol.loc[:, ['real_va', 'price']] = df_baumol.groupby('code', as_index=False)[['real_va', 'price']].transform(lambda x: np.log(x).diff() / (2019 - 1961))
df_baumol = df_baumol.dropna(subset=['real_va', 'price'])

# Initialize the figure
fig, ax = plt.subplots(figsize=(8, 6))

# Set the background color of the figure to transparent
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)

# Plot the data
ax.scatter(df_baumol['real_va'], df_baumol['price'], color=palette[1], edgecolor='k', linewidths=0.75, s=75)

# Plot the OLS regression line
slope, intercept = np.polyfit(df_baumol['real_va'], df_baumol['price'], 1)
x = np.linspace(-0.02, 0.06, 100)
y = slope * x + intercept
ax.plot(x, y, color=palette[0], linestyle='dotted')

# Set the horizontal axis
ax.set_xlim(-0.02, 0.06)
ax.set_xticks(np.arange(-0.02, 0.06 + 0.001, 0.01))
ax.set_xticklabels([str(x) + r'\%' for x in range(-2, 6 + 1, 1)], fontsize=12)
ax.set_xlabel('Annual real GDP growth', fontsize=12)

# Set the vertical axis
ax.set_ylim(0, 0.06)
ax.set_yticks(np.arange(0, 0.06 + 0.01, 0.01))
ax.set_yticklabels([str(x) + r'\%' for x in range(0, 6 + 1, 1)], fontsize=12)
ax.set_ylabel('Annual price growth', fontsize=12, rotation=0, ha='left')
ax.yaxis.set_label_coords(0, 1.01)

# Remove the top and right axes
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, which='major', axis='y', color='gray', linestyle=':', linewidth=0.5)

# Add a note about the data source
ax.text(1, 1.01, 'Source: Statistics Canada', fontsize=8, color='k', ha='right', va='bottom', transform=ax.transAxes)

# Save and close the figure
fig.tight_layout()
fig.savefig(os.path.join(Path(os.getcwd()).parent, 'Figures', 'price_real_va_growth.png'), transparent=True, dpi=300)
plt.close()

########################################################################
# Plot TFP growth against growth in real value-added across industries # 
########################################################################

# Only keep the relevant years and columns
df_baumol = df.loc[(df['year'] == 1962) | (df['year'] == 2019), ['year', 'industry', 'code', 'tfp', 'real_va']]

# Calculate the growth rates
df_baumol.loc[:, ['tfp', 'real_va']] = df_baumol.groupby('code', as_index=False)[['tfp', 'real_va']].transform(lambda x: np.log(x).diff() / (2019 - 1961))
df_baumol = df_baumol.dropna(subset=['tfp', 'real_va'])

# Initialize the figure
fig, ax = plt.subplots(figsize=(8, 6))

# Set the background color of the figure to transparent
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)

# Plot the data
ax.scatter(df_baumol['tfp'], df_baumol['real_va'], color=palette[1], edgecolor='k', linewidths=0.75, s=75)

# Plot the OLS regression line
slope, intercept = np.polyfit(df_baumol['tfp'], df_baumol['real_va'], 1)
x = np.linspace(-0.02, 0.03, 100)
y = slope * x + intercept
ax.plot(x, y, color=palette[0], linestyle='dotted')

# Set the horizontal axis
ax.set_xlim(-0.02, 0.03)
ax.set_xticks(np.arange(-0.02, 0.03 + 0.001, 0.01))
ax.set_xticklabels([str(x) + r'\%' for x in range(-2, 3 + 1, 1)], fontsize=12)
ax.set_xlabel('Annual TFP growth', fontsize=12)

# Set the vertical axis
ax.set_ylim(-0.02, 0.1)
ax.set_yticks(np.arange(-0.02, 0.1 + 0.01, 0.02))
ax.set_yticklabels([str(x) + r'\%' for x in range(-2, 10 + 1, 2)], fontsize=12)
ax.set_ylabel('Annual real GDP growth', fontsize=12, rotation=0, ha='left')
ax.yaxis.set_label_coords(0, 1.01)

# Remove the top and right axes
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, which='major', axis='y', color='gray', linestyle=':', linewidth=0.5)

# Identify the oil and gas extraction industry
position_211 = (df_baumol.loc[df_baumol['code'] == '211', 'tfp'].values[0], df_baumol.loc[df_baumol['code'] == '211', 'real_va'].values[0])
ax.text(position_211[0] + 0.004, position_211[1] - 0.025, 'Oil and gas extraction', fontsize=10, color='k', ha='center', va='center')
ax.annotate('', xy=(position_211[0] + 0.004, position_211[1] - 0.0225), xytext=(position_211[0], position_211[1] - 0.001), arrowprops=dict(arrowstyle='->', color='k', lw=1))

# Identify the computer and electronic product manufacturing industry
position_334 = (df_baumol.loc[df_baumol['code'] == '334', 'tfp'].values[0], df_baumol.loc[df_baumol['code'] == '334', 'real_va'].values[0])
ax.text(position_334[0] - 0.005, position_334[1] + 0.03, 'Computer and electronic\nproduct manufacturing', fontsize=10, color='k', ha='center', va='center')
ax.annotate('', xy=(position_334[0] - 0.005, position_334[1] + 0.025), xytext=(position_334[0], position_334[1] + 0.00075), arrowprops=dict(arrowstyle='->', color='k', lw=1))

# Identify the arts, entertainment and recreation industry
position_71 = (df_baumol.loc[df_baumol['code'] == '71', 'tfp'].values[0], df_baumol.loc[df_baumol['code'] == '71', 'real_va'].values[0])
ax.text(position_71[0] + 0.0065, position_71[1] + 0.025, 'Arts, entertainment\nand recreation', fontsize=10, color='k', ha='center', va='center')
ax.annotate('', xy=(position_71[0] + 0.0065, position_71[1] + 0.02), xytext=(position_71[0] + 0.0003, position_71[1]), arrowprops=dict(arrowstyle='->', color='k', lw=1))

# Identify the wood product manufacturing industry
position_321 = (df_baumol.loc[df_baumol['code'] == '321', 'tfp'].values[0], df_baumol.loc[df_baumol['code'] == '321', 'real_va'].values[0])
ax.text(position_321[0] + 0.005, position_321[1] - 0.02, 'Wood product\nmanufacturing', fontsize=10, color='k', ha='center', va='center')
ax.annotate('', xy=(position_321[0] + 0.005, position_321[1] - 0.015), xytext=(position_321[0], position_321[1] - 0.00125), arrowprops=dict(arrowstyle='->', color='k', lw=1))

# Add a note about the data source
ax.text(1, 1.01, 'Source: Statistics Canada', fontsize=8, color='k', ha='right', va='bottom', transform=ax.transAxes)

# Save and close the figure
fig.tight_layout()
fig.savefig(os.path.join(Path(os.getcwd()).parent, 'Figures', 'real_va_tfp_growth.png'), transparent=True, dpi=300)
plt.close()

########################################################################
# Plot TFP growth against growth in wages across industries            # 
########################################################################

# Only keep the relevant years and columns
df_baumol = df.loc[(df['year'] == 1962) | (df['year'] == 2019), ['year', 'industry', 'code', 'tfp', 'labor_cost', 'hours']]

# Calculate the average wage within each industry
df_baumol['wage'] = df_baumol['labor_cost'] / df_baumol['hours']

# Calculate the growth rates
df_baumol.loc[:, ['tfp', 'wage']] = df_baumol.groupby('code', as_index=False)[['tfp', 'wage']].transform(lambda x: np.log(x).diff() / (2019 - 1961))
df_baumol = df_baumol.dropna(subset=['tfp', 'wage'])

# Initialize the figure
fig, ax = plt.subplots(figsize=(8, 6))

# Set the background color of the figure to transparent
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)

# Plot the data
ax.scatter(df_baumol['tfp'], df_baumol['wage'], color=palette[1], edgecolor='k', linewidths=0.75, s=75)

# Plot the OLS regression line
slope, intercept = np.polyfit(df_baumol['tfp'], df_baumol['wage'], 1)
x = np.linspace(-0.02, 0.03, 100)
y = slope * x + intercept
ax.plot(x, y, color=palette[0], linestyle='dotted')

# Set the horizontal axis
ax.set_xlim(-0.02, 0.03)
ax.set_xticks(np.arange(-0.02, 0.03 + 0.001, 0.01))
ax.set_xticklabels([str(x) + r'\%' for x in range(-2, 3 + 1, 1)], fontsize=12)
ax.set_xlabel('Annual TFP growth', fontsize=12)

# Set the vertical axis
ax.set_ylim(0.02, 0.08)
ax.set_yticks(np.arange(0.02, 0.08 + 0.01, 0.01))
ax.set_yticklabels([str(x) + r'\%' for x in range(2, 8 + 1, 1)], fontsize=12)
ax.set_ylabel('Annual wage growth', fontsize=12, rotation=0, ha='left')
ax.yaxis.set_label_coords(0, 1.01)

# Remove the top and right axes
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, which='major', axis='y', color='gray', linestyle=':', linewidth=0.5)

# Identify the oil and gas extraction industry
position_211 = (df_baumol.loc[df_baumol['code'] == '211', 'tfp'].values[0], df_baumol.loc[df_baumol['code'] == '211', 'wage'].values[0])
ax.text(position_211[0] + 0.004, position_211[1] - 0.015, 'Oil and gas extraction', fontsize=10, color='k', ha='center', va='center')
ax.annotate('', xy=(position_211[0] + 0.004, position_211[1] - 0.0135), xytext=(position_211[0], position_211[1] - 0.0005), arrowprops=dict(arrowstyle='->', color='k', lw=1))

# Identify the computer and electronic product manufacturing industry
position_334 = (df_baumol.loc[df_baumol['code'] == '334', 'tfp'].values[0], df_baumol.loc[df_baumol['code'] == '334', 'wage'].values[0])
ax.text(position_334[0] - 0.005, position_334[1] + 0.01, 'Computer and electronic\nproduct manufacturing', fontsize=10, color='k', ha='center', va='center')
ax.annotate('', xy=(position_334[0] - 0.005, position_334[1] + 0.0075), xytext=(position_334[0], position_334[1] + 0.0005), arrowprops=dict(arrowstyle='->', color='k', lw=1))

# Identify the arts, entertainment and recreation industry
position_71 = (df_baumol.loc[df_baumol['code'] == '71', 'tfp'].values[0], df_baumol.loc[df_baumol['code'] == '71', 'wage'].values[0])
ax.text(position_71[0] + 0.0065, position_71[1] + 0.015, 'Arts, entertainment\nand recreation', fontsize=10, color='k', ha='center', va='center')
ax.annotate('', xy=(position_71[0] + 0.0065, position_71[1] + 0.0125), xytext=(position_71[0] + 0.0003, position_71[1]), arrowprops=dict(arrowstyle='->', color='k', lw=1))

# Identify the wood product manufacturing industry
position_321 = (df_baumol.loc[df_baumol['code'] == '321', 'tfp'].values[0], df_baumol.loc[df_baumol['code'] == '321', 'wage'].values[0])
ax.text(position_321[0] + 0.005, position_321[1] - 0.015, 'Wood product\nmanufacturing', fontsize=10, color='k', ha='center', va='center')
ax.annotate('', xy=(position_321[0] + 0.005, position_321[1] - 0.0125), xytext=(position_321[0], position_321[1] - 0.0005), arrowprops=dict(arrowstyle='->', color='k', lw=1))

# Add a note about the data source
ax.text(1, 1.01, 'Source: Statistics Canada', fontsize=8, color='k', ha='right', va='bottom', transform=ax.transAxes)

# Save and close the figure
fig.tight_layout()
fig.savefig(os.path.join(Path(os.getcwd()).parent, 'Figures', 'wage_tfp_growth.png'), transparent=True, dpi=300)
plt.close()

########################################################################
# Plot TFP growth against growth in value-added across industries for  #
# the two periods of the analysis                                      # 
########################################################################

# Only keep the relevant years and columns
df_baumol_1 = df.loc[(df['year'] == 1962) | (df['year'] == 1980), ['year', 'industry', 'code', 'tfp', 'va']]
df_baumol_2 = df.loc[(df['year'] == 1980) | (df['year'] == 2019), ['year', 'industry', 'code', 'tfp', 'va']]

# Calculate the growth rates
df_baumol_1.loc[:, ['tfp', 'va']] = df_baumol_1.groupby('code', as_index=False)[['tfp', 'va']].transform(lambda x: np.log(x).diff() / (1980 - 1961))
df_baumol_1 = df_baumol_1.dropna(subset=['tfp', 'va'])
df_baumol_2.loc[:, ['tfp', 'va']] = df_baumol_2.groupby('code', as_index=False)[['tfp', 'va']].transform(lambda x: np.log(x).diff() / (2019 - 1980))
df_baumol_2 = df_baumol_2.dropna(subset=['tfp', 'va'])

# Initialize the figure
fig, ax = plt.subplots(figsize=(8, 6))

# Set the background color of the figure to transparent
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)

# Plot the data
ax.scatter(df_baumol_1['tfp'], df_baumol_1['va'], color=palette[1], edgecolor='k', linewidths=0.75, s=75, label='1961-1980')
ax.scatter(df_baumol_2['tfp'], df_baumol_2['va'], color=palette[2], edgecolor='k', linewidths=0.75, s=75, label='1980-2019')

# Plot the OLS regression lines
x = np.linspace(-0.03, 0.04, 100)
slope_1, intercept_1 = np.polyfit(df_baumol_1['tfp'], df_baumol_1['va'], 1)
slope_2, intercept_2 = np.polyfit(df_baumol_2['tfp'], df_baumol_2['va'], 1)
y_1 = slope_1 * x + intercept_1
y_2 = slope_2 * x + intercept_2
ax.plot(x, y_1, color=palette[1], linestyle='dotted')
ax.plot(x, y_2, color=palette[2], linestyle='dotted')

# Set the horizontal axis
ax.set_xlim(-0.03, 0.04)
ax.set_xticks(np.arange(-0.03, 0.04 + 0.001, 0.01))
ax.set_xticklabels([str(x) + r'\%' for x in range(-3, 4 + 1, 1)], fontsize=12)
ax.set_xlabel('Annual TFP growth', fontsize=12)

# Set the vertical axis
ax.set_ylim(-0.02, 0.18)
ax.set_yticks(np.arange(-0.02, 0.18 + 0.01, 0.02))
ax.set_yticklabels([str(x) + r'\%' for x in range(-2, 18 + 1, 2)], fontsize=12)
ax.set_ylabel('Annual GDP growth', fontsize=12, rotation=0, ha='left')
ax.yaxis.set_label_coords(0, 1.01)

# Remove the top and right axes
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, which='major', axis='y', color='gray', linestyle=':', linewidth=0.5)

# Set the legend
ax.legend(loc='upper right', fontsize=12, frameon=False, markerscale=1.5, handlelength=1.5, handletextpad=0.5, borderpad=0.5)

# Add a note about the data source
ax.text(1, 1.01, 'Source: Statistics Canada', fontsize=8, color='k', ha='right', va='bottom', transform=ax.transAxes)

# Save and close the figure
fig.tight_layout()
fig.savefig(os.path.join(Path(os.getcwd()).parent, 'Figures', 'va_tfp_growth_period.png'), transparent=True, dpi=300)
plt.close()

########################################################################
# Plot TFP growth against growth in real value-added across industries #
# for the two periods of the analysis                                  # 
########################################################################

# Only keep the relevant years and columns
df_baumol_1 = df.loc[(df['year'] == 1962) | (df['year'] == 1980), ['year', 'industry', 'code', 'tfp', 'real_va']]
df_baumol_2 = df.loc[(df['year'] == 1980) | (df['year'] == 2019), ['year', 'industry', 'code', 'tfp', 'real_va']]

# Calculate the growth rates
df_baumol_1.loc[:, ['tfp', 'real_va']] = df_baumol_1.groupby('code', as_index=False)[['tfp', 'real_va']].transform(lambda x: np.log(x).diff() / (1980 - 1961))
df_baumol_1 = df_baumol_1.dropna(subset=['tfp', 'real_va'])
df_baumol_2.loc[:, ['tfp', 'real_va']] = df_baumol_2.groupby('code', as_index=False)[['tfp', 'real_va']].transform(lambda x: np.log(x).diff() / (2019 - 1980))
df_baumol_2 = df_baumol_2.dropna(subset=['tfp', 'real_va'])

# Initialize the figure
fig, ax = plt.subplots(figsize=(8, 6))

# Set the background color of the figure to transparent
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)

# Plot the data
ax.scatter(df_baumol_1['tfp'], df_baumol_1['real_va'], color=palette[1], edgecolor='k', linewidths=0.75, s=75, label='1961-1980')
ax.scatter(df_baumol_2['tfp'], df_baumol_2['real_va'], color=palette[2], edgecolor='k', linewidths=0.75, s=75, label='1980-2019')

# Plot the OLS regression lines
x = np.linspace(-0.03, 0.04, 100)
slope_1, intercept_1 = np.polyfit(df_baumol_1['tfp'], df_baumol_1['real_va'], 1)
slope_2, intercept_2 = np.polyfit(df_baumol_2['tfp'], df_baumol_2['real_va'], 1)
y_1 = slope_1 * x + intercept_1
y_2 = slope_2 * x + intercept_2
ax.plot(x, y_1, color=palette[1], linestyle='dotted')
ax.plot(x, y_2, color=palette[2], linestyle='dotted')

# Set the horizontal axis
ax.set_xlim(-0.03, 0.04)
ax.set_xticks(np.arange(-0.03, 0.04 + 0.001, 0.01))
ax.set_xticklabels([str(x) + r'\%' for x in range(-3, 4 + 1, 1)], fontsize=12)
ax.set_xlabel('Annual TFP growth', fontsize=12)

# Set the vertical axis
ax.set_ylim(-0.08, 0.12)
ax.set_yticks(np.arange(-0.08, 0.12 + 0.01, 0.02))
ax.set_yticklabels([str(x) + r'\%' for x in range(-8, 12 + 1, 2)], fontsize=12)
ax.set_ylabel('Annual real GDP growth', fontsize=12, rotation=0, ha='left')
ax.yaxis.set_label_coords(0, 1.01)

# Remove the top and right axes
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, which='major', axis='y', color='gray', linestyle=':', linewidth=0.5)

# Set the legend
ax.legend(loc='lower right', fontsize=12, frameon=False, markerscale=1.5, handlelength=1.5, handletextpad=0.5, borderpad=0.5)

# Add a note about the data source
ax.text(1, 1.01, 'Source: Statistics Canada', fontsize=8, color='k', ha='right', va='bottom', transform=ax.transAxes)

# Save and close the figure
fig.tight_layout()
fig.savefig(os.path.join(Path(os.getcwd()).parent, 'Figures', 'real_va_tfp_growth_period.png'), transparent=True, dpi=300)
plt.close()

########################################################################
# Plot TFP growth against growth in prices across industries for the   #
# two periods of the analysis                                          # 
########################################################################

# Only keep the relevant years and columns
df_baumol_1 = df.loc[(df['year'] == 1962) | (df['year'] == 1980), ['year', 'industry', 'code', 'tfp', 'va', 'real_va']]
df_baumol_2 = df.loc[(df['year'] == 1980) | (df['year'] == 2019), ['year', 'industry', 'code', 'tfp', 'va', 'real_va']]

# Calculate the price in each industry
df_baumol_1['price'] = df_baumol_1['va'] / df_baumol_1['real_va']
df_baumol_2['price'] = df_baumol_2['va'] / df_baumol_2['real_va']

# Calculate the growth rates
df_baumol_1.loc[:, ['tfp', 'price']] = df_baumol_1.groupby('code', as_index=False)[['tfp', 'price']].transform(lambda x: np.log(x).diff() / (1980 - 1961))
df_baumol_1 = df_baumol_1.dropna(subset=['tfp', 'price'])
df_baumol_2.loc[:, ['tfp', 'price']] = df_baumol_2.groupby('code', as_index=False)[['tfp', 'price']].transform(lambda x: np.log(x).diff() / (2019 - 1980))
df_baumol_2 = df_baumol_2.dropna(subset=['tfp', 'price'])

# Initialize the figure
fig, ax = plt.subplots(figsize=(8, 6))

# Set the background color of the figure to transparent
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)

# Plot the data
ax.scatter(df_baumol_1['tfp'], df_baumol_1['price'], color=palette[1], edgecolor='k', linewidths=0.75, s=75, label='1961-1980')
ax.scatter(df_baumol_2['tfp'], df_baumol_2['price'], color=palette[2], edgecolor='k', linewidths=0.75, s=75, label='1980-2019')

# Plot the OLS regression lines
x = np.linspace(-0.03, 0.04, 100)
slope_1, intercept_1 = np.polyfit(df_baumol_1['tfp'], df_baumol_1['price'], 1)
slope_2, intercept_2 = np.polyfit(df_baumol_2['tfp'], df_baumol_2['price'], 1)
y_1 = slope_1 * x + intercept_1
y_2 = slope_2 * x + intercept_2
ax.plot(x, y_1, color=palette[1], linestyle='dotted')
ax.plot(x, y_2, color=palette[2], linestyle='dotted')

# Set the horizontal axis
ax.set_xlim(-0.03, 0.04)
ax.set_xticks(np.arange(-0.03, 0.04 + 0.001, 0.01))
ax.set_xticklabels([str(x) + r'\%' for x in range(-3, 4 + 1, 1)], fontsize=12)
ax.set_xlabel('Annual TFP growth', fontsize=12)

# Set the vertical axis
ax.set_ylim(-0.02, 0.14)
ax.set_yticks(np.arange(-0.02, 0.14 + 0.01, 0.02))
ax.set_yticklabels([str(x) + r'\%' for x in range(-2, 14 + 1, 2)], fontsize=12)
ax.set_ylabel('Annual price growth', fontsize=12, rotation=0, ha='left')
ax.yaxis.set_label_coords(0, 1.01)

# Remove the top and right axes
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, which='major', axis='y', color='gray', linestyle=':', linewidth=0.5)

# Set the legend
ax.legend(loc='upper right', fontsize=12, frameon=False, markerscale=1.5, handlelength=1.5, handletextpad=0.5, borderpad=0.5)

# Add a note about the data source
ax.text(1, 1.01, 'Source: Statistics Canada', fontsize=8, color='k', ha='right', va='bottom', transform=ax.transAxes)

# Save and close the figure
fig.tight_layout()
fig.savefig(os.path.join(Path(os.getcwd()).parent, 'Figures', 'price_tfp_growth_period.png'), transparent=True, dpi=300)
plt.close()

########################################################################
# Plot TFP growth against growth in wages across industries for the    #
# two periods of the analysis                                          # 
########################################################################

# Only keep the relevant years and columns
df_baumol_1 = df.loc[(df['year'] == 1962) | (df['year'] == 1980), ['year', 'industry', 'code', 'tfp', 'labor_cost', 'hours']]
df_baumol_2 = df.loc[(df['year'] == 1980) | (df['year'] == 2019), ['year', 'industry', 'code', 'tfp', 'labor_cost', 'hours']]

# Calculate the average wage within each industry
df_baumol_1['wage'] = df_baumol_1['labor_cost'] / df_baumol_1['hours']
df_baumol_2['wage'] = df_baumol_2['labor_cost'] / df_baumol_2['hours']

# Calculate the growth rates
df_baumol_1.loc[:, ['tfp', 'wage']] = df_baumol_1.groupby('code', as_index=False)[['tfp', 'wage']].transform(lambda x: np.log(x).diff() / (1980 - 1961))
df_baumol_1 = df_baumol_1.dropna(subset=['tfp', 'wage'])
df_baumol_2.loc[:, ['tfp', 'wage']] = df_baumol_2.groupby('code', as_index=False)[['tfp', 'wage']].transform(lambda x: np.log(x).diff() / (2019 - 1980))
df_baumol_2 = df_baumol_2.dropna(subset=['tfp', 'wage'])

# Initialize the figure
fig, ax = plt.subplots(figsize=(8, 6))

# Set the background color of the figure to transparent
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)

# Plot the data
ax.scatter(df_baumol_1['tfp'], df_baumol_1['wage'], color=palette[1], edgecolor='k', linewidths=0.75, s=75, label='1961-1980')
ax.scatter(df_baumol_2['tfp'], df_baumol_2['wage'], color=palette[2], edgecolor='k', linewidths=0.75, s=75, label='1980-2019')

# Plot the OLS regression lines
x = np.linspace(-0.03, 0.04, 100)
slope_1, intercept_1 = np.polyfit(df_baumol_1['tfp'], df_baumol_1['wage'], 1)
slope_2, intercept_2 = np.polyfit(df_baumol_2['tfp'], df_baumol_2['wage'], 1)
y_1 = slope_1 * x + intercept_1
y_2 = slope_2 * x + intercept_2
ax.plot(x, y_1, color=palette[1], linestyle='dotted')
ax.plot(x, y_2, color=palette[2], linestyle='dotted')

# Set the horizontal axis
ax.set_xlim(-0.03, 0.04)
ax.set_xticks(np.arange(-0.03, 0.04 + 0.001, 0.01))
ax.set_xticklabels([str(x) + r'\%' for x in range(-3, 4 + 1, 1)], fontsize=12)
ax.set_xlabel('Annual TFP growth', fontsize=12)

# Set the vertical axis
ax.set_ylim(-0.02, 0.14)
ax.set_yticks(np.arange(-0.02, 0.14 + 0.01, 0.02))
ax.set_yticklabels([str(x) + r'\%' for x in range(-2, 14 + 1, 2)], fontsize=12)
ax.set_ylabel('Annual wage growth', fontsize=12, rotation=0, ha='left')
ax.yaxis.set_label_coords(0, 1.01)

# Remove the top and right axes
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, which='major', axis='y', color='gray', linestyle=':', linewidth=0.5)

# Set the legend
ax.legend(loc='lower right', fontsize=12, frameon=False, markerscale=1.5, handlelength=1.5, handletextpad=0.5, borderpad=0.5)

# Add a note about the data source
ax.text(1, 1.01, 'Source: Statistics Canada', fontsize=8, color='k', ha='right', va='bottom', transform=ax.transAxes)

# Save and close the figure
fig.tight_layout()
fig.savefig(os.path.join(Path(os.getcwd()).parent, 'Figures', 'wage_tfp_growth_period.png'), transparent=True, dpi=300)
plt.close()