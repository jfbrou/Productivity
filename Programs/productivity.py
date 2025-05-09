import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
from stats_can import StatsCan
from datetime import datetime

# Initialize the StatsCan API
sc = StatsCan()

# Set the figures' font
rc('font', **{'family': 'serif', 'serif': ['Palatino']})
rc('text', usetex=True)

# Define my color palette
palette = ['#002855', '#26d07c', '#ff585d', '#f3d03e', '#0072ce', '#eb6fbd', '#00aec7', '#888b8d']

########################################################################
# Prepare the Statistics Canada data                                   # 
########################################################################

# Retrieve the data from Table 36-10-0217-01
df = sc.table_to_df('36-10-0217-01')

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
    'Multifactor productivity based on value-added',
    'Labour input',
    'Capital input',
    'Gross domestic product (GDP)', 
    'Labour compensation',
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
    'Multifactor productivity based on value-added': 'tfp',
    'Labour input': 'labor',
    'Capital input': 'capital',
    'Gross domestic product (GDP)': 'va',
    'Labour compensation': 'labor_cost',
    'Capital cost': 'capital_cost'
})

# Recode the date column to year
df['year'] = df['date'].dt.year
df = df.drop(columns=['date'])
df = df[df['year'] < 2020]

# Calculate the share of value-added of each industry within year
df['va_agg'] = df.groupby('year')['va'].transform('sum')
df['b'] = df['va'] / df['va_agg']
df['b'] = df.groupby('industry')['b'].transform(lambda x: x.rolling(2).mean())
df = df.drop(columns=['va_agg'])

# Calculate the share of value-added of each industry for years 1962, 1980, and 2000
df = pd.merge(df, df.loc[df['year'] == 1962, ['industry', 'b']].rename(columns={'b': 'b_1962'}), on='industry', how='left')
df = pd.merge(df, df.loc[df['year'] == 1980, ['industry', 'b']].rename(columns={'b': 'b_1980'}), on='industry', how='left')
df = pd.merge(df, df.loc[df['year'] == 2000, ['industry', 'b']].rename(columns={'b': 'b_2000'}), on='industry', how='left')

# Calculate the log difference of TFP within each industry
df['tfp_growth'] = df.groupby('industry')['tfp'].transform(lambda x: np.log(x).diff())

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
df_1962_2019 = pd.DataFrame({'year': range(1961, 2019 + 1)})
df['within'] = df['b_1962'] * df['tfp_growth']
df_1962_2019 = pd.merge(df_1962_2019, df.groupby('year', as_index=False).agg({'within': 'sum'}).rename(columns={'within': 'productivity'}), on='year', how='left')
df['between'] = (df['b'] - df['b_1962']) * df['tfp_growth']
df_1962_2019 = pd.merge(df_1962_2019, df.groupby('year', as_index=False).agg({'between': 'sum'}).rename(columns={'between': 'baumol'}), on='year', how='left')
df = df.drop(columns=['within', 'between'])

########################################################################
# Plot the TFP decomposition                                           # 
########################################################################

# Initialize the figure
fig, ax = plt.subplots(figsize=(8, 4))

# Set the background color of the figure to transparent
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)

# Plot the data
ax.plot(df_1962_2019['year'], 100 * (df_1962_2019['productivity'].cumsum() + df_1962_2019['baumol'].cumsum() + 1), label='Total', color=palette[0], linewidth=2)
ax.plot(df_1962_2019['year'], 100 * (df_1962_2019['productivity'].cumsum() + 1), label='Without Baumol', color=palette[1], linewidth=2)

# Set the horizontal axis
ax.set_xlim(1961, 2019)
ax.set_xticks(range(1965, 2015 + 1, 5))
ax.set_xticklabels(range(1965, 2015 + 1, 5), fontsize=12)

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