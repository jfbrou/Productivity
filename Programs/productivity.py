import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
from stats_can import StatsCan
from datetime import datetime
from scipy import sparse
import xlrd

# Initialize the StatsCan API
sc = StatsCan()

# Set the figures' font
rc('font', **{'family': 'serif', 'serif': ['Palatino']})
rc('text', usetex=True)

# Define my color palette
palette = ['#002855', '#26d07c', '#ff585d', '#f3d03e', '#0072ce', '#eb6fbd', '#00aec7', '#888b8d']

# Define a dictionary to map industry names to their corresponding NAICS codes
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
# Prepare the Table 36-10-0217-01 Statistics Canada data               # 
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
    'Multifactor productivity based on value-added': 'tfp',
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
df['naics'] = df['industry'].map(industry_to_naics)

# Rescale the variables to 1961=100
df['tfp'] = df['tfp'] / df.loc[df['year'] == 1961, 'tfp'].values[0] * 100
df['real_va'] = df['real_va'] / df.loc[df['year'] == 1961, 'real_va'].values[0] * 100
df['capital'] = df['capital'] / df.loc[df['year'] == 1961, 'capital'].values[0] * 100
df['labor'] = df['labor'] / df.loc[df['year'] == 1961, 'labor'].values[0] * 100

# Calculate prices and wages
df['price'] = df['va'] / df['real_va']
df['wage'] = df['labor_cost'] / df['hours']
df['capital_price'] = df['capital_cost'] / df['capital']

# Calculate the share of value-added of each industry within year
df['va_agg'] = df.groupby('year')['va'].transform('sum')
df['b'] = df['va'] / df['va_agg']
df['b'] = df.groupby('naics')['b'].transform(lambda x: x.rolling(2).mean())
df = df.drop(columns=['va_agg'])

# Calculate the share of value-added of each industry for years 1961, 1980, and 2000
df = pd.merge(df, df.loc[df['year'] == 1962, ['naics', 'b']].rename(columns={'b': 'b_1961'}), on='naics', how='left')
df = pd.merge(df, df.loc[df['year'] == 1980, ['naics', 'b']].rename(columns={'b': 'b_1980'}), on='naics', how='left')
df = pd.merge(df, df.loc[df['year'] == 2000, ['naics', 'b']].rename(columns={'b': 'b_2000'}), on='naics', how='left')

# Calculate the log difference of TFP, capital, and labor within each industry
df['tfp_growth'] = df.groupby('naics')['tfp'].transform(lambda x: np.log(x).diff())
df['capital_growth'] = df.groupby('naics')['capital'].transform(lambda x: np.log(x).diff())
df['labor_growth'] = df.groupby('naics')['labor'].transform(lambda x: np.log(x).diff())

# Calculate the industry-level output elasticities of capital and labor
df['alpha_k'] = df['capital_cost'] / (df['capital_cost'] + df['labor_cost'])
df['alpha_k'] = df.groupby('naics')['alpha_k'].transform(lambda x: x.rolling(2).mean())
df['alpha_l'] = df['labor_cost'] / (df['capital_cost'] + df['labor_cost'])
df['alpha_l'] = df.groupby('naics')['alpha_l'].transform(lambda x: x.rolling(2).mean())

# Calculate the share of total labor and capital costs of each industry within year
df['capital_cost_agg'] = df.groupby('year')['capital_cost'].transform('sum')
df['omega_k'] = df['capital_cost'] / df['capital_cost_agg']
df['omega_k'] = df.groupby('naics')['omega_k'].transform(lambda x: x.rolling(2).mean())
df['labor_cost_agg'] = df.groupby('year')['labor_cost'].transform('sum')
df['omega_l'] = df['labor_cost'] / df['labor_cost_agg']
df['omega_l'] = df.groupby('naics')['omega_l'].transform(lambda x: x.rolling(2).mean())
df = df.drop(columns=['capital_cost_agg', 'labor_cost_agg'])

# Load the data for the lambda's
df_lambda = pd.read_csv(os.path.join(Path(os.getcwd()).parent, 'Data', 'lambda.csv'))

# Merge the lambda's with the main DataFrame
df = pd.merge(df, df_lambda, on=['naics', 'year'], how='left')

# Calculate the lambda's of each industry for years 1961, 1980, and 2000
df = pd.merge(df, df.loc[df['year'] == 1962, ['naics', 'lambda']].rename(columns={'lambda': 'lambda_1961'}), on='naics', how='left')
df = pd.merge(df, df.loc[df['year'] == 1980, ['naics', 'lambda']].rename(columns={'lambda': 'lambda_1980'}), on='naics', how='left')
df = pd.merge(df, df.loc[df['year'] == 2000, ['naics', 'lambda']].rename(columns={'lambda': 'lambda_2000'}), on='naics', how='left')

########################################################################
# Calculate the TFP growth decomposition for different periods         #
########################################################################

# Calculate the different terms between 1961 and 2019
df_1961_2019 = pd.DataFrame({'year': range(1961, 2019 + 1)})
df_temp = df.copy(deep=True)
df_temp['within_1'] = df_temp['b_1961'] * df_temp['tfp_growth']
df_1961_2019 = pd.merge(df_1961_2019, df_temp.groupby('year', as_index=False).agg({'within_1': 'sum'}).rename(columns={'within_1': 'productivity_1'}), on='year', how='left')
df_temp['between_1'] = (df_temp['b'] - df_temp['b_1961']) * df_temp['tfp_growth']
df_1961_2019 = pd.merge(df_1961_2019, df_temp.groupby('year', as_index=False).agg({'between_1': 'sum'}).rename(columns={'between_1': 'baumol_1'}), on='year', how='left')
df_temp['capital_reallocation'] = (df_temp['b'] * df_temp['alpha_k'] - df_temp['omega_k'] * df_temp['lambda_k']) * df_temp['capital_growth']
df_1961_2019 = pd.merge(df_1961_2019, df_temp.groupby('year', as_index=False).agg({'capital_reallocation': 'sum'}).rename(columns={'capital_reallocation': 'capital'}), on='year', how='left')
df_temp['labor_reallocation'] = (df_temp['b'] * df_temp['alpha_l'] - df_temp['omega_l'] * df_temp['lambda_l']) * df_temp['labor_growth']
df_1961_2019 = pd.merge(df_1961_2019, df_temp.groupby('year', as_index=False).agg({'labor_reallocation': 'sum'}).rename(columns={'labor_reallocation': 'labor'}), on='year', how='left')
df_temp['within_2'] = df_temp['lambda_1961'] * df_temp['tfp_growth']
df_1961_2019 = pd.merge(df_1961_2019, df_temp.groupby('year', as_index=False).agg({'within_2': 'sum'}).rename(columns={'within_2': 'productivity_2'}), on='year', how='left')
df_temp['between_2'] = (df_temp['lambda'] - df_temp['lambda_1961']) * df_temp['tfp_growth']
df_1961_2019 = pd.merge(df_1961_2019, df_temp.groupby('year', as_index=False).agg({'between_2': 'sum'}).rename(columns={'between_2': 'baumol_2'}), on='year', how='left')
df_1961_2019['total'] = df_1961_2019['productivity_1'] + df_1961_2019['baumol_1'] + df_1961_2019['capital'] + df_1961_2019['labor']

# Calculate the different terms between 1961 and 2019 without the oil and gas extraction industry
df_1961_2019_no_oge = pd.DataFrame({'year': range(1961, 2019 + 1)})
df_temp = df[df['naics'] != '211'].copy(deep=True)
df_temp['within_1'] = df_temp['b_1961'] * df_temp['tfp_growth']
df_1961_2019_no_oge = pd.merge(df_1961_2019_no_oge, df_temp.groupby('year', as_index=False).agg({'within_1': 'sum'}).rename(columns={'within_1': 'productivity_1'}), on='year', how='left')
df_temp['between_1'] = (df_temp['b'] - df_temp['b_1961']) * df_temp['tfp_growth']
df_1961_2019_no_oge = pd.merge(df_1961_2019_no_oge, df_temp.groupby('year', as_index=False).agg({'between_1': 'sum'}).rename(columns={'between_1': 'baumol_1'}), on='year', how='left')
df_temp['capital_reallocation'] = (df_temp['b'] * df_temp['alpha_k'] - df_temp['omega_k'] * df_temp['lambda_k']) * df_temp['capital_growth']
df_1961_2019_no_oge = pd.merge(df_1961_2019_no_oge, df_temp.groupby('year', as_index=False).agg({'capital_reallocation': 'sum'}).rename(columns={'capital_reallocation': 'capital'}), on='year', how='left')
df_temp['labor_reallocation'] = (df_temp['b'] * df_temp['alpha_l'] - df_temp['omega_l'] * df_temp['lambda_l']) * df_temp['labor_growth']
df_1961_2019_no_oge = pd.merge(df_1961_2019_no_oge, df_temp.groupby('year', as_index=False).agg({'labor_reallocation': 'sum'}).rename(columns={'labor_reallocation': 'labor'}), on='year', how='left')
df_temp['within_2'] = df_temp['lambda_1961'] * df_temp['tfp_growth']
df_1961_2019_no_oge = pd.merge(df_1961_2019_no_oge, df_temp.groupby('year', as_index=False).agg({'within_2': 'sum'}).rename(columns={'within_2': 'productivity_2'}), on='year', how='left')
df_temp['between_2'] = (df_temp['lambda'] - df_temp['lambda_1961']) * df_temp['tfp_growth']
df_1961_2019_no_oge = pd.merge(df_1961_2019_no_oge, df_temp.groupby('year', as_index=False).agg({'between_2': 'sum'}).rename(columns={'between_2': 'baumol_2'}), on='year', how='left')
df_1961_2019_no_oge['total'] = df_1961_2019_no_oge['productivity_1'] + df_1961_2019_no_oge['baumol_1'] + df_1961_2019_no_oge['capital'] + df_1961_2019_no_oge['labor']

# Calculate the different terms between 1961 and 1980
df_1961_1980 = pd.DataFrame({'year': range(1961, 1980 + 1)})
df_temp = df.copy(deep=True)
df_temp['within_1'] = df_temp['b_1961'] * df_temp['tfp_growth']
df_1961_1980 = pd.merge(df_1961_1980, df_temp.groupby('year', as_index=False).agg({'within_1': 'sum'}).rename(columns={'within_1': 'productivity_1'}), on='year', how='left')
df_temp['between_1'] = (df_temp['b'] - df_temp['b_1961']) * df_temp['tfp_growth']
df_1961_1980 = pd.merge(df_1961_1980, df_temp.groupby('year', as_index=False).agg({'between_1': 'sum'}).rename(columns={'between_1': 'baumol_1'}), on='year', how='left')
df_temp['capital_reallocation'] = (df_temp['b'] * df_temp['alpha_k'] - df_temp['omega_k'] * df_temp['lambda_k']) * df_temp['capital_growth']
df_1961_1980 = pd.merge(df_1961_1980, df_temp.groupby('year', as_index=False).agg({'capital_reallocation': 'sum'}).rename(columns={'capital_reallocation': 'capital'}), on='year', how='left')
df_temp['labor_reallocation'] = (df_temp['b'] * df_temp['alpha_l'] - df_temp['omega_l'] * df_temp['lambda_l']) * df_temp['labor_growth']
df_1961_1980 = pd.merge(df_1961_1980, df_temp.groupby('year', as_index=False).agg({'labor_reallocation': 'sum'}).rename(columns={'labor_reallocation': 'labor'}), on='year', how='left')
df_temp['within_2'] = df_temp['lambda_1961'] * df_temp['tfp_growth']
df_1961_1980 = pd.merge(df_1961_1980, df_temp.groupby('year', as_index=False).agg({'within_2': 'sum'}).rename(columns={'within_2': 'productivity_2'}), on='year', how='left')
df_temp['between_2'] = (df_temp['lambda'] - df_temp['lambda_1961']) * df_temp['tfp_growth']
df_1961_1980 = pd.merge(df_1961_1980, df_temp.groupby('year', as_index=False).agg({'between_2': 'sum'}).rename(columns={'between_2': 'baumol_2'}), on='year', how='left')
df_1961_1980['total'] = df_1961_1980['productivity_1'] + df_1961_1980['baumol_1'] + df_1961_1980['capital'] + df_1961_1980['labor']

# Calculate the different terms between 1961 and 1980 without the oil and gas extraction industry
df_1961_1980_no_oge = pd.DataFrame({'year': range(1961, 1980 + 1)})
df_temp = df[df['naics'] != '211'].copy(deep=True)
df_temp['within_1'] = df_temp['b_1961'] * df_temp['tfp_growth']
df_1961_1980_no_oge = pd.merge(df_1961_1980_no_oge, df_temp.groupby('year', as_index=False).agg({'within_1': 'sum'}).rename(columns={'within_1': 'productivity_1'}), on='year', how='left')
df_temp['between_1'] = (df_temp['b'] - df_temp['b_1961']) * df_temp['tfp_growth']
df_1961_1980_no_oge = pd.merge(df_1961_1980_no_oge, df_temp.groupby('year', as_index=False).agg({'between_1': 'sum'}).rename(columns={'between_1': 'baumol_1'}), on='year', how='left')
df_temp['capital_reallocation'] = (df_temp['b'] * df_temp['alpha_k'] - df_temp['omega_k'] * df_temp['lambda_k']) * df_temp['capital_growth']
df_1961_1980_no_oge = pd.merge(df_1961_1980_no_oge, df_temp.groupby('year', as_index=False).agg({'capital_reallocation': 'sum'}).rename(columns={'capital_reallocation': 'capital'}), on='year', how='left')
df_temp['labor_reallocation'] = (df_temp['b'] * df_temp['alpha_l'] - df_temp['omega_l'] * df_temp['lambda_l']) * df_temp['labor_growth']
df_1961_1980_no_oge = pd.merge(df_1961_1980_no_oge, df_temp.groupby('year', as_index=False).agg({'labor_reallocation': 'sum'}).rename(columns={'labor_reallocation': 'labor'}), on='year', how='left')
df_temp['within_2'] = df_temp['lambda_1961'] * df_temp['tfp_growth']
df_1961_1980_no_oge = pd.merge(df_1961_1980_no_oge, df_temp.groupby('year', as_index=False).agg({'within_2': 'sum'}).rename(columns={'within_2': 'productivity_2'}), on='year', how='left')
df_temp['between_2'] = (df_temp['lambda'] - df_temp['lambda_1961']) * df_temp['tfp_growth']
df_1961_1980_no_oge = pd.merge(df_1961_1980_no_oge, df_temp.groupby('year', as_index=False).agg({'between_2': 'sum'}).rename(columns={'between_2': 'baumol_2'}), on='year', how='left')
df_1961_1980_no_oge['total'] = df_1961_1980_no_oge['productivity_1'] + df_1961_1980_no_oge['baumol_1'] + df_1961_1980_no_oge['capital'] + df_1961_1980_no_oge['labor']

# Calculate the different terms between 1980 and 2000
df_1980_2000 = pd.DataFrame({'year': range(1980, 2000 + 1)})
df_temp = df.copy(deep=True)
df_temp['within_1'] = df_temp['b_1980'] * df_temp['tfp_growth']
df_1980_2000 = pd.merge(df_1980_2000, df_temp.groupby('year', as_index=False).agg({'within_1': 'sum'}).rename(columns={'within_1': 'productivity_1'}), on='year', how='left')
df_temp['between_1'] = (df_temp['b'] - df_temp['b_1980']) * df_temp['tfp_growth']
df_1980_2000 = pd.merge(df_1980_2000, df_temp.groupby('year', as_index=False).agg({'between_1': 'sum'}).rename(columns={'between_1': 'baumol_1'}), on='year', how='left')
df_temp['capital_reallocation'] = (df_temp['b'] * df_temp['alpha_k'] - df_temp['omega_k'] * df_temp['lambda_k']) * df_temp['capital_growth']
df_1980_2000 = pd.merge(df_1980_2000, df_temp.groupby('year', as_index=False).agg({'capital_reallocation': 'sum'}).rename(columns={'capital_reallocation': 'capital'}), on='year', how='left')
df_temp['labor_reallocation'] = (df_temp['b'] * df_temp['alpha_l'] - df_temp['omega_l'] * df_temp['lambda_l']) * df_temp['labor_growth']
df_1980_2000 = pd.merge(df_1980_2000, df_temp.groupby('year', as_index=False).agg({'labor_reallocation': 'sum'}).rename(columns={'labor_reallocation': 'labor'}), on='year', how='left')
df_temp['within_2'] = df_temp['lambda_1980'] * df_temp['tfp_growth']
df_1980_2000 = pd.merge(df_1980_2000, df_temp.groupby('year', as_index=False).agg({'within_2': 'sum'}).rename(columns={'within_2': 'productivity_2'}), on='year', how='left')
df_temp['between_2'] = (df_temp['lambda'] - df_temp['lambda_1980']) * df_temp['tfp_growth']
df_1980_2000 = pd.merge(df_1980_2000, df_temp.groupby('year', as_index=False).agg({'between_2': 'sum'}).rename(columns={'between_2': 'baumol_2'}), on='year', how='left')
df_1980_2000.loc[0, :] = 0
df_1980_2000['total'] = df_1980_2000['productivity_1'] + df_1980_2000['baumol_1'] + df_1980_2000['capital'] + df_1980_2000['labor']

# Calculate the different terms between 1980 and 2000 without the oil and gas extraction industry
df_1980_2000_no_oge = pd.DataFrame({'year': range(1980, 2000 + 1)})
df_temp = df[df['naics'] != '211'].copy(deep=True)
df_temp['within_1'] = df_temp['b_1980'] * df_temp['tfp_growth']
df_1980_2000_no_oge = pd.merge(df_1980_2000_no_oge, df_temp.groupby('year', as_index=False).agg({'within_1': 'sum'}).rename(columns={'within_1': 'productivity_1'}), on='year', how='left')
df_temp['between_1'] = (df_temp['b'] - df_temp['b_1980']) * df_temp['tfp_growth']
df_1980_2000_no_oge = pd.merge(df_1980_2000_no_oge, df_temp.groupby('year', as_index=False).agg({'between_1': 'sum'}).rename(columns={'between_1': 'baumol_1'}), on='year', how='left')
df_temp['capital_reallocation'] = (df_temp['b'] * df_temp['alpha_k'] - df_temp['omega_k'] * df_temp['lambda_k']) * df_temp['capital_growth']
df_1980_2000_no_oge = pd.merge(df_1980_2000_no_oge, df_temp.groupby('year', as_index=False).agg({'capital_reallocation': 'sum'}).rename(columns={'capital_reallocation': 'capital'}), on='year', how='left')
df_temp['labor_reallocation'] = (df_temp['b'] * df_temp['alpha_l'] - df_temp['omega_l'] * df_temp['lambda_l']) * df_temp['labor_growth']
df_1980_2000_no_oge = pd.merge(df_1980_2000_no_oge, df_temp.groupby('year', as_index=False).agg({'labor_reallocation': 'sum'}).rename(columns={'labor_reallocation': 'labor'}), on='year', how='left')
df_temp['within_2'] = df_temp['lambda_1980'] * df_temp['tfp_growth']
df_1980_2000_no_oge = pd.merge(df_1980_2000_no_oge, df_temp.groupby('year', as_index=False).agg({'within_2': 'sum'}).rename(columns={'within_2': 'productivity_2'}), on='year', how='left')
df_temp['between_2'] = (df_temp['lambda'] - df_temp['lambda_1980']) * df_temp['tfp_growth']
df_1980_2000_no_oge = pd.merge(df_1980_2000_no_oge, df_temp.groupby('year', as_index=False).agg({'between_2': 'sum'}).rename(columns={'between_2': 'baumol_2'}), on='year', how='left')
df_1980_2000_no_oge.loc[0, :] = 0
df_1980_2000_no_oge['total'] = df_1980_2000_no_oge['productivity_1'] + df_1980_2000_no_oge['baumol_1'] + df_1980_2000_no_oge['capital'] + df_1980_2000_no_oge['labor']

# Calculate the different terms between 2000 and 2019
df_2000_2019 = pd.DataFrame({'year': range(2000, 2019 + 1)})
df_temp = df.copy(deep=True)
df_temp['within_1'] = df_temp['b_2000'] * df_temp['tfp_growth']
df_2000_2019 = pd.merge(df_2000_2019, df_temp.groupby('year', as_index=False).agg({'within_1': 'sum'}).rename(columns={'within_1': 'productivity_1'}), on='year', how='left')
df_temp['between_1'] = (df_temp['b'] - df_temp['b_2000']) * df_temp['tfp_growth']
df_2000_2019 = pd.merge(df_2000_2019, df_temp.groupby('year', as_index=False).agg({'between_1': 'sum'}).rename(columns={'between_1': 'baumol_1'}), on='year', how='left')
df_temp['capital_reallocation'] = (df_temp['b'] * df_temp['alpha_k'] - df_temp['omega_k'] * df_temp['lambda_k']) * df_temp['capital_growth']
df_2000_2019 = pd.merge(df_2000_2019, df_temp.groupby('year', as_index=False).agg({'capital_reallocation': 'sum'}).rename(columns={'capital_reallocation': 'capital'}), on='year', how='left')
df_temp['labor_reallocation'] = (df_temp['b'] * df_temp['alpha_l'] - df_temp['omega_l'] * df_temp['lambda_l']) * df_temp['labor_growth']
df_2000_2019 = pd.merge(df_2000_2019, df_temp.groupby('year', as_index=False).agg({'labor_reallocation': 'sum'}).rename(columns={'labor_reallocation': 'labor'}), on='year', how='left')
df_temp['within_2'] = df_temp['lambda_2000'] * df_temp['tfp_growth']
df_2000_2019 = pd.merge(df_2000_2019, df_temp.groupby('year', as_index=False).agg({'within_2': 'sum'}).rename(columns={'within_2': 'productivity_2'}), on='year', how='left')
df_temp['between_2'] = (df_temp['lambda'] - df_temp['lambda_2000']) * df_temp['tfp_growth']
df_2000_2019 = pd.merge(df_2000_2019, df_temp.groupby('year', as_index=False).agg({'between_2': 'sum'}).rename(columns={'between_2': 'baumol_2'}), on='year', how='left')
df_2000_2019.loc[0, :] = 0
df_2000_2019['total'] = df_2000_2019['productivity_1'] + df_2000_2019['baumol_1'] + df_2000_2019['capital'] + df_2000_2019['labor']

# Calculate the different terms between 2000 and 2019 without the oil and gas extraction industry
df_2000_2019_no_oge = pd.DataFrame({'year': range(2000, 2019 + 1)})
df_temp = df[df['naics'] != '211'].copy(deep=True)
df_temp['within_1'] = df_temp['b_2000'] * df_temp['tfp_growth']
df_2000_2019_no_oge = pd.merge(df_2000_2019_no_oge, df_temp.groupby('year', as_index=False).agg({'within_1': 'sum'}).rename(columns={'within_1': 'productivity_1'}), on='year', how='left')
df_temp['between_1'] = (df_temp['b'] - df_temp['b_2000']) * df_temp['tfp_growth']
df_2000_2019_no_oge = pd.merge(df_2000_2019_no_oge, df_temp.groupby('year', as_index=False).agg({'between_1': 'sum'}).rename(columns={'between_1': 'baumol_1'}), on='year', how='left')
df_temp['capital_reallocation'] = (df_temp['b'] * df_temp['alpha_k'] - df_temp['omega_k'] * df_temp['lambda_k']) * df_temp['capital_growth']
df_2000_2019_no_oge = pd.merge(df_2000_2019_no_oge, df_temp.groupby('year', as_index=False).agg({'capital_reallocation': 'sum'}).rename(columns={'capital_reallocation': 'capital'}), on='year', how='left')
df_temp['labor_reallocation'] = (df_temp['b'] * df_temp['alpha_l'] - df_temp['omega_l'] * df_temp['lambda_l']) * df_temp['labor_growth']
df_2000_2019_no_oge = pd.merge(df_2000_2019_no_oge, df_temp.groupby('year', as_index=False).agg({'labor_reallocation': 'sum'}).rename(columns={'labor_reallocation': 'labor'}), on='year', how='left')
df_temp['within_2'] = df_temp['lambda_2000'] * df_temp['tfp_growth']
df_2000_2019_no_oge = pd.merge(df_2000_2019_no_oge, df_temp.groupby('year', as_index=False).agg({'within_2': 'sum'}).rename(columns={'within_2': 'productivity_2'}), on='year', how='left')
df_temp['between_2'] = (df_temp['lambda'] - df_temp['lambda_2000']) * df_temp['tfp_growth']
df_2000_2019_no_oge = pd.merge(df_2000_2019_no_oge, df_temp.groupby('year', as_index=False).agg({'between_2': 'sum'}).rename(columns={'between_2': 'baumol_2'}), on='year', how='left')
df_2000_2019_no_oge.loc[0, :] = 0
df_2000_2019_no_oge['total'] = df_2000_2019_no_oge['productivity_1'] + df_2000_2019_no_oge['baumol_1'] + df_2000_2019_no_oge['capital'] + df_2000_2019_no_oge['labor']

########################################################################
# Plot the TFP Baumol effect                                           # 
########################################################################

# Initialize the figure
fig, ax = plt.subplots(figsize=(8, 5))

# Set the background color of the figure to transparent
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)

# Plot the data
ax.plot(df_1961_2019['year'], 100 * (df_1961_2019['total'].cumsum() + 1), label='Total', color=palette[0], linewidth=2)
ax.plot(df_1961_2019['year'], 100 * (df_1961_2019['total'].cumsum() - df_1961_2019['baumol_1'].cumsum() + 1), label='Without Baumol', color=palette[1], linewidth=2)

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

########################################################################
# Plot the TFP Baumol effect without the oil and gas extraction        #
# industry                                                             # 
########################################################################

# Initialize the figure
fig, ax = plt.subplots(figsize=(8, 5))

# Set the background color of the figure to transparent
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)

# Plot the data
ax.plot(df_1961_2019['year'], 100 * (df_1961_2019['total'].cumsum() + 1), label='Total', color=palette[0], linewidth=2)
ax.plot(df_1961_2019['year'], 100 * (df_1961_2019['total'].cumsum() - df_1961_2019['baumol_1'].cumsum() + 1), label='Without Baumol', color=palette[1], linewidth=2)
ax.plot(df_1961_2019_no_oge['year'], 100 * (df_1961_2019_no_oge['total'].cumsum() + 1), label=r'Total (without O\&G)', color=palette[0], linewidth=2, linestyle='dotted')
ax.plot(df_1961_2019_no_oge['year'], 100 * (df_1961_2019_no_oge['total'].cumsum() - df_1961_2019_no_oge['baumol_1'].cumsum() + 1), label=r'Without Baumol (without O\&G)', color=palette[1], linewidth=2, linestyle='dotted')

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
fig.savefig(os.path.join(Path(os.getcwd()).parent, 'Figures', 'baumol_no_oge.png'), transparent=True, dpi=300)
plt.close()

########################################################################
# Plot the TFP allocative efficiency effect effect                     # 
########################################################################

# Initialize the figure
fig, ax = plt.subplots(figsize=(8, 5))

# Set the background color of the figure to transparent
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)

# Plot the data
ax.plot(df_1961_2019['year'], 100 * (df_1961_2019['total'].cumsum() + 1), label='Total', color=palette[0], linewidth=2)
ax.plot(df_1961_2019['year'], 100 * (df_1961_2019['productivity_2'].cumsum() + df_1961_2019['baumol_2'].cumsum() + 1), label='Without misallocation', color=palette[1], linewidth=2)

# Set the horizontal axis
ax.set_xlim(1961, 2019)
ax.set_xticks(range(1965, 2015 + 1, 5))
ax.set_xticklabels(range(1965, 2015 + 1, 5), fontsize=12)

# Set the vertical axis
ax.set_ylim(100, 170)
ax.set_yticks(range(100, 170 + 1, 10))
ax.set_yticklabels(range(100, 170 + 1, 10), fontsize=12)
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
fig.savefig(os.path.join(Path(os.getcwd()).parent, 'Figures', 'misallocation.png'), transparent=True, dpi=300)
plt.close()

########################################################################
# Plot the TFP allocative efficiency effect without the oil and gas    #
# extraction industry                                                  # 
########################################################################

# Initialize the figure
fig, ax = plt.subplots(figsize=(8, 5))

# Set the background color of the figure to transparent
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)

# Plot the data
ax.plot(df_1961_2019['year'], 100 * (df_1961_2019['total'].cumsum() + 1), label='Total', color=palette[0], linewidth=2)
ax.plot(df_1961_2019['year'], 100 * (df_1961_2019['productivity_2'].cumsum() + df_1961_2019['baumol_2'].cumsum() + 1), label='Without misallocation', color=palette[1], linewidth=2)
ax.plot(df_1961_2019_no_oge['year'], 100 * (df_1961_2019_no_oge['total'].cumsum() + 1), label=r'Total (without O\&G)', color=palette[0], linewidth=2, linestyle='dotted')
ax.plot(df_1961_2019_no_oge['year'], 100 * (df_1961_2019_no_oge['productivity_2'].cumsum() + df_1961_2019_no_oge['baumol_2'].cumsum() + 1), label=r'Without misallocation (without O\&G)', color=palette[1], linewidth=2, linestyle='dotted')

# Set the horizontal axis
ax.set_xlim(1961, 2019)
ax.set_xticks(range(1965, 2015 + 1, 5))
ax.set_xticklabels(range(1965, 2015 + 1, 5), fontsize=12)

# Set the vertical axis
ax.set_ylim(100, 180)
ax.set_yticks(range(100, 180 + 1, 10))
ax.set_yticklabels(range(100, 180 + 1, 10), fontsize=12)
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
fig.savefig(os.path.join(Path(os.getcwd()).parent, 'Figures', 'misallocation_no_oge.png'), transparent=True, dpi=300)
plt.close()

########################################################################
# Plot the first TFP decomposition                                     # 
########################################################################

# Initialize the figure
fig, ax = plt.subplots(figsize=(8, 5))

# Set the background color of the figure to transparent
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)

# Plot the data
ax.stackplot(df_1961_2019['year'], df_1961_2019[['productivity_1', 'labor']].cumsum().values.T, colors=palette[0:3], edgecolor='k', linewidth=0.5, zorder=1)
ax.stackplot(df_1961_2019['year'], df_1961_2019[['baumol_1', 'capital']].cumsum().values.T, colors=palette[2:4], edgecolor='k', linewidth=0.5, zorder=1)
ax.plot(df_1961_2019['year'], df_1961_2019['productivity_1'].cumsum() + df_1961_2019['baumol_1'].cumsum() + df_1961_2019['capital'].cumsum() + df_1961_2019['labor'].cumsum(), color='white', linestyle='dotted', linewidth=1.5, zorder=3)

# Set the horizontal axis
ax.set_xlim(1961, 2019)
ax.set_xticks(range(1965, 2015 + 1, 5))
ax.set_xticklabels(range(1965, 2015 + 1, 5), fontsize=12)

# Set the vertical axis
ax.set_ylim(-0.3, 0.6)
ax.set_yticks(np.arange(-0.3, 0.6 + 0.01, 0.1))
ax.set_yticklabels(range(70, 160 + 1, 10), fontsize=12)
ax.set_ylabel('Aggregate TFP decomposition (1961=100)', fontsize=12, rotation=0, ha='left')
ax.yaxis.set_label_coords(0, 1.01)

# Remove the top and right axes
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, which='major', axis='y', color='gray', linestyle=':', linewidth=0.5)

# Set the legend
ax.legend(['Productivity', 'Labor', 'Baumol', 'Capital'], frameon=False, fontsize=12)
ax.text(2019, 0.3, 'Total', fontsize=12, color='white', ha='right', va='bottom')

# Add a note about the data source
ax.text(1, 1.01, 'Source: Statistics Canada', fontsize=8, color='k', ha='right', va='bottom', transform=ax.transAxes)

# Save and close the figure
fig.tight_layout()
fig.savefig(os.path.join(Path(os.getcwd()).parent, 'Figures', 'tfp_decomposition_1.png'), transparent=True, dpi=300)
plt.close()

########################################################################
# Plot the second TFP decomposition                                    # 
########################################################################

# Initialize the figure
fig, ax = plt.subplots(figsize=(8, 5))

# Set the background color of the figure to transparent
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)

# Plot the data
ax.stackplot(df_1961_2019['year'], df_1961_2019['productivity_2'].cumsum().values.T, color=palette[0], edgecolor='k', linewidth=0.5, zorder=1)
ax.stackplot(df_1961_2019['year'], [df_1961_2019['baumol_2'].cumsum().values.T, df_1961_2019['total'].cumsum().values.T - df_1961_2019['productivity_2'].cumsum().values.T - df_1961_2019['baumol_2'].cumsum().values.T], colors=palette[1:3], edgecolor='k', linewidth=0.5, zorder=1)
ax.plot(df_1961_2019['year'], df_1961_2019['total'].cumsum(), color='white', linestyle='dotted', linewidth=1.5, zorder=3)

# Set the horizontal axis
ax.set_xlim(1961, 2019)
ax.set_xticks(range(1965, 2015 + 1, 5))
ax.set_xticklabels(range(1965, 2015 + 1, 5), fontsize=12)

# Set the vertical axis
ax.set_ylim(-0.6, 0.9)
ax.set_yticks(np.arange(-0.6, 0.9 + 0.01, 0.1))
ax.set_yticklabels(range(40, 190 + 1, 10), fontsize=12)
ax.set_ylabel('Aggregate TFP decomposition (1961=100)', fontsize=12, rotation=0, ha='left')
ax.yaxis.set_label_coords(0, 1.01)

# Remove the top and right axes
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, which='major', axis='y', color='gray', linestyle=':', linewidth=0.5)

# Set the legend
ax.legend(['Productivity', 'Baumol', 'Allocative efficiency'], frameon=False, fontsize=12)
ax.text(2019, 0.3, 'Total', fontsize=12, color='white', ha='right', va='bottom')

# Add a note about the data source
ax.text(1, 1.01, 'Source: Statistics Canada', fontsize=8, color='k', ha='right', va='bottom', transform=ax.transAxes)

# Save and close the figure
fig.tight_layout()
fig.savefig(os.path.join(Path(os.getcwd()).parent, 'Figures', 'tfp_decomposition_2.png'), transparent=True, dpi=300)
plt.close()

########################################################################
# Tabulate the first TFP growth decomposition for different periods    # 
########################################################################

# Calculate the different terms for the period 1961-2019
productivity_1961_2019 = 100 * df_1961_2019['productivity_1'].cumsum().iloc[-1] / (2019 - 1961)
baumol_1961_2019 = 100 * df_1961_2019['baumol_1'].cumsum().iloc[-1] / (2019 - 1961)
capital_1961_2019 = 100 * df_1961_2019['capital'].cumsum().iloc[-1] / (2019 - 1961)
labor_1961_2019 = 100 * df_1961_2019['labor'].cumsum().iloc[-1] / (2019 - 1961)
total_1961_2019 = 100 * df_1961_2019['total'].cumsum().iloc[-1] / (2019 - 1961)

# Calculate the different terms for the period 1961-1980
productivity_1961_1980 = 100 * df_1961_1980['productivity_1'].cumsum().iloc[-1] / (1980 - 1961)
baumol_1961_1980 = 100 * df_1961_1980['baumol_1'].cumsum().iloc[-1] / (1980 - 1961)
capital_1961_1980 = 100 * df_1961_1980['capital'].cumsum().iloc[-1] / (1980 - 1961)
labor_1961_1980 = 100 * df_1961_1980['labor'].cumsum().iloc[-1] / (1980 - 1961)
total_1961_1980 = 100 * df_1961_1980['total'].cumsum().iloc[-1] / (1980 - 1961)

# Calculate the different terms for the period 1980-2000
productivity_1980_2000 = 100 * df_1980_2000['productivity_1'].cumsum().iloc[-1] / (2000 - 1980)
baumol_1980_2000 = 100 * df_1980_2000['baumol_1'].cumsum().iloc[-1] / (2000 - 1980)
capital_1980_2000 = 100 * df_1980_2000['capital'].cumsum().iloc[-1] / (2000 - 1980)
labor_1980_2000 = 100 * df_1980_2000['labor'].cumsum().iloc[-1] / (2000 - 1980)
total_1980_2000 = 100 * df_1980_2000['total'].cumsum().iloc[-1] / (2000 - 1980)

# Calculate the different terms for the period 2000-2019
productivity_2000_2019 = 100 * df_2000_2019['productivity_1'].cumsum().iloc[-1] / (2019 - 2000)
baumol_2000_2019 = 100 * df_2000_2019['baumol_1'].cumsum().iloc[-1] / (2019 - 2000)
capital_2000_2019 = 100 * df_2000_2019['capital'].cumsum().iloc[-1] / (2019 - 2000)
labor_2000_2019 = 100 * df_2000_2019['labor'].cumsum().iloc[-1] / (2019 - 2000)
total_2000_2019 = 100 * df_2000_2019['total'].cumsum().iloc[-1] / (2019 - 2000)

# Write a table with the TFP growth decomposition
table = open(os.path.join(Path(os.getcwd()).parent, 'Tables', 'tfp_decomposition_1.tex'), 'w')
lines = [r'\begin{table}[h]',
         r'\centering',
         r'\begin{threeparttable}',
         r'\caption{TFP Growth Decomposition \#1}',
         r'\begin{tabular}{lccccc}',
         r'\hline',
         r'\hline',
         r'& & 1961-2019 & 1961-1980 & 1980-2000 & 2000-2019 \\',
         r'\hline',
         r'Productivity & & ' + '{:.2f}'.format(productivity_1961_2019) + r'\% & ' \
                     + '{:.2f}'.format(productivity_1961_1980) + r'\% & ' \
                     + '{:.2f}'.format(productivity_1980_2000) + r'\% & ' \
                     + '{:.2f}'.format(productivity_2000_2019) + r'\% \\',
         r'Baumol & & ' + '{:.2f}'.format(baumol_1961_2019) + r'\% & ' \
                     + '{:.2f}'.format(baumol_1961_1980) + r'\% & ' \
                     + '{:.2f}'.format(baumol_1980_2000) + r'\% & ' \
                     + '{:.2f}'.format(baumol_2000_2019) + r'\% \\',
         r'Capital & & ' + '{:.2f}'.format(capital_1961_2019) + r'\% & ' \
                     + '{:.2f}'.format(capital_1961_1980) + r'\% & ' \
                     + '{:.2f}'.format(capital_1980_2000) + r'\% & ' \
                     + '{:.2f}'.format(capital_2000_2019) + r'\% \\',
         r'Labor & & ' + '{:.2f}'.format(labor_1961_2019) + r'\% & ' \
                     + '{:.2f}'.format(labor_1961_1980) + r'\% & ' \
                     + '{:.2f}'.format(labor_1980_2000) + r'\% & ' \
                     + '{:.2f}'.format(labor_2000_2019) + r'\% \\',
         r'\hline',
         r'Total & & ' + '{:.2f}'.format(total_1961_2019) + r'\% & ' \
                     + '{:.2f}'.format(total_1961_1980) + r'\% & ' \
                     + '{:.2f}'.format(total_1980_2000) + r'\% & ' \
                     + '{:.2f}'.format(total_2000_2019) + r'\% \\',
         r'\hline',
         r'\hline',
         r'\end{tabular}',
         r'\begin{tablenotes}[flushleft]',
         r'\footnotesize',
         r'\item Note:',
         r'\end{tablenotes}',
         r'\label{tab:tfp_decomposition_1}',
         r'\end{threeparttable}',
         r'\end{table}']
table.write('\n'.join(lines))
table.close()

########################################################################
# Tabulate the first TFP growth decomposition for different periods    #
# without the oil and gas extraction industry                          # 
########################################################################

# Calculate the different terms for the period 1961-2019
productivity_1961_2019 = 100 * df_1961_2019_no_oge['productivity_1'].cumsum().iloc[-1] / (2019 - 1961)
baumol_1961_2019 = 100 * df_1961_2019_no_oge['baumol_1'].cumsum().iloc[-1] / (2019 - 1961)
capital_1961_2019 = 100 * df_1961_2019_no_oge['capital'].cumsum().iloc[-1] / (2019 - 1961)
labor_1961_2019 = 100 * df_1961_2019_no_oge['labor'].cumsum().iloc[-1] / (2019 - 1961)
total_1961_2019 = 100 * df_1961_2019_no_oge['total'].cumsum().iloc[-1] / (2019 - 1961)

# Calculate the different terms for the period 1961-1980
productivity_1961_1980 = 100 * df_1961_1980_no_oge['productivity_1'].cumsum().iloc[-1] / (1980 - 1961)
baumol_1961_1980 = 100 * df_1961_1980_no_oge['baumol_1'].cumsum().iloc[-1] / (1980 - 1961)
capital_1961_1980 = 100 * df_1961_1980_no_oge['capital'].cumsum().iloc[-1] / (1980 - 1961)
labor_1961_1980 = 100 * df_1961_1980_no_oge['labor'].cumsum().iloc[-1] / (1980 - 1961)
total_1961_1980 = 100 * df_1961_1980_no_oge['total'].cumsum().iloc[-1] / (1980 - 1961)

# Calculate the different terms for the period 1980-2000
productivity_1980_2000 = 100 * df_1980_2000_no_oge['productivity_1'].cumsum().iloc[-1] / (2000 - 1980)
baumol_1980_2000 = 100 * df_1980_2000_no_oge['baumol_1'].cumsum().iloc[-1] / (2000 - 1980)
capital_1980_2000 = 100 * df_1980_2000_no_oge['capital'].cumsum().iloc[-1] / (2000 - 1980)
labor_1980_2000 = 100 * df_1980_2000_no_oge['labor'].cumsum().iloc[-1] / (2000 - 1980)
total_1980_2000 = 100 * df_1980_2000_no_oge['total'].cumsum().iloc[-1] / (2000 - 1980)

# Calculate the different terms for the period 2000-2019
productivity_2000_2019 = 100 * df_2000_2019_no_oge['productivity_1'].cumsum().iloc[-1] / (2019 - 2000)
baumol_2000_2019 = 100 * df_2000_2019_no_oge['baumol_1'].cumsum().iloc[-1] / (2019 - 2000)
capital_2000_2019 = 100 * df_2000_2019_no_oge['capital'].cumsum().iloc[-1] / (2019 - 2000)
labor_2000_2019 = 100 * df_2000_2019_no_oge['labor'].cumsum().iloc[-1] / (2019 - 2000)
total_2000_2019 = 100 * df_2000_2019_no_oge['total'].cumsum().iloc[-1] / (2019 - 2000)

# Write a table with the TFP growth decomposition
table = open(os.path.join(Path(os.getcwd()).parent, 'Tables', 'tfp_decomposition_1_no_oge.tex'), 'w')
lines = [r'\begin{table}[h]',
         r'\centering',
         r'\begin{threeparttable}',
         r'\caption{TFP Growth Decomposition \#1 (without O\&G)}',
         r'\begin{tabular}{lccccc}',
         r'\hline',
         r'\hline',
         r'& & 1961-2019 & 1961-1980 & 1980-2000 & 2000-2019 \\',
         r'\hline',
         r'Productivity & & ' + '{:.2f}'.format(productivity_1961_2019) + r'\% & ' \
                     + '{:.2f}'.format(productivity_1961_1980) + r'\% & ' \
                     + '{:.2f}'.format(productivity_1980_2000) + r'\% & ' \
                     + '{:.2f}'.format(productivity_2000_2019) + r'\% \\',
         r'Baumol & & ' + '{:.2f}'.format(baumol_1961_2019) + r'\% & ' \
                     + '{:.2f}'.format(baumol_1961_1980) + r'\% & ' \
                     + '{:.2f}'.format(baumol_1980_2000) + r'\% & ' \
                     + '{:.2f}'.format(baumol_2000_2019) + r'\% \\',
         r'Capital & & ' + '{:.2f}'.format(capital_1961_2019) + r'\% & ' \
                     + '{:.2f}'.format(capital_1961_1980) + r'\% & ' \
                     + '{:.2f}'.format(capital_1980_2000) + r'\% & ' \
                     + '{:.2f}'.format(capital_2000_2019) + r'\% \\',
         r'Labor & & ' + '{:.2f}'.format(labor_1961_2019) + r'\% & ' \
                     + '{:.2f}'.format(labor_1961_1980) + r'\% & ' \
                     + '{:.2f}'.format(labor_1980_2000) + r'\% & ' \
                     + '{:.2f}'.format(labor_2000_2019) + r'\% \\',
         r'\hline',
         r'Total & & ' + '{:.2f}'.format(total_1961_2019) + r'\% & ' \
                     + '{:.2f}'.format(total_1961_1980) + r'\% & ' \
                     + '{:.2f}'.format(total_1980_2000) + r'\% & ' \
                     + '{:.2f}'.format(total_2000_2019) + r'\% \\',
         r'\hline',
         r'\hline',
         r'\end{tabular}',
         r'\begin{tablenotes}[flushleft]',
         r'\footnotesize',
         r'\item Note:',
         r'\end{tablenotes}',
         r'\label{tab:tfp_decomposition_1_no_oge}',
         r'\end{threeparttable}',
         r'\end{table}']
table.write('\n'.join(lines))
table.close()

########################################################################
# Tabulate the second TFP growth decomposition for different periods   # 
########################################################################

# Calculate the different terms for the period 1961-2019
productivity_1961_2019 = 100 * df_1961_2019['productivity_2'].cumsum().iloc[-1] / (2019 - 1961)
baumol_1961_2019 = 100 * df_1961_2019['baumol_2'].cumsum().iloc[-1] / (2019 - 1961)
total_1961_2019 = 100 * df_1961_2019['total'].cumsum().iloc[-1] / (2019 - 1961)
misallocation_1961_2019 = total_1961_2019 - productivity_1961_2019 - baumol_1961_2019

# Calculate the different terms for the period 1961-1980
productivity_1961_1980 = 100 * df_1961_1980['productivity_2'].cumsum().iloc[-1] / (1980 - 1961)
baumol_1961_1980 = 100 * df_1961_1980['baumol_2'].cumsum().iloc[-1] / (1980 - 1961)
total_1961_1980 = 100 * df_1961_1980['total'].cumsum().iloc[-1] / (1980 - 1961)
misallocation_1961_1980 = total_1961_1980 - productivity_1961_1980 - baumol_1961_1980

# Calculate the different terms for the period 1980-2000
productivity_1980_2000 = 100 * df_1980_2000['productivity_2'].cumsum().iloc[-1] / (2000 - 1980)
baumol_1980_2000 = 100 * df_1980_2000['baumol_2'].cumsum().iloc[-1] / (2000 - 1980)
total_1980_2000 = 100 * df_1980_2000['total'].cumsum().iloc[-1] / (2000 - 1980)
misallocation_1980_2000 = total_1980_2000 - productivity_1980_2000 - baumol_1980_2000

# Calculate the different terms for the period 2000-2019
productivity_2000_2019 = 100 * df_2000_2019['productivity_2'].cumsum().iloc[-1] / (2019 - 2000)
baumol_2000_2019 = 100 * df_2000_2019['baumol_2'].cumsum().iloc[-1] / (2019 - 2000)
total_2000_2019 = 100 * df_2000_2019['total'].cumsum().iloc[-1] / (2019 - 2000)
misallocation_2000_2019 = total_2000_2019 - productivity_2000_2019 - baumol_2000_2019

# Write a table with the TFP growth decomposition
table = open(os.path.join(Path(os.getcwd()).parent, 'Tables', 'tfp_decomposition_2.tex'), 'w')
lines = [r'\begin{table}[h]',
         r'\centering',
         r'\begin{threeparttable}',
         r'\caption{TFP Growth Decomposition \#2}',
         r'\begin{tabular}{lccccc}',
         r'\hline',
         r'\hline',
         r'& & 1961-2019 & 1961-1980 & 1980-2000 & 2000-2019 \\',
         r'\hline',
         r'Productivity & & ' + '{:.2f}'.format(productivity_1961_2019) + r'\% & ' \
                     + '{:.2f}'.format(productivity_1961_1980) + r'\% & ' \
                     + '{:.2f}'.format(productivity_1980_2000) + r'\% & ' \
                     + '{:.2f}'.format(productivity_2000_2019) + r'\% \\',
         r'Baumol & & ' + '{:.2f}'.format(baumol_1961_2019) + r'\% & ' \
                     + '{:.2f}'.format(baumol_1961_1980) + r'\% & ' \
                     + '{:.2f}'.format(baumol_1980_2000) + r'\% & ' \
                     + '{:.2f}'.format(baumol_2000_2019) + r'\% \\',
         r'Allocative efficiency & & ' + '{:.2f}'.format(misallocation_1961_2019) + r'\% & ' \
                     + '{:.2f}'.format(misallocation_1961_1980) + r'\% & ' \
                     + '{:.2f}'.format(misallocation_1980_2000) + r'\% & ' \
                     + '{:.2f}'.format(misallocation_2000_2019) + r'\% \\',
         r'\hline',
         r'Total & & ' + '{:.2f}'.format(total_1961_2019) + r'\% & ' \
                     + '{:.2f}'.format(total_1961_1980) + r'\% & ' \
                     + '{:.2f}'.format(total_1980_2000) + r'\% & ' \
                     + '{:.2f}'.format(total_2000_2019) + r'\% \\',
         r'\hline',
         r'\hline',
         r'\end{tabular}',
         r'\begin{tablenotes}[flushleft]',
         r'\footnotesize',
         r'\item Note:',
         r'\end{tablenotes}',
         r'\label{tab:tfp_decomposition_2}',
         r'\end{threeparttable}',
         r'\end{table}']
table.write('\n'.join(lines))
table.close()

########################################################################
# Tabulate the second TFP growth decomposition for different periods   #
# without the oil and gas extraction industry                          # 
########################################################################

# Calculate the different terms for the period 1961-2019
productivity_1961_2019_no_oge = 100 * df_1961_2019_no_oge['productivity_2'].cumsum().iloc[-1] / (2019 - 1961)
baumol_1961_2019_no_oge = 100 * df_1961_2019_no_oge['baumol_2'].cumsum().iloc[-1] / (2019 - 1961)
total_1961_2019_no_oge = 100 * df_1961_2019_no_oge['total'].cumsum().iloc[-1] / (2019 - 1961)
misallocation_1961_2019_no_oge = total_1961_2019_no_oge - productivity_1961_2019_no_oge - baumol_1961_2019_no_oge

# Calculate the different terms for the period 1961-1980
productivity_1961_1980_no_oge = 100 * df_1961_1980_no_oge['productivity_2'].cumsum().iloc[-1] / (1980 - 1961)
baumol_1961_1980_no_oge = 100 * df_1961_1980_no_oge['baumol_2'].cumsum().iloc[-1] / (1980 - 1961)
total_1961_1980_no_oge = 100 * df_1961_1980_no_oge['total'].cumsum().iloc[-1] / (1980 - 1961)
misallocation_1961_1980_no_oge = total_1961_1980_no_oge - productivity_1961_1980_no_oge - baumol_1961_1980_no_oge

# Calculate the different terms for the period 1980-2000
productivity_1980_2000_no_oge = 100 * df_1980_2000_no_oge['productivity_2'].cumsum().iloc[-1] / (2000 - 1980)
baumol_1980_2000_no_oge = 100 * df_1980_2000_no_oge['baumol_2'].cumsum().iloc[-1] / (2000 - 1980)
total_1980_2000_no_oge = 100 * df_1980_2000_no_oge['total'].cumsum().iloc[-1] / (2000 - 1980)
misallocation_1980_2000_no_oge = total_1980_2000_no_oge - productivity_1980_2000_no_oge - baumol_1980_2000_no_oge

# Calculate the different terms for the period 2000-2019
productivity_2000_2019_no_oge = 100 * df_2000_2019_no_oge['productivity_2'].cumsum().iloc[-1] / (2019 - 2000)
baumol_2000_2019_no_oge = 100 * df_2000_2019_no_oge['baumol_2'].cumsum().iloc[-1] / (2019 - 2000)
total_2000_2019_no_oge = 100 * df_2000_2019_no_oge['total'].cumsum().iloc[-1] / (2019 - 2000)
misallocation_2000_2019_no_oge = total_2000_2019_no_oge - productivity_2000_2019_no_oge - baumol_2000_2019_no_oge

# Write a table with the TFP growth decomposition
table = open(os.path.join(Path(os.getcwd()).parent, 'Tables', 'tfp_decomposition_2_no_oge.tex'), 'w')
lines = [r'\begin{table}[h]',
         r'\centering',
         r'\begin{threeparttable}',
         r'\caption{TFP Growth Decomposition \#2 (without O\&G)}',
         r'\begin{tabular}{lccccc}',
         r'\hline',
         r'\hline',
         r'& & 1961-2019 & 1961-1980 & 1980-2000 & 2000-2019 \\',
         r'\hline',
         r'Productivity & & ' + '{:.2f}'.format(productivity_1961_2019_no_oge) + r'\% & ' \
                     + '{:.2f}'.format(productivity_1961_1980_no_oge) + r'\% & ' \
                     + '{:.2f}'.format(productivity_1980_2000_no_oge) + r'\% & ' \
                     + '{:.2f}'.format(productivity_2000_2019_no_oge) + r'\% \\',
         r'Baumol & & ' + '{:.2f}'.format(baumol_1961_2019_no_oge) + r'\% & ' \
                     + '{:.2f}'.format(baumol_1961_1980_no_oge) + r'\% & ' \
                     + '{:.2f}'.format(baumol_1980_2000_no_oge) + r'\% & ' \
                     + '{:.2f}'.format(baumol_2000_2019_no_oge) + r'\% \\',
         r'Allocative efficiency & & ' + '{:.2f}'.format(misallocation_1961_2019_no_oge) + r'\% & ' \
                     + '{:.2f}'.format(misallocation_1961_1980_no_oge) + r'\% & ' \
                     + '{:.2f}'.format(misallocation_1980_2000_no_oge) + r'\% & ' \
                     + '{:.2f}'.format(misallocation_2000_2019_no_oge) + r'\% \\',
         r'\hline',
         r'Total & & ' + '{:.2f}'.format(total_1961_2019_no_oge) + r'\% & ' \
                     + '{:.2f}'.format(total_1961_1980_no_oge) + r'\% & ' \
                     + '{:.2f}'.format(total_1980_2000_no_oge) + r'\% & ' \
                     + '{:.2f}'.format(total_2000_2019_no_oge) + r'\% \\',
         r'\hline',
         r'\hline',
         r'\end{tabular}',
         r'\begin{tablenotes}[flushleft]',
         r'\footnotesize',
         r'\item Note:',
         r'\end{tablenotes}',
         r'\label{tab:tfp_decomposition_2_no_oge}',
         r'\end{threeparttable}',
         r'\end{table}']
table.write('\n'.join(lines))
table.close()

########################################################################
# Plot the TFP contribution of each industry                           #
########################################################################

# Define a mapping for the industry labels
label_map = {
    'Accommodation and food services [72]': 'Accommodation and food services',
    'Administrative and support, waste management and remediation services [56]': 'Admin. and supp., waste mgmt. and remediation services',
    'Arts, entertainment and recreation [71]': 'Arts, entertainment and recreation',
    'Beverage and tobacco product manufacturing [312]': 'Beverage and tobacco product manuf.',
    'Chemical manufacturing [325]': 'Chemical manuf.',
    'Clothing, Leather and allied product manufacturing': 'Clothing, leather and allied product manuf.',
    'Computer and electronic product manufacturing [334]': 'Computer and electronic product manuf.',
    'Construction [23]': 'Construction', 
    'Crop and animal production': 'Crop and animal production',
    'Electrical equipment, appliance and component manufacturing [335]': 'Electrical equip., appliance and component manuf.',
    'Fabricated metal product manufacturing [332]': 'Fabricated metal product manuf.',
    'Finance, insurance, real estate and renting and leasing': 'F.I.R.E.',
    'Fishing, hunting and trapping [114]': 'Fishing, hunting and trapping', 
    'Food manufacturing [311]': 'Food manuf.',
    'Forestry and logging [113]': 'Forestry and logging',
    'Furniture and related product manufacturing [337]': 'Furniture and related product manuf.',
    'Health care and social assistance (except hospitals)': 'Health care and social assistance',
    'Information and cultural industries [51]': 'Information and cultural industries',
    'Machinery manufacturing [333]': 'Machinery manuf.',
    'Mining (except oil and gas) [212]': 'Mining',
    'Miscellaneous manufacturing [339]': 'Miscellaneous manuf.',
    'Non-metallic mineral product manufacturing [327]': 'Non-metallic mineral product manuf.',
    'Oil and gas extraction [211]': 'Oil and gas extract.',
    'Other services (except public administration) [81]': 'Other services',
    'Paper manufacturing [322]': 'Paper manuf.',
    'Petroleum and coal products manufacturing [324]': 'Petroleum and coal products manuf.',
    'Plastics and rubber products manufacturing [326]': 'Plastics and rubber products manuf.',
    'Primary metal manufacturing [331]': 'Primary metal manuf.',
    'Printing and related support activities [323]': 'Printing and related supp. activities',
    'Professional, scientific and technical services [54]': 'Professional, scientific and technical services',
    'Retail trade [44-45]': 'Retail trade',
    'Support activities for agriculture and forestry [115]': 'Supp. activities for agriculture and forestry',
    'Support activities for mining and oil and gas extraction [213]': 'Supp. activities for mining and oil and gas extract.',
    'Textile and textile product mills': 'Textile and textile product mills',
    'Transportation and warehousing [48-49]': 'Transportation and warehousing',
    'Transportation equipment manufacturing [336]': 'Transportation equip. manuf.', 
    'Utilities [221]': 'Utilities',
    'Wholesale trade [41]': 'Wholesale trade', 
    'Wood product manufacturing [321]': 'Wood product manuf.'
}

# Calculate the different terms between 1961 and 2019
df_i = df.copy(deep=True)
df_i['within_1'] = df_i['b_1961'] * df_i['tfp_growth']
df_i['between_1'] = (df_i['b'] - df_i['b_1961']) * df_i['tfp_growth']
df_i['capital_reallocation'] = (df_i['b'] * df_i['alpha_k'] - df_i['omega_k'] * df_i['lambda_k']) * df_i['capital_growth']
df_i['labor_reallocation'] = (df_i['b'] * df_i['alpha_l'] - df_i['omega_l'] * df_i['lambda_l']) * df_i['labor_growth']
df_i['total'] = df_i['within_1'] + df_i['between_1'] + df_i['capital_reallocation'] + df_i['labor_reallocation']
df_i['industry'] = df_i['industry'].map(label_map)
df_i = df_i.loc[df_i['year'] > 1961, ['year', 'naics', 'industry', 'within_1', 'between_1', 'capital_reallocation', 'labor_reallocation', 'total']]
df_i = df_i.groupby('naics', as_index=False).agg({'within_1': 'sum', 'between_1': 'sum', 'capital_reallocation': 'sum', 'labor_reallocation': 'sum', 'total': 'sum', 'industry': lambda x: x.unique()[0]})
df_i = df_i.sort_values(by='total', ascending=False)

# Calculate the growth rates of TFP and value added
df_tfp = df.loc[(df['year'] == 1961) | (df['year'] == 2019), ['industry', 'tfp', 'va']]
df_tfp.loc[:, ['tfp', 'va']] = df_tfp.groupby('industry', as_index=False)[['tfp', 'va']].transform(lambda x: np.log(x).diff() / (2019 - 1961))
df_tfp = df_tfp.dropna(subset=['tfp', 'va'])
df_tfp['industry'] = df_tfp['industry'].map(label_map)
df_i = pd.merge(df_i, df_tfp[['industry', 'tfp', 'va']], on='industry', how='left')

# Initialize the figure
fig, ax = plt.subplots(figsize=(10, 5))

# Set the background color of the figure to transparent
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)

# Plot the data
ax.bar(df_i['industry'], df_i['total'], color=palette[1], linewidth=0.5, edgecolor='k')

# Set the horizontal axis
ax.tick_params(axis='x', labelrotation=90)

# Set the vertical axis
ax.set_ylim(-0.1, 0.06)
ax.set_yticks(np.arange(-0.1, 0.06 + 0.01, 0.02))
ax.set_yticklabels([str(x) + r'\%' for x in range(-10, 6 + 1, 2)], fontsize=12)
ax.set_ylabel('TFP growth contribution', fontsize=12, rotation=0, ha='left')
ax.yaxis.set_label_coords(0, 1.01)

# Remove the top and right axes
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, which='major', axis='y', color='gray', linestyle=':', linewidth=0.5)

# Add a note about the data source
ax.text(1, 1.01, 'Source: Statistics Canada', fontsize=8, color='k', ha='right', va='bottom', transform=ax.transAxes)

# Save and close the figure
fig.savefig(os.path.join(Path(os.getcwd()).parent, 'Figures', 'tfp_contribution_industry.png'), transparent=True, dpi=300, bbox_inches='tight')
plt.close()

# Initialize the figure
fig, ax = plt.subplots(figsize=(10, 5))

# Set the background color of the figure to transparent
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)

# Plot the data
ax.bar(df_i['industry'], df_i['tfp'], color=palette[1], linewidth=0.5, edgecolor='k')

# Set the horizontal axis
ax.tick_params(axis='x', labelrotation=90)

# Set the vertical axis
ax.set_ylim(-0.02, 0.04)
ax.set_yticks(np.arange(-0.02, 0.04 + 0.001, 0.01))
ax.set_yticklabels([str(x) + r'\%' for x in range(-2, 4 + 1, 1)], fontsize=12)
ax.set_ylabel('TFP growth', fontsize=12, rotation=0, ha='left')
ax.yaxis.set_label_coords(0, 1.01)

# Remove the top and right axes
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, which='major', axis='y', color='gray', linestyle=':', linewidth=0.5)

# Add a note about the data source
ax.text(1, 1.01, 'Source: Statistics Canada', fontsize=8, color='k', ha='right', va='bottom', transform=ax.transAxes)

# Save and close the figure
fig.savefig(os.path.join(Path(os.getcwd()).parent, 'Figures', 'tfp_growth_industry.png'), transparent=True, dpi=300, bbox_inches='tight')
plt.close()

# Initialize the figure
fig, ax = plt.subplots(figsize=(10, 5))

# Set the background color of the figure to transparent
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)

# Plot the data
ax.bar(df_i['industry'], df_i['va'], color=palette[1], linewidth=0.5, edgecolor='k')

# Set the horizontal axis
ax.tick_params(axis='x', labelrotation=90)

# Set the vertical axis
ax.set_ylim(0, 0.12)
ax.set_yticks(np.arange(0, 0.12 + 0.01, 0.02))
ax.set_yticklabels([str(x) + r'\%' for x in range(0, 12 + 1, 2)], fontsize=12)
ax.set_ylabel('Nominal GDP growth', fontsize=12, rotation=0, ha='left')
ax.yaxis.set_label_coords(0, 1.01)

# Remove the top and right axes
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, which='major', axis='y', color='gray', linestyle=':', linewidth=0.5)

# Add a note about the data source
ax.text(1, 1.01, 'Source: Statistics Canada', fontsize=8, color='k', ha='right', va='bottom', transform=ax.transAxes)

# Save and close the figure
fig.savefig(os.path.join(Path(os.getcwd()).parent, 'Figures', 'va_growth_industry.png'), transparent=True, dpi=300, bbox_inches='tight')
plt.close()

########################################################################
# Plot TFP growth against several variables by industry                #
########################################################################

# Only keep the relevant years and columns
df_tfp = df.loc[(df['year'] == 1961) | (df['year'] == 2019), ['year', 'naics', 'tfp', 'va', 'real_va', 'price', 'wage', 'capital_price']]

# Calculate the growth rates
df_tfp.loc[:, ['tfp', 'va', 'real_va', 'price', 'wage', 'capital_price']] = df_tfp.groupby('naics', as_index=False)[['tfp', 'va', 'real_va', 'price', 'wage', 'capital_price']].transform(lambda x: np.log(x).diff() / (2019 - 1961))
df_tfp = df_tfp.dropna(subset=['tfp', 'va', 'real_va', 'price', 'wage', 'capital_price'])

# Calculate the average capital and labor costs
df_cost = df.loc[:, ['year', 'naics', 'capital_cost', 'labor_cost']]
df_cost['alpha_k'] = df_cost['capital_cost'] / (df_cost['capital_cost'] + df_cost['labor_cost'])
df_cost['alpha_l'] = df_cost['labor_cost'] / (df_cost['capital_cost'] + df_cost['labor_cost'])
df_cost = df_cost.groupby('naics', as_index=False).agg({'alpha_k': 'mean', 'alpha_l': 'mean'})
df_tfp = pd.merge(df_tfp, df_cost, on='naics', how='left')

# Initialize the figure
fig, ax = plt.subplots(figsize=(8, 6))

# Set the background color of the figure to transparent
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)

# Plot the data
ax.scatter(df_tfp['tfp'], df_tfp['va'], color=palette[1], edgecolor='k', linewidths=0.75, s=75, zorder=3)

# Plot the OLS regression line
slope, intercept = np.polyfit(df_tfp['tfp'], df_tfp['va'], 1)
x = np.linspace(-0.02, 0.04, 100)
y = slope * x + intercept
ax.plot(x, y, color=palette[0], linestyle='dotted')

# Set the horizontal axis
ax.set_xlim(-0.02, 0.04)
ax.set_xticks(np.arange(-0.02, 0.04 + 0.001, 0.01))
ax.set_xticklabels([str(x) + r'\%' for x in range(-2, 4 + 1, 1)], fontsize=14)
ax.set_xlabel('Annual TFP growth', fontsize=14)

# Set the vertical axis
ax.set_ylim(0, 0.12)
ax.set_yticks(np.arange(0, 0.12 + 0.01, 0.02))
ax.set_yticklabels([str(x) + r'\%' for x in range(0, 12 + 1, 2)], fontsize=14)
ax.set_ylabel('Annual GDP growth', fontsize=14, rotation=0, ha='left')
ax.yaxis.set_label_coords(0, 1.01)

# Remove the top and right axes
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, which='major', axis='y', color='gray', linestyle=':', linewidth=0.5)

# Identify the oil and gas extraction industry
position_211 = (df_tfp.loc[df_tfp['naics'] == '211', 'tfp'].values[0], df_tfp.loc[df_tfp['naics'] == '211', 'va'].values[0])
ax.text(position_211[0] + 0.003, position_211[1] - 0.02, 'Oil and gas extraction', fontsize=10, color='k', ha='center', va='center')
ax.annotate('', xy=(position_211[0] + 0.003, position_211[1] - 0.0175), xytext=position_211, arrowprops=dict(arrowstyle='->', color='k', lw=1), zorder=1)

# Identify the computer and electronic product manufacturing industry
position_334 = (df_tfp.loc[df_tfp['naics'] == '334', 'tfp'].values[0], df_tfp.loc[df_tfp['naics'] == '334', 'va'].values[0])
ax.text(position_334[0] + 0.002, position_334[1] + 0.015, 'Computer and electronic\nproduct manufacturing', fontsize=10, color='k', ha='center', va='center')
ax.annotate('', xy=(position_334[0] + 0.002, position_334[1] + 0.01), xytext=position_334, arrowprops=dict(arrowstyle='->', color='k', lw=1), zorder=1)

# Identify the arts, entertainment and recreation industry
position_71 = (df_tfp.loc[df_tfp['naics'] == '71', 'tfp'].values[0], df_tfp.loc[df_tfp['naics'] == '71', 'va'].values[0])
ax.text(position_71[0] + 0.0065, position_71[1] + 0.02, 'Arts, entertainment\nand recreation', fontsize=10, color='k', ha='center', va='center')
ax.annotate('', xy=(position_71[0] + 0.0065, position_71[1] + 0.015), xytext=position_71, arrowprops=dict(arrowstyle='->', color='k', lw=1), zorder=1)

# Identify the wood product manufacturing industry
position_321 = (df_tfp.loc[df_tfp['naics'] == '321', 'tfp'].values[0], df_tfp.loc[df_tfp['naics'] == '321', 'va'].values[0])
ax.text(position_321[0] + 0.0075, position_321[1] - 0.025, 'Wood product\nmanufacturing', fontsize=10, color='k', ha='center', va='center')
ax.annotate('', xy=(position_321[0] + 0.0075, position_321[1] - 0.02), xytext=position_321, arrowprops=dict(arrowstyle='->', color='k', lw=1), zorder=1)

# Add a note about the data source
ax.text(1, 1.01, 'Source: Statistics Canada', fontsize=8, color='k', ha='right', va='bottom', transform=ax.transAxes)

# Save and close the figure
fig.tight_layout()
fig.savefig(os.path.join(Path(os.getcwd()).parent, 'Figures', 'va_tfp_growth.png'), transparent=True, dpi=300)
plt.close()

# Initialize the figure
fig, ax = plt.subplots(figsize=(8, 6))

# Set the background color of the figure to transparent
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)

# Plot the data
ax.scatter(df_tfp['tfp'], df_tfp['real_va'], color=palette[1], edgecolor='k', linewidths=0.75, s=75, zorder=3)

# Plot the OLS regression line
slope, intercept = np.polyfit(df_tfp['tfp'], df_tfp['real_va'], 1)
x = np.linspace(-0.02, 0.04, 100)
y = slope * x + intercept
ax.plot(x, y, color=palette[0], linestyle='dotted')

# Set the horizontal axis
ax.set_xlim(-0.02, 0.04)
ax.set_xticks(np.arange(-0.02, 0.04 + 0.001, 0.01))
ax.set_xticklabels([str(x) + r'\%' for x in range(-2, 4 + 1, 1)], fontsize=14)
ax.set_xlabel('Annual TFP growth', fontsize=14)

# Set the vertical axis
ax.set_ylim(-0.04, 0.08)
ax.set_yticks(np.arange(-0.04, 0.08 + 0.01, 0.02))
ax.set_yticklabels([str(x) + r'\%' for x in range(-2, 10 + 1, 2)], fontsize=14)
ax.set_ylabel('Annual real GDP growth', fontsize=14, rotation=0, ha='left')
ax.yaxis.set_label_coords(0, 1.01)

# Remove the top and right axes
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, which='major', axis='y', color='gray', linestyle=':', linewidth=0.5)

# Identify the oil and gas extraction industry
position_211 = (df_tfp.loc[df_tfp['naics'] == '211', 'tfp'].values[0], df_tfp.loc[df_tfp['naics'] == '211', 'real_va'].values[0])
ax.text(position_211[0] + 0.004, position_211[1] - 0.025, 'Oil and gas extraction', fontsize=10, color='k', ha='center', va='center')
ax.annotate('', xy=(position_211[0] + 0.004, position_211[1] - 0.0225), xytext=position_211, arrowprops=dict(arrowstyle='->', color='k', lw=1), zorder=1)

# Identify the computer and electronic product manufacturing industry
position_334 = (df_tfp.loc[df_tfp['naics'] == '334', 'tfp'].values[0], df_tfp.loc[df_tfp['naics'] == '334', 'real_va'].values[0])
ax.text(position_334[0] + 0.002, position_334[1] + 0.02, 'Computer and electronic\nproduct manufacturing', fontsize=10, color='k', ha='center', va='center')
ax.annotate('', xy=(position_334[0] + 0.002, position_334[1] + 0.015), xytext=position_334, arrowprops=dict(arrowstyle='->', color='k', lw=1), zorder=1)

# Identify the arts, entertainment and recreation industry
position_71 = (df_tfp.loc[df_tfp['naics'] == '71', 'tfp'].values[0], df_tfp.loc[df_tfp['naics'] == '71', 'real_va'].values[0])
ax.text(position_71[0] + 0.0065, position_71[1] + 0.025, 'Arts, entertainment\nand recreation', fontsize=10, color='k', ha='center', va='center')
ax.annotate('', xy=(position_71[0] + 0.0065, position_71[1] + 0.02), xytext=position_71, arrowprops=dict(arrowstyle='->', color='k', lw=1), zorder=1)

# Identify the wood product manufacturing industry
position_321 = (df_tfp.loc[df_tfp['naics'] == '321', 'tfp'].values[0], df_tfp.loc[df_tfp['naics'] == '321', 'real_va'].values[0])
ax.text(position_321[0] + 0.0075, position_321[1] - 0.02, 'Wood product\nmanufacturing', fontsize=10, color='k', ha='center', va='center')
ax.annotate('', xy=(position_321[0] + 0.0075, position_321[1] - 0.015), xytext=position_321, arrowprops=dict(arrowstyle='->', color='k', lw=1), zorder=1)

# Add a note about the data source
ax.text(1, 1.01, 'Source: Statistics Canada', fontsize=8, color='k', ha='right', va='bottom', transform=ax.transAxes)

# Save and close the figure
fig.tight_layout()
fig.savefig(os.path.join(Path(os.getcwd()).parent, 'Figures', 'real_va_tfp_growth.png'), transparent=True, dpi=300)
plt.close()

# Initialize the figure
fig, ax = plt.subplots(figsize=(8, 6))

# Set the background color of the figure to transparent
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)

# Plot the data
ax.scatter(df_tfp['tfp'], df_tfp['price'], color=palette[1], edgecolor='k', linewidths=0.75, s=75, zorder=3)

# Plot the OLS regression line
slope, intercept = np.polyfit(df_tfp['tfp'], df_tfp['price'], 1)
x = np.linspace(-0.02, 0.04, 100)
y = slope * x + intercept
ax.plot(x, y, color=palette[0], linestyle='dotted')

# Set the horizontal axis
ax.set_xlim(-0.02, 0.04)
ax.set_xticks(np.arange(-0.02, 0.04 + 0.001, 0.01))
ax.set_xticklabels([str(x) + r'\%' for x in range(-2, 4 + 1, 1)], fontsize=14)
ax.set_xlabel('Annual TFP growth', fontsize=14)

# Set the vertical axis
ax.set_ylim(0, 0.07)
ax.set_yticks(np.arange(0, 0.07 + 0.01, 0.01))
ax.set_yticklabels([str(x) + r'\%' for x in range(0, 7 + 1, 1)], fontsize=14)
ax.set_ylabel('Annual price growth', fontsize=14, rotation=0, ha='left')
ax.yaxis.set_label_coords(0, 1.01)

# Remove the top and right axes
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, which='major', axis='y', color='gray', linestyle=':', linewidth=0.5)

# Identify the oil and gas extraction industry
position_211 = (df_tfp.loc[df_tfp['naics'] == '211', 'tfp'].values[0], df_tfp.loc[df_tfp['naics'] == '211', 'price'].values[0])
ax.text(position_211[0] + 0.0035, position_211[1] - 0.02, 'Oil and gas extraction', fontsize=10, color='k', ha='center', va='center')
ax.annotate('', xy=(position_211[0] + 0.003, position_211[1] - 0.019), xytext=position_211, arrowprops=dict(arrowstyle='->', color='k', lw=1), zorder=1)

# Identify the computer and electronic product manufacturing industry
position_334 = (df_tfp.loc[df_tfp['naics'] == '334', 'tfp'].values[0], df_tfp.loc[df_tfp['naics'] == '334', 'price'].values[0])
ax.text(position_334[0] - 0.0125, position_334[1], 'Computer and electronic\nproduct manufacturing', fontsize=10, color='k', ha='center', va='center')
ax.annotate('', xy=(position_334[0] - 0.006, position_334[1]), xytext=position_334, arrowprops=dict(arrowstyle='->', color='k', lw=1), zorder=1)

# Identify the arts, entertainment and recreation industry
position_71 = (df_tfp.loc[df_tfp['naics'] == '71', 'tfp'].values[0], df_tfp.loc[df_tfp['naics'] == '71', 'price'].values[0])
ax.text(position_71[0] + 0.01, position_71[1] + 0.0075, 'Arts, entertainment\nand recreation', fontsize=10, color='k', ha='center', va='center')
ax.annotate('', xy=(position_71[0] + 0.01, position_71[1] + 0.005), xytext=position_71, arrowprops=dict(arrowstyle='->', color='k', lw=1), zorder=1)

# Identify the wood product manufacturing industry
position_321 = (df_tfp.loc[df_tfp['naics'] == '321', 'tfp'].values[0], df_tfp.loc[df_tfp['naics'] == '321', 'price'].values[0])
ax.text(position_321[0] + 0.009, position_321[1], 'Wood product\nmanufacturing', fontsize=10, color='k', ha='center', va='center')
ax.annotate('', xy=(position_321[0] + 0.005, position_321[1]), xytext=position_321, arrowprops=dict(arrowstyle='->', color='k', lw=1), zorder=1)

# Add a note about the data source
ax.text(1, 1.01, 'Source: Statistics Canada', fontsize=8, color='k', ha='right', va='bottom', transform=ax.transAxes)

# Save and close the figure
fig.tight_layout()
fig.savefig(os.path.join(Path(os.getcwd()).parent, 'Figures', 'price_tfp_growth.png'), transparent=True, dpi=300)
plt.close()

# Initialize the figure
fig, ax = plt.subplots(figsize=(8, 6))

# Set the background color of the figure to transparent
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)

# Plot the data
ax.scatter(df_tfp['tfp'], df_tfp['wage'], color=palette[1], edgecolor='k', linewidths=0.75, s=75, zorder=3)

# Plot the OLS regression line
slope, intercept = np.polyfit(df_tfp['tfp'], df_tfp['wage'], 1)
x = np.linspace(-0.02, 0.04, 100)
y = slope * x + intercept
ax.plot(x, y, color=palette[0], linestyle='dotted')

# Set the horizontal axis
ax.set_xlim(-0.02, 0.04)
ax.set_xticks(np.arange(-0.02, 0.04 + 0.001, 0.01))
ax.set_xticklabels([str(x) + r'\%' for x in range(-2, 4 + 1, 1)], fontsize=14)
ax.set_xlabel('Annual TFP growth', fontsize=14)

# Set the vertical axis
ax.set_ylim(0.02, 0.09)
ax.set_yticks(np.arange(0.02, 0.09 + 0.01, 0.01))
ax.set_yticklabels([str(x) + r'\%' for x in range(2, 9 + 1, 1)], fontsize=14)
ax.set_ylabel('Annual wage growth', fontsize=14, rotation=0, ha='left')
ax.yaxis.set_label_coords(0, 1.01)

# Remove the top and right axes
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, which='major', axis='y', color='gray', linestyle=':', linewidth=0.5)

# Identify the oil and gas extraction industry
position_211 = (df_tfp.loc[df_tfp['naics'] == '211', 'tfp'].values[0], df_tfp.loc[df_tfp['naics'] == '211', 'wage'].values[0])
ax.text(position_211[0] + 0.004, position_211[1] - 0.015, 'Oil and gas extraction', fontsize=10, color='k', ha='center', va='center')
ax.annotate('', xy=(position_211[0] + 0.004, position_211[1] - 0.0135), xytext=position_211, arrowprops=dict(arrowstyle='->', color='k', lw=1), zorder=1)

# Identify the computer and electronic product manufacturing industry
position_334 = (df_tfp.loc[df_tfp['naics'] == '334', 'tfp'].values[0], df_tfp.loc[df_tfp['naics'] == '334', 'wage'].values[0])
ax.text(position_334[0] + 0.001, position_334[1] + 0.01, 'Computer and electronic\nproduct manufacturing', fontsize=10, color='k', ha='center', va='center')
ax.annotate('', xy=(position_334[0] + 0.001, position_334[1] + 0.0075), xytext=position_334, arrowprops=dict(arrowstyle='->', color='k', lw=1), zorder=1)

# Identify the arts, entertainment and recreation industry
position_71 = (df_tfp.loc[df_tfp['naics'] == '71', 'tfp'].values[0], df_tfp.loc[df_tfp['naics'] == '71', 'wage'].values[0])
ax.text(position_71[0] + 0.0065, position_71[1] + 0.015, 'Arts, entertainment\nand recreation', fontsize=10, color='k', ha='center', va='center')
ax.annotate('', xy=(position_71[0] + 0.0065, position_71[1] + 0.0125), xytext=position_71, arrowprops=dict(arrowstyle='->', color='k', lw=1), zorder=1)

# Identify the wood product manufacturing industry
position_321 = (df_tfp.loc[df_tfp['naics'] == '321', 'tfp'].values[0], df_tfp.loc[df_tfp['naics'] == '321', 'wage'].values[0])
ax.text(position_321[0] + 0.005, position_321[1] - 0.015, 'Wood product\nmanufacturing', fontsize=10, color='k', ha='center', va='center')
ax.annotate('', xy=(position_321[0] + 0.005, position_321[1] - 0.0125), xytext=position_321, arrowprops=dict(arrowstyle='->', color='k', lw=1), zorder=1)

# Add a note about the data source
ax.text(1, 1.01, 'Source: Statistics Canada', fontsize=8, color='k', ha='right', va='bottom', transform=ax.transAxes)

# Save and close the figure
fig.tight_layout()
fig.savefig(os.path.join(Path(os.getcwd()).parent, 'Figures', 'wage_tfp_growth.png'), transparent=True, dpi=300)
plt.close()

# Initialize the figure
fig, ax = plt.subplots(figsize=(8, 6))

# Set the background color of the figure to transparent
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)

# Plot the data
ax.scatter(df_tfp['tfp'], df_tfp['capital_price'], color=palette[1], edgecolor='k', linewidths=0.75, s=75, zorder=3)

# Plot the OLS regression line
slope, intercept = np.polyfit(df_tfp['tfp'], df_tfp['capital_price'], 1)
x = np.linspace(-0.02, 0.04, 100)
y = slope * x + intercept
ax.plot(x, y, color=palette[0], linestyle='dotted')

# Set the horizontal axis
ax.set_xlim(-0.02, 0.04)
ax.set_xticks(np.arange(-0.02, 0.04 + 0.001, 0.01))
ax.set_xticklabels([str(x) + r'\%' for x in range(-2, 4 + 1, 1)], fontsize=14)
ax.set_xlabel('Annual TFP growth', fontsize=14)

# Set the vertical axis
ax.set_ylim(-0.01, 0.07)
ax.set_yticks(np.arange(-0.01, 0.07 + 0.01, 0.01))
ax.set_yticklabels([str(x) + r'\%' for x in range(-1, 7 + 1, 1)], fontsize=14)
ax.set_ylabel('Annual capital price growth', fontsize=14, rotation=0, ha='left')
ax.yaxis.set_label_coords(0, 1.01)

# Remove the top and right axes
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, which='major', axis='y', color='gray', linestyle=':', linewidth=0.5)

# Identify the oil and gas extraction industry
position_211 = (df_tfp.loc[df_tfp['naics'] == '211', 'tfp'].values[0], df_tfp.loc[df_tfp['naics'] == '211', 'capital_price'].values[0])
ax.text(position_211[0] + 0.0035, position_211[1] + 0.015, 'Oil and gas extraction', fontsize=10, color='k', ha='center', va='center')
ax.annotate('', xy=(position_211[0] + 0.003, position_211[1] + 0.014), xytext=position_211, arrowprops=dict(arrowstyle='->', color='k', lw=1), zorder=1)

# Identify the computer and electronic product manufacturing industry
position_334 = (df_tfp.loc[df_tfp['naics'] == '334', 'tfp'].values[0], df_tfp.loc[df_tfp['naics'] == '334', 'capital_price'].values[0])
ax.text(position_334[0], position_334[1] + 0.01, 'Computer and electronic\nproduct manufacturing', fontsize=10, color='k', ha='center', va='center')
ax.annotate('', xy=(position_334[0], position_334[1] + 0.0075), xytext=position_334, arrowprops=dict(arrowstyle='->', color='k', lw=1), zorder=1)

# Identify the arts, entertainment and recreation industry
position_71 = (df_tfp.loc[df_tfp['naics'] == '71', 'tfp'].values[0], df_tfp.loc[df_tfp['naics'] == '71', 'capital_price'].values[0])
ax.text(position_71[0] + 0.0075, position_71[1] + 0.005, 'Arts, entertainment\nand recreation', fontsize=10, color='k', ha='center', va='center')
ax.annotate('', xy=(position_71[0] + 0.0075, position_71[1] + 0.0025), xytext=position_71, arrowprops=dict(arrowstyle='->', color='k', lw=1), zorder=1)

# Identify the wood product manufacturing industry
position_321 = (df_tfp.loc[df_tfp['naics'] == '321', 'tfp'].values[0], df_tfp.loc[df_tfp['naics'] == '321', 'capital_price'].values[0])
ax.text(position_321[0] + 0.009, position_321[1] + 0.01, 'Wood product\nmanufacturing', fontsize=10, color='k', ha='center', va='center')
ax.annotate('', xy=(position_321[0] + 0.009, position_321[1] + 0.0075), xytext=position_321, arrowprops=dict(arrowstyle='->', color='k', lw=1), zorder=1)

# Add a note about the data source
ax.text(1, 1.01, 'Source: Statistics Canada', fontsize=8, color='k', ha='right', va='bottom', transform=ax.transAxes)

# Save and close the figure
fig.tight_layout()
fig.savefig(os.path.join(Path(os.getcwd()).parent, 'Figures', 'capital_price_tfp_growth.png'), transparent=True, dpi=300)
plt.close()

# Initialize the figure
fig, ax = plt.subplots(figsize=(8, 6))

# Set the background color of the figure to transparent
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)

# Plot the data
ax.scatter(df_tfp['tfp'], df_tfp['alpha_k'], color=palette[1], edgecolor='k', linewidths=0.75, s=75, zorder=3)

# Plot the OLS regression line
slope, intercept = np.polyfit(df_tfp['tfp'], df_tfp['alpha_k'], 1)
x = np.linspace(-0.02, 0.04, 100)
y = slope * x + intercept
ax.plot(x, y, color=palette[0], linestyle='dotted')

# Set the horizontal axis
ax.set_xlim(-0.02, 0.04)
ax.set_xticks(np.arange(-0.02, 0.04 + 0.001, 0.01))
ax.set_xticklabels([str(x) + r'\%' for x in range(-2, 4 + 1, 1)], fontsize=14)
ax.set_xlabel('Annual TFP growth', fontsize=14)

# Set the vertical axis
ax.set_ylim(0, 1)
ax.set_yticks(np.arange(0, 1 + 0.01, 0.2))
ax.set_yticklabels([str(x) + r'\%' for x in range(0, 100 + 1, 20)], fontsize=14)
ax.set_ylabel('Average capital cost share', fontsize=14, rotation=0, ha='left')
ax.yaxis.set_label_coords(0, 1.01)

# Remove the top and right axes
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, which='major', axis='y', color='gray', linestyle=':', linewidth=0.5)

# Add a note about the data source
ax.text(1, 1.01, 'Source: Statistics Canada', fontsize=8, color='k', ha='right', va='bottom', transform=ax.transAxes)

# Save and close the figure
fig.tight_layout()
fig.savefig(os.path.join(Path(os.getcwd()).parent, 'Figures', 'capital_share_tfp_growth.png'), transparent=True, dpi=300)
plt.close()

# Initialize the figure
fig, ax = plt.subplots(figsize=(8, 6))

# Set the background color of the figure to transparent
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)

# Plot the data
ax.scatter(df_tfp['tfp'], df_tfp['alpha_l'], color=palette[1], edgecolor='k', linewidths=0.75, s=75, zorder=3)

# Plot the OLS regression line
slope, intercept = np.polyfit(df_tfp['tfp'], df_tfp['alpha_l'], 1)
x = np.linspace(-0.02, 0.04, 100)
y = slope * x + intercept
ax.plot(x, y, color=palette[0], linestyle='dotted')

# Set the horizontal axis
ax.set_xlim(-0.02, 0.04)
ax.set_xticks(np.arange(-0.02, 0.04 + 0.001, 0.01))
ax.set_xticklabels([str(x) + r'\%' for x in range(-2, 4 + 1, 1)], fontsize=14)
ax.set_xlabel('Annual TFP growth', fontsize=14)

# Set the vertical axis
ax.set_ylim(0, 1)
ax.set_yticks(np.arange(0, 1 + 0.01, 0.2))
ax.set_yticklabels([str(x) + r'\%' for x in range(0, 100 + 1, 20)], fontsize=14)
ax.set_ylabel('Average labor cost share', fontsize=14, rotation=0, ha='left')
ax.yaxis.set_label_coords(0, 1.01)

# Remove the top and right axes
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, which='major', axis='y', color='gray', linestyle=':', linewidth=0.5)

# Add a note about the data source
ax.text(1, 1.01, 'Source: Statistics Canada', fontsize=8, color='k', ha='right', va='bottom', transform=ax.transAxes)

# Save and close the figure
fig.tight_layout()
fig.savefig(os.path.join(Path(os.getcwd()).parent, 'Figures', 'labor_share_tfp_growth.png'), transparent=True, dpi=300)
plt.close()

########################################################################
# Plot TFP growth against several variables across industries for the  #
# two periods of the analysis                                          # 
########################################################################

# Only keep the relevant years and columns
df_tfp_1 = df.loc[(df['year'] == 1961) | (df['year'] == 1980), ['year', 'naics', 'tfp', 'va', 'real_va', 'price', 'wage']]
df_tfp_2 = df.loc[(df['year'] == 1980) | (df['year'] == 2000), ['year', 'naics', 'tfp', 'va', 'real_va', 'price', 'wage']]
df_tfp_3 = df.loc[(df['year'] == 2000) | (df['year'] == 2019), ['year', 'naics', 'tfp', 'va', 'real_va', 'price', 'wage']]

# Calculate the growth rates
df_tfp_1.loc[:, ['tfp', 'va', 'real_va', 'price', 'wage']] = df_tfp_1.groupby('naics', as_index=False)[['tfp', 'va', 'real_va', 'price', 'wage']].transform(lambda x: np.log(x).diff() / (1980 - 1961))
df_tfp_1 = df_tfp_1.dropna(subset=['tfp', 'va', 'real_va', 'price', 'wage'])
df_tfp_2.loc[:, ['tfp', 'va', 'real_va', 'price', 'wage']] = df_tfp_2.groupby('naics', as_index=False)[['tfp', 'va', 'real_va', 'price', 'wage']].transform(lambda x: np.log(x).diff() / (2000 - 1980))
df_tfp_2 = df_tfp_2.dropna(subset=['tfp', 'va', 'real_va', 'price', 'wage'])
df_tfp_3.loc[:, ['tfp', 'va', 'real_va', 'price', 'wage']] = df_tfp_3.groupby('naics', as_index=False)[['tfp', 'va', 'real_va', 'price', 'wage']].transform(lambda x: np.log(x).diff() / (2019 - 2000))
df_tfp_3 = df_tfp_3.dropna(subset=['tfp', 'va', 'real_va', 'price', 'wage'])

# Initialize the figure
fig, ax = plt.subplots(figsize=(8, 6))

# Set the background color of the figure to transparent
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)

# Plot the data
ax.scatter(df_tfp_1['tfp'], df_tfp_1['va'], color=palette[0], edgecolor='k', linewidths=0.75, s=75, label='1961-1980')
ax.scatter(df_tfp_2['tfp'], df_tfp_2['va'], color=palette[1], edgecolor='k', linewidths=0.75, s=75, label='1980-2000')
ax.scatter(df_tfp_3['tfp'], df_tfp_3['va'], color=palette[2], edgecolor='k', linewidths=0.75, s=75, label='2000-2019')

# Plot the OLS regression lines
x = np.linspace(-0.04, 0.07, 100)
slope_1, intercept_1 = np.polyfit(df_tfp_1['tfp'], df_tfp_1['va'], 1)
slope_2, intercept_2 = np.polyfit(df_tfp_2['tfp'], df_tfp_2['va'], 1)
slope_3, intercept_3 = np.polyfit(df_tfp_3['tfp'], df_tfp_3['va'], 1)
y_1 = slope_1 * x + intercept_1
y_2 = slope_2 * x + intercept_2
y_3 = slope_3 * x + intercept_3
ax.plot(x, y_1, color=palette[0], linestyle='dotted')
ax.plot(x, y_2, color=palette[1], linestyle='dotted')
ax.plot(x, y_3, color=palette[2], linestyle='dotted')

# Set the horizontal axis
ax.set_xlim(-0.04, 0.07)
ax.set_xticks(np.arange(-0.04, 0.07 + 0.001, 0.01))
ax.set_xticklabels([str(x) + r'\%' for x in range(-4, 7 + 1, 1)], fontsize=12)
ax.set_xlabel('Annual TFP growth', fontsize=12)

# Set the vertical axis
ax.set_ylim(-0.06, 0.18)
ax.set_yticks(np.arange(-0.06, 0.18 + 0.01, 0.04))
ax.set_yticklabels([str(x) + r'\%' for x in range(-6, 18 + 1, 4)], fontsize=12)
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

# Initialize the figure
fig, ax = plt.subplots(figsize=(8, 6))

# Set the background color of the figure to transparent
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)

# Plot the data
ax.scatter(df_tfp_1['tfp'], df_tfp_1['real_va'], color=palette[0], edgecolor='k', linewidths=0.75, s=75, label='1961-1980')
ax.scatter(df_tfp_2['tfp'], df_tfp_2['real_va'], color=palette[1], edgecolor='k', linewidths=0.75, s=75, label='1980-2000')
ax.scatter(df_tfp_3['tfp'], df_tfp_3['real_va'], color=palette[2], edgecolor='k', linewidths=0.75, s=75, label='2000-2019')

# Plot the OLS regression lines
x = np.linspace(-0.04, 0.07, 100)
slope_1, intercept_1 = np.polyfit(df_tfp_1['tfp'], df_tfp_1['real_va'], 1)
slope_2, intercept_2 = np.polyfit(df_tfp_2['tfp'], df_tfp_2['real_va'], 1)
slope_3, intercept_3 = np.polyfit(df_tfp_3['tfp'], df_tfp_3['real_va'], 1)
y_1 = slope_1 * x + intercept_1
y_2 = slope_2 * x + intercept_2
y_3 = slope_3 * x + intercept_3
ax.plot(x, y_1, color=palette[0], linestyle='dotted')
ax.plot(x, y_2, color=palette[1], linestyle='dotted')
ax.plot(x, y_3, color=palette[2], linestyle='dotted')

# Set the horizontal axis
ax.set_xlim(-0.04, 0.07)
ax.set_xticks(np.arange(-0.04, 0.07 + 0.001, 0.01))
ax.set_xticklabels([str(x) + r'\%' for x in range(-4, 7 + 1, 1)], fontsize=12)
ax.set_xlabel('Annual TFP growth', fontsize=12)

# Set the vertical axis
ax.set_ylim(-0.1, 0.14)
ax.set_yticks(np.arange(-0.1, 0.14 + 0.01, 0.04))
ax.set_yticklabels([str(x) + r'\%' for x in range(-10, 14 + 1, 4)], fontsize=12)
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

# Initialize the figure
fig, ax = plt.subplots(figsize=(8, 6))

# Set the background color of the figure to transparent
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)

# Plot the data
ax.scatter(df_tfp_1['tfp'], df_tfp_1['price'], color=palette[0], edgecolor='k', linewidths=0.75, s=75, label='1961-1980')
ax.scatter(df_tfp_2['tfp'], df_tfp_2['price'], color=palette[1], edgecolor='k', linewidths=0.75, s=75, label='1980-2000')
ax.scatter(df_tfp_3['tfp'], df_tfp_3['price'], color=palette[2], edgecolor='k', linewidths=0.75, s=75, label='2000-2019')

# Plot the OLS regression lines
x = np.linspace(-0.04, 0.07, 100)
slope_1, intercept_1 = np.polyfit(df_tfp_1['tfp'], df_tfp_1['price'], 1)
slope_2, intercept_2 = np.polyfit(df_tfp_2['tfp'], df_tfp_2['price'], 1)
slope_3, intercept_3 = np.polyfit(df_tfp_3['tfp'], df_tfp_3['price'], 1)
y_1 = slope_1 * x + intercept_1
y_2 = slope_2 * x + intercept_2
y_3 = slope_3 * x + intercept_3
ax.plot(x, y_1, color=palette[0], linestyle='dotted')
ax.plot(x, y_2, color=palette[1], linestyle='dotted')
ax.plot(x, y_3, color=palette[2], linestyle='dotted')

# Set the horizontal axis
ax.set_xlim(-0.04, 0.07)
ax.set_xticks(np.arange(-0.04, 0.07 + 0.001, 0.01))
ax.set_xticklabels([str(x) + r'\%' for x in range(-4, 7 + 1, 1)], fontsize=12)
ax.set_xlabel('Annual TFP growth', fontsize=12)

# Set the vertical axis
ax.set_ylim(-0.04, 0.14)
ax.set_yticks(np.arange(-0.04, 0.14 + 0.01, 0.02))
ax.set_yticklabels([str(x) + r'\%' for x in range(-4, 14 + 1, 2)], fontsize=12)
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

# Initialize the figure
fig, ax = plt.subplots(figsize=(8, 6))

# Set the background color of the figure to transparent
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)

# Plot the data
ax.scatter(df_tfp_1['tfp'], df_tfp_1['wage'], color=palette[0], edgecolor='k', linewidths=0.75, s=75, label='1961-1980')
ax.scatter(df_tfp_2['tfp'], df_tfp_2['wage'], color=palette[1], edgecolor='k', linewidths=0.75, s=75, label='1980-2000')
ax.scatter(df_tfp_3['tfp'], df_tfp_3['wage'], color=palette[2], edgecolor='k', linewidths=0.75, s=75, label='2000-2019')

# Plot the OLS regression lines
x = np.linspace(-0.04, 0.07, 100)
slope_1, intercept_1 = np.polyfit(df_tfp_1['tfp'], df_tfp_1['wage'], 1)
slope_2, intercept_2 = np.polyfit(df_tfp_2['tfp'], df_tfp_2['wage'], 1)
slope_3, intercept_3 = np.polyfit(df_tfp_3['tfp'], df_tfp_3['wage'], 1)
y_1 = slope_1 * x + intercept_1
y_2 = slope_2 * x + intercept_2
y_3 = slope_3 * x + intercept_3
ax.plot(x, y_1, color=palette[0], linestyle='dotted')
ax.plot(x, y_2, color=palette[1], linestyle='dotted')
ax.plot(x, y_3, color=palette[2], linestyle='dotted')

# Set the horizontal axis
ax.set_xlim(-0.04, 0.07)
ax.set_xticks(np.arange(-0.04, 0.07 + 0.001, 0.01))
ax.set_xticklabels([str(x) + r'\%' for x in range(-4, 7 + 1, 1)], fontsize=12)
ax.set_xlabel('Annual TFP growth', fontsize=12)

# Set the vertical axis
ax.set_ylim(-0.04, 0.14)
ax.set_yticks(np.arange(-0.04, 0.14 + 0.01, 0.02))
ax.set_yticklabels([str(x) + r'\%' for x in range(-4, 14 + 1, 2)], fontsize=12)
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