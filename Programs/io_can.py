import os
from pathlib import Path
import numpy as np
import pandas as pd
from stats_can import StatsCan
from scipy import sparse

# Initialize the StatsCan API
sc = StatsCan()

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

# Define a NAICS aggregation grouping for 2009-2012
naics_agg = {
    '11A': '111-112',
    '111': '111-112',
    '112': '111-112',
    '23A': '23',
    '23B': '23',
    '23C': '23',
    '23D': '23',
    '23E': '23',
    '31A': '313-314',
    '31B': '315-316',
    '410': '41',
    '411': '41',
    '412': '41',
    '413': '41',
    '414': '41',
    '415': '41',
    '416': '41',
    '417': '41',
    '418': '41',
    '419': '41',
    '4A0': '44-45',
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
    '48B': '48-49',
    '481': '48-49',
    '482': '48-49',
    '483': '48-49',
    '484': '48-49',
    '485': '48-49',
    '486': '48-49',
    '488': '48-49',
    '48A': '48-49',
    '49A': '48-49',
    '491': '48-49',
    '492': '48-49',
    '493': '48-49',
    '51A': '51',
    '51B': '51',
    '511': '51',
    '512': '51',
    '515': '51',
    '517': '51',
    '518': '51',
    '519': '51',
    '5A0': '52-53',
    '521': '52-53',
    '522': '52-53',
    '524': '52-53',
    '52A': '52-53',
    '52B': '52-53',
    '53A': '52-53',
    '53B': '52-53',
    '531': '52-53',
    '532': '52-53',
    '533': '52-53',
    '541': '54',
    '561': '56',
    '562': '56',
    '620': '62',
    '621': '62',
    '623': '62',
    '624': '62',
    '710': '71',
    '713': '71',
    '71A': '71',
    '720': '72',
    '721': '72',
    '722': '72',
    '81A': '81',
    '811': '81',
    '812': '81',
    '813': '81',
    '814': '81'
}

# Define a NAICS aggregation grouping for 1997-2008
naics_agg_97_08 = {
    '11A0': '111-112',
    '1130': '113',
    '1140': '114',
    '1150': '115',
    '2111': '211',
    '2121': '212',
    '2122': '212',
    '2123': '212',
    '2131': '213',
    '2211': '221',
    '221A': '221',
    '230A': '23',
    '230X': '23',
    '230H': '23',
    '230I': '23',
    '3111': '311',
    '3113': '311',
    '3114': '311',
    '3115': '311',
    '3116': '311',
    '3117': '311',
    '311A': '311',
    '312A': '312',
    '312B': '312',
    '312C': '312',
    '312D': '312',
    '3122': '312',
    '31A0': '313-314',
    '3150': '315-316',
    '3160': '315-316',
    '3210': '321',
    '3221': '322',
    '3222': '322',
    '3231': '323',
    '3241': '324',
    '3251': '325',
    '3252': '325',
    '3253': '325',
    '3254': '325',
    '325A': '325',
    '3261': '326',
    '3262': '326',
    '3273': '327',
    '327A': '327',
    '3310': '331',
    '3320': '332',
    '3330': '333',
    '3341': '334',
    '334A': '334',
    '3352': '335',
    '335A': '335',
    '3361': '336',
    '3362': '336',
    '3363': '336',
    '3364': '336',
    '3365': '336',
    '3366': '336',
    '3369': '336',
    '3370': '337',
    '3390': '339',
    '4100': '41',
    '4A00': '44-45',
    '4810': '48-49',
    '4820': '48-49',
    '4830': '48-49',
    '4840': '48-49',
    '4850': '48-49',
    '4860': '48-49',
    '48B0': '48-49',
    '49A0': '48-49',
    '4930': '48-49',
    '5120': '51',
    '5131': '51',
    '513A': '51',
    '51A0': '51',
    '51B0': '51',
    '5A01': '52-53',
    '5A02': '52-53',
    '5A03': '52-53',
    '5A04': '52-53',
    '5A05': '52-53',
    '5A06': '52-53',
    '5418': '54',
    '541A': '54',
    '541B': '54',
    '5610': '56',
    '5620': '56',
    '62A0': '62',
    '7100': '71',
    '7200': '72',
    '8110': '81',
    '813A': '81',
    '81A0': '81'
}

# Define a NAICS aggregation grouping for 1961-2008
naics_agg_61_08 = {
    'Accommodation and food services': '72',
    'Administrative and support services': '56',
    'Air, rail, water and scenic and sightseeing transportation and support activities for transportation': '48-49',
    'Arts, entertainment and recreation': '71',
    'Beverage and tobacco product manufacturing': '312',
    'Broadcasting and telecommunications': '51', 
    'Chemical manufacturing': '325',
    'Clothing manufacturing': '315-316',
    'Computer and electronic product manufacturing': '334', 
    'Construction': '23',
    'Crop and animal production': '111-112', 
    'Electric power generation, transmission and distribution': '221',
    'Electrical equipment, appliance and component manufacturing': '335',
    'Fabricated metal product manufacturing': '332',
    'Finance, insurance, real estate and rental and leasing': '52-53',
    'Fishing, hunting and trapping': '114', 
    'Food manufacturing': '311',
    'Forestry and logging': '113',
    'Furniture and related product manufacturing': '337',
    'Health care and social assistance': '62',
    'Leather and allied product manufacturing': '315-316',
    'Machinery manufacturing': '333', 
    'Mining (except oil and gas)': '212',
    'Miscellaneous manufacturing': '339',
    'Motion picture and sound recording industries': '51',
    'Natural gas distribution, water and other systems': '221',
    'Non-metallic mineral product manufacturing': '327',
    'Oil and gas extraction': '211',
    'Operating, office, cafeteria and laboratory supplies': 'FC1',
    'Paper manufacturing': '322',
    'Personal and laundry services and private households': '81',
    'Petroleum and coal products manufacturing': '324',
    'Pipeline transportation': '48-49',
    'Plastics and rubber products manufacturing': '326',
    'Postal service and couriers and messengers': '48-49',
    'Primary metal manufacturing': '331',
    'Printing and related support activities': '323',
    'Professional, scientific and technical services': '54',
    'Publishing industries, information services and data processing services': '51',
    'Publishing, broadcasting, telecommunications, and other information services': '51',
    'Repair and maintenance': '81', 
    'Retail trade': '44-45',
    'Support activities for agriculture and forestry': '115',
    'Support activities for mining and oil and gas extraction': '213',
    'Textile and textile product mills': '313-314', 
    'Transit and ground passenger transportation': '48-49',
    'Transportation equipment manufacturing': '336', 
    'Transportation margins': 'FC2',
    'Travel, entertainment, advertising and promotion': 'FC3',
    'Truck transportation': '48-49', 
    'Warehousing and storage': '48-49',
    'Waste management and remediation services': '56', 
    'Wholesale trade': '41',
    'Wood product manufacturing': '321'
}

########################################################################
# Prepare the Table 36-10-0217-01 Statistics Canada data               # 
########################################################################

# Retrieve the data from Table 36-10-0217-01
df_cl = sc.table_to_df('36-10-0217-01')

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
df_cl = df_cl[~df_cl['North American Industry Classification System (NAICS)'].isin(drop_list)]

# Keep the relevant variables
df_cl = df_cl[df_cl['Multifactor productivity and related variables'].isin(['Gross output', 'Gross domestic product (GDP)', 'Capital cost', 'Labour compensation',])]

# Reshape the DataFrame
df_cl = df_cl.pivot_table(index=['North American Industry Classification System (NAICS)', 'REF_DATE'], 
                          columns='Multifactor productivity and related variables', 
                          values='VALUE').reset_index().rename_axis(None, axis=1)

# Rename the columns
df_cl = df_cl.rename(columns={
    'North American Industry Classification System (NAICS)': 'industry',
    'REF_DATE': 'date',
    'Gross output': 'sales',
    'Gross domestic product (GDP)': 'va', 
    'Capital cost': 'capital_cost',
    'Labour compensation': 'labor_cost'
})

# Recode the date column to year
df_cl['year'] = df_cl['date'].dt.year
df_cl = df_cl.drop(columns=['date'])
df_cl = df_cl[df_cl['year'] < 2020]

# Map the industry name to the NAICS code
df_cl['naics'] = df_cl['industry'].map(industry_to_naics)

########################################################################
# Prepare the Table 36-10-0001-01 Statistics Canada data (2013-2019)   # 
########################################################################

# Retrieve the data from Table 36-10-0001-01
df = sc.table_to_df('36-10-0001-01')

# Restrict on basic prices
df = df[df['Valuation'] == 'Basic price']

# Keep the relevant columns
df = df[['REF_DATE', 'Supply', 'Use', 'VALUE']].rename(columns={'REF_DATE': 'date', 'Supply': 'supply', 'Use': 'use', 'VALUE': 'value'})

# Recode the date column to year
df['year'] = df['date'].dt.year
df = df.drop(columns=['date'])
df = df[df['year'] < 2020]

# Identify the "supply" and "use" codes
df['supply_naics'] = df['supply'].str[-9:-1]
df['use_naics'] = df['use'].str[-9:-1]

# Only keep the supply and use codes that start with "BS" (business sector)
df = df[df['supply_naics'].str.startswith('BS')]
df = df[df['use_naics'].str.startswith('BS')]

# Drop the supply and use codes "BS551113" and "BS610000"
df = df[~df['supply_naics'].isin(['BS551113', 'BS610000'])]
df = df[~df['use_naics'].isin(['BS551113', 'BS610000'])]

# Define aggregated codes at the 3-digit NAICS level
df['supply_naics_agg'] = df['supply_naics'].str[2:5]
df['use_naics_agg'] = df['use_naics'].str[2:5]

# Aggregate the data frame at the 3-digit NAICS level
df = df.groupby(['supply_naics_agg', 'use_naics_agg', 'year'], as_index=False).agg({'value': 'sum'})

# Map the aggregation grouping
df.loc[df['supply_naics_agg'].isin(naics_agg.keys()), 'supply_naics_agg'] = df.loc[df['supply_naics_agg'].isin(naics_agg.keys()), 'supply_naics_agg'].map(naics_agg)
df.loc[df['use_naics_agg'].isin(naics_agg.keys()), 'use_naics_agg'] = df.loc[df['use_naics_agg'].isin(naics_agg.keys()), 'use_naics_agg'].map(naics_agg)

# Aggregate the data frame at the coarser 3-digit NAICS level
df = df.groupby(['supply_naics_agg', 'use_naics_agg', 'year'], as_index=False).agg({'value': 'sum'})

# Create a DataFrame with all possible combinations of codes
all_naics = list(set(df['supply_naics_agg'].unique()) | set(df['use_naics_agg'].unique())) + ['capital', 'labor']
df_all = pd.DataFrame([(supply, use, year) for supply in all_naics for use in all_naics for year in range(2013, 2019 + 1)], columns=['supply_naics_agg', 'use_naics_agg', 'year'])
df = pd.merge(df_all, df, on=['supply_naics_agg', 'use_naics_agg', 'year'], how='left')

# Include the capital and labor costs
df_capital = df_cl[['capital_cost', 'naics', 'year']].rename(columns={'naics': 'use_naics_agg'})
df_capital['supply_naics_agg'] = 'capital'
df_capital['capital_cost'] = df_capital['capital_cost'] * 1000
df = pd.merge(df, df_capital, on=['use_naics_agg', 'supply_naics_agg', 'year'], how='left')
df.loc[(df['supply_naics_agg'] == 'capital') & ~df['use_naics_agg'].isin(['capital', 'labor']), 'value'] = df.loc[(df['supply_naics_agg'] == 'capital') & ~df['use_naics_agg'].isin(['capital', 'labor']), 'capital_cost']
df = df.drop(columns=['capital_cost'])
df_labor = df_cl[['labor_cost', 'naics', 'year']].rename(columns={'naics': 'use_naics_agg'})
df_labor['supply_naics_agg'] = 'labor'
df_labor['labor_cost'] = df_labor['labor_cost'] * 1000
df = pd.merge(df, df_labor, on=['supply_naics_agg', 'use_naics_agg', 'year'], how='left')
df.loc[(df['supply_naics_agg'] == 'labor') & ~df['use_naics_agg'].isin(['capital', 'labor']), 'value'] = df.loc[(df['supply_naics_agg'] == 'labor') & ~df['use_naics_agg'].isin(['capital', 'labor']), 'labor_cost']
df = df.drop(columns=['labor_cost'])

# Fill in the missing values with 0
df.loc[df['value'].isna(), 'value'] = 0

# Calculate the cost share of each industry
df['cost_share'] = df.groupby(['year', 'use_naics_agg'])['value'].transform(lambda x: x / x.sum())
df.loc[df['cost_share'].isna(), 'cost_share'] = 0

# Sort the data frame by year, use_naics_agg, and supply_naics_agg
df_13_19 = df.sort_values(by=['year', 'use_naics_agg', 'supply_naics_agg'])

########################################################################
# Prepare the I-O tables from Statistics Canada data for 2010-2012     # 
########################################################################

# Create an empty DataFrame
df_10_12 = pd.DataFrame()

# Iterate over the years 2010 to 2012
for year in range(2010, 2012 + 1):
    # Load the data
    df = pd.read_excel(os.path.join(Path(os.getcwd()).parent, 'Data', 'IOTs national symmetric domestic and imports L97 ' + str(year) + '.xlsx'), sheet_name="Domestic", header=None)

    # Drop useless rows and columns
    df = df.iloc[10:, :].drop(index=11).reset_index(drop=True)
    df = df.drop(df.columns[[0, 2]], axis=1).iloc[:, :-1]

    # Reshape the data
    df.columns = df.iloc[0]
    df = df[1:].reset_index(drop=True)
    df = df.rename(columns={df.columns[0]: 'supply_naics'})
    df = pd.melt(
        df,
        id_vars='supply_naics',
        var_name='use_naics',
        value_name='value'
    )

    # Convert the values to float
    df['value'] = pd.to_numeric(df['value'], errors='coerce')

    # Only keep the supply and use codes that start with "BS" (business sector)
    df = df[df['supply_naics'].str.startswith('BS')]
    df = df[df['use_naics'].str.startswith('BS')]

    # Drop the supply and use codes "BS551113" and "BS610000"
    df = df[~df['supply_naics'].isin(['BS551113', 'BS610000'])]
    df = df[~df['use_naics'].isin(['BS551113', 'BS610000'])]

    # Define aggregated codes at the 3-digit NAICS level
    df['supply_naics_agg'] = df['supply_naics'].str[2:5]
    df['use_naics_agg'] = df['use_naics'].str[2:5]

    # Aggregate the data frame at the 3-digit NAICS level
    df = df.groupby(['supply_naics_agg', 'use_naics_agg'], as_index=False).agg({'value': 'sum'})

    # Map the aggregation grouping
    df.loc[df['supply_naics_agg'].isin(naics_agg.keys()), 'supply_naics_agg'] = df.loc[df['supply_naics_agg'].isin(naics_agg.keys()), 'supply_naics_agg'].map(naics_agg)
    df.loc[df['use_naics_agg'].isin(naics_agg.keys()), 'use_naics_agg'] = df.loc[df['use_naics_agg'].isin(naics_agg.keys()), 'use_naics_agg'].map(naics_agg)

    # Aggregate the data frame at the coarser 3-digit NAICS level
    df = df.groupby(['supply_naics_agg', 'use_naics_agg'], as_index=False).agg({'value': 'sum'})

    # Create a DataFrame with all possible combinations of codes
    all_naics = list(set(df['supply_naics_agg'].unique()) | set(df['use_naics_agg'].unique())) + ['capital', 'labor']
    df_all = pd.DataFrame([(supply, use) for supply in all_naics for use in all_naics], columns=['supply_naics_agg', 'use_naics_agg'])
    df = pd.merge(df_all, df, on=['supply_naics_agg', 'use_naics_agg'], how='left')

    # Include the capital and labor costs
    df_capital = df_cl.loc[df_cl['year'] == year, ['capital_cost', 'naics']].rename(columns={'naics': 'use_naics_agg'})
    df_capital['supply_naics_agg'] = 'capital'
    df_capital['capital_cost'] = df_capital['capital_cost'] * 1000
    df = pd.merge(df, df_capital, on=['use_naics_agg', 'supply_naics_agg'], how='left')
    df.loc[(df['supply_naics_agg'] == 'capital') & ~df['use_naics_agg'].isin(['capital', 'labor']), 'value'] = df.loc[(df['supply_naics_agg'] == 'capital') & ~df['use_naics_agg'].isin(['capital', 'labor']), 'capital_cost']
    df = df.drop(columns=['capital_cost'])
    df_labor = df_cl.loc[df_cl['year'] == year, ['labor_cost', 'naics']].rename(columns={'naics': 'use_naics_agg'})
    df_labor['supply_naics_agg'] = 'labor'
    df_labor['labor_cost'] = df_labor['labor_cost'] * 1000
    df = pd.merge(df, df_labor, on=['supply_naics_agg', 'use_naics_agg'], how='left')
    df.loc[(df['supply_naics_agg'] == 'labor') & ~df['use_naics_agg'].isin(['capital', 'labor']), 'value'] = df.loc[(df['supply_naics_agg'] == 'labor') & ~df['use_naics_agg'].isin(['capital', 'labor']), 'labor_cost']
    df = df.drop(columns=['labor_cost'])

    # Fill in the missing values with 0
    df.loc[df['value'].isna(), 'value'] = 0

    # Calculate the cost share of each industry
    df['cost_share'] = df.groupby('use_naics_agg')['value'].transform(lambda x: x / x.sum())
    df.loc[df['cost_share'].isna(), 'cost_share'] = 0

    # Sort the data frame by year, use_naics_agg, and supply_naics_agg
    df = df.sort_values(by=['use_naics_agg', 'supply_naics_agg'])

    # Create the year column
    df['year'] = year

    # Append the data to the DataFrame
    df_10_12 = pd.concat([df_10_12, df], ignore_index=True)

# Sort the data frame by year, use_naics_agg, and supply_naics_agg
df_10_12 = df_10_12.sort_values(by=['year', 'use_naics_agg', 'supply_naics_agg'])

########################################################################
# Prepare the I-O table from Statistics Canada data for 2009           # 
########################################################################

# Load the data
df = pd.read_excel(os.path.join(Path(os.getcwd()).parent, 'Data', 'IOTs national symmetric domestic and imports L61 2009.xls'), sheet_name="Domestic", header=None)

# Drop useless rows and columns
df = df.iloc[10:, :].drop(index=11).reset_index(drop=True)
df = df.drop(df.columns[[0, 2]], axis=1).iloc[:, :-1]

# Reshape the data
df.columns = df.iloc[0]
df = df[1:].reset_index(drop=True)
df = df.rename(columns={df.columns[0]: 'supply_naics'})
df = pd.melt(
    df,
    id_vars='supply_naics',
    var_name='use_naics',
    value_name='value'
)

# Convert the values to float
df['value'] = pd.to_numeric(df['value'], errors='coerce')

# Only keep the supply and use codes that start with "BS" (business sector)
df = df[df['supply_naics'].str.startswith('BS')]
df = df[df['use_naics'].str.startswith('BS')]

# Drop the supply and use code "BS61000"
df = df[~df['supply_naics'].isin(['BS61000'])]
df = df[~df['use_naics'].isin(['BS61000'])]

# Define aggregated codes at the 3-digit NAICS level
df['supply_naics_agg'] = df['supply_naics'].str[2:5]
df['use_naics_agg'] = df['use_naics'].str[2:5]

# Aggregate the data frame at the 3-digit NAICS level
df = df.groupby(['supply_naics_agg', 'use_naics_agg'], as_index=False).agg({'value': 'sum'})

# Map the aggregation grouping
df.loc[df['supply_naics_agg'].isin(naics_agg.keys()), 'supply_naics_agg'] = df.loc[df['supply_naics_agg'].isin(naics_agg.keys()), 'supply_naics_agg'].map(naics_agg)
df.loc[df['use_naics_agg'].isin(naics_agg.keys()), 'use_naics_agg'] = df.loc[df['use_naics_agg'].isin(naics_agg.keys()), 'use_naics_agg'].map(naics_agg)

# Aggregate the data frame at the coarser 3-digit NAICS level
df = df.groupby(['supply_naics_agg', 'use_naics_agg'], as_index=False).agg({'value': 'sum'})

# Create a DataFrame with all possible combinations of codes
all_naics = list(set(df['supply_naics_agg'].unique()) | set(df['use_naics_agg'].unique())) + ['capital', 'labor']
df_all = pd.DataFrame([(supply, use) for supply in all_naics for use in all_naics], columns=['supply_naics_agg', 'use_naics_agg'])
df = pd.merge(df_all, df, on=['supply_naics_agg', 'use_naics_agg'], how='left')

# Include the capital and labor costs
df_capital = df_cl.loc[df_cl['year'] == 2009, ['capital_cost', 'naics']].rename(columns={'naics': 'use_naics_agg'})
df_capital['supply_naics_agg'] = 'capital'
df_capital['capital_cost'] = df_capital['capital_cost'] * 1000
df = pd.merge(df, df_capital, on=['use_naics_agg', 'supply_naics_agg'], how='left')
df.loc[(df['supply_naics_agg'] == 'capital') & ~df['use_naics_agg'].isin(['capital', 'labor']), 'value'] = df.loc[(df['supply_naics_agg'] == 'capital') & ~df['use_naics_agg'].isin(['capital', 'labor']), 'capital_cost']
df = df.drop(columns=['capital_cost'])
df_labor = df_cl.loc[df_cl['year'] == 2009, ['labor_cost', 'naics']].rename(columns={'naics': 'use_naics_agg'})
df_labor['supply_naics_agg'] = 'labor'
df_labor['labor_cost'] = df_labor['labor_cost'] * 1000
df = pd.merge(df, df_labor, on=['supply_naics_agg', 'use_naics_agg'], how='left')
df.loc[(df['supply_naics_agg'] == 'labor') & ~df['use_naics_agg'].isin(['capital', 'labor']), 'value'] = df.loc[(df['supply_naics_agg'] == 'labor') & ~df['use_naics_agg'].isin(['capital', 'labor']), 'labor_cost']
df = df.drop(columns=['labor_cost'])

# Fill in the missing values with 0
df.loc[df['value'].isna(), 'value'] = 0

# Calculate the cost share of each industry
df['cost_share'] = df.groupby('use_naics_agg')['value'].transform(lambda x: x / x.sum())
df.loc[df['cost_share'].isna(), 'cost_share'] = 0

# Sort the data frame by year, use_naics_agg, and supply_naics_agg
df_09 = df.sort_values(by=['use_naics_agg', 'supply_naics_agg'])

# Create the year column
df_09['year'] = 2009

########################################################################
# Prepare the I-O tables from Statistics Canada data for 1997-2008     # 
########################################################################

# Create an empty DataFrame
df_97_08 = pd.DataFrame()

# Iterate over the years 1997 to 2008
for year in range(1997, 2008 + 1):
    # Load the data
    df = pd.read_excel(os.path.join(Path(os.getcwd()).parent, 'Data', 'IOTs national symmetric domestic and imports L-Public ' + str(year) + '.xls'), sheet_name="Domestic", header=None)

    # Drop useless rows and columns
    df = df.iloc[13:107, :].drop(index=14).reset_index(drop=True)
    df = df.drop(df.columns[[0, 2]], axis=1).iloc[:, :93]

    # Reshape the data
    df.columns = df.iloc[0]
    df = df[1:].reset_index(drop=True)
    df = df.rename(columns={df.columns[0]: 'supply_naics'})
    df = pd.melt(
        df,
        id_vars='supply_naics',
        var_name='use_naics',
        value_name='value'
    )

    # Convert the values to float
    df['value'] = pd.to_numeric(df['value'], errors='coerce')

    # Drop the supply and use code "611A"
    df = df[~df['supply_naics'].isin(['611A'])]
    df = df[~df['use_naics'].isin(['611A'])]

    # Rename the supply and use codes
    df = df.rename(columns={'supply_naics': 'supply_naics_agg', 'use_naics': 'use_naics_agg'})

    # Map the aggregation grouping
    df.loc[df['supply_naics_agg'].isin(naics_agg_97_08.keys()), 'supply_naics_agg'] = df.loc[df['supply_naics_agg'].isin(naics_agg_97_08.keys()), 'supply_naics_agg'].map(naics_agg_97_08)
    df.loc[df['use_naics_agg'].isin(naics_agg_97_08.keys()), 'use_naics_agg'] = df.loc[df['use_naics_agg'].isin(naics_agg_97_08.keys()), 'use_naics_agg'].map(naics_agg_97_08)

    # Aggregate the data frame at the coarser 3-digit NAICS level
    df = df.groupby(['supply_naics_agg', 'use_naics_agg'], as_index=False).agg({'value': 'sum'})

    # Create a DataFrame with all possible combinations of codes
    all_naics = list(set(df['supply_naics_agg'].unique()) | set(df['use_naics_agg'].unique())) + ['capital', 'labor']
    df_all = pd.DataFrame([(supply, use) for supply in all_naics for use in all_naics], columns=['supply_naics_agg', 'use_naics_agg'])
    df = pd.merge(df_all, df, on=['supply_naics_agg', 'use_naics_agg'], how='left')

    # Include the capital and labor costs
    df_capital = df_cl.loc[df_cl['year'] == year, ['capital_cost', 'naics']].rename(columns={'naics': 'use_naics_agg'})
    df_capital['supply_naics_agg'] = 'capital'
    df_capital['capital_cost'] = df_capital['capital_cost'] * 1000
    df = pd.merge(df, df_capital, on=['use_naics_agg', 'supply_naics_agg'], how='left')
    df.loc[(df['supply_naics_agg'] == 'capital') & ~df['use_naics_agg'].isin(['capital', 'labor']), 'value'] = df.loc[(df['supply_naics_agg'] == 'capital') & ~df['use_naics_agg'].isin(['capital', 'labor']), 'capital_cost']
    df = df.drop(columns=['capital_cost'])
    df_labor = df_cl.loc[df_cl['year'] == year, ['labor_cost', 'naics']].rename(columns={'naics': 'use_naics_agg'})
    df_labor['supply_naics_agg'] = 'labor'
    df_labor['labor_cost'] = df_labor['labor_cost'] * 1000
    df = pd.merge(df, df_labor, on=['supply_naics_agg', 'use_naics_agg'], how='left')
    df.loc[(df['supply_naics_agg'] == 'labor') & ~df['use_naics_agg'].isin(['capital', 'labor']), 'value'] = df.loc[(df['supply_naics_agg'] == 'labor') & ~df['use_naics_agg'].isin(['capital', 'labor']), 'labor_cost']
    df = df.drop(columns=['labor_cost'])

    # Fill in the missing values with 0
    df.loc[df['value'].isna(), 'value'] = 0

    # Calculate the cost share of each industry
    df['cost_share'] = df.groupby('use_naics_agg')['value'].transform(lambda x: x / x.sum())
    df.loc[df['cost_share'].isna(), 'cost_share'] = 0

    # Sort the data frame by year, use_naics_agg, and supply_naics_agg
    df = df.sort_values(by=['use_naics_agg', 'supply_naics_agg'])

    # Create the year column
    df['year'] = year

    # Append the data to the DataFrame
    df_97_08 = pd.concat([df_97_08, df], ignore_index=True)

# Sort the data frame by year, use_naics_agg, and supply_naics_agg
df_97_08 = df_97_08.sort_values(by=['year', 'use_naics_agg', 'supply_naics_agg'])

########################################################################
# Prepare the I-O tables from Statistics Canada data for 1961-2008     #
########################################################################

# Retrieve the data from Table 36-10-0407-01
df = sc.table_to_df('36-10-0407-01')

# Keep the relevant columns
df = df[['REF_DATE', 'Inputs-outputs', 'North American Industry Classification System (NAICS)', 'Commodity', 'VALUE']].rename(columns={'REF_DATE': 'date', 'Inputs-outputs': 'io', 'North American Industry Classification System (NAICS)': 'naics', 'Commodity': 'commodity', 'VALUE': 'value'})

# Recode the date column to year
df['year'] = df['date'].dt.year
df = df.drop(columns=['date'])
df = df[df['year'] < 2020]

# Create an input and an output DataFrame
df_i = df[df['io'] == 'Inputs'].drop(columns=['io'])
df_o = df[df['io'] == 'Outputs'].drop(columns=['io'])

# Map the aggregation grouping
df_i = df_i[df_i['naics'].isin(naics_agg_61_08.keys())]
df_i['naics'] = df_i['naics'].map(naics_agg_61_08)
df_o = df_o[df_o['naics'].isin(naics_agg_61_08.keys())]
df_o['naics'] = df_o['naics'].map(naics_agg_61_08)

# Drop the "Total commodities" rows
df_i = df_i[df_i['commodity'] != 'Total commodities']
df_o = df_o[df_o['commodity'] != 'Total commodities']

# Recode "Transportation margins" as "Other transportation and storage"
df_i = df_i[df_i['naics'] != 'FC2']
df_i.loc[df_i['commodity'] == 'Transportation margins', 'commodity'] = 'Other transportation and storage'
df_i = df_i.groupby(['year', 'naics', 'commodity'], as_index=False).aggregate({'value': 'sum'})
df_o = df_o[df_o['naics'] != 'FC2']

# Reallocate the intermediate inputs of the first fictive industry to actual industries
fc1_output = df_o.loc[df_o['naics'] == 'FC1', 'commodity'].unique()[0]
df_o = df_o[df_o['naics'] != 'FC1']
df_i_fc1 = df_i[df_i['naics'] == 'FC1']
df_i = df_i[df_i['naics'] != 'FC1']
df_i_fc1['share'] = df_i_fc1.groupby('year', as_index=False)['value'].transform(lambda x: x / x.sum())
df_i = pd.merge(df_i.loc[df_i['commodity'] != fc1_output, :], df_i.loc[df_i['commodity'] == fc1_output, ['year', 'naics', 'value']].rename(columns={'value': 'value_fc1'}), how='left', on=['year', 'naics'])
df_i['value_fc1'] = df_i['value_fc1'].fillna(0)
df_i = pd.merge(df_i, df_i_fc1[['year', 'commodity', 'share']], how='left', on=['year', 'commodity'])
df_i['share'] = df_i['share'].fillna(0)
df_i['value'] = df_i['value'] + df_i['value_fc1'] * df_i['share']
df_i = df_i.drop(columns=['value_fc1', 'share'])

# Reallocate the intermediate inputs of the third fictive industry to actual industries
fc3_output = df_o.loc[df_o['naics'] == 'FC3', 'commodity'].unique()[0]
df_o = df_o[df_o['naics'] != 'FC3']
df_i_fc3 = df_i[df_i['naics'] == 'FC3']
df_i = df_i[df_i['naics'] != 'FC3']
df_i_fc3['share'] = df_i_fc3.groupby('year', as_index=False)['value'].transform(lambda x: x / x.sum())
df_i = pd.merge(df_i.loc[df_i['commodity'] != fc3_output, :], df_i.loc[df_i['commodity'] == fc3_output, ['year', 'naics', 'value']].rename(columns={'value': 'value_fc3'}), how='left', on=['year', 'naics'])
df_i['value_fc3'] = df_i['value_fc3'].fillna(0)
df_i = pd.merge(df_i, df_i_fc3[['year', 'commodity', 'share']], how='left', on=['year', 'commodity'])
df_i['share'] = df_i['share'].fillna(0)
df_i['value'] = df_i['value'] + df_i['value_fc3'] * df_i['share']
df_i = df_i.drop(columns=['value_fc3', 'share'])

# Drop the non-common commodities
df_o = df_o[~df_o['commodity'].isin(set(df_o['commodity'].unique()) - set(df_i['commodity'].unique()))]
df_i = df_i[~df_i['commodity'].isin(set(df_i['commodity'].unique()) - set(df_o['commodity'].unique()))]

# Create an empty DataFrame
df_61_08_ita = pd.DataFrame()
df_61_08_cta = pd.DataFrame()

# Iterate over the years 1961 to 2008
for y in range(1961, 2008 + 1):
    # Create the input and output DataFrames for the current year
    U = df_i.loc[df_i['year'] == y].pivot_table(index='commodity', columns='naics', values='value', aggfunc='sum', fill_value=0).astype(float)
    V = df_o.loc[df_o['year'] == y].pivot_table(index='commodity', columns='naics', values='value', aggfunc='sum', fill_value=0).astype(float)
    U, V = U.align(V, join='outer', axis=0, fill_value=0)
    x_ita = V.sum(axis=0)
    x_cta = V.sum(axis=1)
    Xinv_ita = 1 / x_ita.replace(0, float('inf'))
    Xinv_cta = 1 / x_cta.replace(0, float('inf'))
    B_ita = V.mul(Xinv_ita, axis=1)
    B_cta = V.mul(Xinv_cta, axis=0)
    Z_ita = B_ita.T.dot(U)
    Z_cta = B_cta.T.dot(U)
    Z_ita.index.name = 'supply_naics_agg'
    Z_cta.index.name = 'supply_naics_agg'
    Z_ita = Z_ita.reset_index().melt(id_vars='supply_naics_agg', var_name='use_naics_agg', value_name='value')
    Z_cta = Z_cta.reset_index().melt(id_vars='supply_naics_agg', var_name='use_naics_agg', value_name='value')

    # Create a DataFrame with all possible combinations of codes
    all_naics_ita = list(set(Z_ita['supply_naics_agg'].unique()) | set(Z_ita['use_naics_agg'].unique())) + ['capital', 'labor']
    all_naics_cta = list(set(Z_cta['supply_naics_agg'].unique()) | set(Z_cta['use_naics_agg'].unique())) + ['capital', 'labor']
    Z_all_ita = pd.DataFrame([(supply, use) for supply in all_naics_ita for use in all_naics_ita], columns=['supply_naics_agg', 'use_naics_agg'])
    Z_all_cta = pd.DataFrame([(supply, use) for supply in all_naics_cta for use in all_naics_cta], columns=['supply_naics_agg', 'use_naics_agg'])
    Z_ita = pd.merge(Z_all_ita, Z_ita, on=['supply_naics_agg', 'use_naics_agg'], how='left')
    Z_cta = pd.merge(Z_all_cta, Z_cta, on=['supply_naics_agg', 'use_naics_agg'], how='left')

    # Include the capital and labor costs
    df_capital = df_cl.loc[df_cl['year'] == y, ['capital_cost', 'naics']].rename(columns={'naics': 'use_naics_agg'})
    df_capital['supply_naics_agg'] = 'capital'
    df_capital['capital_cost'] = df_capital['capital_cost']
    Z_ita = pd.merge(Z_ita, df_capital, on=['use_naics_agg', 'supply_naics_agg'], how='left')
    Z_cta = pd.merge(Z_cta, df_capital, on=['use_naics_agg', 'supply_naics_agg'], how='left')
    Z_ita.loc[(Z_ita['supply_naics_agg'] == 'capital') & ~Z_ita['use_naics_agg'].isin(['capital', 'labor']), 'value'] = Z_ita.loc[(Z_ita['supply_naics_agg'] == 'capital') & ~Z_ita['use_naics_agg'].isin(['capital', 'labor']), 'capital_cost']
    Z_cta.loc[(Z_cta['supply_naics_agg'] == 'capital') & ~Z_cta['use_naics_agg'].isin(['capital', 'labor']), 'value'] = Z_cta.loc[(Z_cta['supply_naics_agg'] == 'capital') & ~Z_cta['use_naics_agg'].isin(['capital', 'labor']), 'capital_cost']
    Z_ita = Z_ita.drop(columns=['capital_cost'])
    Z_cta = Z_cta.drop(columns=['capital_cost'])
    df_labor = df_cl.loc[df_cl['year'] == y, ['labor_cost', 'naics']].rename(columns={'naics': 'use_naics_agg'})
    df_labor['supply_naics_agg'] = 'labor'
    df_labor['labor_cost'] = df_labor['labor_cost']
    Z_ita = pd.merge(Z_ita, df_labor, on=['supply_naics_agg', 'use_naics_agg'], how='left')
    Z_cta = pd.merge(Z_cta, df_labor, on=['supply_naics_agg', 'use_naics_agg'], how='left')
    Z_ita.loc[(Z_ita['supply_naics_agg'] == 'labor') & ~Z_ita['use_naics_agg'].isin(['capital', 'labor']), 'value'] = Z_ita.loc[(Z_ita['supply_naics_agg'] == 'labor') & ~Z_ita['use_naics_agg'].isin(['capital', 'labor']), 'labor_cost']
    Z_cta.loc[(Z_cta['supply_naics_agg'] == 'labor') & ~Z_cta['use_naics_agg'].isin(['capital', 'labor']), 'value'] = Z_cta.loc[(Z_cta['supply_naics_agg'] == 'labor') & ~Z_cta['use_naics_agg'].isin(['capital', 'labor']), 'labor_cost']
    Z_ita = Z_ita.drop(columns=['labor_cost'])
    Z_cta = Z_cta.drop(columns=['labor_cost'])

    # Fill in the missing values with 0
    Z_ita.loc[Z_ita['value'].isna(), 'value'] = 0
    Z_cta.loc[Z_cta['value'].isna(), 'value'] = 0

    # Calculate the cost share of each industry
    Z_ita['cost_share'] = Z_ita.groupby('use_naics_agg')['value'].transform(lambda x: x / x.sum())
    Z_cta['cost_share'] = Z_cta.groupby('use_naics_agg')['value'].transform(lambda x: x / x.sum())
    Z_ita.loc[Z_ita['cost_share'].isna(), 'cost_share'] = 0
    Z_cta.loc[Z_cta['cost_share'].isna(), 'cost_share'] = 0

    # Sort the data frame by year, use_naics_agg, and supply_naics_agg
    Z_ita = Z_ita.sort_values(by=['use_naics_agg', 'supply_naics_agg'])
    Z_cta = Z_cta.sort_values(by=['use_naics_agg', 'supply_naics_agg'])

    # Append the data to the DataFrame
    df_61_08_ita = pd.concat([df_61_08_ita, Z_ita.assign(year=y)], ignore_index=True)
    df_61_08_cta = pd.concat([df_61_08_cta, Z_cta.assign(year=y)], ignore_index=True)

# Sort the data frame by year, use_naics_agg, and supply_naics_agg
df_61_08_ita = df_61_08_ita.sort_values(by=['year', 'use_naics_agg', 'supply_naics_agg'])
df_61_08_cta = df_61_08_cta.sort_values(by=['year', 'use_naics_agg', 'supply_naics_agg'])

# Only keep the years 1961 to 1996 from the CTA approach
df_61_96 = df_61_08_cta[df_61_08_cta['year'] < 1997]

########################################################################
# Append the I-O tables across all years and calculate the lambda's    # 
########################################################################

# Concatenate the data frames
df = pd.concat([df_13_19, df_10_12, df_09, df_97_08, df_61_96], ignore_index=True)
df = pd.merge(df, df_cl[['year', 'naics', 'sales']].rename(columns={'naics': 'use_naics_agg'}), how='left', on=['year', 'use_naics_agg'])
df = df.sort_values(by=['year', 'use_naics_agg', 'supply_naics_agg'])

# Create the cost-based IO matrices for each year and calculate the lambda's
df_lambda = pd.DataFrame([(year, naics) for naics in df_cl['naics'].unique() for year in df_cl['year'].unique()], columns=['year', 'naics']).sort_values(by=['year', 'naics'])
df_lambda['lambda'] = np.nan
df_lambda['lambda_k'] = np.nan
df_lambda['lambda_l'] = np.nan
df_lambda['wedge'] = np.nan
for year in df_lambda['year'].unique():
    df_year = df[df['year'] == year]
    if year < 1997:
        df_year['revenue_share'] = df_year['value'] / df_year['sales']
    else:
        df_year['revenue_share'] = df_year['value'] / (1000 * df_year['sales'])
    df_year.loc[df_year['revenue_share'].isna(), 'revenue_share'] = 0
    df_year_cost = df_year.pivot(index='use_naics_agg', columns='supply_naics_agg', values='cost_share')
    df_year_revenue = df_year.pivot(index='use_naics_agg', columns='supply_naics_agg', values='revenue_share')
    naics_list = df_year_cost.index.tolist()
    Omega_tilde = sparse.csr_matrix(df_year_cost.values)
    Omega = sparse.csr_matrix(df_year_revenue.values)
    b = df_cl.loc[df_cl['year'] == year, ['naics', 'va']].sort_values(by=['naics'])['va'].values
    b = b / b.sum()
    b = np.append(b, [0, 0])
    lambda_tilde = np.matmul(b.transpose(), np.linalg.inv(np.eye(Omega_tilde.shape[0]) - Omega_tilde))
    numerator = np.sum(np.matmul(Omega_tilde[:-2, :-2].todense(), Omega[:-2, :-2].todense()), axis=1)
    denominator = np.sum(np.matmul(Omega[:-2, :-2].todense(), Omega[:-2, :-2].todense()), axis=1)
    wedge = numerator / denominator
    d_lambda = dict([(naics_list[i], lambda_tilde[0, i]) for i in range(len(naics_list) - 2)])
    d_wedge = dict([(naics_list[i], wedge.flatten()[0, i]) for i in range(len(naics_list) - 2)])
    df_lambda.loc[df_lambda['year'] == year, 'lambda'] = df_lambda[df_lambda['year'] == year]['naics'].map(d_lambda)
    df_lambda.loc[df_lambda['year'] == year, 'lambda_k'] = lambda_tilde[0, -2]
    df_lambda.loc[df_lambda['year'] == year, 'lambda_l'] = lambda_tilde[0, -1]
    df_lambda.loc[df_lambda['year'] == year, 'wedge'] = df_lambda[df_lambda['year'] == year]['naics'].map(d_wedge)

# Take the average of successive years
df_lambda['lambda'] = df_lambda.groupby('naics')['lambda'].transform(lambda x: x.rolling(2).mean())
df_lambda['lambda_k'] = df_lambda.groupby('naics')['lambda_k'].transform(lambda x: x.rolling(2).mean())
df_lambda['lambda_l'] = df_lambda.groupby('naics')['lambda_l'].transform(lambda x: x.rolling(2).mean())

# Save the data frame to a CSV file
df_lambda.to_csv(os.path.join(Path(os.getcwd()).parent, 'Data', 'lambda.csv'), index=False)