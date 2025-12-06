import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse

########################################################################
# Prepare the value-added tables from BEA data for 1963-1986           # 
########################################################################

# Define a mapping from SIC to NAICS for the value-added data
sic72_to_naics_map = {
    'Farms': '111-112',
    'Agricultural services, forestry, and fishing': '113-115',
    'Metal mining': '212',
    'Coal mining': '212',
    'Oil and gas extraction': '211',
    'Nonmetallic minerals, except fuels': '212',
    'Construction': '23',
    'Lumber and wood products': '321',
    'Furniture and fixtures': '337',
    'Stone, clay, and glass products': '327',
    'Primary metal industries': '331',
    'Fabricated metal products': '332',
    'Machinery, except electrical': '333',
    'Electric and electronic equipment': '334-335',
    'Motor vehicles and equipment': '336',
    'Other transportation equipment': '336',
    'Instruments and related products': ['334', '339'],
    'Miscellaneous manufacturing industries': '339',
    'Food and kindred products': '311',
    'Tobacco products': '312',
    'Textile mill products': '313-314',
    'Apparel and other textile products': '315-316',
    'Paper and allied products': '322',
    # 'Printing' is 323, while 'Publishing' is part of 51
    'Printing and publishing': ['323', '51'],
    'Chemicals and allied products': '325',
    'Petroleum and coal products': '324',
    'Rubber and miscellaneous plastics products': '326',
    'Leather and leather products': '315-316',

    # Transportation, Utilities, Trade (NAICS 22x, 41x-49x)
    'Railroad transportation': '48-49',
    'Local and interurban passenger transit': '48-49',
    'Trucking and warehousing': '48-49',
    'Water transportation': '48-49',
    'Transportation by air': '48-49',
    'Pipelines, except natural gas': '48-49',
    'Transportation services': '48-49',
    'Communications': '51',
    'Telephone and telegraph': '51',
    'Radio and television': '51',
    'Electric, gas, and sanitary services': '221',
    'Wholesale trade': '41',
    'Retail trade': '44-45',

    # Finance, Insurance, Real Estate (NAICS 52x-53x)
    'Banking': '52-53',
    'Credit agencies other than banks': '52-53',
    'Security and commodity brokers': '52-53',
    'Insurance carriers': '52-53',
    'Insurance agents, brokers, and service': '52-53',
    'Real estate /2/': '52-53',
    'Holding and other investment offices': '52-53',

    # Services (NAICS 54x-81x)
    'Hotels and other lodging places': '72',
    'Personal services': '81',
    # 'Business services' maps to both professional (54) and admin (56)
    'Business services': ['54', '56'],
    'Auto repair, services, and parking': '81',
    'Miscellaneous repair services': '81',
    'Motion pictures': '71',
    'Amusement and recreation services': '71',
    'Health services': '62',
    'Legal services': '54',
    'Social services': '62',
    'Membership organizations': '81',
    'Miscellaneous professional services': '54',
    'Private households': '81',
}

# Load the data
df_va_63_86 = pd.read_excel(os.path.join(Path(os.getcwd()).parent, 'Data', 'va_components_47-87.xls'), sheet_name='72SIC_VA, GO, II')
df_va_c_63_86 = pd.read_excel(os.path.join(Path(os.getcwd()).parent, 'Data', 'va_components_47-87.xls'), sheet_name='72SIC_Components of VA')

# Delete the trailing whitespace characters in the 'Industry Title' column
df_va_63_86['Industry Title'] = df_va_63_86['Industry Title'].str.strip()
df_va_c_63_86['Industry Title'] = df_va_c_63_86['Industry Title'].str.strip()

# Only keep the relevant columns
df_va_63_86 = df_va_63_86[df_va_63_86['Code'] == 'VA']

########################################################################
# Prepare the I-O tables from BEA data for 1963-1996                   # 
########################################################################

# Define a NAICS aggregation grouping for 1963-1996
naics_agg_63_96 = {
    '111CA':  '111-112', 
    '113FF':  '113-115', 
    '211':    '211', 
    '212':    '212', 
    '213':    '213', 
    '22':     '221', 
    '23':     '23', 
    '321':    '321', 
    '327':    '327',
    '331':    '331', 
    '332':    '332', 
    '333':    '333', 
    '334':    '334', 
    '335':    '335', 
    '3361MV': '336', 
    '3364OT': '336', 
    '337':    '337',
    '339':    '339',
    '311FT':  '311-312',
    '313TT':  '313-314',
    '315AL':  '315-316',
    '322':    '322',
    '323':    '323',
    '324':    '324',
    '325':    '325',
    '326':    '326',
    '42':     '41',
    '44RT':   '44-45',
    '481':    '48-49',
    '482':    '48-49',
    '483':    '48-49',
    '484':    '48-49',
    '485':    '48-49',
    '486':    '48-49',
    '487OS':  '48-49',
    '493':    '48-49',
    '511':    '51',
    '512':    '51',
    '513':    '51',
    '514':    '51',
    '521CI':  '52-53',
    '523':    '52-53',
    '524':    '52-53',
    '525':    '52-53',
    '531':    '52-53',
    '532RL':  '52-53',
    '5411':   '54',
    '5415':   '54',
    '5412OP': '54',
    '55':     '55',
    '561':    '56',
    '562':    '56',
    '61':     '61',
    '621':    '62',
    '624':    '62',
    '711AS':  '71',
    '713':    '71',
    '721':    '72',
    '722':    '72',
    '81':     '81'
}

# Create an empty DataFrame
df_63_96_ita = pd.DataFrame()
df_63_96_cta = pd.DataFrame()

# Iterate through the years 1963 to 1996
years = range(1963, 1997)
for year in years:
    # Load the output data
    df_o = pd.read_excel(os.path.join(Path(os.getcwd()).parent, 'Data', 'IOMake_Before_Redefinitions_1963-1996_Summary.xlsx'), sheet_name=str(year), skiprows=6)
    df_o.drop(columns=df_o.columns[1], inplace=True)
    df_o.replace("...", 0, inplace=True)

    # Load the input data
    df_i = pd.read_excel(os.path.join(Path(os.getcwd()).parent, 'Data', 'IOUse_Before_Redefinitions_PRO_1963-1996_Summary.xlsx'), sheet_name=str(year), skiprows=6)
    df_i.drop(columns=df_i.columns[1], inplace=True)
    df_i.replace("...", 0, inplace=True)

    # Reshape to long format
    df_o = df_o.melt(id_vars=["Code"], var_name="commodity", value_name="value").rename(columns={"Code": "naics"})
    df_i = df_i.melt(id_vars=["Code"], var_name="naics", value_name="value").rename(columns={"Code": "commodity"})

    # Drop the following codes in the 'commodity' column: ['622HO', 'GFG', 'GFE', 'GSLG', 'GSLE', 'Used', 'Other', 'T008', 'T005', 'T006']
    df_o = df_o[~df_o['commodity'].isin(['622HO', 'GFG', 'GFE', 'GSLG', 'GSLE', 'Used', 'Other', 'T008', 'T005', 'T006'])]
    df_i = df_i[~df_i['commodity'].isin(['622HO', 'GFG', 'GFE', 'GSLG', 'GSLE', 'Used', 'Other', 'T008', 'T005', 'T006'])]

    # Map the industry codes to NAICS
    df_o['naics'] = df_o['naics'].map(naics_agg_63_96)
    df_i['naics'] = df_i['naics'].map(naics_agg_63_96)
    df_o = df_o.dropna(subset=['naics'])
    df_i = df_i.dropna(subset=['naics'])
    df_o = df_o.groupby(['naics', 'commodity'], as_index=False).agg({'value': 'sum'})
    df_i = df_i.groupby(['naics', 'commodity'], as_index=False).agg({'value': 'sum'})

    # Create the input and output DataFrames for the current year
    U = df_i.pivot_table(index='commodity', columns='naics', values='value', aggfunc='sum', fill_value=0.0).astype(float)
    V = df_o.pivot_table(index='commodity', columns='naics', values='value', aggfunc='sum', fill_value=0.0).astype(float)
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
    df_capital = df_cl.loc[df_cl['year'] == year, ['capital_cost', 'naics']].rename(columns={'naics': 'use_naics_agg'})
    df_capital['supply_naics_agg'] = 'capital'
    df_capital['capital_cost'] = df_capital['capital_cost']
    Z_ita = pd.merge(Z_ita, df_capital, on=['use_naics_agg', 'supply_naics_agg'], how='left')
    Z_cta = pd.merge(Z_cta, df_capital, on=['use_naics_agg', 'supply_naics_agg'], how='left')
    Z_ita.loc[(Z_ita['supply_naics_agg'] == 'capital') & ~Z_ita['use_naics_agg'].isin(['capital', 'labor']), 'value'] = Z_ita.loc[(Z_ita['supply_naics_agg'] == 'capital') & ~Z_ita['use_naics_agg'].isin(['capital', 'labor']), 'capital_cost']
    Z_cta.loc[(Z_cta['supply_naics_agg'] == 'capital') & ~Z_cta['use_naics_agg'].isin(['capital', 'labor']), 'value'] = Z_cta.loc[(Z_cta['supply_naics_agg'] == 'capital') & ~Z_cta['use_naics_agg'].isin(['capital', 'labor']), 'capital_cost']
    Z_ita = Z_ita.drop(columns=['capital_cost'])
    Z_cta = Z_cta.drop(columns=['capital_cost'])
    df_labor = df_cl.loc[df_cl['year'] == year, ['labor_cost', 'naics']].rename(columns={'naics': 'use_naics_agg'})
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
    df_63_96_ita = pd.concat([df_63_96_ita, Z_ita.assign(year=year)], ignore_index=True)
    df_63_96_cta = pd.concat([df_63_96_cta, Z_cta.assign(year=year)], ignore_index=True)