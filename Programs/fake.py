# Import libraries
import os
import dotenv
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Load environment variables from .env file
dotenv.load_dotenv()

# Define the path to the project's directory
path = os.getenv('path')

# For reproducibility
np.random.seed(42)

# Parameters for panel simulation
n_firms = 1000 # Number of unique firms
available_years = np.arange(2000, 2018)

# Helper function to generate a random date within a given year
def random_date_in_year(year, start_month=1, end_month=12):
    start = datetime(year, start_month, 1)
    # Use day 28 for simplicity to avoid month-end issues
    end = datetime(year, end_month, 28)
    return start + timedelta(days=np.random.randint(0, (end - start).days))

# Helper function for monetary variables using a lognormal distribution
def sim_monetary(mean_log, sigma):
    return np.random.lognormal(mean=mean_log, sigma=sigma)

# Define categorical lists and constants
provinces = ['AB', 'BC', 'MB', 'NB', 'NL', 'NS', 'NT', 'NU', 'ON', 'PE', 'QC', 'SK', 'YT']
legal_types = ['1', '2', '3', '4', '5', '6', '9']
nonprofit_codes = [0, 1, 2]
boolean_flags = [0, -1]
naics_codes = [str(i).zfill(2) for i in range(11, 100)]
ownership_gender = [0, 1, 2, 3]
foreign_country_codes = ['US', 'MX', 'GB', 'FR', 'DE', 'JP']

# Create a list of unique firm IDs
firm_ids = [f'FIRM{str(i).zfill(5)}' for i in range(n_firms)]

# Container for panel data rows
rows = []

# Simulate panel data: each firm is observed continuously from its entry year to its exit year.
for firm in firm_ids:
    # Select an entry year randomly from available years
    entry_year = np.random.choice(available_years)
    # Select an exit year uniformly from all available years at or after the entry year
    possible_exit_years = available_years[available_years >= entry_year]
    exit_year = np.random.choice(possible_exit_years)
    
    # Randomly assign an incorporation year (must be on or before the entry year)
    incorporation_year = np.random.choice(np.arange(1990, entry_year + 1))
    incorporation_date = random_date_in_year(incorporation_year)
    
    # Generate one observation for each year from entry_year to exit_year (unbalanced panel)
    for obs_year in range(entry_year, exit_year + 1):
        row = {}
        row['EntID'] = firm
        # Fiscal dates are generated within the observation year
        row['FiscalStartDate'] = random_date_in_year(obs_year)
        # FiscalEndDate is set 300-400 days after FiscalStartDate
        row['FiscalEndDate'] = row['FiscalStartDate'] + pd.to_timedelta(np.random.randint(300, 400), unit='d')
        # Incorporation date remains constant for the firm
        row['IncorporationDate'] = incorporation_date

        # ---------------- Base Variables ----------------
        # T4_Payroll: total payroll – using lognormal; median roughly exp(12) ~ 1.6e5
        row['T4_Payroll'] = sim_monetary(12, 0.8)
        # Average employees from PD7; using Poisson to get a count (plus 1 to avoid zero)
        row['PD7_AvgEmp_12'] = np.random.poisson(lam=50) + 1
        row['PD7_AvgEmp_NonZero'] = np.random.poisson(lam=40) + 1

        # Total assets: scale to a median around several million dollars
        row['total_assets'] = sim_monetary(16, 0.9)
        row['total_liabilities'] = sim_monetary(15, 0.9)
        row['total_shareholder_equity'] = sim_monetary(15, 1.0)
        row['total_current_assets'] = sim_monetary(15, 0.9)
        row['total_tangible_assets'] = sim_monetary(15, 1.0)
        # Accumulated amortization as a fraction of tangible assets
        row['tot_acum_amort_tangible_assets'] = row['total_tangible_assets'] * np.random.uniform(0.1, 0.5)
        row['total_intangible_assets'] = sim_monetary(14, 1.0)
        row['tot_acum_amort_intang_assets'] = row['total_intangible_assets'] * np.random.uniform(0.1, 0.5)
        row['total_current_liabilities'] = sim_monetary(14, 1.0)
        # For land, buildings, and machinery, use lower means
        row['land'] = sim_monetary(12, 1.0)
        row['buildings'] = sim_monetary(13, 1.0)
        row['machinery_and_equipment'] = sim_monetary(13, 1.0)
        # Revenue and expenses
        row['total_revenue'] = sim_monetary(15, 0.9)
        row['total_expenses'] = sim_monetary(15, 0.9)
        # Farming revenue/expenses (smaller scale)
        row['farm_total_revenue'] = sim_monetary(13, 1.0)
        row['farm_total_expenses'] = sim_monetary(13, 1.0)
        # Farm net income: set as difference plus some noise
        row['farm_net_income'] = row['farm_total_revenue'] - row['farm_total_expenses'] + np.random.normal(0, 1e5)
        row['total_cost_of_sales'] = sim_monetary(14, 0.9)
        row['gross_profits'] = sim_monetary(14, 0.9)
        # Net income before tax/extraneous items: allow negative values
        row['net_income_befor_taxextraitems'] = np.random.normal(0, 1e6)
        row['sales_goods_and_services'] = sim_monetary(15, 0.9)
        row['net_income_after_taxextraitems'] = np.random.normal(0, 1e6)
        row['opening_inventory'] = sim_monetary(12, 1.0)
        row['closing_inventory'] = sim_monetary(12, 1.0)
        row['total_operating_expenses'] = sim_monetary(14, 1.0)
        row['amortization_tangible_assets'] = sim_monetary(11, 1.0)
        row['amortization_intangible_assets'] = sim_monetary(11, 1.0)
        # SR&ED expenditures and related items; using a lower scale
        row['SRED_Expenditures'] = sim_monetary(10, 1.0)
        row['SRED_ITC_Earned'] = sim_monetary(10, 1.0)
        row['SRED_ITC_Current_at_35Percent'] = sim_monetary(10, 1.0)
        row['SRED_ITC_Capital_at_35Percent'] = sim_monetary(10, 1.0)
        row['SRED_ITC_Current_at_20Percent'] = sim_monetary(10, 1.0)
        row['SRED_ITC_Capital_at_20Percent'] = sim_monetary(10, 1.0)
        row['SRED_Deducted_PartI'] = sim_monetary(10, 1.0)
        row['SRED_from_partnership'] = sim_monetary(10, 1.0)
        row['SRED_refunded'] = sim_monetary(10, 1.0)
        row['SRED_carried_back_1year'] = sim_monetary(10, 1.0)
        row['SRED_carried_back_2years'] = sim_monetary(10, 1.0)
        row['SRED_carried_back_3years'] = sim_monetary(10, 1.0)
        row['OPAddressProvince'] = np.random.choice(provinces)
        row['LegalTypeCode'] = np.random.choice(legal_types)
        row['NonProfitCode'] = np.random.choice(nonprofit_codes)
        row['NAICS'] = np.random.choice(naics_codes)
        row['EntMultiEstablishmentFlag'] = np.random.choice(boolean_flags)
        row['EntMultiLocationFlag'] = np.random.choice(boolean_flags)
        row['EntMultiProvinceFlag'] = np.random.choice(boolean_flags)
        row['EntMultiActivityFlag'] = np.random.choice(boolean_flags)
        # BirthDate: choose a date between incorporation and the fiscal start date
        row['BirthDate'] = random_date_in_year(max(incorporation_year, row['FiscalStartDate'].year))
        row['BusinessStatusCode'] = np.random.randint(0, 8)
        row['Purchases_cost_of_materials'] = sim_monetary(14, 1.0)
        row['capital_cost_allowance'] = sim_monetary(11, 1.0)
        row['NbBN_filedT4'] = np.random.randint(0, 10)
        row['NbBN_filedPD7'] = np.random.randint(0, 10)
        row['NbBN_filedT2'] = np.random.randint(0, 10)
        row['CCPC'] = np.random.randint(0, 2)
        
        # ---------------- Analytic Variables ----------------
        # Gross output is defined as the sum of total_revenue and farm_total_revenue.
        row['gross_output'] = row['total_revenue'] + row['farm_total_revenue']
        # Value-added measures: using definitions from the data dictionary.
        # value_added_cca = net_income_befor_taxextraitems + T4_Payroll + capital_cost_allowance
        row['value_added_cca'] = max(row['net_income_befor_taxextraitems'] + row['T4_Payroll'] + row['capital_cost_allowance'], 0)
        # value_added_amort = net_income_befor_taxextraitems + T4_Payroll + amortization_tangible_assets
        row['value_added_amort'] = max(row['net_income_befor_taxextraitems'] + row['T4_Payroll'] + row['amortization_tangible_assets'], 0)
        row['int_inputs_cca'] = row['gross_output'] - row['value_added_cca']
        row['int_inputs_amort'] = row['gross_output'] - row['value_added_amort']
        row['lp_go'] = row['gross_output'] / (row['PD7_AvgEmp_12'] + 1)
        row['lp_va_cca'] = row['value_added_cca'] / (row['PD7_AvgEmp_12'] + 1)
        row['lp_va_amort'] = row['value_added_amort'] / (row['PD7_AvgEmp_12'] + 1)
        row['total_tangible_net_stock'] = row['total_tangible_assets'] - row['tot_acum_amort_tangible_assets']
        row['total_intangible_net_stock'] = row['total_intangible_assets'] - row['tot_acum_amort_intang_assets']
        row['total_assets_d'] = row['total_assets'] if row['total_assets'] > 0 else 0
        row['total_liabilities_d'] = row['total_liabilities'] if row['total_liabilities'] > 0 else 0
        row['total_current_assets_d'] = row['total_current_assets'] if row['total_current_assets'] > 0 else 0
        row['total_current_liabilities_d'] = row['total_current_liabilities'] if row['total_current_liabilities'] > 0 else 0
        row['total_expenses_d'] = row['total_expenses'] if row['total_expenses'] > 0 else 0
        row['farm_total_expenses_d'] = row['farm_total_expenses'] if row['farm_total_expenses'] > 0 else 0
        row['total_cost_of_sales_d'] = row['total_cost_of_sales'] if row['total_cost_of_sales'] > 0 else 0
        row['sales_goods_and_services_d'] = row['sales_goods_and_services'] if row['sales_goods_and_services'] > 0 else 0
        row['total_operating_expenses_d'] = row['total_operating_expenses'] if row['total_operating_expenses'] > 0 else 0
        row['amortization_tangible_assets_d'] = row['amortization_tangible_assets'] if row['amortization_tangible_assets'] > 0 else 0
        row['amortization_intangible_assets_d'] = row['amortization_intangible_assets'] if row['amortization_intangible_assets'] > 0 else 0
        # Investment variables: simulated on a lower scale
        row['investment_BLDG'] = sim_monetary(10, 1.0)
        row['investment_ME'] = sim_monetary(10, 1.0)
        row['investment_INTANGIBLE'] = sim_monetary(10, 1.0)
        row['investment_NOCLASS'] = sim_monetary(10, 1.0)
        row['investment_JUNK'] = sim_monetary(10, 1.0)
        row['total_tangible_net_investment'] = (row['investment_BLDG'] + row['investment_ME'] +
                                                row['investment_NOCLASS'] + row['investment_JUNK'])
        row['CountryOfControl_Nalmf'] = 'CAN'
        row['OwnershipGender'] = np.random.choice(ownership_gender)
        # Age is calculated as the difference between the fiscal start year and the incorporation year.
        row['age'] = row['FiscalStartDate'].year - row['IncorporationDate'].year
        row['NAICS_DOMINANT'] = np.random.choice(naics_codes)
        
        # ---------------- TEC (Exports) Variables ----------------
        def gen_list_and_values(codes):
            k = np.random.randint(1, 4)
            code_list = ','.join(np.random.choice(codes, k, replace=False))
            vals = [sim_monetary(10, 1.0) for _ in range(k)]
            while len(vals) < 3:
                vals.append(np.nan)
            return code_list, vals[0], vals[1], vals[2]
        
        export_countries, row['first_export_country_value'], row['second_export_country_value'], row['third_export_country_value'] = gen_list_and_values(foreign_country_codes)
        row['list_of_export_countries'] = export_countries
        export_products, row['first_export_product_value'], row['second_export_product_value'], row['third_export_product_value'] = gen_list_and_values([str(x) for x in range(10, 100)])
        row['list_of_export_products'] = export_products
        row['total_exports'] = sim_monetary(10, 1.0)
        
        # ---------------- TIC (Imports) Variables ----------------
        import_countries, row['first_import_country_value'], row['second_import_country_value'], row['third_import_country_value'] = gen_list_and_values(foreign_country_codes)
        row['list_of_import_countries'] = import_countries
        import_products, row['first_import_product_value'], row['second_import_product_value'], row['third_import_product_value'] = gen_list_and_values([str(x) for x in range(10, 100)])
        row['list_of_import_products'] = import_products
        row['total_imports'] = sim_monetary(10, 1.0)
        
        # ---------------- SR&ED Expenditures Variables ----------------
        row['RD_out'] = sim_monetary(10, 1.0)
        row['RD_inv'] = sim_monetary(10, 1.0)
        row['RD_purchase'] = sim_monetary(10, 1.0)
        row['RD_thirdparty'] = sim_monetary(10, 1.0)
        row['RD_inhouse'] = sim_monetary(10, 1.0)
        row['L0534_0'] = sim_monetary(10, 1.0)
        row['L0536_0'] = sim_monetary(10, 1.0)
        row['L0430_0'] = sim_monetary(10, 1.0)
        row['L0517_0'] = sim_monetary(10, 1.0)
        row['L0518_0'] = sim_monetary(10, 1.0)
        row['L0432_0'] = sim_monetary(10, 1.0)
        row['L0380_0'] = sim_monetary(10, 1.0)
        row['L0502_0'] = sim_monetary(10, 1.0)
        row['L0390_0'] = sim_monetary(10, 1.0)
        row['L0504_0'] = sim_monetary(10, 1.0)
        row['L0340_0'] = sim_monetary(10, 1.0)
        row['L0345_0'] = sim_monetary(10, 1.0)
        row['L0370_0'] = sim_monetary(10, 1.0)
        
        # ---------------- KLEMS Variables ----------------
        row['IFPA'] = np.random.uniform(0.8, 1.2)
        row['IFPK'] = np.random.uniform(0.8, 1.2)
        row['IFPL'] = np.random.uniform(0.8, 1.2)
        row['IFPV'] = np.random.uniform(0.8, 1.2)
        
        rows.append(row)

# Create the DataFrame from the list of rows
df = pd.DataFrame(rows)
print(df.head())

# Optionally, save the dataset to CSV
df.to_csv(os.path.join(path, 'Data/synthetic.csv'), index=False)