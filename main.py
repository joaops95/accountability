import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
from difflib import SequenceMatcher

# Set style for plots
plt.style.use('default')  # Using default style instead of seaborn
sns.set_theme()  # This will apply seaborn styling

def similar(a, b):
    """Calculate string similarity ratio"""
    return SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()

def get_normalized_company_name(name):
    """Normalize company names to handle different formats"""
    # Dictionary of known name mappings
    name_mappings = {
        'ASSOCIA«√O EMPREENDEDORIS': 'AIEL',
        'ASSOCIAÇÃO EMPREENDEDORIS': 'AIEL',
        'AIEL - ASSOCIAÇÃO PARA A INOVAÇÃO E EMPREENDEDORISMO DE LISBOA': 'AIEL',
        'TRF. ASSOCIA«√O EMPREENDEDORIS': 'AIEL',
        'WONDERBREEZE': 'WONDERBREEZE UNIPESSOAL LDA',
        'WONDERBREEZE UNIPESSOAL': 'WONDERBREEZE UNIPESSOAL LDA',
        'CASE SOFTWARE': 'CASE SOFTWARE SOLUCOES INFORMATICAS LDA',
        'INVOICEXPRESS': 'INVOICEXPRESS, LDA',
        'SARAMED': 'SARAMED IMPORTACAO E EXPORTACAO LDA',
        'TRF.P/ BANK OF AMERICA': 'BANK OF AMERICA',
        'TRF.P/ XE EUROPE': 'XE EUROPE BV'
    }
    
    # Clean the input name
    clean_name = str(name).strip().upper()
    
    # Check if we have a direct mapping
    for key, value in name_mappings.items():
        if key.upper() in clean_name:
            return value.upper()
    
    # Remove common prefixes
    prefixes_to_remove = ['TRF.P/', 'TRF. ', 'TR-', 'TRF ', 'TRANSFERENCIA ', 'PAGAMENTO ']
    for prefix in prefixes_to_remove:
        if clean_name.startswith(prefix.upper()):
            clean_name = clean_name[len(prefix):].strip()
    
    return clean_name

def is_duplicate_transaction(row, existing_transactions, threshold=0.6):
    """
    Check if a transaction from e-fatura is already in bank statements
    Parameters:
        row: e-fatura transaction (standardized format)
        existing_transactions: DataFrame of bank transactions
        threshold: similarity threshold for company names
    """
    date = row['Date']
    amount = row['Amount']  # Already negative
    
    # Look for transactions within 3 days of the e-fatura date
    date_mask = (
        (existing_transactions['Date'] >= date - timedelta(days=3)) &
        (existing_transactions['Date'] <= date + timedelta(days=3))
    )
    
    # Look for transactions with the same amount (with small tolerance for rounding)
    amount_tolerance = 0.01  # 1 cent tolerance
    amount_mask = (existing_transactions['Amount'].between(
        amount - amount_tolerance,
        amount + amount_tolerance
    ))
    
    potential_matches = existing_transactions[date_mask & amount_mask]
    
    if len(potential_matches) == 0:
        return False
    
    # Normalize the e-fatura company name
    efatura_company = get_normalized_company_name(row['Description'])
    
    # Check company name similarity
    for _, match in potential_matches.iterrows():
        bank_company = get_normalized_company_name(match['Description'])
        
        # First try exact match with normalized names
        if efatura_company == bank_company:
            return True
        
        # Then try similarity match
        if similar(efatura_company, bank_company) > threshold:
            return True
            
    return False

def load_revolut_transactions(file_path):
    # Read Revolut CSV file
    df = pd.read_csv(file_path)
    
    # Convert date columns to datetime
    df['Date completed (UTC)'] = pd.to_datetime(df['Date completed (UTC)'])
    
    # Ensure Amount is float
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
    
    # Standardize columns
    df_standardized = pd.DataFrame({
        'Date': df['Date completed (UTC)'],
        'Description': df['Description'],
        'Amount': df['Amount'],
        'Currency': df['Payment currency'],
        'Type': df['Type'],
        'Source': 'Revolut'
    })
    
    # Print some sample values for verification
    print("\nSample of Revolut transactions:")
    print(df_standardized[['Date', 'Description', 'Amount']].head())
    
    return df_standardized

def load_montepio_transactions(file_path):
    try:
        # Read Montepio CSV file with proper encoding
        df = pd.read_csv(file_path, encoding='utf-8-sig')
        
        # Convert date column to datetime with error handling
        def parse_date(date_str):
            try:
                # First try the standard format
                return pd.to_datetime(date_str, format='%m/%d/%Y')
            except:
                try:
                    # If that fails, try parsing with dayfirst=True
                    return pd.to_datetime(date_str, dayfirst=True)
                except:
                    print(f"Warning: Could not parse date: {date_str}")
                    return None
        
        # Convert dates and check for invalid dates
        df['DATA MOV.'] = df['DATA MOV.'].apply(parse_date)
        
        # Remove rows with invalid dates or dates in the wrong year
        invalid_dates = df['DATA MOV.'].isna() | (df['DATA MOV.'].dt.year > 2024)
        if invalid_dates.any():
            print(f"Warning: Removing {invalid_dates.sum()} transactions with invalid dates:")
            print(df[invalid_dates][['DATA MOV.', 'DESCRI«√O', 'IMPORT¬NCIA']])
            df = df[~invalid_dates]
        
        # Convert amount string to float (handling Portuguese number format)
        def convert_amount(value):
            try:
                # Remove any spaces
                value = str(value).strip()
                # If the value contains a comma and a dot, handle Portuguese format
                if '.' in value and ',' in value:
                    # Remove dots (thousand separators) and replace comma with dot
                    value = value.replace('.', '').replace(',', '.')
                # If only comma exists, replace it with dot
                elif ',' in value:
                    value = value.replace(',', '.')
                return float(value)
            except:
                print(f"Warning: Could not convert value: {value}")
                return 0.0
        
        # Convert amounts using the new function
        df['IMPORT¬NCIA'] = df['IMPORT¬NCIA'].apply(convert_amount)
        
        # Standardize columns
        df_standardized = pd.DataFrame({
            'Date': df['DATA MOV.'],
            'Description': df['DESCRI«√O'],
            'Amount': df['IMPORT¬NCIA'],
            'Currency': 'EUR',
            'Type': 'BANK_TRANSFER',  # Default type for Montepio transactions
            'Source': 'Montepio'
        })
        
        # Print some sample values for verification
        print("\nSample of converted Montepio transactions:")
        print(df_standardized[['Date', 'Description', 'Amount']].head())
        print("\nDate range in Montepio data:")
        print(f"From: {df_standardized['Date'].min()}")
        print(f"To: {df_standardized['Date'].max()}")
        
        return df_standardized
    except Exception as e:
        print(f"Error loading Montepio transactions: {str(e)}")
        return pd.DataFrame()  # Return empty DataFrame if there's an error

def convert_amount(value):
    """Convert Portuguese formatted number string to float"""
    try:
        # Remove currency symbol and spaces
        value = str(value).replace('€', '').strip()
        
        # Handle Portuguese number format (1.234,56 -> 1234.56)
        if '.' in value and ',' in value:
            # Remove dots (thousand separators) and replace comma with dot
            value = value.replace('.', '').replace(',', '.')
        # If only comma exists, replace it with dot
        elif ',' in value:
            value = value.replace(',', '.')
            
        return float(value)
    except:
        print(f"Warning: Could not convert value: {value}")
        return 0.0

def load_efatura_transactions(file_path):
    """Load and process e-fatura data"""
    try:
        # Read CSV file with proper encoding and separator
        df = pd.read_csv(file_path, encoding='utf-8', sep=';')
        
        # Convert amount strings to float using the helper function
        df['Total'] = df['Total'].apply(convert_amount)
        df['IVA'] = df['IVA'].apply(convert_amount)
        df['Base Tributável'] = df['Base Tributável'].apply(convert_amount)
        
        # Convert date with flexible format handling
        def parse_date(date_str):
            try:
                # Try ISO format first (YYYY-MM-DD)
                return pd.to_datetime(date_str, format='%Y-%m-%d')
            except:
                try:
                    # Try Portuguese format (DD/MM/YY)
                    return pd.to_datetime(date_str, format='%d/%m/%y')
                except:
                    try:
                        # Try Portuguese format with full year (DD/MM/YYYY)
                        return pd.to_datetime(date_str, format='%d/%m/%Y')
                    except:
                        print(f"Warning: Could not parse date: {date_str}")
                        return None

        df['Data Emissão'] = df['Data Emissão'].apply(parse_date)
        
        # Remove rows with invalid dates
        invalid_dates = df['Data Emissão'].isna()
        if invalid_dates.any():
            print(f"Warning: Removing {invalid_dates.sum()} transactions with invalid dates")
            df = df[~invalid_dates]
        
        # Filter out canceled documents
        df = df[df['Situação'] != 'Documento Anulado pelo Emitente']
        
        # Create standardized DataFrame
        df_standardized = pd.DataFrame({
            'Date': df['Data Emissão'],
            'Description': df['Emitente'].apply(lambda x: x.split(' - ')[1] if ' - ' in x else x),
            'Amount': -df['Total'],  # Negative since these are expenses
            'Currency': 'EUR',
            'Type': df['Setor'].fillna('Other'),
            'Source': 'E-fatura',
            'VAT': df['IVA'],
            'Base': df['Base Tributável']
        })
        
        # Print some verification info
        print("\nE-fatura data summary:")
        print(f"Total transactions: {len(df_standardized)}")
        print(f"Date range: {df_standardized['Date'].min()} to {df_standardized['Date'].max()}")
        print("\nSample of e-fatura transactions:")
        print(df_standardized[['Date', 'Description', 'Amount', 'VAT']].head())
        
        return df_standardized
    except Exception as e:
        print(f"Error loading e-fatura transactions: {str(e)}")
        print("Full traceback:", e.__traceback__)
        return pd.DataFrame()

def merge_all_transactions(revolut_df, montepio_df, efatura_df):
    """Merge all transactions while avoiding duplicates"""
    # First merge bank transactions
    bank_transactions = pd.concat([revolut_df, montepio_df], ignore_index=True)
    
    # For each bank transaction, check if it exists in e-fatura
    bank_expenses = bank_transactions[bank_transactions['Amount'] < 0].copy()
    bank_income = bank_transactions[bank_transactions['Amount'] >= 0].copy()
    
    print("\nChecking for duplicates...")
    duplicates_found = 0
    
    # Keep only bank expenses that don't have a matching e-fatura entry
    unique_bank_expenses = []
    
    for _, bank_row in bank_expenses.iterrows():
        # Look for matching e-fatura entry
        date = bank_row['Date']
        amount = bank_row['Amount']
        
        # Find potential matches in e-fatura
        date_mask = (
            (efatura_df['Date'] >= date - timedelta(days=3)) &
            (efatura_df['Date'] <= date + timedelta(days=3))
        )
        
        # Look for transactions with the same amount (with small tolerance for rounding)
        amount_tolerance = 0.01  # 1 cent tolerance
        amount_mask = (efatura_df['Amount'].between(
            amount - amount_tolerance,
            amount + amount_tolerance
        ))
        
        potential_matches = efatura_df[date_mask & amount_mask]
        
        if len(potential_matches) == 0:
            # No matching e-fatura entry found, keep the bank transaction
            unique_bank_expenses.append(bank_row)
        else:
            duplicates_found += 1
            print(f"Found expense in e-fatura: {date.strftime('%Y-%m-%d')} - {bank_row['Description']} - €{abs(amount):,.2f}")
    
    # Create final merged dataset
    unique_bank_expenses_df = pd.DataFrame(unique_bank_expenses) if unique_bank_expenses else pd.DataFrame()
    
    # Merge e-fatura expenses, bank income, and unique bank expenses
    merged_df = pd.concat([
        efatura_df,  # All e-fatura entries (with VAT info)
        bank_income,  # All income transactions
        unique_bank_expenses_df  # Only bank expenses without e-fatura match
    ], ignore_index=True)
    
    # Sort by date
    merged_df = merged_df.sort_values('Date')
    
    # Add time period columns
    merged_df['Week'] = merged_df['Date'].dt.isocalendar().week
    merged_df['Month'] = merged_df['Date'].dt.month
    merged_df['Year'] = merged_df['Date'].dt.year
    
    # Print summary of merged data
    print("\nMerged transactions summary:")
    print(f"Total transactions: {len(merged_df)}")
    print(f"From e-fatura: {len(efatura_df)}")
    print(f"From bank (income): {len(bank_income)}")
    print(f"From bank (unique expenses): {len(unique_bank_expenses_df)}")
    print(f"Duplicate expenses (using e-fatura): {duplicates_found}")
    print(f"Date range: {merged_df['Date'].min()} to {merged_df['Date'].max()}")
    
    return merged_df

def analyze_transactions(df):
    # Separate income and expenses
    expenses = df[df['Amount'] < 0].copy()
    income = df[df['Amount'] > 0].copy()
    
    # Weekly analysis
    weekly_spending = expenses.groupby(['Year', 'Week'])['Amount'].sum().abs()
    weekly_income = income.groupby(['Year', 'Week'])['Amount'].sum()
    
    # Monthly analysis
    monthly_spending = expenses.groupby(['Year', 'Month'])['Amount'].sum().abs()
    monthly_income = income.groupby(['Year', 'Month'])['Amount'].sum()
    
    # Source analysis
    source_spending = expenses.groupby('Source')['Amount'].sum().abs()
    source_income = income.groupby('Source')['Amount'].sum()
    
    # Type analysis
    type_spending = expenses.groupby('Type')['Amount'].sum().abs()
    type_income = income.groupby('Type')['Amount'].sum()
    
    return {
        'weekly': (weekly_spending, weekly_income),
        'monthly': (monthly_spending, monthly_income),
        'source': (source_spending, source_income),
        'type': (type_spending, type_income)
    }

def generate_plots(analysis_results):
    os.makedirs('financial_reports', exist_ok=True)
    
    # Monthly P&L plot
    plt.figure(figsize=(15, 7))
    monthly_spending, monthly_income = analysis_results['monthly']
    
    # Align the indices of monthly_spending and monthly_income
    aligned_spending, aligned_income = monthly_spending.align(monthly_income, fill_value=0)
    
    # Create month labels from the aligned data
    month_labels = aligned_spending.index.map(lambda x: f"{x[0]}-{x[1]:02d}")
    x = range(len(month_labels))
    
    plt.bar(x, aligned_income.values, label='Income', alpha=0.7)
    plt.bar(x, -aligned_spending.values, label='Expenses', alpha=0.7)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.title('Monthly Profit & Loss')
    plt.xlabel('Month')
    plt.ylabel('Amount (EUR)')
    plt.xticks(x, month_labels, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('financial_reports/monthly_pnl.png')
    plt.close()
    
    # Source distribution plot
    plt.figure(figsize=(12, 6))
    source_spending, source_income = analysis_results['source']
    
    # Align source data
    aligned_source_spending, aligned_source_income = source_spending.align(source_income, fill_value=0)
    source_net = aligned_source_income.add(aligned_source_spending, fill_value=0)
    
    source_net.plot(kind='bar')
    plt.title('Net Cash Flow by Source')
    plt.xlabel('Source')
    plt.ylabel('Amount (EUR)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('financial_reports/source_distribution.png')
    plt.close()

def generate_pnl_report(df, analysis_results):
    monthly_spending, monthly_income = analysis_results['monthly']
    source_spending, source_income = analysis_results['source']
    type_spending, type_income = analysis_results['type']
    
    # Align monthly data
    aligned_monthly_spending, aligned_monthly_income = monthly_spending.align(monthly_income, fill_value=0)
    monthly_net = aligned_monthly_income.add(aligned_monthly_spending.multiply(-1), fill_value=0)
    
    # Align source data
    aligned_source_spending, aligned_source_income = source_spending.align(source_income, fill_value=0)
    source_net = aligned_source_income.add(aligned_source_spending, fill_value=0)
    
    # Align type data
    aligned_type_spending, aligned_type_income = type_spending.align(type_income, fill_value=0)
    
    total_income = df[df['Amount'] > 0]['Amount'].sum()
    total_expenses = df[df['Amount'] < 0]['Amount'].sum()
    net_profit = total_income + total_expenses
    
    report = f"""Profit & Loss Report
==================

Summary:
--------
Total Income: €{total_income:,.2f}
Total Expenses: €{abs(total_expenses):,.2f}
Net Profit/Loss: €{net_profit:,.2f}

Monthly Breakdown:
----------------
Net Profit/Loss by Month:
{monthly_net.to_string()}

Source Analysis:
--------------
Net by Source:
{source_net.to_string()}

Transaction Types:
---------------
Income by Type:
{aligned_type_income.sort_values(ascending=False).to_string()}

Expenses by Type:
{aligned_type_spending.sort_values(ascending=False).to_string()}
"""
    
    with open('financial_reports/pnl_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    # Create detailed transaction log
    df.to_csv('financial_reports/detailed_transactions.csv', index=False, encoding='utf-8-sig')

def identify_client_income(description):
    """Identify if a transaction is client income and extract client name"""
    # List of known client identifiers
    client_prefixes = {
        # 'TR-CONNECTRSCOM': 'Connectrs',
        'TR-ONESTAT': 'Onestat',
        'TR-TRUE ODYSSEYS': 'True Odysseys',
        # 'TR-ESTR-GRUPO GLOBAL': 'Grupo Global',
        'TR-IPS-Mystic': 'Mystic Protocol',
        'TR-POLTAPP': 'Poltapp',
        'TR-MUTABLE POTENTIAL': 'Mutable Potential',
        'WONDERBREEZE': 'Wonderbreeze',
        'TR-LUIS MARCAL': 'Luis Marcal'
    }
    
    description = str(description).upper()
    for prefix, client_name in client_prefixes.items():
        if prefix.upper() in description:
            return client_name
    return None

def calculate_vat_summary(df):
    """Calculate VAT summary by quarter"""
    # Add quarter column
    df['Quarter'] = df['Date'].dt.quarter
    
    # Identify client income transactions
    df['Client'] = df['Description'].apply(identify_client_income)
    
    # Split into income and expenses
    client_income = df[df['Client'].notna()].copy()
    expenses = df[df['Amount'] < 0].copy()
    
    # Calculate VAT for income (23% of income)
    client_income['VAT_Collected'] = client_income['Amount'] * 0.23
    
    # Use VAT from e-fatura for expenses when available, otherwise estimate
    expenses['VAT_Paid'] = expenses.apply(
        lambda x: x.get('VAT', -x['Amount'] * 0.23) if x['Source'] == 'E-fatura' else -x['Amount'] * 0.23,
        axis=1
    )
    
    # Group by quarter for income
    quarterly_income_vat = client_income.groupby('Quarter').agg({
        'Amount': 'sum',
        'VAT_Collected': 'sum',
        'Client': lambda x: len(x.unique())
    }).rename(columns={
        'Amount': 'Total Income',
        'Client': 'Number of Clients'
    })
    
    # Group by quarter for expenses
    quarterly_expense_vat = expenses.groupby('Quarter').agg({
        'Amount': 'sum',
        'VAT_Paid': 'sum'
    }).rename(columns={
        'Amount': 'Total Expenses'
    })
    
    # Combine quarterly summaries
    quarterly_vat = pd.concat([
        quarterly_income_vat,
        quarterly_expense_vat
    ], axis=1).fillna(0)
    
    # Calculate net VAT
    quarterly_vat['Net_VAT'] = quarterly_vat['VAT_Collected'] + quarterly_vat['VAT_Paid']
    
    # Calculate yearly totals
    yearly_totals = {
        'Total Income': quarterly_vat['Total Income'].sum(),
        'Total Expenses': quarterly_vat['Total Expenses'].sum(),
        'VAT Collected': quarterly_vat['VAT_Collected'].sum(),
        'VAT Paid': quarterly_vat['VAT_Paid'].sum(),
        'Net VAT': quarterly_vat['Net_VAT'].sum(),
        'Unique Clients': len(client_income['Client'].unique())
    }
    
    # Group by client and quarter
    client_summary = client_income.groupby(['Quarter', 'Client']).agg({
        'Amount': 'sum',
        'VAT_Collected': 'sum'
    }).round(2)
    
    return quarterly_vat, client_summary, yearly_totals

def generate_vat_report(quarterly_vat, client_summary, yearly_totals):
    """Generate VAT report"""
    report = """VAT and Client Income Report
==========================

Yearly Summary:
-------------
Total Income: €{:,.2f}
Total Expenses: €{:,.2f}
Total VAT Collected: €{:,.2f}
Total VAT Paid: €{:,.2f}
Net VAT (To Pay/Receive): €{:,.2f}
Total Unique Clients: {}

Quarterly Summary:
----------------
""".format(
        yearly_totals['Total Income'],
        yearly_totals['Total Expenses'],
        yearly_totals['VAT Collected'],
        yearly_totals['VAT Paid'],
        yearly_totals['Net VAT'],
        yearly_totals['Unique Clients']
    )
    
    for quarter, row in quarterly_vat.iterrows():
        report += f"\nQ{quarter}:\n"
        report += f"Total Income: €{row['Total Income']:,.2f}\n"
        report += f"VAT Collected: €{row['VAT_Collected']:,.2f}\n"
        report += f"Total Expenses: €{row['Total Expenses']:,.2f}\n"
        report += f"VAT Paid: €{row['VAT_Paid']:,.2f}\n"
        report += f"Net VAT (To Pay/Receive): €{row['Net_VAT']:,.2f}\n"
        report += f"Number of Clients: {row['Number of Clients']}\n"
        
        report += "\nClient Breakdown:\n"
        quarter_clients = client_summary.loc[quarter]
        for client, data in quarter_clients.iterrows():
            report += f"  {client}:\n"
            report += f"    Income: €{data['Amount']:,.2f}\n"
            report += f"    VAT: €{data['VAT_Collected']:,.2f}\n"
    
    with open('financial_reports/vat_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    # Save detailed data to CSV
    quarterly_vat.to_csv('financial_reports/quarterly_vat_summary.csv')
    client_summary.to_csv('financial_reports/client_summary.csv')

def main():
    try:
        # Load transactions from all sources
        revolut_df = load_revolut_transactions('revolut_bank_extract.csv')
        montepio_df = load_montepio_transactions('montepio_bank_extract.csv')
        efatura_df = load_efatura_transactions('financas_extract.csv')
        
        # Merge all transactions
        merged_df = merge_all_transactions(revolut_df, montepio_df, efatura_df)
        
        # Analyze transactions
        analysis_results = analyze_transactions(merged_df)
        
        # Generate visualizations
        generate_plots(analysis_results)
        
        # Generate PNL report
        generate_pnl_report(merged_df, analysis_results)
        
        # Calculate VAT summary
        quarterly_vat, client_summary, yearly_totals = calculate_vat_summary(merged_df)
        
        # Generate VAT report
        generate_vat_report(quarterly_vat, client_summary, yearly_totals)
        
        print("Financial reports and visualizations have been generated in the 'financial_reports' directory.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
