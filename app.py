import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

st.set_page_config(page_title="Financial Transactions", layout="wide")

def identify_client_income(description):
    """Identify if a transaction is client income and extract client name"""
    # List of known client identifiers
    client_prefixes = {
        'TR-ONESTAT': 'Onestat',
        'TR-TRUE ODYSSEYS': 'True Odysseys',
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

def load_transactions():
    # Read the CSV file
    df = pd.read_csv('financial_reports/detailed_transactions.csv', parse_dates=['Date'])
    
    # Add client identification
    df['Client'] = df['Description'].apply(identify_client_income)
    
    return df

def get_quarter_dates(df, quarter):
    year = 2024
    quarter_starts = {
        1: f"{year}-01-01",
        2: f"{year}-04-01",
        3: f"{year}-07-01",
        4: f"{year}-10-01"
    }
    quarter_ends = {
        1: f"{year}-03-31",
        2: f"{year}-06-30",
        3: f"{year}-09-30",
        4: f"{year}-12-31"
    }
    return pd.to_datetime(quarter_starts[quarter]), pd.to_datetime(quarter_ends[quarter])

def create_quarterly_summary(df, start_date, end_date):
    # Filter data for the quarter
    quarter_data = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    
    # Monthly summary
    monthly = quarter_data.set_index('Date').resample('M').agg({
        'Amount': [
            ('Income', lambda x: x[x > 0].sum()),
            ('Expenses', lambda x: x[x < 0].sum()),
            ('Net', 'sum')
        ]
    })
    monthly.columns = monthly.columns.droplevel(0)
    
    return quarter_data, monthly

def load_client_summary():
    return pd.read_csv('financial_reports/client_summary.csv')

def show_vat_section(quarter_num, quarter_data, show_total=False):
    st.subheader("VAT Analysis")
    
    # Add warning about VAT calculations
    st.warning("""
        Note: These calculations are for reference only:
        - Income VAT: 23% of client income (to be paid)
        - Expense VAT: Actual VAT amounts from e-fatura (deductible)
        Please consult with an accountant for official VAT declarations.
    """)
    
    # Filter data
    client_income = quarter_data[quarter_data['Client'].notna()].copy()
    efatura_expenses = quarter_data[
        (quarter_data['Amount'] < 0) & 
        (quarter_data['Source'] == 'E-fatura') & 
        (quarter_data['VAT'].notna())
    ].copy()
    
    # Calculate VAT metrics
    total_income = client_income['Amount'].sum()
    vat_to_pay = total_income * 0.23
    income_with_vat = total_income * 1.23
    
    # Group e-fatura expenses by company
    efatura_summary = efatura_expenses.groupby('Description').agg({
        'Amount': 'sum',
        'VAT': 'sum',
        'Base': 'sum'
    }).round(2)
    
    total_deductible_vat = efatura_summary['VAT'].sum()
    net_vat = vat_to_pay - abs(total_deductible_vat)
    
    # Display summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Client Income (without VAT)", f"€{total_income:,.2f}")
        st.metric("Client Income (with VAT)", f"€{income_with_vat:,.2f}")
    with col2:
        st.metric("VAT to Pay (23%)", f"€{vat_to_pay:,.2f}")
        st.metric("VAT Deductible", f"€{abs(total_deductible_vat):,.2f}")
    with col3:
        st.metric("Net VAT to Pay", f"€{net_vat:,.2f}")
        st.metric("Number of Clients", client_income['Client'].nunique())
    
    # Show client income breakdown
    st.subheader("Client Income and VAT")
    client_summary = client_income.groupby('Client').agg({
        'Amount': 'sum',
        'Date': 'count'
    }).rename(columns={
        'Date': 'Number of Transactions'
    })
    client_summary['VAT (23%)'] = client_summary['Amount'] * 0.23
    client_summary['Total with VAT'] = client_summary['Amount'] * 1.23
    
    st.dataframe(
        client_summary.style.format({
            'Amount': '€{:,.2f}',
            'VAT (23%)': '€{:,.2f}',
            'Total with VAT': '€{:,.2f}',
            'Number of Transactions': '{:,.0f}'
        }),
        use_container_width=True
    )
    
    # Show e-fatura expenses breakdown
    st.subheader("E-fatura Expenses and VAT")
    
    # Add percentage calculations
    efatura_summary['VAT %'] = (efatura_summary['VAT'] / efatura_summary['Base'] * 100).round(1)
    efatura_summary = efatura_summary.sort_values('VAT', ascending=False)
    
    st.dataframe(
        efatura_summary.style.format({
            'Amount': '€{:,.2f}',
            'VAT': '€{:,.2f}',
            'Base': '€{:,.2f}',
            'VAT %': '{:.1f}%'
        }),
        use_container_width=True
    )
    
    # Show VAT summary chart
    st.subheader("VAT Summary")
    fig = go.Figure()
    
    # Add VAT to pay bar
    fig.add_trace(go.Bar(
        name="VAT to Pay",
        x=["VAT Analysis"],
        y=[vat_to_pay],
        marker_color='red',
        opacity=0.7
    ))
    
    # Add VAT deductible bar
    fig.add_trace(go.Bar(
        name="VAT Deductible",
        x=["VAT Analysis"],
        y=[-abs(total_deductible_vat)],
        marker_color='green',
        opacity=0.7
    ))
    
    # Add net VAT line
    fig.add_trace(go.Scatter(
        name="Net VAT",
        x=["VAT Analysis"],
        y=[net_vat],
        mode='markers',
        marker=dict(size=12, color='blue')
    ))
    
    fig.update_layout(
        title="VAT Components",
        barmode='relative',
        showlegend=True,
        yaxis_title="Amount (EUR)"
    )
    
    st.plotly_chart(fig, use_container_width=True)

def style_dataframe(df):
    """Apply conditional styling to the DataFrame"""
    def color_amount(val):
        try:
            amount = float(str(val).replace('€', '').replace(',', ''))
            color = 'red' if amount < 0 else 'green'
            return f'color: {color}; font-weight: bold'
        except:
            return ''

    def highlight_row(row):
        try:
            # Check if Client column exists and has a value
            if 'Client' in row and pd.notna(row.get('Client')):
                return ['background-color: rgba(255, 255, 0, 0.1)'] * len(row)
        except:
            pass
        return [''] * len(row)

    # Format numbers and dates
    format_dict = {
        'Amount': '€{:,.2f}',
        'Date': lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else ''
    }
    
    # Add VAT formatting if the column exists
    if 'VAT' in df.columns:
        format_dict['VAT'] = lambda x: f'€{x:,.2f}' if pd.notna(x) else ''
    
    # Create styled dataframe
    styled = df.style.format(format_dict)
    
    # Apply colors to Amount column
    if 'Amount' in df.columns:
        styled = styled.applymap(color_amount, subset=['Amount'])
    
    # Highlight client transactions only if Client column exists
    if 'Client' in df.columns:
        styled = styled.apply(highlight_row, axis=1)
    
    # Add alternating row colors
    styled = styled.set_properties(**{
        'background-color': 'rgba(0, 0, 0, 0.02)',
        'padding': '8px',
        'border-bottom': '1px solid #ddd'
    }, subset=pd.IndexSlice[df.index[::2], :])
    
    # Add hover effect
    styled.set_table_styles([{
        'selector': 'tr:hover',
        'props': [('background-color', 'rgba(0, 0, 0, 0.05)')]
    }])
    
    return styled

def main():
    st.title("Financial Transactions Dashboard")
    
    # Load data
    df = load_transactions()
    
    # Add view selector
    view_type = st.radio(
        "Select View",
        ["Full Year", "Quarterly"],
        horizontal=True
    )
    
    if view_type == "Full Year":
        # Summary metrics for the full year
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Transactions", len(df))
        with col2:
            year_income = df[df['Amount'] > 0]['Amount'].sum()
            st.metric("Year Income", f"€{year_income:,.2f}")
        with col3:
            year_expenses = df[df['Amount'] < 0]['Amount'].sum()
            st.metric("Year Expenses", f"€{abs(year_expenses):,.2f}")
        with col4:
            st.metric("Year Net", f"€{(year_income + year_expenses):,.2f}")
        
        # Monthly summary plot for full year
        st.subheader("Monthly Summary")
        monthly_data = df.set_index('Date').resample('M').agg({
            'Amount': [
                ('Income', lambda x: x[x > 0].sum()),
                ('Expenses', lambda x: x[x < 0].sum()),
                ('Net', 'sum')
            ]
        })
        monthly_data.columns = monthly_data.columns.droplevel(0)
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Bar(
                name="Income",
                x=monthly_data.index,
                y=monthly_data['Income'],
                marker_color='green',
                opacity=0.7
            )
        )
        
        fig.add_trace(
            go.Bar(
                name="Expenses",
                x=monthly_data.index,
                y=-monthly_data['Expenses'],
                marker_color='red',
                opacity=0.7
            )
        )
        
        fig.add_trace(
            go.Scatter(
                name="Net",
                x=monthly_data.index,
                y=monthly_data['Net'],
                line=dict(color='blue', width=2),
                mode='lines+markers'
            ),
            secondary_y=True
        )
        
        fig.update_layout(
            title="Monthly Summary for Full Year",
            barmode='relative',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Transaction type analysis for full year
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Income by Type")
            income_by_type = df[df['Amount'] > 0].groupby('Type')['Amount'].sum()
            fig = px.pie(
                values=income_by_type.values,
                names=income_by_type.index,
                title="Income Distribution by Type"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Expenses by Type")
            expenses_by_type = df[df['Amount'] < 0].groupby('Type')['Amount'].sum().abs()
            fig = px.pie(
                values=expenses_by_type.values,
                names=expenses_by_type.index,
                title="Expense Distribution by Type"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Scatter plot of all transactions
        st.subheader("Transaction Timeline")
        fig = go.Figure()
        
        for source in df['Source'].unique():
            source_data = df[df['Source'] == source]
            fig.add_trace(go.Scatter(
                x=source_data['Date'],
                y=source_data['Amount'],
                mode='markers',
                name=source,
                hovertemplate=(
                    "<b>%{text}</b><br>" +
                    "Date: %{x}<br>" +
                    "Amount: €%{y:,.2f}<br>" +
                    "<extra></extra>"
                ),
                text=source_data['Description']
            ))
        
        fig.update_layout(
            title="All Transactions Over Time",
            xaxis_title="Date",
            yaxis_title="Amount (EUR)",
            hovermode='closest',
            showlegend=True
        )
        
        # Add a horizontal line at y=0
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # VAT analysis for full year
        st.markdown("---")
        show_vat_section(0, df, show_total=True)
        
    else:
        # Quarterly view code
        quarters = ["Q1 (Jan-Mar)", "Q2 (Apr-Jun)", "Q3 (Jul-Sep)", "Q4 (Oct-Dec)"]
        selected_quarter = st.radio("Select Quarter", quarters, horizontal=True)
        quarter_num = quarters.index(selected_quarter) + 1
        
        # Get date range for selected quarter
        start_date, end_date = get_quarter_dates(df, quarter_num)
        quarter_data, monthly_summary = create_quarterly_summary(df, start_date, end_date)
        
        # Summary metrics for the quarter
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Transactions", len(quarter_data))
        with col2:
            quarter_income = quarter_data[quarter_data['Amount'] > 0]['Amount'].sum()
            st.metric("Quarter Income", f"€{quarter_income:,.2f}")
        with col3:
            quarter_expenses = quarter_data[quarter_data['Amount'] < 0]['Amount'].sum()
            st.metric("Quarter Expenses", f"€{abs(quarter_expenses):,.2f}")
        with col4:
            st.metric("Quarter Net", f"€{(quarter_income + quarter_expenses):,.2f}")
        
        # Monthly summary plot
        st.subheader("Monthly Summary")
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Bar(
                name="Income",
                x=monthly_summary.index,
                y=monthly_summary['Income'],
                marker_color='green',
                opacity=0.7
            )
        )
        
        fig.add_trace(
            go.Bar(
                name="Expenses",
                x=monthly_summary.index,
                y=-monthly_summary['Expenses'],
                marker_color='red',
                opacity=0.7
            )
        )
        
        fig.add_trace(
            go.Scatter(
                name="Net",
                x=monthly_summary.index,
                y=monthly_summary['Net'],
                line=dict(color='blue', width=2),
                mode='lines+markers'
            ),
            secondary_y=True
        )
        
        fig.update_layout(
            title=f"Monthly Summary for {selected_quarter}",
            barmode='relative',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Transaction type analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Income by Type")
            income_by_type = quarter_data[quarter_data['Amount'] > 0].groupby('Type')['Amount'].sum()
            fig = px.pie(
                values=income_by_type.values,
                names=income_by_type.index,
                title="Income Distribution by Type"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Expenses by Type")
            expenses_by_type = quarter_data[quarter_data['Amount'] < 0].groupby('Type')['Amount'].sum().abs()
            fig = px.pie(
                values=expenses_by_type.values,
                names=expenses_by_type.index,
                title="Expense Distribution by Type"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # VAT section for quarter
        st.markdown("---")
        show_vat_section(quarter_num, quarter_data)
    
    # Common transaction table section
    st.markdown("---")
    st.subheader("Detailed Transactions")
    
    # Get the appropriate dataset based on view type
    display_df = quarter_data if view_type == "Quarterly" else df
    
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        # Source filter
        sources = ['All'] + list(display_df['Source'].unique())
        selected_source = st.selectbox("Filter by Source", sources)
    
    with col2:
        # Transaction type filter
        types = ['All'] + list(display_df['Type'].unique())
        selected_type = st.selectbox("Filter by Type", types)
    
    with col3:
        # Search in descriptions
        search = st.text_input("Search in descriptions")
    
    # Apply filters
    filtered_data = display_df.copy()
    if selected_source != 'All':
        filtered_data = filtered_data[filtered_data['Source'] == selected_source]
    if selected_type != 'All':
        filtered_data = filtered_data[filtered_data['Type'] == selected_type]
    if search:
        filtered_data = filtered_data[filtered_data['Description'].str.contains(search, case=False, na=False)]
    
    # Sort options
    col1, col2 = st.columns(2)
    with col1:
        sort_col = st.selectbox("Sort by", ['Date', 'Amount', 'Description', 'Source', 'Type'])
    with col2:
        sort_order = st.radio("Sort order", ['Ascending', 'Descending'])
    
    # Apply sorting
    filtered_data = filtered_data.sort_values(
        by=sort_col,
        ascending=(sort_order == 'Ascending')
    )
    
    # Column selection
    default_columns = ['Date', 'Description', 'Amount']
    if 'Type' in filtered_data.columns:
        default_columns.append('Type')
    if 'Source' in filtered_data.columns:
        default_columns.append('Source')
    if 'Client' in filtered_data.columns:
        default_columns.append('Client')
    
    columns_to_display = st.multiselect(
        "Select columns to display",
        options=filtered_data.columns.tolist(),
        default=default_columns,
    )
    
    if columns_to_display:
        display_data = filtered_data[columns_to_display]
    else:
        display_data = filtered_data
    
    # Display the table with enhanced styling
    try:
        styled_df = style_dataframe(display_data)
        st.dataframe(
            styled_df,
            use_container_width=True,
            height=400
        )
    except Exception as e:
        st.error(f"Error displaying table: {str(e)}")
        st.dataframe(display_data)  # Fallback to unstyled display

if __name__ == "__main__":
    main() 