import streamlit as st
import pandas as pd
import numpy as np
from utils.financial_data import format_currency, format_percentage, format_ratio

def render_metrics_comparison(comparison_data):
    """
    Render the financial metrics comparison table and summary
    
    Args:
        comparison_data (dict): Dictionary containing financial data for each company
    """
    if not comparison_data:
        st.error("No comparison data available")
        return
    
    # Create tabs for different metric categories
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", "💰 Valuation", "📈 Profitability", 
        "💵 Financial Performance", "🏦 Financial Health"
    ])
    
    with tab1:
        render_overview_metrics(comparison_data)
    
    with tab2:
        render_valuation_metrics(comparison_data)
    
    with tab3:
        render_profitability_metrics(comparison_data)
    
    with tab4:
        render_performance_metrics(comparison_data)
    
    with tab5:
        render_financial_health_metrics(comparison_data)

def render_overview_metrics(comparison_data):
    """Render overview metrics including basic company info"""
    st.subheader("Company Overview")
    
    # Create overview dataframe
    overview_data = {}
    for company, data in comparison_data.items():
        overview_data[company] = {
            'Company Name': data.get('Company Name', 'N/A'),
            'Sector': data.get('Sector', 'N/A'),
            'Industry': data.get('Industry', 'N/A'),
            'Market Cap': format_currency(data.get('Market Cap')),
            'Current Price': format_currency(data.get('Current Price')),
            'P/E Ratio': format_ratio(data.get('P/E Ratio')),
            '52W High': format_currency(data.get('52 Week High')),
            '52W Low': format_currency(data.get('52 Week Low'))
        }
    
    df_overview = pd.DataFrame(overview_data).T
    st.dataframe(df_overview, use_container_width=True)
    
    # Quick comparison metrics
    st.subheader("Quick Comparison")
    
    companies = list(comparison_data.keys())
    
    if len(companies) >= 2:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Largest by Market Cap",
                value=get_largest_by_metric(comparison_data, 'Market Cap'),
                help="Company with the highest market capitalization"
            )
        
        with col2:
            st.metric(
                label="Lowest P/E Ratio",
                value=get_smallest_by_metric(comparison_data, 'P/E Ratio'),
                help="Company with the lowest P/E ratio (potentially undervalued)"
            )
        
        with col3:
            st.metric(
                label="Highest Profit Margin",
                value=get_largest_by_metric(comparison_data, 'Profit Margin'),
                help="Company with the highest profit margin"
            )

def render_valuation_metrics(comparison_data):
    """Render valuation metrics"""
    st.subheader("Valuation Metrics")
    
    valuation_data = {}
    for company, data in comparison_data.items():
        valuation_data[company] = {
            'Market Cap': format_currency(data.get('Market Cap')),
            'P/E Ratio': format_ratio(data.get('P/E Ratio')),
            'Forward P/E': format_ratio(data.get('Forward P/E')),
            'P/B Ratio': format_ratio(data.get('P/B Ratio')),
            'EV/EBITDA': format_ratio(data.get('EV/EBITDA')),
            'Dividend Yield': format_percentage(data.get('Dividend Yield')),
            'Dividend Rate': format_currency(data.get('Dividend Rate'))
        }
    
    df_valuation = pd.DataFrame(valuation_data).T
    st.dataframe(df_valuation, use_container_width=True)
    
    # Valuation insights
    render_valuation_insights(comparison_data)

def render_profitability_metrics(comparison_data):
    """Render profitability metrics"""
    st.subheader("Profitability Metrics")
    
    profitability_data = {}
    for company, data in comparison_data.items():
        profitability_data[company] = {
            'Gross Margin': format_percentage(data.get('Gross Margin')),
            'Operating Margin': format_percentage(data.get('Operating Margin')),
            'Profit Margin': format_percentage(data.get('Profit Margin')),
            'ROE': format_percentage(data.get('ROE')),
            'ROA': format_percentage(data.get('ROA')),
            'Revenue Growth': format_percentage(data.get('Revenue Growth')),
            'Earnings Growth': format_percentage(data.get('Earnings Growth'))
        }
    
    df_profitability = pd.DataFrame(profitability_data).T
    st.dataframe(df_profitability, use_container_width=True)
    
    # Profitability ranking
    render_profitability_ranking(comparison_data)

def render_performance_metrics(comparison_data):
    """Render financial performance metrics"""
    st.subheader("Financial Performance (TTM)")
    
    performance_data = {}
    for company, data in comparison_data.items():
        performance_data[company] = {
            'Revenue': format_currency(data.get('Revenue (TTM)')),
            'Gross Profit': format_currency(data.get('Gross Profit (TTM)')),
            'Net Income': format_currency(data.get('Net Income (TTM)')),
            'Operating Cash Flow': format_currency(data.get('Operating Cash Flow (TTM)')),
            'Free Cash Flow': format_currency(data.get('Free Cash Flow (TTM)'))
        }
    
    df_performance = pd.DataFrame(performance_data).T
    st.dataframe(df_performance, use_container_width=True)
    
    # Performance comparison
    render_performance_comparison(comparison_data)

def render_financial_health_metrics(comparison_data):
    """Render financial health metrics"""
    st.subheader("Financial Health")
    
    health_data = {}
    for company, data in comparison_data.items():
        health_data[company] = {
            'Current Ratio': format_ratio(data.get('Current Ratio')),
            'Quick Ratio': format_ratio(data.get('Quick Ratio')),
            'Debt to Equity': format_ratio(data.get('Debt to Equity')),
            'Total Debt': format_currency(data.get('Total Debt')),
            'Total Cash': format_currency(data.get('Total Cash')),
            'Payout Ratio': format_percentage(data.get('Payout Ratio'))
        }
    
    df_health = pd.DataFrame(health_data).T
    st.dataframe(df_health, use_container_width=True)
    
    # Financial health analysis
    render_financial_health_analysis(comparison_data)

def get_largest_by_metric(comparison_data, metric):
    """Get company with largest value for a specific metric"""
    try:
        max_value = -float('inf')
        max_company = 'N/A'
        
        for company, data in comparison_data.items():
            value = data.get(metric)
            if pd.notna(value) and value > max_value:
                max_value = value
                max_company = company
        
        return max_company if max_company != 'N/A' else 'N/A'
    except:
        return 'N/A'

def get_smallest_by_metric(comparison_data, metric):
    """Get company with smallest value for a specific metric"""
    try:
        min_value = float('inf')
        min_company = 'N/A'
        
        for company, data in comparison_data.items():
            value = data.get(metric)
            if pd.notna(value) and value > 0 and value < min_value:
                min_value = value
                min_company = company
        
        return min_company if min_company != 'N/A' else 'N/A'
    except:
        return 'N/A'

def render_valuation_insights(comparison_data):
    """Render valuation insights"""
    st.subheader("💡 Valuation Insights")
    
    # Calculate average P/E ratio
    pe_ratios = []
    for company, data in comparison_data.items():
        pe = data.get('P/E Ratio')
        if pd.notna(pe) and pe > 0:
            pe_ratios.append((company, pe))
    
    if pe_ratios:
        avg_pe = sum(pe for _, pe in pe_ratios) / len(pe_ratios)
        
        st.write(f"**Average P/E Ratio:** {avg_pe:.2f}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Below Average P/E (Potentially Undervalued):**")
            below_avg = [company for company, pe in pe_ratios if pe < avg_pe]
            for company in below_avg:
                pe = next(pe for c, pe in pe_ratios if c == company)
                st.write(f"• {company}: {pe:.2f}")
        
        with col2:
            st.write("**Above Average P/E (Potentially Overvalued):**")
            above_avg = [company for company, pe in pe_ratios if pe >= avg_pe]
            for company in above_avg:
                pe = next(pe for c, pe in pe_ratios if c == company)
                st.write(f"• {company}: {pe:.2f}")

def render_profitability_ranking(comparison_data):
    """Render profitability ranking"""
    st.subheader("🏆 Profitability Ranking")
    
    # Rank by profit margin
    profit_margins = []
    for company, data in comparison_data.items():
        margin = data.get('Profit Margin')
        if pd.notna(margin):
            profit_margins.append((company, margin * 100 if margin < 1 else margin))
    
    if profit_margins:
        profit_margins.sort(key=lambda x: x[1], reverse=True)
        
        st.write("**Ranked by Profit Margin:**")
        for i, (company, margin) in enumerate(profit_margins, 1):
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
            st.write(f"{medal} {company}: {margin:.2f}%")

def render_performance_comparison(comparison_data):
    """Render performance comparison"""
    st.subheader("📊 Revenue Comparison")
    
    # Revenue comparison
    revenues = []
    for company, data in comparison_data.items():
        revenue = data.get('Revenue (TTM)')
        if pd.notna(revenue):
            revenues.append((company, revenue))
    
    if revenues:
        revenues.sort(key=lambda x: x[1], reverse=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**By Revenue Size:**")
            for company, revenue in revenues:
                st.write(f"• {company}: {format_currency(revenue)}")
        
        with col2:
            # Calculate revenue market share
            total_revenue = sum(revenue for _, revenue in revenues)
            st.write("**Revenue Market Share:**")
            for company, revenue in revenues:
                share = (revenue / total_revenue) * 100
                st.write(f"• {company}: {share:.1f}%")

def render_financial_health_analysis(comparison_data):
    """Render financial health analysis"""
    st.subheader("🏥 Financial Health Analysis")
    
    # Debt analysis
    debt_ratios = []
    for company, data in comparison_data.items():
        debt_equity = data.get('Debt to Equity')
        if pd.notna(debt_equity):
            debt_ratios.append((company, debt_equity))
    
    if debt_ratios:
        avg_debt_ratio = sum(ratio for _, ratio in debt_ratios) / len(debt_ratios)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Low Debt (Conservative):**")
            low_debt = [company for company, ratio in debt_ratios if ratio < avg_debt_ratio]
            for company in low_debt:
                ratio = next(r for c, r in debt_ratios if c == company)
                st.write(f"• {company}: {ratio:.2f}")
        
        with col2:
            st.write("**High Debt (Aggressive):**")
            high_debt = [company for company, ratio in debt_ratios if ratio >= avg_debt_ratio]
            for company in high_debt:
                ratio = next(r for c, r in debt_ratios if c == company)
                st.write(f"• {company}: {ratio:.2f}")
