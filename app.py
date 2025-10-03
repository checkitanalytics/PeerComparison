import streamlit as st
import pandas as pd
import numpy as np
from components.company_selector import render_company_selector
from components.metrics_display import render_metrics_comparison
from components.charts import render_comparison_charts
from components.historical_trends import render_historical_trends
from utils.financial_data import get_company_data, get_financial_metrics
from utils.calculations import calculate_percentage_differences

# Page configuration
st.set_page_config(
    page_title="Financial Data Comparison",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'selected_companies' not in st.session_state:
    st.session_state.selected_companies = []
if 'comparison_data' not in st.session_state:
    st.session_state.comparison_data = {}

# Main title
st.title("📊 Financial Data Comparison Platform")
st.markdown("Compare key financial metrics and performance across peer companies")

# Sidebar for company selection
with st.sidebar:
    st.header("🏢 Company Selection")
    selected_companies = render_company_selector()

# Main content area
if len(selected_companies) < 2:
    st.info("Please select at least 2 companies to start comparison")
    
    # Show example of available metrics when no companies selected
    st.subheader("Available Metrics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📈 Financial Performance**")
        st.markdown("- Revenue (TTM)")
        st.markdown("- Gross Profit")
        st.markdown("- Net Income")
        st.markdown("- Operating Cash Flow")
    
    with col2:
        st.markdown("**📊 Profitability Ratios**")
        st.markdown("- Gross Margin")
        st.markdown("- Operating Margin")
        st.markdown("- Net Profit Margin")
        st.markdown("- Return on Equity (ROE)")
    
    with col3:
        st.markdown("**💰 Valuation Metrics**")
        st.markdown("- Market Capitalization")
        st.markdown("- P/E Ratio")
        st.markdown("- P/B Ratio")
        st.markdown("- EV/EBITDA")

else:
    # Fetch data for selected companies
    with st.spinner("Fetching financial data..."):
        comparison_data = {}
        failed_companies = []
        
        for company in selected_companies:
            try:
                data = get_company_data(company)
                if data is not None:
                    metrics = get_financial_metrics(data)
                    comparison_data[company] = metrics
                else:
                    failed_companies.append(company)
            except Exception as e:
                st.error(f"Failed to fetch data for {company}: {str(e)}")
                failed_companies.append(company)
    
    # Remove failed companies from selection
    if failed_companies:
        st.warning(f"Could not fetch data for: {', '.join(failed_companies)}")
        comparison_data = {k: v for k, v in comparison_data.items() if k not in failed_companies}
    
    if len(comparison_data) >= 2:
        st.session_state.comparison_data = comparison_data
        
        # Create tabs for different analysis views
        analysis_tab1, analysis_tab2 = st.tabs([
            "📊 Current Comparison", 
            "📈 Historical Trends"
        ])
        
        with analysis_tab1:
            # Display metrics comparison
            st.subheader("📊 Financial Metrics Comparison")
            render_metrics_comparison(comparison_data)
            
            # Display comparison charts
            st.subheader("📈 Interactive Comparison Charts")
            render_comparison_charts(comparison_data)
            
            # Percentage differences analysis
            st.subheader("🔍 Comparative Analysis")
            percentage_diffs = calculate_percentage_differences(comparison_data)
        
            if percentage_diffs is not None:
                st.markdown("**Performance Differences (% vs Average)**")
                
                # Create columns for better display
                companies = list(comparison_data.keys())
                cols = st.columns(len(companies))
                
                for i, company in enumerate(companies):
                    with cols[i]:
                        st.markdown(f"**{company}**")
                        company_diffs = percentage_diffs[company]
                        
                        for metric, diff in company_diffs.items():
                            if not pd.isna(diff):
                                color = "green" if diff > 0 else "red"
                                st.markdown(f"{metric}: <span style='color:{color}'>{diff:+.1f}%</span>", 
                                          unsafe_allow_html=True)
        
        with analysis_tab2:
            # Display historical trends
            render_historical_trends(selected_companies)
    
    elif len(comparison_data) == 1:
        st.info("Only one company's data was successfully fetched. Please add more companies for comparison.")
    else:
        st.error("Could not fetch data for any of the selected companies. Please try different ticker symbols.")

# Footer
st.markdown("---")
st.markdown("*Data provided by Yahoo Finance. This application is for informational purposes only and should not be considered as investment advice.*")
