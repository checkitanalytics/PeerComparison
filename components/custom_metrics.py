import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils.calculations import calculate_financial_ratios

def render_custom_metrics(comparison_data):
    """
    Render custom metric calculations and ratios
    
    Args:
        comparison_data (dict): Dictionary containing financial data for each company
    """
    st.subheader("🧮 Custom Metric Calculations")
    
    if not comparison_data or len(comparison_data) < 1:
        st.info("Select at least one company to calculate custom metrics")
        return
    
    # Create tabs for different custom metrics
    metric_tab1, metric_tab2, metric_tab3 = st.tabs([
        "📊 Calculated Ratios",
        "💡 Create Custom Formula",
        "📈 Advanced Metrics"
    ])
    
    with metric_tab1:
        render_calculated_ratios(comparison_data)
    
    with metric_tab2:
        render_custom_formula_builder(comparison_data)
    
    with metric_tab3:
        render_advanced_metrics(comparison_data)

def render_calculated_ratios(comparison_data):
    """Display additional calculated financial ratios"""
    st.markdown("### Additional Calculated Ratios")
    st.markdown("These ratios are calculated from the base financial metrics")
    
    # Calculate additional ratios for each company
    calculated_data = {}
    
    for company, data in comparison_data.items():
        additional_ratios = calculate_financial_ratios(data)
        
        # Combine original and calculated metrics
        combined = {}
        
        # Price to Sales
        if 'P/S Ratio' in additional_ratios:
            combined['Price/Sales'] = additional_ratios['P/S Ratio']
        
        # Enterprise Value
        if 'Enterprise Value' in additional_ratios:
            combined['Enterprise Value'] = additional_ratios['Enterprise Value']
        
        # Calculate additional custom ratios
        
        # Operating Cash Flow to Revenue
        ocf = data.get('Operating Cash Flow (TTM)')
        revenue = data.get('Revenue (TTM)')
        if pd.notna(ocf) and pd.notna(revenue) and revenue != 0:
            combined['OCF/Revenue'] = ocf / revenue
        
        # Free Cash Flow to Revenue
        fcf = data.get('Free Cash Flow (TTM)')
        if pd.notna(fcf) and pd.notna(revenue) and revenue != 0:
            combined['FCF/Revenue'] = fcf / revenue
        
        # Earnings Per Dollar of Debt
        net_income = data.get('Net Income (TTM)')
        total_debt = data.get('Total Debt')
        if pd.notna(net_income) and pd.notna(total_debt) and total_debt != 0:
            combined['Earnings/Debt'] = net_income / total_debt
        
        # Cash to Debt Ratio
        total_cash = data.get('Total Cash')
        if pd.notna(total_cash) and pd.notna(total_debt) and total_debt != 0:
            combined['Cash/Debt'] = total_cash / total_debt
        
        # Market Cap to Revenue (another way)
        market_cap = data.get('Market Cap')
        if pd.notna(market_cap) and pd.notna(revenue) and revenue != 0:
            combined['Market Cap/Revenue'] = market_cap / revenue
        
        # FCF Yield
        if pd.notna(fcf) and pd.notna(market_cap) and market_cap != 0:
            combined['FCF Yield'] = fcf / market_cap
        
        # Dividend Payout from FCF
        dividend_rate = data.get('Dividend Rate')
        if pd.notna(dividend_rate) and pd.notna(fcf) and fcf != 0:
            # Approximate annual dividend
            annual_dividend = dividend_rate
            combined['Dividend/FCF'] = annual_dividend / fcf
        
        calculated_data[company] = combined
    
    if calculated_data:
        # Create DataFrame
        df_calculated = pd.DataFrame(calculated_data).T
        
        # Format the DataFrame
        formatted_df = df_calculated.copy()
        for col in formatted_df.columns:
            if col in ['Enterprise Value', 'Price/Sales', 'Market Cap/Revenue']:
                formatted_df[col] = formatted_df[col].apply(lambda x: f"{x/1e9:.2f}B" if pd.notna(x) and abs(x) >= 1e9 else f"{x/1e6:.2f}M" if pd.notna(x) and abs(x) >= 1e6 else f"{x:.2f}" if pd.notna(x) else "N/A")
            elif col in ['OCF/Revenue', 'FCF/Revenue', 'FCF Yield', 'Dividend/FCF']:
                formatted_df[col] = formatted_df[col].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "N/A")
            else:
                formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
        
        st.dataframe(formatted_df, use_container_width=True)
        
        # Visualization of selected ratio
        st.markdown("#### Ratio Comparison Chart")
        
        available_ratios = [col for col in df_calculated.columns if df_calculated[col].notna().any()]
        
        if available_ratios:
            selected_ratio = st.selectbox(
                "Select Ratio to Visualize",
                options=available_ratios,
                index=0
            )
            
            # Create bar chart
            companies = []
            values = []
            
            for company in df_calculated.index:
                value = df_calculated.loc[company, selected_ratio]
                if pd.notna(value):
                    companies.append(company)
                    values.append(value)
            
            if companies:
                fig = go.Figure(data=[go.Bar(
                    x=companies,
                    y=values,
                    text=[f"{v:.2f}" for v in values],
                    textposition='auto',
                    marker_color='lightblue'
                )])
                
                fig.update_layout(
                    title=f"{selected_ratio} Comparison",
                    xaxis_title="Companies",
                    yaxis_title=selected_ratio,
                    showlegend=False,
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)

def render_custom_formula_builder(comparison_data):
    """Allow users to create custom formulas"""
    st.markdown("### Custom Formula Builder")
    st.markdown("Create your own metrics using available financial data")
    
    # Get available metrics
    all_metrics = set()
    for data in comparison_data.values():
        all_metrics.update([k for k, v in data.items() if isinstance(v, (int, float)) and pd.notna(v)])
    
    numeric_metrics = sorted(list(all_metrics))
    
    if not numeric_metrics:
        st.warning("No numeric metrics available")
        return
    
    st.markdown("#### Formula Builder")
    
    col1, col2 = st.columns(2)
    
    with col1:
        metric1 = st.selectbox(
            "Numerator (Metric 1)",
            options=numeric_metrics,
            index=0,
            key="custom_metric1"
        )
    
    with col2:
        metric2 = st.selectbox(
            "Denominator (Metric 2)",
            options=numeric_metrics,
            index=1 if len(numeric_metrics) > 1 else 0,
            key="custom_metric2"
        )
    
    custom_name = st.text_input(
        "Custom Metric Name",
        value=f"{metric1}/{metric2}",
        help="Give your custom metric a meaningful name"
    )
    
    multiply_by_100 = st.checkbox(
        "Multiply by 100 (for percentage)",
        value=False
    )
    
    if st.button("Calculate Custom Metric"):
        if metric1 == metric2:
            st.warning("Please select different metrics for numerator and denominator")
        else:
            # Calculate custom metric
            results = {}
            
            for company, data in comparison_data.items():
                val1 = data.get(metric1)
                val2 = data.get(metric2)
                
                if pd.notna(val1) and pd.notna(val2) and val2 != 0:
                    result = val1 / val2
                    if multiply_by_100:
                        result *= 100
                    results[company] = result
            
            if results:
                st.success(f"Calculated {custom_name} for {len(results)} companies")
                
                # Display results
                df_results = pd.DataFrame.from_dict(results, orient='index', columns=[custom_name])
                df_results = df_results.sort_values(by=custom_name, ascending=False)
                
                # Format display
                formatted_results = df_results.copy()
                if multiply_by_100:
                    formatted_results[custom_name] = formatted_results[custom_name].apply(lambda x: f"{x:.2f}%")
                else:
                    formatted_results[custom_name] = formatted_results[custom_name].apply(lambda x: f"{x:.4f}")
                
                st.dataframe(formatted_results, use_container_width=True)
                
                # Visualize
                fig = go.Figure(data=[go.Bar(
                    x=list(results.keys()),
                    y=list(results.values()),
                    marker_color='lightcoral'
                )])
                
                y_label = f"{custom_name} (%)" if multiply_by_100 else custom_name
                
                fig.update_layout(
                    title=f"{custom_name} Comparison",
                    xaxis_title="Companies",
                    yaxis_title=y_label,
                    showlegend=False,
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("No valid results. Check if the selected metrics have data.")

def render_advanced_metrics(comparison_data):
    """Display advanced financial metrics and analysis"""
    st.markdown("### Advanced Metrics")
    
    # DuPont Analysis
    st.markdown("#### DuPont Analysis (ROE Decomposition)")
    st.markdown("ROE = Net Profit Margin × Asset Turnover × Equity Multiplier")
    st.info("Note: This analysis uses approximations as not all balance sheet data is available from the API")
    
    dupont_data = {}
    
    for company, data in comparison_data.items():
        profit_margin = data.get('Profit Margin')
        roe = data.get('ROE')
        revenue = data.get('Revenue (TTM)')
        market_cap = data.get('Market Cap')
        total_debt = data.get('Total Debt')
        
        # Calculate components
        components = {}
        
        # Net Profit Margin (already have this)
        if pd.notna(profit_margin):
            components['Net Profit Margin (%)'] = profit_margin * 100 if profit_margin < 1 else profit_margin
        
        # Asset Turnover Approximation (Revenue / Market Cap as proxy for assets)
        # This is an approximation since we don't have book value of assets
        if pd.notna(revenue) and pd.notna(market_cap) and market_cap != 0:
            asset_turnover_approx = revenue / market_cap
            components['Asset Turnover (approx)'] = asset_turnover_approx
        
        # Equity Multiplier Approximation using Debt/Equity
        debt_equity = data.get('Debt to Equity')
        if pd.notna(debt_equity):
            # Equity Multiplier = 1 + Debt/Equity
            equity_multiplier = 1 + (debt_equity / 100 if debt_equity > 10 else debt_equity)
            components['Equity Multiplier (approx)'] = equity_multiplier
        
        # Actual ROE for comparison
        if pd.notna(roe):
            components['ROE (%)'] = roe * 100 if roe < 1 else roe
        
        # Calculate implied ROE from components if available
        if 'Net Profit Margin (%)' in components and 'Asset Turnover (approx)' in components and 'Equity Multiplier (approx)' in components:
            npm = components['Net Profit Margin (%)'] / 100
            at = components['Asset Turnover (approx)']
            em = components['Equity Multiplier (approx)']
            implied_roe = npm * at * em * 100
            components['Implied ROE (%)'] = implied_roe
        
        if components:
            dupont_data[company] = components
    
    if dupont_data:
        df_dupont = pd.DataFrame(dupont_data).T
        
        # Format for display
        formatted_dupont = df_dupont.copy()
        for col in formatted_dupont.columns:
            if '(%)' in col:
                formatted_dupont[col] = formatted_dupont[col].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
            else:
                formatted_dupont[col] = formatted_dupont[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
        
        st.dataframe(formatted_dupont, use_container_width=True)
        
        # Explanation
        with st.expander("Understanding DuPont Analysis"):
            st.markdown("""
            The DuPont analysis breaks down ROE into three components:
            
            1. **Net Profit Margin**: How much profit is generated per dollar of revenue
            2. **Asset Turnover**: How efficiently assets are used to generate revenue
            3. **Equity Multiplier**: How much financial leverage is used
            
            **Note**: Due to API limitations, we use approximations:
            - Asset Turnover uses Market Cap as a proxy for Total Assets
            - Equity Multiplier is derived from Debt/Equity ratio
            - Implied ROE may differ from reported ROE due to these approximations
            """)
    
    # Magic Formula Metrics (Earnings Yield + ROIC approximation)
    st.markdown("#### Magic Formula Metrics")
    st.markdown("Earnings Yield and Return on Capital approximations")
    
    magic_data = {}
    
    for company, data in comparison_data.items():
        # Earnings Yield = 1 / P/E
        pe_ratio = data.get('P/E Ratio')
        if pd.notna(pe_ratio) and pe_ratio != 0:
            earnings_yield = (1 / pe_ratio) * 100
        else:
            earnings_yield = np.nan
        
        # Use ROE as proxy for ROIC
        roe = data.get('ROE')
        
        magic_data[company] = {
            'Earnings Yield (%)': earnings_yield,
            'ROE (%)': roe * 100 if pd.notna(roe) and roe < 1 else roe if pd.notna(roe) else np.nan
        }
    
    if magic_data:
        df_magic = pd.DataFrame(magic_data).T
        
        # Format for display
        formatted_magic = df_magic.copy()
        for col in formatted_magic.columns:
            formatted_magic[col] = formatted_magic[col].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
        
        st.dataframe(formatted_magic, use_container_width=True)
        
        # Scatter plot of Earnings Yield vs ROE
        companies = []
        ey_values = []
        roe_values = []
        
        for company in df_magic.index:
            ey = df_magic.loc[company, 'Earnings Yield (%)']
            roe_val = df_magic.loc[company, 'ROE (%)']
            
            if pd.notna(ey) and pd.notna(roe_val):
                companies.append(company)
                ey_values.append(ey)
                roe_values.append(roe_val)
        
        if companies:
            fig = go.Figure(data=[go.Scatter(
                x=ey_values,
                y=roe_values,
                mode='markers+text',
                text=companies,
                textposition='top center',
                marker=dict(size=12, color='lightgreen'),
                hovertemplate='<b>%{text}</b><br>Earnings Yield: %{x:.2f}%<br>ROE: %{y:.2f}%<extra></extra>'
            )])
            
            fig.update_layout(
                title="Magic Formula: Earnings Yield vs ROE",
                xaxis_title="Earnings Yield (%)",
                yaxis_title="ROE (%)",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Altman Z-Score approximation
    st.markdown("#### Financial Health Indicators")
    
    health_data = {}
    
    for company, data in comparison_data.items():
        current_ratio = data.get('Current Ratio')
        debt_equity = data.get('Debt to Equity')
        roe = data.get('ROE')
        
        # Simple health score based on available metrics
        health_score = 0
        criteria = []
        
        if pd.notna(current_ratio):
            if current_ratio >= 2.0:
                health_score += 2
                criteria.append("Strong Liquidity")
            elif current_ratio >= 1.0:
                health_score += 1
                criteria.append("Adequate Liquidity")
        
        if pd.notna(debt_equity):
            if debt_equity < 0.5:
                health_score += 2
                criteria.append("Low Leverage")
            elif debt_equity < 1.0:
                health_score += 1
                criteria.append("Moderate Leverage")
        
        if pd.notna(roe):
            roe_pct = roe * 100 if roe < 1 else roe
            if roe_pct >= 15:
                health_score += 2
                criteria.append("Strong Profitability")
            elif roe_pct >= 10:
                health_score += 1
                criteria.append("Good Profitability")
        
        health_data[company] = {
            'Health Score': health_score,
            'Max Score': 6,
            'Rating': 'Excellent' if health_score >= 5 else 'Good' if health_score >= 3 else 'Fair' if health_score >= 1 else 'Weak',
            'Strengths': ', '.join(criteria) if criteria else 'None'
        }
    
    if health_data:
        df_health = pd.DataFrame(health_data).T
        st.dataframe(df_health, use_container_width=True)
        
        # Bar chart of health scores
        fig = go.Figure(data=[go.Bar(
            x=list(health_data.keys()),
            y=[v['Health Score'] for v in health_data.values()],
            text=[v['Rating'] for v in health_data.values()],
            textposition='auto',
            marker_color=['green' if v['Health Score'] >= 5 else 'yellow' if v['Health Score'] >= 3 else 'orange' for v in health_data.values()]
        )])
        
        fig.update_layout(
            title="Financial Health Score Comparison",
            xaxis_title="Companies",
            yaxis_title="Health Score (out of 6)",
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
