import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots

def render_comparison_charts(comparison_data):
    """
    Render interactive comparison charts using Plotly
    
    Args:
        comparison_data (dict): Dictionary containing financial data for each company
    """
    if not comparison_data or len(comparison_data) < 2:
        st.error("Need at least 2 companies for comparison charts")
        return
    
    # Create chart selection tabs
    chart_tab1, chart_tab2, chart_tab3, chart_tab4 = st.tabs([
        "📊 Key Metrics", "💰 Valuation", "📈 Performance", "🔍 Custom Analysis"
    ])
    
    with chart_tab1:
        render_key_metrics_charts(comparison_data)
    
    with chart_tab2:
        render_valuation_charts(comparison_data)
    
    with chart_tab3:
        render_performance_charts(comparison_data)
    
    with chart_tab4:
        render_custom_analysis_charts(comparison_data)

def render_key_metrics_charts(comparison_data):
    """Render key financial metrics charts"""
    st.subheader("Key Financial Metrics Comparison")
    
    # Market Cap comparison
    fig_market_cap = create_bar_chart(
        comparison_data, 
        'Market Cap', 
        "Market Capitalization Comparison",
        format_as_currency=True
    )
    if fig_market_cap:
        st.plotly_chart(fig_market_cap, use_container_width=True)
    
    # P/E Ratio comparison
    fig_pe = create_bar_chart(
        comparison_data, 
        'P/E Ratio', 
        "P/E Ratio Comparison"
    )
    if fig_pe:
        st.plotly_chart(fig_pe, use_container_width=True)
    
    # Multi-metric radar chart
    fig_radar = create_radar_chart(comparison_data)
    if fig_radar:
        st.subheader("Multi-Metric Comparison (Normalized)")
        st.plotly_chart(fig_radar, use_container_width=True)

def render_valuation_charts(comparison_data):
    """Render valuation-focused charts"""
    st.subheader("Valuation Analysis")
    
    # Valuation metrics comparison
    valuation_metrics = ['P/E Ratio', 'P/B Ratio', 'EV/EBITDA']
    fig_valuation = create_grouped_bar_chart(
        comparison_data, 
        valuation_metrics, 
        "Valuation Ratios Comparison"
    )
    if fig_valuation:
        st.plotly_chart(fig_valuation, use_container_width=True)
    
    # Price vs fundamentals scatter plot
    fig_scatter = create_price_fundamentals_scatter(comparison_data)
    if fig_scatter:
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Dividend analysis
    fig_dividend = create_dividend_chart(comparison_data)
    if fig_dividend:
        st.plotly_chart(fig_dividend, use_container_width=True)

def render_performance_charts(comparison_data):
    """Render performance-focused charts"""
    st.subheader("Financial Performance Analysis")
    
    # Revenue and profitability
    fig_revenue = create_revenue_profitability_chart(comparison_data)
    if fig_revenue:
        st.plotly_chart(fig_revenue, use_container_width=True)
    
    # Profitability margins
    margin_metrics = ['Gross Margin', 'Operating Margin', 'Profit Margin']
    fig_margins = create_grouped_bar_chart(
        comparison_data, 
        margin_metrics, 
        "Profitability Margins Comparison (%)",
        format_as_percentage=True
    )
    if fig_margins:
        st.plotly_chart(fig_margins, use_container_width=True)
    
    # Cash flow analysis
    fig_cashflow = create_cashflow_chart(comparison_data)
    if fig_cashflow:
        st.plotly_chart(fig_cashflow, use_container_width=True)

def render_custom_analysis_charts(comparison_data):
    """Render custom analysis charts with user selection"""
    st.subheader("Custom Analysis")
    
    # Get available numeric metrics
    numeric_metrics = get_numeric_metrics(comparison_data)
    
    if not numeric_metrics:
        st.warning("No numeric metrics available for custom analysis")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        x_metric = st.selectbox(
            "X-Axis Metric",
            options=numeric_metrics,
            index=0 if 'Market Cap' in numeric_metrics else 0
        )
    
    with col2:
        y_metric = st.selectbox(
            "Y-Axis Metric",
            options=numeric_metrics,
            index=1 if len(numeric_metrics) > 1 else 0
        )
    
    if x_metric and y_metric and x_metric != y_metric:
        fig_custom = create_custom_scatter_plot(comparison_data, x_metric, y_metric)
        if fig_custom:
            st.plotly_chart(fig_custom, use_container_width=True)
    
    # Custom bar chart
    st.subheader("Custom Bar Chart")
    selected_metrics = st.multiselect(
        "Select metrics to compare",
        options=numeric_metrics,
        default=numeric_metrics[:3] if len(numeric_metrics) >= 3 else numeric_metrics
    )
    
    if selected_metrics:
        fig_custom_bar = create_multi_metric_bar_chart(comparison_data, selected_metrics)
        if fig_custom_bar:
            st.plotly_chart(fig_custom_bar, use_container_width=True)

def create_bar_chart(data, metric, title, format_as_currency=False):
    """Create a bar chart for a single metric"""
    try:
        companies = []
        values = []
        
        for company, company_data in data.items():
            value = company_data.get(metric)
            if pd.notna(value) and value != 0:
                companies.append(company)
                values.append(value)
        
        if not companies:
            return None
        
        fig = go.Figure(data=[
            go.Bar(
                x=companies,
                y=values,
                text=[f"${v/1e9:.2f}B" if format_as_currency and v >= 1e9 
                      else f"${v/1e6:.2f}M" if format_as_currency and v >= 1e6 
                      else f"{v:.2f}" for v in values],
                textposition='auto',
                marker_color=px.colors.qualitative.Set3[:len(companies)]
            )
        ])
        
        fig.update_layout(
            title=title,
            xaxis_title="Companies",
            yaxis_title=metric,
            showlegend=False,
            height=500
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating bar chart for {metric}: {str(e)}")
        return None

def create_grouped_bar_chart(data, metrics, title, format_as_percentage=False):
    """Create a grouped bar chart for multiple metrics"""
    try:
        companies = list(data.keys())
        
        fig = go.Figure()
        
        for metric in metrics:
            values = []
            for company in companies:
                value = data[company].get(metric)
                if pd.notna(value):
                    # Convert to percentage if needed
                    if format_as_percentage and value < 1:
                        value = value * 100
                    values.append(value)
                else:
                    values.append(0)
            
            fig.add_trace(go.Bar(
                name=metric,
                x=companies,
                y=values,
                text=[f"{v:.2f}%" if format_as_percentage else f"{v:.2f}" for v in values],
                textposition='auto'
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Companies",
            yaxis_title="Values",
            barmode='group',
            height=500
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating grouped bar chart: {str(e)}")
        return None

def create_radar_chart(data):
    """Create a radar chart for multi-metric comparison"""
    try:
        # Select key metrics for radar chart
        radar_metrics = ['P/E Ratio', 'ROE', 'Profit Margin', 'Current Ratio', 'Revenue Growth']
        
        # Normalize data for radar chart (0-100 scale)
        normalized_data = {}
        
        for metric in radar_metrics:
            values = []
            for company_data in data.values():
                value = company_data.get(metric)
                if pd.notna(value):
                    values.append(value)
            
            if values:
                min_val, max_val = min(values), max(values)
                if max_val > min_val:
                    for company, company_data in data.items():
                        if company not in normalized_data:
                            normalized_data[company] = {}
                        
                        value = company_data.get(metric)
                        if pd.notna(value):
                            # Normalize to 0-100 scale
                            normalized_data[company][metric] = ((value - min_val) / (max_val - min_val)) * 100
                        else:
                            normalized_data[company][metric] = 0
        
        if not normalized_data:
            return None
        
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set3
        
        for i, (company, company_metrics) in enumerate(normalized_data.items()):
            values = [company_metrics.get(metric, 0) for metric in radar_metrics]
            values.append(values[0])  # Close the radar chart
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=radar_metrics + [radar_metrics[0]],
                fill='toself',
                name=company,
                line_color=colors[i % len(colors)]
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            title="Multi-Metric Radar Comparison (Normalized 0-100)",
            height=600
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating radar chart: {str(e)}")
        return None

def create_price_fundamentals_scatter(data):
    """Create scatter plot comparing price metrics vs fundamentals"""
    try:
        companies = []
        pe_ratios = []
        profit_margins = []
        market_caps = []
        
        for company, company_data in data.items():
            pe = company_data.get('P/E Ratio')
            margin = company_data.get('Profit Margin')
            mc = company_data.get('Market Cap')
            
            if all(pd.notna(x) for x in [pe, margin, mc]):
                companies.append(company)
                pe_ratios.append(pe)
                # Convert margin to percentage if needed
                profit_margins.append(margin * 100 if margin < 1 else margin)
                market_caps.append(mc)
        
        if not companies:
            return None
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=pe_ratios,
            y=profit_margins,
            mode='markers+text',
            text=companies,
            textposition='top center',
            marker=dict(
                size=[np.log10(mc) * 3 for mc in market_caps],  # Size based on market cap
                color=market_caps,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Market Cap")
            ),
            hovertemplate='<b>%{text}</b><br>P/E: %{x:.2f}<br>Profit Margin: %{y:.2f}%<extra></extra>'
        ))
        
        fig.update_layout(
            title="P/E Ratio vs Profit Margin (Bubble size = Market Cap)",
            xaxis_title="P/E Ratio",
            yaxis_title="Profit Margin (%)",
            height=500
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating scatter plot: {str(e)}")
        return None

def create_dividend_chart(data):
    """Create dividend analysis chart"""
    try:
        companies = []
        dividend_yields = []
        dividend_rates = []
        
        for company, company_data in data.items():
            dy = company_data.get('Dividend Yield')
            dr = company_data.get('Dividend Rate')
            
            if pd.notna(dy) or pd.notna(dr):
                companies.append(company)
                dividend_yields.append(dy * 100 if pd.notna(dy) and dy < 1 else dy if pd.notna(dy) else 0)
                dividend_rates.append(dr if pd.notna(dr) else 0)
        
        if not companies:
            return None
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Dividend Yield (%)', 'Dividend Rate ($)'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        fig.add_trace(
            go.Bar(x=companies, y=dividend_yields, name="Dividend Yield", marker_color='lightblue'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=companies, y=dividend_rates, name="Dividend Rate", marker_color='lightgreen'),
            row=1, col=2
        )
        
        fig.update_layout(
            title="Dividend Analysis",
            height=400,
            showlegend=False
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating dividend chart: {str(e)}")
        return None

def create_revenue_profitability_chart(data):
    """Create revenue and profitability comparison chart"""
    try:
        companies = []
        revenues = []
        net_incomes = []
        
        for company, company_data in data.items():
            revenue = company_data.get('Revenue (TTM)')
            net_income = company_data.get('Net Income (TTM)')
            
            if pd.notna(revenue) and pd.notna(net_income):
                companies.append(company)
                revenues.append(revenue / 1e9)  # Convert to billions
                net_incomes.append(net_income / 1e9)  # Convert to billions
        
        if not companies:
            return None
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Revenue (Billions $)', 'Net Income (Billions $)')
        )
        
        fig.add_trace(
            go.Bar(x=companies, y=revenues, name="Revenue", marker_color='skyblue'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=companies, y=net_incomes, name="Net Income", marker_color='lightcoral'),
            row=1, col=2
        )
        
        fig.update_layout(
            title="Revenue vs Profitability",
            height=400,
            showlegend=False
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating revenue profitability chart: {str(e)}")
        return None

def create_cashflow_chart(data):
    """Create cash flow analysis chart"""
    try:
        companies = []
        operating_cf = []
        free_cf = []
        
        for company, company_data in data.items():
            ocf = company_data.get('Operating Cash Flow (TTM)')
            fcf = company_data.get('Free Cash Flow (TTM)')
            
            if pd.notna(ocf) or pd.notna(fcf):
                companies.append(company)
                operating_cf.append(ocf / 1e9 if pd.notna(ocf) else 0)  # Convert to billions
                free_cf.append(fcf / 1e9 if pd.notna(fcf) else 0)  # Convert to billions
        
        if not companies:
            return None
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Operating Cash Flow',
            x=companies,
            y=operating_cf,
            marker_color='lightblue'
        ))
        
        fig.add_trace(go.Bar(
            name='Free Cash Flow',
            x=companies,
            y=free_cf,
            marker_color='lightgreen'
        ))
        
        fig.update_layout(
            title='Cash Flow Comparison (Billions $)',
            xaxis_title='Companies',
            yaxis_title='Cash Flow (Billions $)',
            barmode='group',
            height=500
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating cash flow chart: {str(e)}")
        return None

def create_custom_scatter_plot(data, x_metric, y_metric):
    """Create custom scatter plot with user-selected metrics"""
    try:
        companies = []
        x_values = []
        y_values = []
        
        for company, company_data in data.items():
            x_val = company_data.get(x_metric)
            y_val = company_data.get(y_metric)
            
            if pd.notna(x_val) and pd.notna(y_val):
                companies.append(company)
                x_values.append(x_val)
                y_values.append(y_val)
        
        if not companies:
            return None
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=x_values,
            y=y_values,
            mode='markers+text',
            text=companies,
            textposition='top center',
            marker=dict(size=12, color=px.colors.qualitative.Set3[:len(companies)]),
            hovertemplate=f'<b>%{{text}}</b><br>{x_metric}: %{{x}}<br>{y_metric}: %{{y}}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'{x_metric} vs {y_metric}',
            xaxis_title=x_metric,
            yaxis_title=y_metric,
            height=500
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating custom scatter plot: {str(e)}")
        return None

def create_multi_metric_bar_chart(data, metrics):
    """Create bar chart with multiple metrics"""
    try:
        companies = list(data.keys())
        
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set3
        
        for i, metric in enumerate(metrics):
            values = []
            for company in companies:
                value = data[company].get(metric)
                if pd.notna(value):
                    values.append(value)
                else:
                    values.append(0)
            
            fig.add_trace(go.Bar(
                name=metric,
                x=companies,
                y=values,
                marker_color=colors[i % len(colors)]
            ))
        
        fig.update_layout(
            title='Multi-Metric Comparison',
            xaxis_title='Companies',
            yaxis_title='Values',
            barmode='group',
            height=500
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating multi-metric bar chart: {str(e)}")
        return None

def get_numeric_metrics(data):
    """Get list of numeric metrics available in the data"""
    numeric_metrics = set()
    
    for company_data in data.values():
        for metric, value in company_data.items():
            if isinstance(value, (int, float)) and pd.notna(value):
                numeric_metrics.add(metric)
    
    # Filter out non-meaningful metrics for comparison
    excluded_metrics = {'Company Name', 'Sector', 'Industry'}
    numeric_metrics = numeric_metrics - excluded_metrics
    
    return sorted(list(numeric_metrics))
