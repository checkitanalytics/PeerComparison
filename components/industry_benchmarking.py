import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

def render_industry_benchmarking(comparison_data):
    """
    Render industry benchmarking and sector averages
    
    Args:
        comparison_data (dict): Dictionary containing financial data for each company
    """
    st.subheader("🏭 Industry Benchmarking & Sector Analysis")
    
    if not comparison_data or len(comparison_data) < 1:
        st.info("Select at least one company to view industry benchmarking")
        return
    
    # Group companies by sector
    sectors = {}
    for company, data in comparison_data.items():
        sector = data.get('Sector', 'Unknown')
        if sector not in sectors:
            sectors[sector] = []
        sectors[sector].append(company)
    
    # Display sector distribution
    st.subheader("📊 Sector Distribution")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**Companies by Sector:**")
        for sector, companies in sectors.items():
            st.markdown(f"**{sector}:** {len(companies)} companies")
            for company in companies:
                st.markdown(f"  • {company}")
    
    with col2:
        # Sector pie chart
        sector_counts = {sector: len(companies) for sector, companies in sectors.items()}
        
        fig_sector = go.Figure(data=[go.Pie(
            labels=list(sector_counts.keys()),
            values=list(sector_counts.values()),
            hole=0.3,
            hovertemplate='<b>%{label}</b><br>Companies: %{value}<br>Percentage: %{percent}<extra></extra>'
        )])
        
        fig_sector.update_layout(
            title="Sector Distribution",
            height=400
        )
        
        st.plotly_chart(fig_sector, use_container_width=True)
    
    # Calculate sector averages
    st.subheader("📈 Sector Average Metrics")
    
    sector_averages = calculate_sector_averages(comparison_data, sectors)
    
    if sector_averages:
        display_sector_averages(sector_averages)
    
    # Industry comparison
    st.subheader("🔍 Industry Peer Comparison")
    
    # Allow user to select a sector for detailed analysis
    if len(sectors) > 1:
        selected_sector = st.selectbox(
            "Select Sector for Detailed Analysis",
            options=list(sectors.keys())
        )
        
        sector_companies = sectors[selected_sector]
        sector_data = {k: v for k, v in comparison_data.items() if k in sector_companies}
        
        if sector_data:
            display_sector_detailed_analysis(sector_data, selected_sector)
    else:
        # If all companies are in same sector, show detailed analysis directly
        sector_name = list(sectors.keys())[0]
        display_sector_detailed_analysis(comparison_data, sector_name)
    
    # Competitive positioning
    st.subheader("🎯 Competitive Positioning Matrix")
    
    display_competitive_positioning(comparison_data)

def calculate_sector_averages(comparison_data, sectors):
    """Calculate average metrics for each sector"""
    sector_averages = {}
    
    key_metrics = [
        'Market Cap', 'P/E Ratio', 'P/B Ratio', 'EV/EBITDA',
        'Gross Margin', 'Operating Margin', 'Profit Margin',
        'ROE', 'ROA', 'Revenue (TTM)', 'Net Income (TTM)',
        'Debt to Equity', 'Current Ratio'
    ]
    
    for sector, companies in sectors.items():
        sector_metrics = {}
        
        for metric in key_metrics:
            values = []
            for company in companies:
                if company in comparison_data:
                    value = comparison_data[company].get(metric)
                    if pd.notna(value):
                        values.append(value)
            
            if values:
                sector_metrics[metric] = {
                    'average': np.mean(values),
                    'median': np.median(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'std': np.std(values)
                }
        
        if sector_metrics:
            sector_averages[sector] = sector_metrics
    
    return sector_averages

def display_sector_averages(sector_averages):
    """Display sector average metrics in a table"""
    
    # Select metrics to display
    display_metrics = st.multiselect(
        "Select Metrics to Compare Across Sectors",
        options=['P/E Ratio', 'Profit Margin', 'ROE', 'Debt to Equity', 'Market Cap'],
        default=['P/E Ratio', 'Profit Margin', 'ROE']
    )
    
    if not display_metrics:
        st.info("Select at least one metric to compare")
        return
    
    # Create comparison table
    comparison_table = {}
    
    for sector, metrics in sector_averages.items():
        sector_row = {}
        for metric in display_metrics:
            if metric in metrics:
                avg_value = metrics[metric]['average']
                
                # Format based on metric type
                if metric in ['Gross Margin', 'Operating Margin', 'Profit Margin', 'ROE', 'ROA']:
                    if avg_value < 1:
                        avg_value *= 100
                    sector_row[f"{metric} (Avg)"] = f"{avg_value:.2f}%"
                elif metric in ['Market Cap', 'Revenue (TTM)', 'Net Income (TTM)']:
                    if avg_value >= 1e9:
                        sector_row[f"{metric} (Avg)"] = f"${avg_value/1e9:.2f}B"
                    elif avg_value >= 1e6:
                        sector_row[f"{metric} (Avg)"] = f"${avg_value/1e6:.2f}M"
                    else:
                        sector_row[f"{metric} (Avg)"] = f"${avg_value:.2f}"
                else:
                    sector_row[f"{metric} (Avg)"] = f"{avg_value:.2f}"
        
        comparison_table[sector] = sector_row
    
    if comparison_table:
        df_comparison = pd.DataFrame(comparison_table).T
        st.dataframe(df_comparison, use_container_width=True)
        
        # Create bar chart for sector comparison
        for metric in display_metrics:
            metric_key = f"{metric} (Avg)"
            
            if any(metric_key in row for row in comparison_table.values()):
                fig = go.Figure()
                
                sectors_list = []
                values_list = []
                
                for sector, metrics_dict in sector_averages.items():
                    if metric in metrics_dict:
                        sectors_list.append(sector)
                        value = metrics_dict[metric]['average']
                        
                        # Convert percentages if needed
                        if metric in ['Gross Margin', 'Operating Margin', 'Profit Margin', 'ROE', 'ROA'] and value < 1:
                            value *= 100
                        
                        values_list.append(value)
                
                if sectors_list:
                    fig.add_trace(go.Bar(
                        x=sectors_list,
                        y=values_list,
                        text=[f"{v:.2f}" for v in values_list],
                        textposition='auto',
                        marker_color=px.colors.qualitative.Set3[:len(sectors_list)]
                    ))
                    
                    y_label = f"{metric} (%)" if metric in ['Gross Margin', 'Operating Margin', 'Profit Margin', 'ROE', 'ROA'] else metric
                    
                    fig.update_layout(
                        title=f"Sector Average: {metric}",
                        xaxis_title="Sector",
                        yaxis_title=y_label,
                        showlegend=False,
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)

def display_sector_detailed_analysis(sector_data, sector_name):
    """Display detailed analysis for a specific sector"""
    st.markdown(f"### {sector_name} Sector Analysis")
    
    if len(sector_data) < 2:
        st.info("Need at least 2 companies in the sector for meaningful comparison")
        return
    
    # Calculate sector statistics
    st.markdown("#### Sector Statistics")
    
    key_metrics = ['P/E Ratio', 'Profit Margin', 'ROE', 'Revenue (TTM)', 'Market Cap']
    
    stats_data = {}
    
    for metric in key_metrics:
        values = []
        for company, data in sector_data.items():
            value = data.get(metric)
            if pd.notna(value):
                values.append(value)
        
        if values:
            stats_data[metric] = {
                'Average': np.mean(values),
                'Median': np.median(values),
                'Std Dev': np.std(values),
                'Min': np.min(values),
                'Max': np.max(values)
            }
    
    if stats_data:
        df_stats = pd.DataFrame(stats_data).T
        
        # Format the dataframe
        for col in df_stats.columns:
            df_stats[col] = df_stats[col].apply(lambda x: f"{x:.2f}")
        
        st.dataframe(df_stats, use_container_width=True)
    
    # Company rankings within sector
    st.markdown("#### Company Rankings within Sector")
    
    ranking_metric = st.selectbox(
        "Select Metric for Ranking",
        options=['Market Cap', 'Revenue (TTM)', 'Profit Margin', 'ROE', 'P/E Ratio'],
        index=0
    )
    
    # Create ranking
    rankings = []
    for company, data in sector_data.items():
        value = data.get(ranking_metric)
        if pd.notna(value):
            rankings.append((company, value))
    
    # Sort based on metric (higher is better for most metrics, lower for P/E)
    reverse_sort = ranking_metric != 'P/E Ratio'
    rankings.sort(key=lambda x: x[1], reverse=reverse_sort)
    
    # Display rankings
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**Rankings by {ranking_metric}:**")
        for i, (company, value) in enumerate(rankings, 1):
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
            
            if ranking_metric in ['Market Cap', 'Revenue (TTM)', 'Net Income (TTM)']:
                value_str = f"${value/1e9:.2f}B" if value >= 1e9 else f"${value/1e6:.2f}M"
            elif ranking_metric in ['Profit Margin', 'ROE', 'ROA']:
                value_str = f"{value*100:.2f}%" if value < 1 else f"{value:.2f}%"
            else:
                value_str = f"{value:.2f}"
            
            st.markdown(f"{medal} {company}: {value_str}")
    
    with col2:
        # Deviation from sector average
        if rankings:
            values = [v for _, v in rankings]
            avg_value = np.mean(values)
            
            st.markdown("**Deviation from Sector Average:**")
            for company, value in rankings:
                if avg_value != 0:
                    deviation = ((value - avg_value) / abs(avg_value)) * 100
                    color = "green" if deviation > 0 else "red"
                    st.markdown(f"{company}: <span style='color:{color}'>{deviation:+.1f}%</span>", 
                              unsafe_allow_html=True)
                else:
                    st.markdown(f"{company}: N/A (sector average is zero)")

def display_competitive_positioning(comparison_data):
    """Display competitive positioning matrix"""
    
    st.markdown("Compare companies on two key dimensions:")
    
    col1, col2 = st.columns(2)
    
    numeric_metrics = [
        'Market Cap', 'P/E Ratio', 'Profit Margin', 'ROE', 'Revenue (TTM)',
        'Debt to Equity', 'Revenue Growth', 'Gross Margin'
    ]
    
    with col1:
        x_metric = st.selectbox(
            "X-Axis (Horizontal)",
            options=numeric_metrics,
            index=0
        )
    
    with col2:
        y_metric = st.selectbox(
            "Y-Axis (Vertical)",
            options=numeric_metrics,
            index=2
        )
    
    if x_metric == y_metric:
        st.warning("Please select different metrics for X and Y axes")
        return
    
    # Create scatter plot
    companies = []
    x_values = []
    y_values = []
    market_caps = []
    sectors = []
    
    for company, data in comparison_data.items():
        x_val = data.get(x_metric)
        y_val = data.get(y_metric)
        mc = data.get('Market Cap', 0)
        sector = data.get('Sector', 'Unknown')
        
        if pd.notna(x_val) and pd.notna(y_val):
            companies.append(company)
            
            # Convert percentages if needed
            if x_metric in ['Profit Margin', 'ROE', 'ROA', 'Gross Margin'] and x_val < 1:
                x_val *= 100
            if y_metric in ['Profit Margin', 'ROE', 'ROA', 'Gross Margin'] and y_val < 1:
                y_val *= 100
            
            x_values.append(x_val)
            y_values.append(y_val)
            market_caps.append(mc)
            sectors.append(sector)
    
    if not companies:
        st.warning("No data available for the selected metrics")
        return
    
    # Create the scatter plot
    fig = go.Figure()
    
    # Color by sector
    unique_sectors = list(set(sectors))
    colors = px.colors.qualitative.Set3[:len(unique_sectors)]
    sector_color_map = {sector: colors[i] for i, sector in enumerate(unique_sectors)}
    
    for i, company in enumerate(companies):
        fig.add_trace(go.Scatter(
            x=[x_values[i]],
            y=[y_values[i]],
            mode='markers+text',
            name=sectors[i],
            text=[company],
            textposition='top center',
            marker=dict(
                size=max(10, min(30, np.log10(market_caps[i]) * 3)) if market_caps[i] > 0 else 10,
                color=sector_color_map[sectors[i]],
                line=dict(width=2, color='white')
            ),
            hovertemplate=f'<b>{company}</b><br>{x_metric}: %{{x:.2f}}<br>{y_metric}: %{{y:.2f}}<br>Sector: {sectors[i]}<extra></extra>',
            showlegend=sectors[i] not in [sectors[j] for j in range(i)]  # Show legend only once per sector
        ))
    
    # Add average lines
    avg_x = np.mean(x_values)
    avg_y = np.mean(y_values)
    
    fig.add_hline(y=avg_y, line_dash="dash", line_color="gray", 
                  annotation_text="Average", annotation_position="right")
    fig.add_vline(x=avg_x, line_dash="dash", line_color="gray",
                  annotation_text="Average", annotation_position="top")
    
    # Add quadrant labels
    x_range = max(x_values) - min(x_values)
    y_range = max(y_values) - min(y_values)
    
    x_label = f"{x_metric} (%)" if x_metric in ['Profit Margin', 'ROE', 'ROA', 'Gross Margin'] else x_metric
    y_label = f"{y_metric} (%)" if y_metric in ['Profit Margin', 'ROE', 'ROA', 'Gross Margin'] else y_metric
    
    fig.update_layout(
        title=f"Competitive Positioning: {x_metric} vs {y_metric}",
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=600,
        hovermode='closest'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Quadrant analysis
    st.markdown("#### Quadrant Analysis")
    
    quadrants = {
        'High-High': [],
        'High-Low': [],
        'Low-High': [],
        'Low-Low': []
    }
    
    for i, company in enumerate(companies):
        x_high = x_values[i] >= avg_x
        y_high = y_values[i] >= avg_y
        
        if x_high and y_high:
            quadrants['High-High'].append(company)
        elif x_high and not y_high:
            quadrants['High-Low'].append(company)
        elif not x_high and y_high:
            quadrants['Low-High'].append(company)
        else:
            quadrants['Low-Low'].append(company)
    
    cols = st.columns(2)
    
    with cols[0]:
        st.markdown(f"**High {x_metric}, High {y_metric}:**")
        if quadrants['High-High']:
            for company in quadrants['High-High']:
                st.markdown(f"✅ {company}")
        else:
            st.markdown("_None_")
        
        st.markdown(f"**Low {x_metric}, High {y_metric}:**")
        if quadrants['Low-High']:
            for company in quadrants['Low-High']:
                st.markdown(f"⚠️ {company}")
        else:
            st.markdown("_None_")
    
    with cols[1]:
        st.markdown(f"**High {x_metric}, Low {y_metric}:**")
        if quadrants['High-Low']:
            for company in quadrants['High-Low']:
                st.markdown(f"⚠️ {company}")
        else:
            st.markdown("_None_")
        
        st.markdown(f"**Low {x_metric}, Low {y_metric}:**")
        if quadrants['Low-Low']:
            for company in quadrants['Low-Low']:
                st.markdown(f"❌ {company}")
        else:
            st.markdown("_None_")
