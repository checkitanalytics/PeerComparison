import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from utils.financial_data import get_historical_data, extract_historical_metrics

def render_historical_trends(selected_companies):
    """
    Render historical trend analysis with time-series comparisons
    
    Args:
        selected_companies (list): List of company ticker symbols
    """
    st.subheader("📈 Historical Trend Analysis")
    
    if not selected_companies or len(selected_companies) < 1:
        st.info("Select at least one company to view historical trends")
        return
    
    # Period selector
    col1, col2 = st.columns([2, 1])
    
    with col1:
        period = st.selectbox(
            "Select Time Period",
            options=["1mo", "3mo", "6mo", "1y", "2y", "5y"],
            index=3,
            help="Choose the historical period to analyze"
        )
    
    with col2:
        chart_type = st.selectbox(
            "Chart Type",
            options=["Stock Price", "Financial Metrics"],
            index=0
        )
    
    # Fetch historical data for all companies
    with st.spinner("Fetching historical data..."):
        historical_data = {}
        failed_companies = []
        
        for company in selected_companies:
            data = get_historical_data(company, period)
            if data:
                historical_data[company] = data
            else:
                failed_companies.append(company)
        
        if failed_companies:
            st.warning(f"Could not fetch historical data for: {', '.join(failed_companies)}")
    
    if not historical_data:
        st.error("No historical data available for the selected companies")
        return
    
    # Display charts based on selection
    if chart_type == "Stock Price":
        render_price_trends(historical_data)
    else:
        render_financial_metric_trends(historical_data)

def render_price_trends(historical_data):
    """Render stock price trend charts"""
    st.subheader("Stock Price Trends")
    
    # Create price comparison chart
    fig = go.Figure()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    for i, (company, data) in enumerate(historical_data.items()):
        price_history = data.get('price_history')
        
        if price_history is not None and not price_history.empty:
            fig.add_trace(go.Scatter(
                x=price_history.index,
                y=price_history['Close'],
                mode='lines',
                name=company,
                line=dict(color=colors[i % len(colors)], width=2),
                hovertemplate=f'<b>{company}</b><br>Date: %{{x}}<br>Price: $%{{y:.2f}}<extra></extra>'
            ))
    
    fig.update_layout(
        title="Stock Price Comparison Over Time",
        xaxis_title="Date",
        yaxis_title="Stock Price ($)",
        hovermode='x unified',
        height=500,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Price performance metrics
    st.subheader("Price Performance Metrics")
    
    performance_data = {}
    
    for company, data in historical_data.items():
        price_history = data.get('price_history')
        
        if price_history is not None and not price_history.empty:
            first_price = price_history['Close'].iloc[0]
            last_price = price_history['Close'].iloc[-1]
            max_price = price_history['Close'].max()
            min_price = price_history['Close'].min()
            
            price_change = last_price - first_price
            price_change_pct = (price_change / first_price) * 100 if first_price > 0 else 0
            
            performance_data[company] = {
                'Start Price': f"${first_price:.2f}",
                'End Price': f"${last_price:.2f}",
                'Change ($)': f"${price_change:.2f}",
                'Change (%)': f"{price_change_pct:+.2f}%",
                'High': f"${max_price:.2f}",
                'Low': f"${min_price:.2f}",
                'Volatility': f"{price_history['Close'].std():.2f}"
            }
    
    if performance_data:
        df_performance = pd.DataFrame(performance_data).T
        st.dataframe(df_performance, use_container_width=True)
    
    # Volume comparison
    st.subheader("Trading Volume Trends")
    
    fig_volume = go.Figure()
    
    for i, (company, data) in enumerate(historical_data.items()):
        price_history = data.get('price_history')
        
        if price_history is not None and not price_history.empty and 'Volume' in price_history.columns:
            fig_volume.add_trace(go.Bar(
                x=price_history.index,
                y=price_history['Volume'],
                name=company,
                marker_color=colors[i % len(colors)],
                hovertemplate=f'<b>{company}</b><br>Date: %{{x}}<br>Volume: %{{y:,.0f}}<extra></extra>'
            ))
    
    fig_volume.update_layout(
        title="Trading Volume Over Time",
        xaxis_title="Date",
        yaxis_title="Volume",
        barmode='group',
        height=400
    )
    
    st.plotly_chart(fig_volume, use_container_width=True)

def render_financial_metric_trends(historical_data):
    """Render financial metric trend charts"""
    st.subheader("Financial Metrics Trends (Quarterly)")
    
    # Extract financial metrics for all companies
    metrics_data = {}
    
    for company, data in historical_data.items():
        metrics = extract_historical_metrics(data)
        if metrics is not None:
            metrics_data[company] = metrics
    
    if not metrics_data:
        st.warning("No quarterly financial data available for the selected companies")
        return
    
    # Get available metrics
    all_metrics = set()
    for metrics_df in metrics_data.values():
        all_metrics.update(metrics_df.columns.tolist())
    
    if not all_metrics:
        st.warning("No financial metrics available")
        return
    
    # Metric selector
    selected_metric = st.selectbox(
        "Select Metric to Compare",
        options=sorted(list(all_metrics)),
        index=0 if 'Revenue' not in all_metrics else sorted(list(all_metrics)).index('Revenue')
    )
    
    # Create trend chart for selected metric
    fig = go.Figure()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    for i, (company, metrics_df) in enumerate(metrics_data.items()):
        if selected_metric in metrics_df.columns:
            # Sort by date ascending for proper time series display
            metric_series = metrics_df[selected_metric].sort_index()
            
            # Format values based on metric type
            if selected_metric in ['Revenue', 'Net Income', 'Gross Profit']:
                y_values = metric_series / 1e9  # Convert to billions
                y_label = f"{selected_metric} (Billions $)"
                hover_template = f'<b>{company}</b><br>Date: %{{x}}<br>{selected_metric}: $%{{y:.2f}}B<extra></extra>'
            elif selected_metric == 'Profit Margin':
                y_values = metric_series
                y_label = f"{selected_metric} (%)"
                hover_template = f'<b>{company}</b><br>Date: %{{x}}<br>{selected_metric}: %{{y:.2f}}%<extra></extra>'
            else:
                y_values = metric_series
                y_label = selected_metric
                hover_template = f'<b>{company}</b><br>Date: %{{x}}<br>{selected_metric}: %{{y:.2f}}<extra></extra>'
            
            fig.add_trace(go.Scatter(
                x=metric_series.index,
                y=y_values,
                mode='lines+markers',
                name=company,
                line=dict(color=colors[i % len(colors)], width=2),
                marker=dict(size=8),
                hovertemplate=hover_template
            ))
    
    fig.update_layout(
        title=f"{selected_metric} Trends Comparison",
        xaxis_title="Quarter",
        yaxis_title=y_label,
        hovermode='x unified',
        height=500,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Growth rate analysis
    st.subheader("Quarter-over-Quarter Growth Rates")
    
    growth_data = {}
    
    for company, metrics_df in metrics_data.items():
        if selected_metric in metrics_df.columns:
            metric_series = metrics_df[selected_metric].sort_index()
            
            # Calculate QoQ growth rate
            growth_rate = metric_series.pct_change() * 100
            
            if not growth_rate.empty and growth_rate.notna().any():
                avg_growth = growth_rate.mean()
                latest_growth = growth_rate.iloc[-1] if not pd.isna(growth_rate.iloc[-1]) else 0
                
                growth_data[company] = {
                    'Latest QoQ Growth': f"{latest_growth:+.2f}%",
                    'Average QoQ Growth': f"{avg_growth:+.2f}%",
                    'Max Growth': f"{growth_rate.max():+.2f}%",
                    'Min Growth': f"{growth_rate.min():+.2f}%"
                }
    
    if growth_data:
        df_growth = pd.DataFrame(growth_data).T
        st.dataframe(df_growth, use_container_width=True)
    
    # Multiple metrics comparison grid
    st.subheader("All Metrics Overview")
    
    # Create small multiples for all metrics
    available_metrics = sorted(list(all_metrics))
    
    for metric in available_metrics:
        with st.expander(f"View {metric} Trend"):
            fig_metric = go.Figure()
            
            for i, (company, metrics_df) in enumerate(metrics_data.items()):
                if metric in metrics_df.columns:
                    metric_series = metrics_df[metric].sort_index()
                    
                    fig_metric.add_trace(go.Scatter(
                        x=metric_series.index,
                        y=metric_series,
                        mode='lines+markers',
                        name=company,
                        line=dict(color=colors[i % len(colors)])
                    ))
            
            fig_metric.update_layout(
                title=f"{metric} Over Time",
                xaxis_title="Quarter",
                yaxis_title=metric,
                height=300
            )
            
            st.plotly_chart(fig_metric, use_container_width=True)
