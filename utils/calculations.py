import pandas as pd
import numpy as np

def calculate_percentage_differences(comparison_data):
    """
    Calculate percentage differences from average for each metric
    
    Args:
        comparison_data (dict): Dictionary with company data
    
    Returns:
        dict: Percentage differences for each company and metric
    """
    if len(comparison_data) < 2:
        return None
    
    try:
        # Create DataFrame from comparison data
        df = pd.DataFrame(comparison_data).T
        
        # Select only numeric columns for comparison
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Calculate averages for numeric columns
        averages = df[numeric_cols].mean(axis=0)
        
        # Calculate percentage differences
        percentage_diffs = {}
        
        for company in comparison_data.keys():
            company_diffs = {}
            
            for metric in numeric_cols:
                company_value = df.loc[company, metric]
                avg_value = averages[metric]
                
                if pd.notna(company_value) and pd.notna(avg_value) and avg_value != 0:
                    diff_pct = ((company_value - avg_value) / abs(avg_value)) * 100
                    company_diffs[metric] = diff_pct
                else:
                    company_diffs[metric] = np.nan
            
            percentage_diffs[company] = company_diffs
        
        return percentage_diffs
    
    except Exception as e:
        return None

def calculate_financial_ratios(metrics):
    """
    Calculate additional financial ratios from basic metrics
    
    Args:
        metrics (dict): Dictionary of financial metrics
    
    Returns:
        dict: Additional calculated ratios
    """
    ratios = {}
    
    try:
        # Price to Sales ratio
        if 'Market Cap' in metrics and 'Revenue (TTM)' in metrics:
            market_cap = metrics['Market Cap']
            revenue = metrics['Revenue (TTM)']
            if pd.notna(market_cap) and pd.notna(revenue) and revenue != 0:
                ratios['P/S Ratio'] = market_cap / revenue
        
        # Enterprise Value
        if all(k in metrics for k in ['Market Cap', 'Total Debt', 'Total Cash']):
            market_cap = metrics['Market Cap']
            total_debt = metrics['Total Debt']
            total_cash = metrics['Total Cash']
            
            if all(pd.notna(x) for x in [market_cap, total_debt, total_cash]):
                ratios['Enterprise Value'] = market_cap + total_debt - total_cash
        
        # Asset Turnover
        if 'Revenue (TTM)' in metrics and 'Total Assets' in metrics:
            revenue = metrics['Revenue (TTM)']
            assets = metrics.get('Total Assets')
            if pd.notna(revenue) and pd.notna(assets) and assets != 0:
                ratios['Asset Turnover'] = revenue / assets
        
        # Financial Leverage
        if 'Total Debt' in metrics and 'Market Cap' in metrics:
            debt = metrics['Total Debt']
            market_cap = metrics['Market Cap']
            if pd.notna(debt) and pd.notna(market_cap) and market_cap != 0:
                ratios['Financial Leverage'] = debt / market_cap
    
    except Exception as e:
        pass
    
    return ratios

def rank_companies(comparison_data, metric, ascending=False):
    """
    Rank companies based on a specific metric
    
    Args:
        comparison_data (dict): Dictionary with company data
        metric (str): Metric to rank by
        ascending (bool): Whether to rank in ascending order
    
    Returns:
        list: Companies ranked by the metric
    """
    try:
        company_values = {}
        
        for company, data in comparison_data.items():
            if metric in data and pd.notna(data[metric]):
                company_values[company] = data[metric]
        
        if not company_values:
            return list(comparison_data.keys())
        
        # Sort companies by metric value
        sorted_companies = sorted(company_values.items(), 
                                key=lambda x: x[1], 
                                reverse=not ascending)
        
        return [company for company, value in sorted_companies]
    
    except Exception as e:
        return list(comparison_data.keys())

def calculate_growth_metrics(historical_data):
    """
    Calculate growth metrics from historical data
    
    Args:
        historical_data (pd.DataFrame): Historical financial data
    
    Returns:
        dict: Growth metrics
    """
    growth_metrics = {}
    
    try:
        if historical_data.empty or len(historical_data.columns) < 2:
            return growth_metrics
        
        # Calculate year-over-year growth for various metrics
        metrics_to_calculate = ['Total Revenue', 'Net Income', 'Total Assets', 'Gross Profit']
        
        for metric in metrics_to_calculate:
            if metric in historical_data.index:
                metric_data = historical_data.loc[metric]
                # Remove NaN values and sort by date
                metric_data = metric_data.dropna().sort_index()
                
                if len(metric_data) >= 2:
                    # Calculate compound annual growth rate (CAGR)
                    start_value = metric_data.iloc[0]
                    end_value = metric_data.iloc[-1]
                    num_periods = len(metric_data) - 1
                    
                    if start_value > 0 and end_value > 0 and num_periods > 0:
                        cagr = ((end_value / start_value) ** (1/num_periods)) - 1
                        growth_metrics[f'{metric} CAGR'] = cagr * 100
                    
                    # Calculate latest year growth
                    if len(metric_data) >= 2:
                        latest_growth = ((metric_data.iloc[-1] / metric_data.iloc[-2]) - 1) * 100
                        growth_metrics[f'{metric} YoY Growth'] = latest_growth
    
    except Exception as e:
        pass
    
    return growth_metrics

def identify_outliers(comparison_data, metric, threshold=2):
    """
    Identify outlier companies for a specific metric using standard deviation
    
    Args:
        comparison_data (dict): Dictionary with company data
        metric (str): Metric to analyze
        threshold (float): Standard deviation threshold for outliers
    
    Returns:
        dict: Companies categorized as outliers or normal
    """
    try:
        values = []
        companies = []
        
        for company, data in comparison_data.items():
            if metric in data and pd.notna(data[metric]):
                values.append(data[metric])
                companies.append(company)
        
        if len(values) < 3:  # Need at least 3 companies for meaningful outlier detection
            return {'outliers': [], 'normal': companies}
        
        values = np.array(values)
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        outliers = []
        normal = []
        
        for i, company in enumerate(companies):
            z_score = abs(values[i] - mean_val) / std_val
            if z_score > threshold:
                outliers.append(company)
            else:
                normal.append(company)
        
        return {'outliers': outliers, 'normal': normal}
    
    except Exception as e:
        return {'outliers': [], 'normal': list(comparison_data.keys())}
