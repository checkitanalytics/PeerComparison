import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_company_data(ticker):
    """
    Fetch company data from Yahoo Finance
    
    Args:
        ticker (str): Stock ticker symbol
    
    Returns:
        yf.Ticker object or None if failed
    """
    try:
        stock = yf.Ticker(ticker)
        # Test if ticker is valid by trying to get info
        info = stock.info
        if not info or info.get('regularMarketPrice') is None:
            return None
        return stock
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None

def get_financial_metrics(stock_data):
    """
    Extract key financial metrics from stock data
    
    Args:
        stock_data (yf.Ticker): Yahoo Finance ticker object
    
    Returns:
        dict: Dictionary containing financial metrics
    """
    try:
        info = stock_data.info
        financials = stock_data.financials
        balance_sheet = stock_data.balance_sheet
        cash_flow = stock_data.cashflow
        
        metrics = {}
        
        # Basic company info
        metrics['Company Name'] = info.get('longName', 'N/A')
        metrics['Sector'] = info.get('sector', 'N/A')
        metrics['Industry'] = info.get('industry', 'N/A')
        
        # Market data
        metrics['Market Cap'] = info.get('marketCap', np.nan)
        metrics['Current Price'] = info.get('regularMarketPrice', np.nan)
        metrics['52 Week High'] = info.get('fiftyTwoWeekHigh', np.nan)
        metrics['52 Week Low'] = info.get('fiftyTwoWeekLow', np.nan)
        
        # Valuation ratios
        metrics['P/E Ratio'] = info.get('trailingPE', np.nan)
        metrics['Forward P/E'] = info.get('forwardPE', np.nan)
        metrics['P/B Ratio'] = info.get('priceToBook', np.nan)
        metrics['EV/EBITDA'] = info.get('enterpriseToEbitda', np.nan)
        
        # Profitability metrics
        metrics['Gross Margin'] = info.get('grossMargins', np.nan)
        metrics['Operating Margin'] = info.get('operatingMargins', np.nan)
        metrics['Profit Margin'] = info.get('profitMargins', np.nan)
        metrics['ROE'] = info.get('returnOnEquity', np.nan)
        metrics['ROA'] = info.get('returnOnAssets', np.nan)
        
        # Financial performance (TTM - Trailing Twelve Months)
        metrics['Revenue (TTM)'] = info.get('totalRevenue', np.nan)
        metrics['Gross Profit (TTM)'] = info.get('grossProfits', np.nan)
        metrics['Net Income (TTM)'] = info.get('netIncomeToCommon', np.nan)
        metrics['Operating Cash Flow (TTM)'] = info.get('operatingCashflow', np.nan)
        metrics['Free Cash Flow (TTM)'] = info.get('freeCashflow', np.nan)
        
        # Growth metrics
        metrics['Revenue Growth'] = info.get('revenueGrowth', np.nan)
        metrics['Earnings Growth'] = info.get('earningsGrowth', np.nan)
        
        # Dividend information
        metrics['Dividend Yield'] = info.get('dividendYield', np.nan)
        metrics['Dividend Rate'] = info.get('dividendRate', np.nan)
        metrics['Payout Ratio'] = info.get('payoutRatio', np.nan)
        
        # Financial strength
        metrics['Current Ratio'] = info.get('currentRatio', np.nan)
        metrics['Quick Ratio'] = info.get('quickRatio', np.nan)
        metrics['Debt to Equity'] = info.get('debtToEquity', np.nan)
        metrics['Total Debt'] = info.get('totalDebt', np.nan)
        metrics['Total Cash'] = info.get('totalCash', np.nan)
        
        # Additional metrics from financial statements
        if not financials.empty and len(financials.columns) > 0:
            latest_financials = financials.iloc[:, 0]  # Most recent year
            
            if 'Total Revenue' in financials.index:
                metrics['Revenue (Latest)'] = latest_financials.get('Total Revenue', np.nan)
            if 'Gross Profit' in financials.index:
                metrics['Gross Profit (Latest)'] = latest_financials.get('Gross Profit', np.nan)
            if 'Net Income' in financials.index:
                metrics['Net Income (Latest)'] = latest_financials.get('Net Income', np.nan)
        
        return metrics
        
    except Exception as e:
        st.error(f"Error processing financial metrics: {str(e)}")
        return {}

def format_currency(value, in_billions=False):
    """
    Format currency values for display
    
    Args:
        value: Numeric value
        in_billions (bool): Whether to display in billions
    
    Returns:
        str: Formatted currency string
    """
    if pd.isna(value) or value is None:
        return "N/A"
    
    try:
        if in_billions and abs(value) >= 1e9:
            return f"${value/1e9:.2f}B"
        elif abs(value) >= 1e9:
            return f"${value/1e9:.2f}B"
        elif abs(value) >= 1e6:
            return f"${value/1e6:.2f}M"
        elif abs(value) >= 1e3:
            return f"${value/1e3:.2f}K"
        else:
            return f"${value:.2f}"
    except:
        return "N/A"

def format_percentage(value):
    """
    Format percentage values for display
    
    Args:
        value: Numeric value (as decimal)
    
    Returns:
        str: Formatted percentage string
    """
    if pd.isna(value) or value is None:
        return "N/A"
    
    try:
        if isinstance(value, (int, float)):
            if value > 1:  # Assume it's already in percentage form
                return f"{value:.2f}%"
            else:  # Convert from decimal to percentage
                return f"{value * 100:.2f}%"
        return "N/A"
    except:
        return "N/A"

def format_ratio(value):
    """
    Format ratio values for display
    
    Args:
        value: Numeric value
    
    Returns:
        str: Formatted ratio string
    """
    if pd.isna(value) or value is None:
        return "N/A"
    
    try:
        return f"{float(value):.2f}"
    except:
        return "N/A"

@st.cache_data(ttl=600)  # Cache for 10 minutes
def get_historical_data(ticker, period="1y"):
    """
    Fetch historical price and financial data for a company
    
    Args:
        ticker (str): Stock ticker symbol
        period (str): Time period (1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, max)
    
    Returns:
        dict: Dictionary with historical data
    """
    try:
        stock = yf.Ticker(ticker)
        
        # Get historical price data
        hist = stock.history(period=period)
        
        if hist.empty:
            return None
        
        # Get quarterly financials
        quarterly_financials = stock.quarterly_financials
        quarterly_balance_sheet = stock.quarterly_balance_sheet
        quarterly_cashflow = stock.quarterly_cashflow
        
        return {
            'price_history': hist,
            'quarterly_financials': quarterly_financials,
            'quarterly_balance_sheet': quarterly_balance_sheet,
            'quarterly_cashflow': quarterly_cashflow
        }
    except Exception as e:
        st.error(f"Error fetching historical data for {ticker}: {str(e)}")
        return None

def extract_historical_metrics(historical_data):
    """
    Extract key metrics from historical financial statements
    
    Args:
        historical_data (dict): Dictionary containing historical data
    
    Returns:
        pd.DataFrame: DataFrame with historical metrics over time
    """
    try:
        metrics_over_time = {}
        
        financials = historical_data.get('quarterly_financials')
        
        if financials is not None and not financials.empty:
            # Extract revenue
            if 'Total Revenue' in financials.index:
                metrics_over_time['Revenue'] = financials.loc['Total Revenue']
            
            # Extract net income
            if 'Net Income' in financials.index:
                metrics_over_time['Net Income'] = financials.loc['Net Income']
            
            # Extract gross profit
            if 'Gross Profit' in financials.index:
                metrics_over_time['Gross Profit'] = financials.loc['Gross Profit']
            
            # Calculate profit margin
            if 'Net Income' in financials.index and 'Total Revenue' in financials.index:
                revenue = financials.loc['Total Revenue']
                net_income = financials.loc['Net Income']
                metrics_over_time['Profit Margin'] = (net_income / revenue * 100).replace([np.inf, -np.inf], np.nan)
        
        if metrics_over_time:
            df = pd.DataFrame(metrics_over_time)
            # Sort by date (most recent first)
            df = df.sort_index(ascending=False)
            return df
        
        return None
        
    except Exception as e:
        return None
