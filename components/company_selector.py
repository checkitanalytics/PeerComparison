import streamlit as st

def render_company_selector():
    """
    Render the company selection interface in the sidebar
    
    Returns:
        list: Selected company ticker symbols
    """
    st.markdown("Enter ticker symbols for companies you want to compare:")
    
    # Text input for adding companies
    new_company = st.text_input(
        "Add Company (Ticker Symbol)",
        placeholder="e.g., AAPL, GOOGL, MSFT",
        help="Enter a stock ticker symbol and press Enter"
    ).upper().strip()
    
    # Add company button
    if st.button("Add Company") and new_company:
        if new_company not in st.session_state.selected_companies:
            if len(st.session_state.selected_companies) < 10:  # Limit to 10 companies
                st.session_state.selected_companies.append(new_company)
                st.success(f"Added {new_company}")
                st.rerun()
            else:
                st.warning("Maximum of 10 companies allowed for comparison")
        else:
            st.warning(f"{new_company} is already selected")
    
    # Quick add buttons for popular stocks
    st.markdown("**Quick Add Popular Stocks:**")
    
    # Tech stocks
    col1, col2 = st.columns(2)
    with col1:
        if st.button("AAPL", key="add_aapl"):
            if "AAPL" not in st.session_state.selected_companies:
                st.session_state.selected_companies.append("AAPL")
                st.rerun()
        
        if st.button("GOOGL", key="add_googl"):
            if "GOOGL" not in st.session_state.selected_companies:
                st.session_state.selected_companies.append("GOOGL")
                st.rerun()
        
        if st.button("MSFT", key="add_msft"):
            if "MSFT" not in st.session_state.selected_companies:
                st.session_state.selected_companies.append("MSFT")
                st.rerun()
        
        if st.button("AMZN", key="add_amzn"):
            if "AMZN" not in st.session_state.selected_companies:
                st.session_state.selected_companies.append("AMZN")
                st.rerun()
    
    with col2:
        if st.button("TSLA", key="add_tsla"):
            if "TSLA" not in st.session_state.selected_companies:
                st.session_state.selected_companies.append("TSLA")
                st.rerun()
        
        if st.button("META", key="add_meta"):
            if "META" not in st.session_state.selected_companies:
                st.session_state.selected_companies.append("META")
                st.rerun()
        
        if st.button("NVDA", key="add_nvda"):
            if "NVDA" not in st.session_state.selected_companies:
                st.session_state.selected_companies.append("NVDA")
                st.rerun()
        
        if st.button("NFLX", key="add_nflx"):
            if "NFLX" not in st.session_state.selected_companies:
                st.session_state.selected_companies.append("NFLX")
                st.rerun()
    
    # Banking stocks section
    if st.checkbox("Show Banking Stocks"):
        st.markdown("**Banking Stocks:**")
        bank_col1, bank_col2 = st.columns(2)
        
        with bank_col1:
            if st.button("JPM", key="add_jpm"):
                if "JPM" not in st.session_state.selected_companies:
                    st.session_state.selected_companies.append("JPM")
                    st.rerun()
            
            if st.button("BAC", key="add_bac"):
                if "BAC" not in st.session_state.selected_companies:
                    st.session_state.selected_companies.append("BAC")
                    st.rerun()
        
        with bank_col2:
            if st.button("WFC", key="add_wfc"):
                if "WFC" not in st.session_state.selected_companies:
                    st.session_state.selected_companies.append("WFC")
                    st.rerun()
            
            if st.button("GS", key="add_gs"):
                if "GS" not in st.session_state.selected_companies:
                    st.session_state.selected_companies.append("GS")
                    st.rerun()
    
    # Display selected companies
    if st.session_state.selected_companies:
        st.markdown("**Selected Companies:**")
        
        companies_to_remove = []
        for i, company in enumerate(st.session_state.selected_companies):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"{i+1}. {company}")
            with col2:
                if st.button("❌", key=f"remove_{company}"):
                    companies_to_remove.append(company)
        
        # Remove companies that were marked for removal
        for company in companies_to_remove:
            st.session_state.selected_companies.remove(company)
            st.rerun()
        
        # Clear all button
        if st.button("Clear All"):
            st.session_state.selected_companies = []
            st.rerun()
        
        # Show count
        count = len(st.session_state.selected_companies)
        if count >= 2:
            st.success(f"Ready to compare {count} companies!")
        else:
            st.info(f"Add {2-count} more company/companies to start comparison")
    
    else:
        st.info("No companies selected yet")
    
    # Additional options
    st.markdown("---")
    st.markdown("**Comparison Options:**")
    
    # Metric selection for primary comparison
    primary_metrics = st.multiselect(
        "Focus Metrics (optional)",
        options=[
            "Market Cap", "P/E Ratio", "Revenue (TTM)", "Profit Margin",
            "ROE", "Debt to Equity", "Dividend Yield"
        ],
        default=["Market Cap", "P/E Ratio", "Revenue (TTM)", "Profit Margin"],
        help="Select key metrics to highlight in comparisons"
    )
    
    # Store in session state
    st.session_state.focus_metrics = primary_metrics
    
    return st.session_state.selected_companies

def validate_ticker(ticker):
    """
    Basic validation for ticker symbols
    
    Args:
        ticker (str): Ticker symbol to validate
    
    Returns:
        bool: True if ticker appears valid
    """
    if not ticker:
        return False
    
    # Basic validation rules
    ticker = ticker.upper().strip()
    
    # Must be alphanumeric and between 1-5 characters for most exchanges
    if not ticker.isalnum():
        return False
    
    if len(ticker) < 1 or len(ticker) > 10:
        return False
    
    return True

def suggest_peer_companies(company_ticker, sector_info=None):
    """
    Suggest peer companies based on sector/industry
    This is a basic implementation - in a production app you might use
    a more sophisticated recommendation system
    
    Args:
        company_ticker (str): Primary company ticker
        sector_info (str): Sector information
    
    Returns:
        list: Suggested peer company tickers
    """
    # Basic peer suggestions based on common knowledge
    peer_suggestions = {
        # Tech giants
        'AAPL': ['MSFT', 'GOOGL', 'META', 'AMZN'],
        'MSFT': ['AAPL', 'GOOGL', 'META', 'ORCL'],
        'GOOGL': ['AAPL', 'MSFT', 'META', 'AMZN'],
        'META': ['GOOGL', 'SNAP', 'TWTR', 'PINS'],
        'AMZN': ['AAPL', 'GOOGL', 'WMT', 'COST'],
        
        # Banking
        'JPM': ['BAC', 'WFC', 'C', 'GS'],
        'BAC': ['JPM', 'WFC', 'C', 'USB'],
        'WFC': ['JPM', 'BAC', 'C', 'USB'],
        
        # Automotive
        'TSLA': ['F', 'GM', 'TM', 'RIVN'],
        'F': ['GM', 'TSLA', 'STLA', 'TM'],
        'GM': ['F', 'TSLA', 'STLA', 'TM'],
        
        # Retail
        'WMT': ['TGT', 'COST', 'AMZN', 'HD'],
        'TGT': ['WMT', 'COST', 'DG', 'KSS'],
        
        # Streaming/Entertainment
        'NFLX': ['DIS', 'PARA', 'WBD', 'ROKU'],
        'DIS': ['NFLX', 'PARA', 'WBD', 'CMCSA']
    }
    
    return peer_suggestions.get(company_ticker.upper(), [])
