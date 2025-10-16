# Peer Company Comparison Platform

## Overview

This is a Flask-based peer company comparison platform that enables users to analyze and compare key financial metrics across multiple companies. The application uses OpenAI's GPT-4 to automatically identify peer companies and fetches real-time financial data from Yahoo Finance. It provides interactive visualizations and detailed time series analysis.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture

**Framework Choice: Flask with Embedded HTML**
- **Problem**: Need for a simple, integrated web application without complex frontend build processes
- **Solution**: Flask serves both API endpoints and HTML/JavaScript frontend in a single file
- **Rationale**: Rapid development, no build step required, easy deployment on Replit
- **Trade-offs**: Less separation of concerns than SPA frameworks, but simpler for single-page applications

**UI Components**
- Tailwind CSS for responsive styling via CDN
- Chart.js for interactive financial charts (bar charts, line charts)
- Vanilla JavaScript for dynamic content rendering
- Single-page application with dynamic content updates

**State Management**
- Client-side JavaScript manages application state
- Asynchronous API calls to Flask backend
- Dynamic DOM manipulation for results display

### Backend Architecture

**Web Framework: Flask**
- Lightweight Python web framework
- RESTful API endpoints for peer discovery and metrics fetching
- CORS enabled for development flexibility

**Specialty Peer Groups**
- **MEGA7 (Magnificent 7 Tech Giants)**: AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA
- **eVTOL / Urban Air Mobility**: EH, JOBY, ACHR, BLDE
- **Electric Vehicle Manufacturers**: RIVN, LCID, NIO, XPEV, LI, ZK, PSNY, BYDDY, VFS, WKHS
- **Ticker Aliases**: GOOG→GOOGL, FB→META, SRTA→BLDE, BLADE→BLDE
- When a user searches for a company in a specialty group, peers are automatically selected from that group

**AI-Powered Analysis (Multi-Provider with Fallbacks)**
- **Primary Provider**: DeepSeek API (deepseek-v3.2-exp model)
- **Fallback Provider**: Perplexity AI (llama-3.1-sonar-small-128k-online model)
- **Final Fallback**: Local deterministic text builder
- **Purpose**: Generate analyst-style financial conclusions and translations
- **Input**: Company financial metrics and peer comparison data
- **Output**: Concise, factual financial analysis with ranking insights

**Data Fetching Layer**
- **Library**: yfinance (Yahoo Finance API wrapper)
- **Retry Logic**: Up to 3 attempts with exponential backoff (4s, 8s delays)
- **Rate Limiting Protection**: 
  - Custom User-Agent headers to avoid blocking
  - Delays between requests (1s before fetch, 0.5s between different API calls)
  - Detects HTTP 429 errors and automatically retries
- **Data Retrieved**: Quarterly income statements and cash flow statements

**Financial Metrics Calculation**
- **Total Revenue**: Primary revenue fields with fallbacks
- **Gross Margin %**: Calculated from Gross Profit and Total Revenue (when available)
- **Operating Expense**: Direct fields or calculated from components (SG&A + R&D, or SG&A + Selling & Marketing, or SG&A alone)
- **EBIT**: Direct fields or calculated (Gross Profit - OpEx, or Revenue - COGS - OpEx, or Pretax Income as proxy)
- **Net Income**: Primary income statement fields
- **Free Cash Flow**: Operating Cash Flow + Capital Expenditure
- Handles different company types:
  - Standard tech/manufacturing companies: All 6 metrics available
  - Fintech/financial services: 5 metrics (Gross Margin % N/A due to no COGS)
- Robust fallback calculations for missing direct fields

### API Endpoints

**POST /api/find-peers**
- Accepts: `{ "ticker": "AAPL" }`
- Returns: JSON with primary company, industry/group, and 2 peer companies
- **Peer Selection Logic**:
  1. Check if ticker is in specialty groups (MEGA7, eVTOL, EV) → return group peers
  2. Otherwise, use industry-based peer matching (same industry → same sector → market cap)
- Supports ticker aliases (GOOG→GOOGL, FB→META, BLADE→BLDE)

**POST /api/get-metrics**
- Accepts: `{ "tickers": ["AAPL", "MSFT", "GOOGL", "AMZN"] }`
- Returns: Financial metrics for each ticker (5 quarters of data)
- Filters out companies with missing/incomplete data

**POST /api/peer-key-metrics-conclusion**
- Accepts: `{ "primary": "TSLA", "latest_quarter": {...}, "time_series": {...} }`
- Returns: AI-generated financial analysis with multi-provider fallback
- Analysis includes:
  - **Enhanced peer snapshot with explanation**: Rankings shown as "#rank/total, where #1 is best except OpEx"
  - **Latest quarter metrics with actual values**: Revenue, GM%, EBIT, Net Income, FCF
  - 5-quarter time series trend analysis (revenue, margins, profitability)
  - Latest quarter peer comparison with rankings
  - Notable strengths and concerns
- **AI Provider Cascade**:
  1. DeepSeek API (primary)
  2. Perplexity AI (fallback)
  3. Local deterministic builder (final fallback)
- Response indicates which provider was used: `"llm": "deepseek|perplexity|local-fallback"`

**GET /api/health**
- Health check endpoint
- Verifies AI provider configurations
- Returns: 
  ```json
  {
    "status": "healthy",
    "deepseek_configured": true/false,
    "deepseek_model": "deepseek-v3.2-exp",
    "perplexity_configured": true/false,
    "perplexity_model": "llama-3.1-sonar-small-128k-online"
  }
  ```

### Visualization Features

**1. Total Revenue Comparison**
- Bar chart comparing revenue across companies
- Data shown in billions of dollars
- Up to 5 quarters displayed
- Color-coded by company

**2. Gross Margin % Trend**
- Line chart showing margin trends over time
- Percentage-based comparison
- Helps identify operational efficiency patterns

**3. Latest Quarter Metrics Table**
- Cross-company comparison for most recent quarter
- Shows actual quarter/year (e.g., "2025Q2 Metrics")
- Metrics: Total Revenue, Gross Margin %, Operating Expense, EBIT, Net Income, Free Cash Flow
- Automatically filters out companies with missing data (N/A values)

**4. Time Series Tables (NEW)**
- Individual table for each company showing 5 quarters of historical data
- Quarter columns: Most recent to oldest (e.g., 2025Q2, 2025Q1, 2024Q4, 2024Q3, 2024Q2)
- Metric rows: Total Revenue, Operating Expense, Gross Margin %, EBIT, Net Income, Free Cash Flow
- Raw numerical values for detailed analysis
- Gross Margin % shown as decimal values
- Other metrics shown with full precision

### Data Handling

**Missing Data Strategy**
- Companies with no available data are automatically filtered from comparison tables and charts
- Individual metrics may show "N/A" if specific data points are unavailable
- Yahoo Finance API sometimes has incomplete data for certain companies/quarters

**Data Format**
- Quarters formatted as "YYYYQX" (e.g., "2024Q3")
- Revenue/expense values stored as integers
- Gross Margin % stored as floating-point percentage values
- Up to 5 quarters of historical data retrieved per company

### External Dependencies

**AI Service Providers (Multi-Provider Architecture)**

**DeepSeek API (Primary)**
- **Purpose**: Financial analysis and conclusion generation
- **Model**: deepseek-v3.2-exp
- **Authentication**: API key stored in DEEPSEEK_API_KEY environment variable
- **Usage**: Primary AI provider for all text generation

**Perplexity AI (Fallback)**
- **Purpose**: Backup AI provider when DeepSeek is unavailable
- **Model**: llama-3.1-sonar-small-128k-online
- **Authentication**: API key stored in PERPLEXITY_API_KEY environment variable
- **Usage**: Automatic fallback when DeepSeek fails or is unconfigured

**Local Deterministic Builder (Final Fallback)**
- **Purpose**: Guaranteed text generation when no AI provider is available
- **Implementation**: Template-based conclusion builder using pure Python
- **Usage**: Final fallback to ensure the app always returns results

**Yahoo Finance API (via yfinance)**
- **Purpose**: Real-time quarterly financial data
- **Data Sources**: Income statements, cash flow statements
- **Rate Limiting**: Aggressive rate limits requiring retry logic and delays
- **Reliability**: Some companies may have missing or incomplete data

**Python Package Dependencies**
- `flask`: Web application framework
- `flask-cors`: Cross-origin resource sharing support
- `yfinance`: Yahoo Finance API wrapper
- `pandas`: Data manipulation and analysis
- `requests`: HTTP client for AI API calls (DeepSeek, Perplexity)

### Design Patterns

**Error Handling**
- Retry logic with exponential backoff for Yahoo Finance rate limiting
- Graceful degradation when data is unavailable
- User-friendly error messages
- Backend logging for debugging

**Performance Optimization**
- Parallel data fetching for multiple companies
- Client-side chart rendering
- Responsive UI with loading indicators

**Recent Updates (October 16, 2025)**

**Critical Bug Fix - Peer Finding (October 16, 2025):**
- **Root Cause**: The `yfinance` library's `tickers_sp500()` and `tickers_nasdaq()` functions were unavailable/deprecated, causing the universe to be empty
- **Impact**: NO company could find peers - all showed empty peer lists
- **Solution**: 
  - Added comprehensive fallback ticker list with 100+ major US companies across all sectors
  - Includes: Tech (AAPL, MSFT), Retail (HD, LOW), Finance (JPM, BAC), Healthcare, Energy, etc.
  - Fixed peer deduplication to prevent duplicate peers when falling back across tiers
- **Verification**: Tested HD (finds LOW), JPM (finds BAC, WFC), DIS (finds NFLX, WBD) - all working ✅
- **Peer Matching Logic**:
  1. Specialty groups (MEGA7, eVTOL, EV) checked first
  2. Same industry match (e.g., HD → LOW both in Home Improvement Retail)
  3. Fallback to same sector if insufficient peers
  4. Final fallback to market cap proximity

**Enhanced Primary Company Analysis:**
- Added explanation of peer ranking system: "(Rankings #rank/total, where #1 is best except OpEx)"
- Added "Latest quarter metrics" line showing actual values from the comparison table
- Includes: Revenue (in B/M), Gross Margin %, EBIT, Net Income, Free Cash Flow
- Makes peer comparison more intuitive and informative for users
- Applied to both AI-generated and local fallback analysis

**Specialty Peer Groups Integration:**
- Added 3 self-defined specialty peer groups for better peer matching
- **MEGA7**: Magnificent 7 tech giants (AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA)
- **eVTOL Group**: Urban air mobility companies (EH, JOBY, ACHR, BLDE)
- **EV Group**: Electric vehicle manufacturers (RIVN, LCID, NIO, XPEV, LI, ZK, PSNY, BYDDY, VFS, WKHS)
- Ticker aliases support: GOOG→GOOGL, FB→META, SRTA→BLDE, BLADE→BLDE
- Peer selection prioritizes specialty groups before falling back to industry/sector matching

**Perplexity AI Fallback Integration:**
- Added Perplexity AI as a fallback provider for financial analysis
- Multi-provider cascade: DeepSeek → Perplexity → Local Builder
- Perplexity uses llama-3.1-sonar-small-128k-online model
- API configuration via PERPLEXITY_API_KEY environment variable
- Accurate tracking of which AI provider generated each conclusion
- Updated health endpoint to show all provider configurations

**Previous Updates (October 15, 2025)**

**Critical Bug Fix - Yahoo Finance Data Fetching:**
- Fixed pandas DataFrame boolean evaluation error in calculate_metrics function
- Issue: Using `or` operator with DataFrames caused "The truth value of a DataFrame is ambiguous" error
- Solution: Replaced `getattr(s, "quarterly_financials", None) or getattr(s, "financials", None)` with explicit None checks
- Impact: Yahoo Finance data fetching now works correctly for all tickers

**Enhanced Metrics Extraction for All Company Types:**
- Improved Yahoo Finance data extraction with multiple field name variations
- Added calculation fallbacks for Operating Expense (from SG&A, R&D, Selling & Marketing components)
- Added calculation fallbacks for EBIT (from Gross Profit, Revenue, COGS, or Pretax Income)
- Now supports different company structures:
  - Standard tech/manufacturing (AAPL, NVDA, TSLA): All 6 metrics
  - Fintech/financial services (UPST): 5 metrics (Gross Margin % appropriately N/A)
- Gracefully handles missing fields with intelligent fallback calculations

**AI-Powered Primary Company Conclusion:**
- Implemented comprehensive AI analysis using OpenAI GPT-4
- Analyzes 5-quarter time series trends (revenue growth, margin changes, profitability patterns)
- Compares primary company with peers in latest quarter
- Identifies notable strengths, concerns, and competitive positioning
- Provides 3-4 sentence actionable insights
- Handles missing metrics gracefully in analysis
- Includes fallback to basic summary if OpenAI API fails

**Previous Updates:**
- Added retry logic with exponential backoff for Yahoo Finance API
- Implemented custom User-Agent headers to avoid blocking
- Fixed missing data handling - companies with no data are filtered out
- Changed table headers to show actual quarter/year instead of "Latest Quarter"
- Added time series tables showing 5 quarters of historical data per company
