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

**AI-Powered Peer Discovery**
- **Library**: OpenAI API (GPT-4)
- **Purpose**: Automatically identify 3 peer companies based on industry and market cap
- **Input**: User-provided stock ticker symbol
- **Output**: JSON response with peer company tickers and names

**Data Fetching Layer**
- **Library**: yfinance (Yahoo Finance API wrapper)
- **Retry Logic**: Up to 3 attempts with exponential backoff (4s, 8s delays)
- **Rate Limiting Protection**: 
  - Custom User-Agent headers to avoid blocking
  - Delays between requests (1s before fetch, 0.5s between different API calls)
  - Detects HTTP 429 errors and automatically retries
- **Data Retrieved**: Quarterly income statements and cash flow statements

**Financial Metrics Calculation**
- Gross Margin % (calculated from Gross Profit and Total Revenue)
- Operating Expense (sum of SG&A and R&D)
- EBIT (from Operating Income)
- Free Cash Flow (Operating Cash Flow + Capital Expenditure)
- Handles missing data gracefully with fallback calculations

### API Endpoints

**POST /api/find-peers**
- Accepts: `{ "ticker": "AAPL" }`
- Returns: JSON with primary company, industry, and 3 peer companies
- Uses OpenAI GPT-4 for intelligent peer matching

**POST /api/get-metrics**
- Accepts: `{ "tickers": ["AAPL", "MSFT", "GOOGL", "AMZN"] }`
- Returns: Financial metrics for each ticker (5 quarters of data)
- Filters out companies with missing/incomplete data

**GET /api/health**
- Health check endpoint
- Verifies OpenAI API configuration

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

**OpenAI API**
- **Purpose**: Intelligent peer company identification
- **Model**: GPT-4
- **Authentication**: API key stored in OPENAI_API_KEY environment variable
- **Usage**: Single API call per peer discovery request

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
- `openai`: OpenAI API client
- `httpx==0.23.3`: HTTP client (pinned version for OpenAI compatibility)

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

**Recent Updates (October 2025)**
- Added retry logic with exponential backoff for Yahoo Finance API
- Implemented custom User-Agent headers to avoid blocking
- Fixed missing data handling - companies with no data are filtered out
- Changed table headers to show actual quarter/year instead of "Latest Quarter"
- Added time series tables showing 5 quarters of historical data per company
- **Critical Bug Fix**: Fixed pandas DataFrame boolean evaluation error in calculate_metrics function
  - Issue: Using `or` operator with DataFrames caused "The truth value of a DataFrame is ambiguous" error
  - Solution: Replaced `getattr(s, "quarterly_financials", None) or getattr(s, "financials", None)` with explicit None checks
  - Impact: Yahoo Finance data fetching now works correctly for all tickers
