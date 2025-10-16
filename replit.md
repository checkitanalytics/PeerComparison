# Peer Company Comparison Platform

## Overview

This Flask-based platform analyzes and compares financial metrics of companies, utilizing OpenAI's GPT-4 (and fallbacks) to identify peers and Yahoo Finance for real-time data. It provides interactive visualizations and detailed time series analysis to offer concise, factual financial insights and rankings, supporting investment analysis and competitive intelligence.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture

The frontend is a Flask-served HTML/JavaScript application using Tailwind CSS for styling and Chart.js for interactive financial charts. It's designed as a single-page application with client-side state management and dynamic DOM updates via asynchronous API calls to the Flask backend.

### Backend Architecture

The backend is a Flask application providing RESTful API endpoints for peer discovery and financial data retrieval.

**Key Features:**

*   **AI-Powered Analysis (Multi-Provider with Fallbacks):** Uses DeepSeek API (primary), Perplexity AI (fallback), and a local deterministic text builder (final fallback) to generate analyst-style financial conclusions and translations based on company financial metrics and peer comparison data.
*   **Data Fetching:** Utilizes `yfinance` to fetch quarterly income and cash flow statements from Yahoo Finance, incorporating retry logic with exponential backoff, custom User-Agent headers, and delays to handle rate limiting and 429 errors.
*   **Financial Metrics Calculation:** Robustly calculates key metrics (Total Revenue, Gross Margin %, Operating Expense, EBIT, Net Income, Free Cash Flow) with extensive fallbacks for missing direct fields, accommodating different company types (e.g., fintech without Gross Margin %).
*   **Peer Selection Logic:** Prioritizes pre-defined specialty peer groups (e.g., MEGA7, eVTOL, EV) and ticker aliases. Otherwise, it uses industry-based matching, falling back to sector and then market capitalization proximity.
*   **API Endpoints:**
    *   `POST /api/find-peers`: Discovers peer companies for a given ticker.
    *   `POST /api/get-metrics`: Retrieves financial metrics for specified tickers.
    *   `POST /api/peer-key-metrics-conclusion`: Generates AI-powered financial analysis.
    *   `GET /api/primary-company-analysis`: Public endpoint for comprehensive primary company analysis with peer comparisons, latest quarter metrics, and multi-language support (English/Chinese).
    *   `GET /api/health`: Provides a health check and AI provider configuration status.

### Visualization Features

*   **Total Revenue Comparison:** Bar chart showing revenue across companies over up to 5 quarters.
*   **Gross Margin % Trend:** Line chart displaying gross margin trends over time.
*   **Latest Quarter Metrics Table:** Cross-company comparison for the most recent quarter, including Market Cap and calculated metrics.
*   **Time Series Tables:** Individual tables for each company showing 5 quarters of historical data for all key financial metrics.

### Data Handling

*   **Missing Data Strategy:** Companies with no available data are filtered out; individual missing metrics show "N/A."
*   **Data Format:** Quarters are "YYYYQX," numerical values in integers, and Gross Margin % as floating-point.

### Design Patterns

*   **Error Handling:** Includes retry logic, graceful degradation for unavailable data, and user-friendly error messages.
*   **Performance Optimization:** Features parallel data fetching and client-side chart rendering.

## External Dependencies

*   **DeepSeek API:** Primary AI provider for financial analysis and text generation (`deepseek-v3.2-exp` model).
*   **Perplexity AI:** Fallback AI provider for financial analysis (`llama-3.1-sonar-small-128k-online` model).
*   **Yahoo Finance API (via `yfinance`):** Provides real-time quarterly financial data (income and cash flow statements).
*   **Python Packages:** `flask`, `flask-cors`, `yfinance`, `pandas`, `requests`.