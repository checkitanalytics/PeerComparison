# Financial Data Comparison Platform

## Overview

This is a Streamlit-based financial data comparison platform that enables users to analyze and compare key financial metrics across multiple companies. The application fetches real-time financial data from Yahoo Finance and provides interactive visualizations, historical trend analysis, industry benchmarking, and customizable reporting capabilities. Users can save comparison sets to a PostgreSQL database for future reference.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture

**Framework Choice: Streamlit**
- **Problem**: Need for rapid development of an interactive financial analytics dashboard with minimal frontend complexity
- **Solution**: Streamlit provides a Python-native approach to building web applications with built-in state management and reactive components
- **Rationale**: Eliminates need for separate frontend framework (React/Vue), reduces development time, and allows data scientists to build UI without JavaScript knowledge
- **Trade-offs**: Less flexibility than traditional web frameworks, but sufficient for data-focused applications with standard UI patterns

**Component-Based Structure**
- Modular component architecture with separation of concerns
- Components organized by functionality: company_selector, metrics_display, charts, historical_trends, industry_benchmarking, export_reports, custom_metrics, saved_sets
- Each component is self-contained and renders specific UI sections
- Promotes code reusability and maintainability

**State Management**
- Uses Streamlit's session_state for client-side state persistence
- Key state variables: `selected_companies`, `comparison_data`, `db_initialized`
- State persists across reruns within a user session

**Layout Pattern**
- Wide layout with sidebar for controls and main content area for visualizations
- Responsive column-based layouts using Streamlit's column system
- Tab-based organization for different metric categories and analysis views

### Backend Architecture

**Data Fetching Layer**
- **Library Choice: yfinance (Yahoo Finance API wrapper)**
- Provides real-time and historical financial data for publicly traded companies
- Implements caching strategy with 5-minute TTL to reduce API calls and improve performance
- Graceful error handling for invalid tickers or API failures

**Business Logic Layer (Utils)**
- `financial_data.py`: Data acquisition and metric extraction
- `calculations.py`: Financial ratio calculations and percentage difference computations
- `database.py`: PostgreSQL database operations for saved comparison sets
- Clear separation between data fetching, processing, and persistence

**Calculation Engine**
- Computes derived financial ratios from raw metrics
- Calculates percentage differences from peer group averages
- Supports custom formula creation for advanced analysis

### Data Storage

**PostgreSQL Database**
- **Purpose**: Persistent storage for user-saved comparison sets
- **Schema Design**: Single table `comparison_sets` with fields:
  - `id` (SERIAL PRIMARY KEY)
  - `name` (VARCHAR 255) - User-defined set name
  - `description` (TEXT) - Optional description
  - `companies` (TEXT) - Serialized list of ticker symbols
  - `created_at`, `updated_at` (TIMESTAMP)
- **Connection Pattern**: Environment variable-based configuration using standard PostgreSQL connection parameters (PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD)
- **Design Decision**: Simple relational schema adequate for storing comparison sets; companies stored as text field rather than separate table due to simplicity and lack of need for complex querying

**Caching Strategy**
- Streamlit's `@st.cache_data` decorator with 5-minute TTL on financial data fetches
- Reduces redundant API calls for the same ticker within cache window
- Improves application responsiveness and reduces external API load

### Visualization Layer

**Library Choice: Plotly**
- **Problem**: Need for interactive, publication-quality financial charts
- **Solution**: Plotly provides interactive charts with hover details, zoom, pan capabilities
- **Alternatives Considered**: Matplotlib (static charts, less interactive), Altair (limited customization)
- **Pros**: Rich interactivity, professional appearance, extensive chart types
- **Cons**: Larger bundle size compared to simpler charting libraries

**Chart Types Implemented**
- Bar charts for metric comparisons
- Line charts for historical trends
- Pie charts for sector distribution
- Multi-axis subplots for combined visualizations

### Report Generation

**Export Formats**
- **Excel (.xlsx)**: Using openpyxl for structured spreadsheet reports with formatting
- **PDF Reports**: Using fpdf library for formatted PDF generation
- Downloadable reports include comparison tables, calculated metrics, and metadata

### External Dependencies

**Yahoo Finance API (via yfinance library)**
- **Purpose**: Primary data source for real-time and historical financial data
- **Data Accessed**: Stock prices, company info, financial statements, balance sheets, cash flow statements
- **Integration Pattern**: Direct API calls through Python library wrapper
- **Rate Limiting**: Managed through caching layer

**PostgreSQL Database**
- **Purpose**: Persistent storage for user-saved comparison sets
- **Connection**: Standard psycopg2 connection with environment variable configuration
- **Deployment Consideration**: Requires PostgreSQL instance with credentials configured in environment

**Python Package Dependencies**
- `streamlit`: Web application framework
- `yfinance`: Yahoo Finance API wrapper
- `pandas`: Data manipulation and analysis
- `numpy`: Numerical computations
- `plotly`: Interactive visualization
- `psycopg2`: PostgreSQL database adapter
- `openpyxl`: Excel file generation
- `fpdf`: PDF report generation

**Design Pattern: Multi-Company Comparison**
- Supports 2-10 companies for simultaneous comparison
- Implements validation to ensure minimum comparison threshold
- Data structures use company ticker as key for efficient lookups
- Percentage difference calculations normalize metrics across different scales