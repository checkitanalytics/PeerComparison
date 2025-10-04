"""
Peer Company Comparison - Complete Replit Application
This serves both the API and the frontend HTML
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import os
from openai import OpenAI
import json
import time
from functools import lru_cache

app = Flask(__name__, static_folder='static')
CORS(app)

# Initialize OpenAI client
api_key = os.environ.get('OPENAI_API_KEY')
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")
client = OpenAI(api_key=api_key)

def calculate_metrics(ticker, max_retries=3):
    """Calculate key financial metrics for a given ticker with retry logic"""
    for attempt in range(max_retries):
        try:
            print(f"Fetching data for {ticker} (attempt {attempt + 1}/{max_retries})...")
            if attempt > 0:
                wait_time = (2 ** attempt) * 2
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)

            stock = yf.Ticker(ticker)
            if not hasattr(stock, "_session_configured"):
                stock.session.headers["User-Agent"] = (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                )
                stock._session_configured = True

            time.sleep(1)
            try:
                income_quarterly = stock.quarterly_income_stmt
                time.sleep(0.5)
                cash_flow_quarterly = stock.quarterly_cashflow
            except Exception as fetch_error:
                error_msg = str(fetch_error)
                if "429" in error_msg and attempt < max_retries - 1:
                    print("Rate limited! Will retry...")
                    continue
                print(f"Error fetching data for {ticker}: {error_msg}")
                return None

            print(f"Income statement columns: {len(income_quarterly.columns) if not income_quarterly.empty else 0}")
            print(f"Cash flow columns: {len(cash_flow_quarterly.columns) if not cash_flow_quarterly.empty else 0}")

            if income_quarterly.empty or cash_flow_quarterly.empty:
                if attempt < max_retries - 1:
                    print(f"Empty data for {ticker}, retrying...")
                    continue
                print(f"No data available for {ticker} (empty dataframes)")
                return None

            if "Gross Profit" in income_quarterly.index:
                income_quarterly.loc["Gross Margin %"] = (
                    income_quarterly.loc["Gross Profit"] * 100.0
                    / income_quarterly.loc["Total Revenue"]
                )
            else:
                income_quarterly.loc["Gross Margin %"] = (
                    (income_quarterly.loc["Total Revenue"] - income_quarterly.loc["Cost Of Revenue"]) * 100.0
                    / income_quarterly.loc["Total Revenue"]
                )

            if "Operating Expense" not in income_quarterly.index:
                if (
                    "Selling General And Administration" in income_quarterly.index
                    and "Research And Development" in income_quarterly.index
                ):
                    income_quarterly.loc["Operating Expense"] = (
                        income_quarterly.loc["Selling General And Administration"]
                        + income_quarterly.loc["Research And Development"]
                    )

            if "EBIT" not in income_quarterly.index and "Operating Income" in income_quarterly.index:
                income_quarterly.loc["EBIT"] = income_quarterly.loc["Operating Income"]

            if "Free Cash Flow" not in cash_flow_quarterly.index:
                if (
                    "Operating Cash Flow" in cash_flow_quarterly.index
                    and "Capital Expenditure" in cash_flow_quarterly.index
                ):
                    cash_flow_quarterly.loc["Free Cash Flow"] = (
                        cash_flow_quarterly.loc["Operating Cash Flow"]
                        + cash_flow_quarterly.loc["Capital Expenditure"]
                    )

            metrics_to_extract = [
                "Total Revenue",
                "Operating Expense",
                "Gross Margin %",
                "EBIT",
                "Net Income",
            ]

            filtered_income = income_quarterly[
                income_quarterly.index.isin(metrics_to_extract)
            ].iloc[:, 0:5]
            filtered_cash_flow = cash_flow_quarterly[
                cash_flow_quarterly.index == "Free Cash Flow"
            ].iloc[:, 0:5]

            if filtered_income.isna().sum().sum() == 0 and filtered_cash_flow.isna().sum().sum() == 0:
                result = pd.concat([filtered_income, filtered_cash_flow], axis=0)
                result.columns = result.columns.to_period("Q").astype(str)

                result_dict = {}
                for metric in result.index:
                    result_dict[metric] = {}
                    for quarter in result.columns:
                        value = result.loc[metric, quarter]
                        if pd.notna(value):
                            result_dict[metric][quarter] = (
                                float(value) if metric == "Gross Margin %" else int(value)
                            )
                        else:
                            result_dict[metric][quarter] = None

                print(f"Successfully processed data for {ticker}")
                return result_dict

            print(f"Data contains NaN values for {ticker}")
            return None
        except Exception as e:
            print(f"Error calculating metrics for {ticker}: {str(e)}")
            return None

    return None


@app.route('/')
def index():
    """Serve the main HTML page"""
    return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Peer Company Comparison</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
</head>
<body class="bg-gradient-to-br from-blue-50 to-indigo-100 min-h-screen p-8">
    <div class="max-w-7xl mx-auto">
        <div class="bg-white rounded-lg shadow-xl p-8 mb-8">
            <h1 class="text-4xl font-bold text-gray-800 mb-2">Peer Company Key Metrics Comparison</h1>
            <p class="text-gray-600 mb-6">Enter a company ticker to find and compare key financial metrics with peer companies</p>
            
            <div class="flex gap-4 mb-6">
                <input type="text" id="tickerInput" placeholder="Enter ticker (e.g., TSLA)" 
                    class="flex-1 px-4 py-3 border-2 border-gray-300 rounded-lg focus:outline-none focus:border-indigo-500 text-lg uppercase">
                <button onclick="findPeers()" id="findButton"
                    class="px-8 py-3 bg-indigo-600 text-white rounded-lg font-semibold hover:bg-indigo-700 disabled:bg-gray-400 transition-colors">
                    Find Peers
                </button>
            </div>

            <div id="error" class="hidden bg-red-50 border-l-4 border-red-500 p-4 mb-6">
                <p class="text-red-700"></p>
            </div>
            <div id="loading" class="hidden text-center py-4">
                <div class="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600"></div>
                <p class="mt-2 text-gray-600">Analyzing companies and fetching data...</p>
            </div>
            <div id="peerInfo" class="hidden bg-indigo-50 rounded-lg p-6 mb-8"></div>
        </div>
        <div id="results" class="hidden space-y-8"></div>
    </div>

    <script>
        async function findPeers() {
            const ticker = document.getElementById('tickerInput').value.trim().toUpperCase();
            if (!ticker) {
                showError('Please enter a company ticker symbol');
                return;
            }

            showLoading(true);
            hideError();
            document.getElementById('peerInfo').classList.add('hidden');
            document.getElementById('results').classList.add('hidden');

            try {
                const response = await fetch('/api/find-peers', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ ticker })
                });

                if (!response.ok) throw new Error('Failed to find peers');
                const peerData = await response.json();
                
                displayPeers(peerData);
                await fetchMetrics(peerData);
            } catch (err) {
                showError(err.message);
            } finally {
                showLoading(false);
            }
        }

        function displayPeers(data) {
            const html = `
                <h2 class="text-2xl font-semibold text-gray-800 mb-4">Peer Companies in ${data.industry}</h2>
                <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
                    <div class="bg-white p-4 rounded-lg shadow">
                        <div class="text-sm text-gray-600 mb-1">Primary Company</div>
                        <div class="text-xl font-bold text-indigo-600">${data.primary_company}</div>
                    </div>
                    ${data.peers.map((peer, idx) => `
                        <div class="bg-white p-4 rounded-lg shadow">
                            <div class="text-sm text-gray-600 mb-1">Peer ${idx + 1}</div>
                            <div class="text-xl font-bold text-gray-800">${peer.ticker}</div>
                            <div class="text-sm text-gray-600">${peer.name}</div>
                        </div>
                    `).join('')}
                </div>
            `;
            document.getElementById('peerInfo').innerHTML = html;
            document.getElementById('peerInfo').classList.remove('hidden');
        }

        async function fetchMetrics(peerData) {
            const tickers = [peerData.primary_company, ...peerData.peers.map(p => p.ticker)];
            
            const response = await fetch('/api/get-metrics', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ tickers })
            });

            if (!response.ok) throw new Error('Failed to fetch metrics');
            const metricsData = await response.json();
            
            displayResults(metricsData, tickers);
        }

        function displayResults(data, tickers) {
            const quarters = new Set();
            Object.values(data).forEach(company => {
                if (company['Total Revenue']) {
                    Object.keys(company['Total Revenue']).forEach(q => quarters.add(q));
                }
            });
            const sortedQuarters = Array.from(quarters).sort().reverse().slice(0, 5);

            const resultsDiv = document.getElementById('results');
            resultsDiv.innerHTML = `
                <div class="bg-white rounded-lg shadow-xl p-6">
                    <h3 class="text-2xl font-semibold text-gray-800 mb-4">Total Revenue Comparison</h3>
                    <canvas id="revenueChart" height="80"></canvas>
                </div>
                <div class="bg-white rounded-lg shadow-xl p-6">
                    <h3 class="text-2xl font-semibold text-gray-800 mb-4">Gross Margin % Trend</h3>
                    <canvas id="marginChart" height="80"></canvas>
                </div>
                <div class="bg-white rounded-lg shadow-xl p-6 overflow-x-auto">
                    <h3 class="text-2xl font-semibold text-gray-800 mb-4">Latest Quarter Metrics</h3>
                    <table class="w-full" id="metricsTable"></table>
                </div>
            `;
            resultsDiv.classList.remove('hidden');

            createCharts(data, tickers, sortedQuarters);
            createTable(data, tickers, sortedQuarters);
        }

        function createCharts(data, tickers, quarters) {
            const colors = ['#8884d8', '#82ca9d', '#ffc658', '#ff8042'];
            
            // Revenue Chart
            const revenueData = {
                labels: quarters,
                datasets: tickers.map((ticker, idx) => ({
                    label: ticker,
                    data: quarters.map(q => (data[ticker]['Total Revenue']?.[q] || 0) / 1000000000),
                    backgroundColor: colors[idx]
                }))
            };

            new Chart(document.getElementById('revenueChart'), {
                type: 'bar',
                data: revenueData,
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            ticks: { callback: value => '$' + value.toFixed(1) + 'B' }
                        }
                    }
                }
            });

            // Margin Chart
            const marginData = {
                labels: quarters,
                datasets: tickers.map((ticker, idx) => ({
                    label: ticker,
                    data: quarters.map(q => data[ticker]['Gross Margin %']?.[q] || 0),
                    borderColor: colors[idx],
                    backgroundColor: colors[idx],
                    fill: false
                }))
            };

            new Chart(document.getElementById('marginChart'), {
                type: 'line',
                data: marginData,
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            ticks: { callback: value => value.toFixed(1) + '%' }
                        }
                    }
                }
            });
        }

        function createTable(data, tickers, quarters) {
            const metrics = ['Total Revenue', 'Gross Margin %', 'Operating Expense', 'EBIT', 'Net Income', 'Free Cash Flow'];
            const latestQ = quarters[0];
            
            let html = '<thead><tr class="border-b-2 border-gray-300"><th class="text-left py-3 px-4">Metric</th>';
            tickers.forEach(t => html += `<th class="text-right py-3 px-4">${t}</th>`);
            html += '</tr></thead><tbody>';

            metrics.forEach(metric => {
                html += '<tr class="border-b border-gray-200 hover:bg-gray-50"><td class="py-3 px-4 font-medium">' + metric + '</td>';
                tickers.forEach(ticker => {
                    const value = data[ticker][metric]?.[latestQ];
                    const formatted = value !== undefined && value !== null
                        ? (metric === 'Gross Margin %' ? value.toFixed(2) + '%' : '$' + (value/1000000000).toFixed(2) + 'B')
                        : 'N/A';
                    html += `<td class="text-right py-3 px-4">${formatted}</td>`;
                });
                html += '</tr>';
            });

            html += '</tbody>';
            document.getElementById('metricsTable').innerHTML = html;
        }

        function showError(msg) {
            const errorDiv = document.getElementById('error');
            errorDiv.querySelector('p').textContent = msg;
            errorDiv.classList.remove('hidden');
        }

        function hideError() {
            document.getElementById('error').classList.add('hidden');
        }

        function showLoading(show) {
            document.getElementById('loading').classList.toggle('hidden', !show);
            document.getElementById('findButton').disabled = show;
        }

        document.getElementById('tickerInput').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') findPeers();
        });
    </script>
</body>
</html>
    '''


@app.route('/api/find-peers', methods=['POST'])
def find_peers():
    """Use OpenAI API to find peer companies"""
    try:
        data = request.json
        ticker = data.get('ticker', '').upper()
        
        if not ticker:
            return jsonify({'error': 'Ticker is required'}), 400
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a financial analyst. Respond only with valid JSON."},
                {"role": "user", "content": f"""Given ticker "{ticker}", identify 3 peer companies with similar market cap and industry.

Respond ONLY with valid JSON:
{{
  "primary_company": "{ticker}",
  "industry": "industry name",
  "peers": [
    {{"ticker": "TICKER1", "name": "Company Name 1"}},
    {{"ticker": "TICKER2", "name": "Company Name 2"}},
    {{"ticker": "TICKER3", "name": "Company Name 3"}}
  ]
}}"""}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        response_text = response.choices[0].message.content.strip()
        response_text = response_text.replace('```json', '').replace('```', '').strip()
        peer_data = json.loads(response_text)
        
        return jsonify(peer_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/get-metrics', methods=['POST'])
def get_metrics():
    """Fetch financial metrics for multiple companies"""
    try:
        data = request.json
        tickers = data.get('tickers', [])
        
        if not tickers:
            return jsonify({'error': 'Tickers are required'}), 400
        
        results = {}
        for ticker in tickers:
            metrics = calculate_metrics(ticker)
            results[ticker] = metrics if metrics else {'error': 'Unable to fetch data'}
        
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/health')
def health():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'openai_configured': bool(os.environ.get('OPENAI_API_KEY'))
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)