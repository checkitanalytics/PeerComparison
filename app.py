"""
Peer Company Comparison - Complete Replit Application
Serves both the API and the frontend HTML
- NEW: S3-backed ticker<->company resolver (CSV/JSON under s3://checkitanalytics/tickers/)
- NEW: /api/resolve to normalize user input to a ticker
- Minor hardening for yfinance / retries / logging
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os, json, time, io
from functools import lru_cache

import pandas as pd
import yfinance as yf

# Optional: boto3 if AWS creds are provided (for S3 ticker map)
try:
    import boto3  # type: ignore
    BOTO3_AVAILABLE = True
except Exception:
    BOTO3_AVAILABLE = False

from openai import OpenAI

# -----------------------------
# App / CORS
# -----------------------------
app = Flask(__name__, static_folder='static')
CORS(app)

# -----------------------------
# OpenAI
# -----------------------------
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")
client = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------------
# S3 Ticker Map (optional)
# Expects CSV/JSON files in s3://checkitanalytics/tickers/
# Will build bi-directional maps: ticker -> name, name -> ticker
# -----------------------------
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
S3_BUCKET = os.environ.get("S3_TICKER_BUCKET", "checkitanalytics")
S3_PREFIX = os.environ.get("S3_TICKER_PREFIX", "tickers/")

_ticker_to_name = {}
_name_to_ticker = {}
_s3_loaded = False
_s3_error = None


def _normalize_key(s: str) -> str:
    """case-insensitive, strip punctuation/spaces for matching."""
    return ''.join(ch for ch in s.strip().lower() if ch.isalnum())


def _ingest_rows(df: pd.DataFrame):
    # Accept common column variants
    cols = {c.lower(): c for c in df.columns}
    ticker_col = cols.get("ticker") or cols.get("tic") or cols.get("symbol")
    name_col = cols.get("name") or cols.get("comn") or cols.get("company") or cols.get("company_name")

    if not ticker_col:
        return
    # Name may be missing in some rows; handle gracefully
    for _, row in df.iterrows():
        t = str(row.get(ticker_col, "") or "").strip().upper()
        n = str(row.get(name_col, "") or "").strip()
        if not t:
            continue
        if n:
            _ticker_to_name[t] = n
            _name_to_ticker[_normalize_key(n)] = t
        # Also map ticker-text itself for reverse lookup convenience
        _name_to_ticker[_normalize_key(t)] = t


def _load_from_csv_bytes(b: bytes):
    try:
        df = pd.read_csv(io.BytesIO(b))
        _ingest_rows(df)
    except Exception:
        # Try TSV just in case
        try:
            df = pd.read_csv(io.BytesIO(b), sep="\t")
            _ingest_rows(df)
        except Exception:
            pass


def _load_from_json_bytes(b: bytes):
    try:
        obj = json.loads(b.decode("utf-8"))
        # obj could be list[dict] or dict of lists
        if isinstance(obj, list):
            df = pd.DataFrame(obj)
        elif isinstance(obj, dict):
            df = pd.DataFrame(obj)
        else:
            return
        _ingest_rows(df)
    except Exception:
        pass


def _load_ticker_map_from_s3():
    global _s3_loaded, _s3_error
    if _s3_loaded:
        return
    if not BOTO3_AVAILABLE:
        _s3_error = "boto3 not installed/available"
        _s3_loaded = True
        return
    # Only try if creds present
    if not (os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY")):
        _s3_error = "AWS credentials not provided"
        _s3_loaded = True
        return

    try:
        s3 = boto3.client("s3", region_name=AWS_REGION)
        paginator = s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_PREFIX)

        found_any = False
        for page in pages:
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if not key.lower().endswith((".csv", ".json")):
                    continue
                found_any = True
                body = s3.get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()
                if key.lower().endswith(".csv"):
                    _load_from_csv_bytes(body)
                else:
                    _load_from_json_bytes(body)

        if not found_any:
            _s3_error = f"No CSV/JSON found under s3://{S3_BUCKET}/{S3_PREFIX}"
    except Exception as e:
        _s3_error = f"S3 load error: {e}"
    finally:
        _s3_loaded = True


# Attempt load at startup (non-fatal)
_load_ticker_map_from_s3()


def resolve_input_to_ticker(user_input: str) -> dict:
    """
    Resolves arbitrary user input (ticker or company name) to a canonical ticker.
    Returns: { "input": "...", "ticker": "TSLA", "name": "Tesla, Inc.", "source": "s3|guess|input" }
    """
    raw = (user_input or "").strip()
    if not raw:
        return {"error": "Input is empty"}
    norm = _normalize_key(raw)

    # 1) Direct S3 map by normalized name
    if norm in _name_to_ticker:
        t = _name_to_ticker[norm]
        n = _ticker_to_name.get(t)
        return {"input": raw, "ticker": t, "name": n, "source": "s3"}

    # 2) If they typed a ticker exactly, prefer it
    if raw.isalpha() and raw.upper() in _ticker_to_name:
        t = raw.upper()
        return {"input": raw, "ticker": t, "name": _ticker_to_name.get(t), "source": "s3"}

    # 3) Fallback: if they typed something that *looks* like a ticker, try it
    if raw.isalpha() and 1 <= len(raw) <= 6:
        t = raw.upper()
        return {"input": raw, "ticker": t, "name": _ticker_to_name.get(t), "source": "input"}

    # 4) Last resort: return input back, client can decide to proceed
    return {"input": raw, "ticker": raw.upper(), "name": _ticker_to_name.get(raw.upper()), "source": "guess"}


# -----------------------------
# Financial metrics (yfinance)
# -----------------------------
def _ensure_yf_session_headers(t: yf.Ticker):
    try:
        # yfinance maintains a session inside; set a user-agent once
        if not hasattr(t, "_session_configured"):
            t.session.headers["User-Agent"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            t._session_configured = True
    except Exception:
        pass


@lru_cache(maxsize=128)
def calculate_metrics(ticker: str, max_retries: int = 3):
    """Calculate key financial metrics for a given ticker with retry logic (cached)"""
    ticker = (ticker or "").upper().strip()
    if not ticker:
        return None

    for attempt in range(max_retries):
        try:
            print(f"[metrics] {ticker} (attempt {attempt + 1}/{max_retries})")
            if attempt > 0:
                wait_time = (2 ** attempt) * 2
                time.sleep(wait_time)

            stock = yf.Ticker(ticker)
            _ensure_yf_session_headers(stock)
            time.sleep(0.5)

            try:
                income_quarterly = stock.quarterly_income_stmt or pd.DataFrame()
                time.sleep(0.2)
                cash_flow_quarterly = stock.quarterly_cashflow or pd.DataFrame()
            except Exception as fetch_error:
                if "429" in str(fetch_error) and attempt < max_retries - 1:
                    print("[metrics] rate-limited, retrying…")
                    continue
                print(f"[metrics] fetch error for {ticker}: {fetch_error}")
                return None

            if income_quarterly.empty or cash_flow_quarterly.empty:
                if attempt < max_retries - 1:
                    print(f"[metrics] empty frames for {ticker}, retrying…")
                    continue
                print(f"[metrics] no data for {ticker}")
                return None

            # Compute derived rows if needed
            if "Gross Profit" in income_quarterly.index:
                income_quarterly.loc["Gross Margin %"] = (
                    income_quarterly.loc["Gross Profit"] * 100.0 / income_quarterly.loc["Total Revenue"]
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
                        cash_flow_quarterly.loc["Operating Cash Flow"] + cash_flow_quarterly.loc["Capital Expenditure"]
                    )

            metrics_to_extract = ["Total Revenue", "Operating Expense", "Gross Margin %", "EBIT", "Net Income"]
            filtered_income = income_quarterly[income_quarterly.index.isin(metrics_to_extract)].iloc[:, 0:5]
            filtered_cash_flow = cash_flow_quarterly[cash_flow_quarterly.index == "Free Cash Flow"].iloc[:, 0:5]

            result = pd.concat([filtered_income, filtered_cash_flow], axis=0)
            if result.empty:
                return None

            # Pretty quarter labels
            result.columns = result.columns.to_period("Q").astype(str)

            result_dict = {}
            for metric in result.index:
                result_dict[metric] = {}
                for quarter in result.columns:
                    value = result.loc[metric, quarter]
                    if pd.notna(value):
                        result_dict[metric][quarter] = float(value) if metric == "Gross Margin %" else int(value)
                    else:
                        result_dict[metric][quarter] = None

            print(f"[metrics] success {ticker}")
            return result_dict

        except Exception as e:
            print(f"[metrics] error {ticker}: {e}")
            return None
    return None


# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def index():
    """Serve the main HTML page"""
    return '''
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Peer Company Comparison</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
</head>
<body class="bg-gradient-to-br from-blue-50 to-indigo-100 min-h-screen p-8">
  <div class="max-w-7xl mx-auto">
    <div class="bg-white rounded-lg shadow-xl p-8 mb-8">
      <h1 class="text-4xl font-bold text-gray-800 mb-2">Peer Company Key Metrics Comparison</h1>
      <p class="text-gray-600 mb-6">Enter a <b>ticker or company name</b> to find and compare key financial metrics with peer companies</p>

      <div class="flex gap-4 mb-6">
        <input type="text" id="tickerInput" placeholder="e.g., TSLA or Tesla"
          class="flex-1 px-4 py-3 border-2 border-gray-300 rounded-lg focus:outline-none focus:border-indigo-500 text-lg">
        <button onclick="resolveAndFind()" id="findButton"
          class="px-8 py-3 bg-indigo-600 text-white rounded-lg font-semibold hover:bg-indigo-700 disabled:bg-gray-400 transition-colors">
          Find Peers
        </button>
      </div>

      <div id="error" class="hidden bg-red-50 border-l-4 border-red-500 p-4 mb-6"><p class="text-red-700"></p></div>
      <div id="loading" class="hidden text-center py-4">
        <div class="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600"></div>
        <p class="mt-2 text-gray-600">Resolving input, analyzing peers, and fetching data...</p>
      </div>
      <div id="peerInfo" class="hidden bg-indigo-50 rounded-lg p-6 mb-8"></div>
    </div>
    <div id="results" class="hidden space-y-8"></div>
  </div>

  <script>
    async function resolveAndFind() {
      const raw = document.getElementById('tickerInput').value.trim();
      if (!raw) return showError('Please enter a ticker or company name');

      showLoading(true); hideError();
      document.getElementById('peerInfo').classList.add('hidden');
      document.getElementById('results').classList.add('hidden');

      try {
        const r = await fetch('/api/resolve', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ input: raw })});
        if (!r.ok) throw new Error('Failed to resolve input');
        const res = await r.json();
        if (res.error) throw new Error(res.error);

        // Proceed with the resolved ticker
        await findPeers(res.ticker, res.name);
      } catch (err) {
        showError(err.message);
      } finally {
        showLoading(false);
      }
    }

    async function findPeers(ticker, name) {
      try {
        showLoading(true);
        const response = await fetch('/api/find-peers', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ticker }) });
        if (!response.ok) throw new Error('Failed to find peers');
        const peerData = await response.json();

        // Attach pretty name (if we had it)
        if (name && peerData && peerData.primary_company === ticker) {
          peerData.primary_name = name;
        }

        displayPeers(peerData);
        await fetchMetrics(peerData);
      } catch (err) {
        showError(err.message);
      } finally {
        showLoading(false);
      }
    }

    function displayPeers(data) {
      const primaryLabel = data.primary_name ? data.primary_company + ' — ' + data.primary_name : data.primary_company;
      const html = `
        <h2 class="text-2xl font-semibold text-gray-800 mb-4">Peer Companies in ${data.industry}</h2>
        <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div class="bg-white p-4 rounded-lg shadow">
            <div class="text-sm text-gray-600 mb-1">Primary Company</div>
            <div class="text-xl font-bold text-indigo-600">${primaryLabel}</div>
          </div>
          ${data.peers.map((peer, idx) => `
            <div class="bg-white p-4 rounded-lg shadow">
              <div class="text-sm text-gray-600 mb-1">Peer ${idx + 1}</div>
              <div class="text-xl font-bold text-gray-800">${peer.ticker}</div>
              <div class="text-sm text-gray-600">${peer.name || ''}</div>
            </div>
          `).join('')}
        </div>
      `;
      document.getElementById('peerInfo').innerHTML = html;
      document.getElementById('peerInfo').classList.remove('hidden');
    }

    async function fetchMetrics(peerData) {
      const tickers = [peerData.primary_company, ...peerData.peers.map(p => p.ticker)];
      const response = await fetch('/api/get-metrics', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ tickers }) });
      if (!response.ok) throw new Error('Failed to fetch metrics');
      const metricsData = await response.json();
      displayResults(metricsData, tickers);
    }

    function displayResults(data, tickers) {
      const validTickers = tickers.filter(t => data[t] && !data[t].error && data[t]['Total Revenue']);
      if (validTickers.length === 0) return showError('No valid data found for any companies. Please try different tickers.');

      const quarters = new Set();
      validTickers.forEach(t => {
        if (data[t]['Total Revenue']) Object.keys(data[t]['Total Revenue']).forEach(q => quarters.add(q));
      });
      const sortedQuarters = Array.from(quarters).sort().reverse().slice(0, 5);
      const latestQuarter = sortedQuarters[0] || 'N/A';

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
          <h3 class="text-2xl font-semibold text-gray-800 mb-4">${latestQuarter} Metrics</h3>
          <table class="w-full" id="metricsTable"></table>
        </div>
        <div id="timeSeriesTables" class="space-y-6"></div>
      `;
      resultsDiv.classList.remove('hidden');

      createCharts(data, validTickers, sortedQuarters);
      createTable(data, validTickers, sortedQuarters);
      createTimeSeriesTables(data, validTickers, sortedQuarters);
    }

    function createCharts(data, tickers, quarters) {
      const colors = ['#8884d8', '#82ca9d', '#ffc658', '#ff8042', '#a4de6c', '#d0ed57'];

      const revenueData = {
        labels: quarters,
        datasets: tickers.map((t, i) => ({
          label: t,
          data: quarters.map(q => (data[t]['Total Revenue']?.[q] || 0) / 1_000_000_000),
          backgroundColor: colors[i % colors.length]
        }))
      };
      new Chart(document.getElementById('revenueChart'), {
        type: 'bar',
        data: revenueData,
        options: { responsive: true, scales: { y: { ticks: { callback: v => '$' + v.toFixed(1) + 'B' } } } }
      });

      const marginData = {
        labels: quarters,
        datasets: tickers.map((t, i) => ({
          label: t,
          data: quarters.map(q => data[t]['Gross Margin %']?.[q] || 0),
          borderColor: colors[i % colors.length],
          backgroundColor: colors[i % colors.length],
          fill: false
        }))
      };
      new Chart(document.getElementById('marginChart'), {
        type: 'line',
        data: marginData,
        options: { responsive: true, scales: { y: { ticks: { callback: v => v.toFixed(1) + '%' } } } }
      });
    }

    function createTable(data, tickers, quarters) {
      const metrics = ['Total Revenue', 'Gross Margin %', 'Operating Expense', 'EBIT', 'Net Income', 'Free Cash Flow'];
      const latestQ = quarters[0];
      let html = '<thead><tr class="border-b-2 border-gray-300"><th class="text-left py-3 px-4">Metric</th>';
      tickers.forEach(t => html += `<th class="text-right py-3 px-4">${t}</th>`); html += '</tr></thead><tbody>';

      metrics.forEach(metric => {
        html += `<tr class="border-b border-gray-200 hover:bg-gray-50"><td class="py-3 px-4 font-medium">${metric}</td>`;
        tickers.forEach(t => {
          const v = data[t][metric]?.[latestQ];
          const formatted = (v !== undefined && v !== null)
            ? (metric === 'Gross Margin %' ? v.toFixed(2) + '%' : '$' + (v/1_000_000_000).toFixed(2) + 'B')
            : 'N/A';
          html += `<td class="text-right py-3 px-4">${formatted}</td>`;
        });
        html += '</tr>';
      });

      html += '</tbody>';
      document.getElementById('metricsTable').innerHTML = html;
    }

    function createTimeSeriesTables(data, tickers, quarters) {
      const metrics = ['Total Revenue', 'Operating Expense', 'Gross Margin %', 'EBIT', 'Net Income', 'Free Cash Flow'];
      const container = document.getElementById('timeSeriesTables');
      let html = '';
      tickers.forEach(ticker => {
        html += `
          <div class="bg-white rounded-lg shadow-xl p-6 overflow-x-auto">
            <h3 class="text-2xl font-semibold text-gray-800 mb-4">${ticker} - 5 Quarter Time Series</h3>
            <table class="w-full">
              <thead><tr class="border-b-2 border-gray-300">
                <th class="text-left py-3 px-4"></th>
                ${quarters.map(q => `<th class="text-right py-3 px-4">${q}</th>`).join('')}
              </tr></thead>
              <tbody>`;
        metrics.forEach(metric => {
          html += `<tr class="border-b border-gray-200 hover:bg-gray-50"><td class="py-3 px-4 font-medium">${metric}</td>`;
          quarters.forEach(q => {
            const v = data[ticker][metric]?.[q];
            let formatted = 'N/A';
            if (v !== undefined && v !== null) {
              formatted = (metric === 'Gross Margin %') ? v.toFixed(2) + '%' : '$' + (v/1_000_000).toFixed(1) + 'M';
            }
            html += `<td class="text-right py-3 px-4">${formatted}</td>`;
          });
          html += '</tr>';
        });
        html += `</tbody></table></div>`;
      });
      container.innerHTML = html;
    }

    function showError(msg){ const d=document.getElementById('error'); d.querySelector('p').textContent=msg; d.classList.remove('hidden'); }
    function hideError(){ document.getElementById('error').classList.add('hidden'); }
    function showLoading(b){ document.getElementById('loading').classList.toggle('hidden',!b); document.getElementById('findButton').disabled=b; }
    document.getElementById('tickerInput').addEventListener('keypress', e => { if (e.key === 'Enter') resolveAndFind(); });
  </script>
</body>
</html>
    '''


@app.route('/api/resolve', methods=['POST'])
def api_resolve():
    """Resolve user input to a canonical ticker using S3 map if present"""
    try:
        data = request.json or {}
        user_input = data.get("input", "")
        result = resolve_input_to_ticker(user_input)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/find-peers', methods=['POST'])
def find_peers():
    """Use OpenAI API to find peer companies"""
    try:
        data = request.json or {}
        ticker = (data.get('ticker') or '').upper().strip()
        if not ticker:
            return jsonify({'error': 'Ticker is required'}), 400

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a financial analyst. Respond only with valid JSON."},
                {"role": "user", "content": f'''Given ticker "{ticker}", identify 3 peer companies with similar market cap and industry.
Respond ONLY with valid JSON:
{{
  "primary_company": "{ticker}",
  "industry": "industry name",
  "peers": [
    {{"ticker": "TICKER1", "name": "Company Name 1"}},
    {{"ticker": "TICKER2", "name": "Company Name 2"}},
    {{"ticker": "TICKER3", "name": "Company Name 3"}}
  ]
}}'''}
            ],
            temperature=0.3,
            max_tokens=400
        )

        response_text = response.choices[0].message.content.strip()
        # Clean up accidental code fences if present
        response_text = response_text.replace('```json', '').replace('```', '').strip()
        peer_data = json.loads(response_text)
        return jsonify(peer_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/get-metrics', methods=['POST'])
def get_metrics():
    """Fetch financial metrics for multiple companies"""
    try:
        data = request.json or {}
        tickers = data.get('tickers', [])
        if not tickers:
            return jsonify({'error': 'Tickers are required'}), 400

        results = {}
        for t in tickers:
            metrics = calculate_metrics(t)
            results[t] = metrics if metrics else {'error': 'Unable to fetch data'}
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/health')
def health():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'openai_configured': bool(os.environ.get('OPENAI_API_KEY')),
        's3_loaded': _s3_loaded,
        's3_error': _s3_error,
        'ticker_map_counts': {
            'ticker_to_name': len(_ticker_to_name),
            'name_to_ticker': len(_name_to_ticker)
        }
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    # debug=True is fine for Replit; switch off in prod
    app.run(host='0.0.0.0', port=port, debug=True)
