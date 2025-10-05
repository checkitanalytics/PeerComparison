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


def _verify_ticker_with_yfinance(ticker: str) -> dict:
    """
    Verify if a ticker is valid using yfinance and return its info.
    Returns: {"ticker": "TSLA", "name": "Tesla, Inc."} or None
    """
    try:
        t = ticker.upper().strip()
        stock = yf.Ticker(t)
        _ensure_yf_session_headers(stock)
        
        # Try to get basic info
        info = stock.get_info() or {}
        long_name = info.get("longName") or info.get("shortName")
        
        # If we got a name, consider it valid
        if long_name:
            return {"ticker": t, "name": long_name}
        
        # Check if there's any data available
        if info.get("symbol") == t:
            return {"ticker": t, "name": None}
            
        return None
    except Exception:
        return None


def _search_ticker_with_openai(company_name: str) -> dict:
    """
    Use OpenAI to resolve a company name to a ticker symbol.
    Returns: {"ticker": "TSLA", "name": "Tesla, Inc."} or None
    """
    try:
        prompt = f"""
You are a stock ticker expert. Given the company name "{company_name}", return ONLY the stock ticker symbol and full company name.
If it's a publicly-traded company, respond with JSON: {{"ticker":"SYMBOL","name":"Full Company Name"}}
If it's a private company or doesn't exist, respond with: {{"error":"Private or not found"}}

Company name: {company_name}
"""
        response = client.chat.completions.create(
            model="gpt-4",
            temperature=0.0,
            max_tokens=100,
            messages=[
                {"role": "system", "content": "Respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ]
        )
        raw = response.choices[0].message.content.strip().replace("```json", "").replace("```", "")
        obj = json.loads(raw)
        
        if obj.get("error"):
            return None
        
        ticker = obj.get("ticker", "").upper().strip()
        name = obj.get("name", "").strip()
        
        if not ticker:
            return None
        
        # Verify the ticker with yfinance
        verified = _verify_ticker_with_yfinance(ticker)
        if verified:
            return verified
        
        return None
    except Exception:
        return None


def _search_ticker_with_yfinance(query: str) -> dict:
    """
    Search for a ticker using company name via yfinance.
    Returns: {"ticker": "TSLA", "name": "Tesla, Inc."} or None
    """
    if not query:
        return None
    
    # Common company name to ticker mappings (fast lookup for known companies)
    common_mappings = {
        "tesla": "TSLA",
        "apple": "AAPL",
        "microsoft": "MSFT",
        "amazon": "AMZN",
        "google": "GOOGL",
        "alphabet": "GOOGL",
        "meta": "META",
        "facebook": "META",
        "nvidia": "NVDA",
        "netflix": "NFLX",
        "twitter": "TWTR",
        "x corp": "TWTR",
        "boeing": "BA",
        "airbus": "AIR.PA",
        "intel": "INTC",
        "amd": "AMD",
        "walmart": "WMT",
        "disney": "DIS",
        "coca cola": "KO",
        "pepsi": "PEP",
        "mcdonalds": "MCD",
        "starbucks": "SBUX",
        "visa": "V",
        "mastercard": "MA",
        "jpmorgan": "JPM",
        "jp morgan": "JPM",
        "bank of america": "BAC",
        "wells fargo": "WFC",
        "goldman sachs": "GS",
        "morgan stanley": "MS",
        "berkshire hathaway": "BRK.B",
        "exxon": "XOM",
        "chevron": "CVX",
        "pfizer": "PFE",
        "johnson & johnson": "JNJ",
        "johnson and johnson": "JNJ",
        "unitedhealth": "UNH",
        "procter & gamble": "PG",
        "procter and gamble": "PG",
        "home depot": "HD",
        "salesforce": "CRM",
        "adobe": "ADBE",
        "cisco": "CSCO",
        "oracle": "ORCL",
        "ibm": "IBM",
        "paypal": "PYPL",
        "uber": "UBER",
        "lyft": "LYFT",
        "airbnb": "ABNB",
        "zoom": "ZM",
        "slack": "WORK",
        "spotify": "SPOT",
        "snapchat": "SNAP",
        "pinterest": "PINS",
        "square": "SQ",
        "robinhood": "HOOD",
        "coinbase": "COIN",
        "snowflake": "SNOW",
        "palantir": "PLTR",
        "databricks": None,  # Private
        "spacex": None,  # Private company
        "stripe": None,  # Private company
    }
    
    # Normalize the query
    norm = query.strip().lower()
    
    # Check common mappings first (fast path)
    if norm in common_mappings:
        ticker = common_mappings[norm]
        if ticker:
            # Verify and get name
            result = _verify_ticker_with_yfinance(ticker)
            if result:
                return result
        else:
            return None  # Private company
    
    # Try as-is if it looks like a ticker
    if len(query) <= 6 and query.isalpha():
        result = _verify_ticker_with_yfinance(query)
        if result:
            return result
    
    # For multi-word company names, try OpenAI as fallback
    if ' ' in query or len(query) > 6:
        result = _search_ticker_with_openai(query)
        if result:
            return result
    
    return None


def resolve_input_to_ticker(user_input: str) -> dict:
    """
    Resolves arbitrary user input (ticker or company name) to a canonical ticker.
    Returns: { "input": "...", "ticker": "TSLA", "name": "Tesla, Inc.", "source": "s3|guess|input|yfinance" }
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

    # 3) Try yfinance search to resolve company name to ticker
    yf_result = _search_ticker_with_yfinance(raw)
    if yf_result:
        return {"input": raw, "ticker": yf_result["ticker"], "name": yf_result.get("name"), "source": "yfinance"}

    # 4) Fallback: if they typed something that *looks* like a ticker, try it
    if raw.isalpha() and 1 <= len(raw) <= 6:
        t = raw.upper()
        # Verify it's a valid ticker using yfinance
        verified = _verify_ticker_with_yfinance(t)
        if verified:
            return {"input": raw, "ticker": verified["ticker"], "name": verified.get("name"), "source": "input"}
        return {"input": raw, "ticker": t, "name": _ticker_to_name.get(t), "source": "input"}

    # 5) Last resort: return input back, client can decide to proceed
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
                income_quarterly = stock.quarterly_income_stmt if stock.quarterly_income_stmt is not None else pd.DataFrame()
                time.sleep(0.2)
                cash_flow_quarterly = stock.quarterly_cashflow if stock.quarterly_cashflow is not None else pd.DataFrame()
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


# ============================================================
# STEP 1: Peer Selection Add-Ons (constants & helpers)
# ============================================================

# Aliases only for peer logic (doesn't affect your resolver)
PEER_TICKER_ALIAS = {
    "GOOG": "GOOGL",
    "FB":   "META",
    "SRTA": "BLDE",   # Blade Air Mobility alias
    "BLADE": "BLDE",
}
def _normalize_peer_ticker(t: str) -> str:
    u = (t or "").upper().strip()
    return PEER_TICKER_ALIAS.get(u, u)

# Mega 7 (include TSLA)
MEGA7 = [
    {"ticker": "AAPL",  "name": "Apple Inc."},
    {"ticker": "MSFT",  "name": "Microsoft Corporation"},
    {"ticker": "GOOGL", "name": "Alphabet Inc. (Class A)"},
    {"ticker": "AMZN",  "name": "Amazon.com, Inc."},
    {"ticker": "META",  "name": "Meta Platforms, Inc."},
    {"ticker": "NVDA",  "name": "NVIDIA Corporation"},
    {"ticker": "TSLA",  "name": "Tesla, Inc."},
]
MEGA7_TICKERS = {x["ticker"] for x in MEGA7}

# eVTOL group
EVTOL_GROUP = [
    {"ticker": "EH",    "name": "EHang Holdings Limited"},
    {"ticker": "JOBY",  "name": "Joby Aviation, Inc."},
    {"ticker": "ACHR",  "name": "Archer Aviation Inc."},
    {"ticker": "BLDE",  "name": "Blade Air Mobility, Inc."},
]
EVTOL_TICKERS = {x["ticker"] for x in EVTOL_GROUP}

# Tunables
PEER_LIMIT = 6
MARKET_CAP_RATIO_LIMIT = 10.0  # >10x difference -> exclude

def _same_industry(a: str, b: str) -> bool:
    return bool(a and b and a.strip().lower() == b.strip().lower())

def _mc_ratio_ok(mc_a, mc_b, limit=MARKET_CAP_RATIO_LIMIT) -> bool:
    """If both MCs present and ratio > limit -> not ok; if either missing -> allow."""
    if mc_a is None or mc_b is None:
        return True
    small = min(mc_a, mc_b)
    big = max(mc_a, mc_b)
    if small <= 0:
        return False
    return (big / small) <= float(limit)

@lru_cache(maxsize=512)
def fetch_profile(ticker: str, max_retries: int = 2) -> dict:
    """
    Get {ticker, name, industry, market_cap} using yfinance.
    Uses fast_info for market cap, and get_info for industry/name.
    Cached to cut repeated calls.
    """
    t = (ticker or "").upper().strip()
    if not t:
        return {"ticker": ticker, "name": None, "industry": None, "market_cap": None}

    stock = yf.Ticker(t)
    _ensure_yf_session_headers(stock)

    market_cap = None
    name = t
    industry = None

    for attempt in range(max_retries):
        try:
            # market cap via fast_info
            try:
                fi = getattr(stock, "fast_info", {}) or {}
                market_cap = fi.get("market_cap", None) if hasattr(fi, "get") else getattr(fi, "market_cap", None)
            except Exception:
                market_cap = None

            # name/industry via get_info (may be slower)
            info = {}
            try:
                info = stock.get_info() or {}
            except Exception:
                info = {}

            name = info.get("longName") or info.get("shortName") or t
            industry = info.get("industry") or info.get("sector")

            return {
                "ticker": t,
                "name": name,
                "industry": industry,
                "market_cap": market_cap
            }
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(0.4 * (attempt + 1))
                continue
            break

    return {"ticker": t, "name": name, "industry": industry, "market_cap": market_cap}


# ============================================================
# STEP 2: Candidate Generators (S3 + optional OpenAI)
# ============================================================

def _s3_universe_candidates(base_ticker: str, base_industry: str, limit: int = 120):
    """
    Propose candidates from S3 universe (same industry). Offline & safe.
    """
    try:
        universe = list(_ticker_to_name.keys())
    except NameError:
        universe = []
    out = []
    count = 0
    for t in universe:
        if t == base_ticker:
            continue
        prof = fetch_profile(t)
        if _same_industry(base_industry or "", prof.get("industry") or ""):
            out.append({"ticker": prof["ticker"], "name": prof.get("name")})
        count += 1
        if count >= limit:
            break
    return out


def _openai_candidates(base_ticker: str, count: int = 16):
    """
    Optional: ask OpenAI to propose commonly compared names; we still filter by rules after.
    """
    try:
        prompt = f"""
List {count} publicly-traded companies that are commonly compared as peers to {base_ticker},
operating in the same *industry* (not just sector). Return ONLY JSON:
{{"peers":[{{"ticker":"T1","name":"Name 1"}}, ...]}}
Exclude {base_ticker}. Prefer US-listed if available.
"""
        ai = client.chat.completions.create(
            model="gpt-4",
            temperature=0.1,
            max_tokens=400,
            messages=[
                {"role": "system", "content": "Respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ]
        )
        raw = ai.choices[0].message.content.strip().replace("```json", "").replace("```", "")
        obj = json.loads(raw)
        peers = obj.get("peers", [])
        out = []
        for p in peers:
            ct = _normalize_peer_ticker(p.get("ticker", ""))
            if ct and ct != base_ticker:
                out.append({"ticker": ct, "name": p.get("name")})
        return out
    except Exception:
        return []


# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def index():
    """Serve the main HTML page (now with manual add-company support)"""
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

      <!-- Search boxes and buttons - responsive layout -->
      <div class="mb-6">
        <div class="flex flex-col md:flex-row gap-3 items-stretch md:items-center">
          <input type="text" id="tickerInput" placeholder="e.g., TSLA or Tesla"
            class="flex-1 px-4 py-3 border-2 border-gray-300 rounded-lg focus:outline-none focus:border-indigo-500 text-lg">
          <button onclick="resolveAndFind()" id="findButton"
            class="px-6 py-3 bg-indigo-600 text-white rounded-lg font-semibold hover:bg-indigo-700 disabled:bg-gray-400 transition-colors whitespace-nowrap">
            Find Peers
          </button>
          <input type="text" id="manualInput" placeholder="Add company (ticker or name)"
            class="flex-1 px-4 py-3 border-2 border-gray-300 rounded-lg focus:outline-none focus:border-indigo-500 text-lg">
          <button onclick="addCompany()" id="addButton"
            class="px-6 py-3 bg-emerald-600 text-white rounded-lg font-semibold hover:bg-emerald-700 disabled:bg-gray-400 transition-colors whitespace-nowrap">
            Add company
          </button>
        </div>
        <p class="text-sm text-gray-500 mt-2">Tip: Try adding AAPL, MSFT, AMZN, etc.</p>
      </div>

      <div id="error" class="hidden bg-red-50 border-l-4 border-red-500 p-4 mb-6"><p class="text-red-700"></p></div>
      <div id="loading" class="hidden text-center py-4">
        <div class="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600"></div>
        <p class="mt-2 text-gray-600">Working…</p>
      </div>
      <div id="peerInfo" class="hidden bg-indigo-50 rounded-lg p-6 mb-8"></div>
    </div>

    <div id="results" class="hidden space-y-8"></div>
  </div>

  <script>
    // ---------- Global state ----------
    let _metricsData = {};         // ticker -> metrics dict
    let _tickers = [];             // ordered list for charts/tables
    let _quarters = [];            // sorted newest->oldest (max 5)
    let _combinedChart = null;

    const COLORS = ['#8884d8', '#82ca9d', '#ffc658', '#ff8042', '#a4de6c', '#d0ed57', '#8dd1e1', '#a28dd1'];

    // ---------- Utilities ----------
    function showError(msg){ const d=document.getElementById('error'); d.querySelector('p').textContent=msg; d.classList.remove('hidden'); }
    function hideError(){ document.getElementById('error').classList.add('hidden'); }
    function showLoading(b){
      document.getElementById('loading').classList.toggle('hidden',!b);
      document.getElementById('findButton').disabled=b;
      document.getElementById('addButton').disabled=b;
    }

    function uniqLower(arr){ const s=new Set(); const out=[]; for(const x of arr){const y=(x||'').toUpperCase(); if(!s.has(y)){s.add(y); out.push(y);} } return out; }

    // ---------- Initial peer flow ----------
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

        await findPeers(res.ticker, res.name);
      } catch (err) { showError(err.message); }
      finally { showLoading(false); }
    }

    async function findPeers(ticker, name) {
      try {
        showLoading(true);
        const response = await fetch('/api/find-peers', {
          method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ticker })
        });
        if (!response.ok) throw new Error('Failed to find peers');
        const peerData = await response.json();
        if (name && peerData && peerData.primary_company === ticker) peerData.primary_name = name;

        displayPeers(peerData);

        const tickers = [peerData.primary_company, ...peerData.peers.map(p => p.ticker)];
        const metricsData = await fetchMetricsForTickers(tickers);

        // Initialize global state
        _metricsData = metricsData;
        // Always include primary company, filter peers only
        const primaryTicker = peerData.primary_company;
        const validPeers = tickers.slice(1).filter(t => metricsData[t] && !metricsData[t].error && metricsData[t]['Total Revenue']);
        _tickers = [primaryTicker, ...validPeers];
        _quarters = computeQuarters(metricsData, _tickers);

        renderAll();
      } catch (err) { showError(err.message); }
      finally { showLoading(false); }
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

    // ---------- Manual add flow ----------
    async function addCompany() {
      const raw = document.getElementById('manualInput').value.trim();
      if (!raw) return showError('Please enter a ticker or company name to add.');
      hideError();
      try {
        showLoading(true);
        // Resolve to a ticker
        const r = await fetch('/api/resolve', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ input: raw }) });
        if (!r.ok) throw new Error('Failed to resolve input');
        const res = await r.json();
        if (res.error) throw new Error(res.error);
        const newTicker = (res.ticker || '').toUpperCase();

        if (!newTicker) throw new Error('Could not resolve to a ticker.');
        if (_tickers.includes(newTicker)) {
          return showError(`${newTicker} is already included.`);
        }

        // Fetch metrics only for the new ticker
        const oneMetrics = await fetchMetricsForTickers([newTicker]);
        if (!oneMetrics[newTicker] || oneMetrics[newTicker].error || !oneMetrics[newTicker]['Total Revenue']) {
          return showError(`No usable data for ${newTicker}.`);
        }

        // Merge into global state
        _metricsData[newTicker] = oneMetrics[newTicker];
        _tickers.push(newTicker);
        _tickers = uniqLower(_tickers);

        // Recompute unified quarter set (top 5 newest)
        _quarters = computeQuarters(_metricsData, _tickers);

        // Re-render charts & tables with the new ticker
        renderAll();

        // Clear input
        document.getElementById('manualInput').value = '';
        document.getElementById('results').classList.remove('hidden');
      } catch (err) {
        showError(err.message);
      } finally {
        showLoading(false);
      }
    }

    // ---------- Data helpers ----------
    async function fetchMetricsForTickers(tickers){
      const response = await fetch('/api/get-metrics', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ tickers })
      });
      if (!response.ok) throw new Error('Failed to fetch metrics');
      return await response.json();
    }

    function computeQuarters(data, tickers){
      const qset = new Set();
      tickers.forEach(t => {
        const m = data[t];
        if (m && m['Total Revenue']) Object.keys(m['Total Revenue']).forEach(q => qset.add(q));
      });
      return Array.from(qset).sort().reverse().slice(0,5);
    }

    // ---------- Rendering ----------
    function renderAll(){
      const resultsDiv = document.getElementById('results');
      resultsDiv.innerHTML = `
        <div class="bg-white rounded-lg shadow-xl p-6">
          <h3 class="text-2xl font-semibold text-gray-800 mb-4">Total Revenue & Gross Margin % Trend</h3>
          <canvas id="combinedChart" height="80"></canvas>
        </div>
        <div class="bg-white rounded-lg shadow-xl p-6 overflow-x-auto">
          <h3 class="text-2xl font-semibold text-gray-800 mb-4">Latest Quarter Metrics</h3>
          <table class="w-full" id="metricsTable"></table>
        </div>
        <div id="timeSeriesTables" class="space-y-6"></div>
      `;
      resultsDiv.classList.remove('hidden');

      renderCharts();
      renderTable();
      renderTimeSeriesTables();
    }

    function renderCharts(){
      const labels = _quarters;

      const datasetsRevenue = _tickers.map((t, i) => ({
        label: t + ' Revenue',
        data: labels.map(q => (((_metricsData[t] || {})['Total Revenue']?.[q] || 0) / 1_000_000_000)),
        backgroundColor: COLORS[i % COLORS.length],
        type: 'bar',
        yAxisID: 'y'
      }));

      const datasetsMargin = _tickers.map((t, i) => ({
        label: t + ' Margin %',
        data: labels.map(q => ((_metricsData[t] || {})['Gross Margin %']?.[q] || 0)),
        borderColor: COLORS[i % COLORS.length],
        backgroundColor: COLORS[i % COLORS.length],
        type: 'line',
        yAxisID: 'y1',
        fill: false,
        borderWidth: 2,
        pointRadius: 4,
        pointHoverRadius: 6
      }));

      const combinedDatasets = [...datasetsRevenue, ...datasetsMargin];

      if (_combinedChart) _combinedChart.destroy();
      _combinedChart = new Chart(document.getElementById('combinedChart'), {
        type: 'bar',
        data: { labels, datasets: combinedDatasets },
        options: {
          responsive: true,
          interaction: {
            mode: 'index',
            intersect: false
          },
          scales: {
            y: {
              type: 'linear',
              display: true,
              position: 'left',
              title: {
                display: true,
                text: 'Total Revenue (Billions $)'
              },
              ticks: {
                callback: v => '$' + Number(v).toFixed(1) + 'B'
              }
            },
            y1: {
              type: 'linear',
              display: true,
              position: 'right',
              title: {
                display: true,
                text: 'Gross Margin %'
              },
              ticks: {
                callback: v => Number(v).toFixed(1) + '%'
              },
              grid: {
                drawOnChartArea: false
              }
            }
          }
        }
      });
    }

    function renderTable(){
      const metrics = ['Total Revenue', 'Gross Margin %', 'Operating Expense', 'EBIT', 'Net Income', 'Free Cash Flow'];
      
      // Get the latest quarter available for each company
      const tickerLatestQuarters = _tickers.map(t => {
        const tData = _metricsData[t] || {};
        const quarters = tData['Total Revenue'] ? Object.keys(tData['Total Revenue']).sort().reverse() : [];
        return quarters[0] || 'N/A';
      });
      
      let html = '<thead><tr class="border-b-2 border-gray-300"><th class="text-left py-3 px-4">Metric</th>';
      _tickers.forEach((t, i) => {
        html += `<th class="text-right py-3 px-4">${t}<br/><span class="text-xs text-gray-500">${tickerLatestQuarters[i]}</span></th>`;
      });
      html += '</tr></thead><tbody>';

      metrics.forEach(metric => {
        html += `<tr class="border-b border-gray-200 hover:bg-gray-50"><td class="py-3 px-4 font-medium">${metric}</td>`;
        _tickers.forEach((t, i) => {
          const latestQ = tickerLatestQuarters[i];
          const v = (_metricsData[t] || {})[metric]?.[latestQ];
          const formatted = (v !== undefined && v !== null)
            ? (metric === 'Gross Margin %'
                ? Number(v).toFixed(2) + '%'
                : '$' + (Number(v)/1_000_000_000).toFixed(2) + 'B')
            : 'N/A';
          html += `<td class="text-right py-3 px-4">${formatted}</td>`;
        });
        html += '</tr>';
      });

      html += '</tbody>';
      document.getElementById('metricsTable').innerHTML = html;
    }

    function renderTimeSeriesTables(){
      const metrics = ['Total Revenue', 'Operating Expense', 'Gross Margin %', 'EBIT', 'Net Income', 'Free Cash Flow'];
      const container = document.getElementById('timeSeriesTables');
      let html = '';
      _tickers.forEach(ticker => {
        html += `
          <div class="bg-white rounded-lg shadow-xl p-6 overflow-x-auto">
            <h3 class="text-2xl font-semibold text-gray-800 mb-4">${ticker} - 5 Quarter Time Series</h3>
            <table class="w-full">
              <thead><tr class="border-b-2 border-gray-300">
                <th class="text-left py-3 px-4"></th>
                ${_quarters.map(q => `<th class="text-right py-3 px-4">${q}</th>`).join('')}
              </tr></thead>
              <tbody>`;
        metrics.forEach(metric => {
          html += `<tr class="border-b border-gray-200 hover:bg-gray-50"><td class="py-3 px-4 font-medium">${metric}</td>`;
          _quarters.forEach(q => {
            const v = (_metricsData[ticker] || {})[metric]?.[q];
            let formatted = 'N/A';
            if (v !== undefined && v !== null) {
              formatted = (metric === 'Gross Margin %') ? Number(v).toFixed(2) + '%'
                        : '$' + (Number(v)/1_000_000).toFixed(1) + 'M';
            }
            html += `<td class="text-right py-3 px-4">${formatted}</td>`;
          });
          html += '</tr>';
        });
        html += `</tbody></table></div>`;
      });
      container.innerHTML = html;
    }

    // Enter-to-submit helpers
    document.getElementById('tickerInput').addEventListener('keypress', e => { if (e.key === 'Enter') resolveAndFind(); });
    document.getElementById('manualInput').addEventListener('keypress', e => { if (e.key === 'Enter') addCompany(); });
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


@app.route('/api/diagnostics', methods=['GET'])
def api_diagnostics():
    """Return diagnostic information about S3 connection and ticker map status"""
    return jsonify({
        "s3_loaded": _s3_loaded,
        "s3_error": _s3_error,
        "boto3_available": BOTO3_AVAILABLE,
        "ticker_count": len(_ticker_to_name),
        "company_name_count": len(_name_to_ticker),
        "s3_bucket": S3_BUCKET,
        "s3_prefix": S3_PREFIX,
        "aws_region": AWS_REGION,
        "aws_credentials_present": bool(os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"))
    })


# ============================================================
# STEP 3: Revised Peer Selection Route (with Mega 7 + eVTOL)
# ============================================================
@app.route('/api/find-peers', methods=['POST'])
def find_peers():
    """
    Revised peer selection:
      - Special cases:
          * Mega 7: if base in MEGA7 -> peers = other six members (TSLA included in the set).
          * eVTOL: if base in EVTOL_TICKERS -> peers = rest of eVTOL group.
      - General:
          * Build candidate list (S3 + optional OpenAI).
          * Keep only same-industry.
          * If both MCs present and ratio > 10x -> exclude.
          * Else include.
          * Sort by MC closeness, return up to PEER_LIMIT.
    """
    try:
        data = request.json or {}
        base_ticker = _normalize_peer_ticker((data.get('ticker') or '').upper().strip())
        if not base_ticker:
            return jsonify({'error': 'Ticker is required'}), 400

        base_prof = fetch_profile(base_ticker)
        base_ind  = base_prof.get("industry")
        base_mc   = base_prof.get("market_cap")

        # --- Special: Mega 7
        if base_ticker in MEGA7_TICKERS:
            peers = [p for p in MEGA7 if p["ticker"] != base_ticker]
            return jsonify({
                "primary_company": base_prof["ticker"],
                "industry": base_ind or "N/A",
                "peers": peers
            })

        # --- Special: eVTOL
        if base_ticker in EVTOL_TICKERS:
            peers = [p for p in EVTOL_GROUP if p["ticker"] != base_ticker]
            return jsonify({
                "primary_company": base_prof["ticker"],
                "industry": base_ind or "N/A",
                "peers": peers
            })

        # --- Candidate pool (merge S3 + OpenAI, then unique) ---
        cand = _s3_universe_candidates(base_ticker, base_ind or "", limit=120)
        more = _openai_candidates(base_ticker, count=16)
        seen = set()
        candidates = []
        for c in cand + more:
            ct = _normalize_peer_ticker(c.get("ticker"))
            if not ct or ct == base_ticker:
                continue
            if ct in seen:
                continue
            seen.add(ct)
            candidates.append({"ticker": ct, "name": c.get("name")})

        # --- Filter by rules ---
        valid = []
        for c in candidates:
            prof = fetch_profile(c["ticker"])
            c_ind = prof.get("industry")
            c_mc  = prof.get("market_cap")

            if not _same_industry(base_ind or "", c_ind or ""):
                continue
            if not _mc_ratio_ok(base_mc, c_mc, MARKET_CAP_RATIO_LIMIT):
                continue

            valid.append({
                "ticker": prof["ticker"],
                "name": prof.get("name") or c.get("name") or prof["ticker"],
                "market_cap": c_mc
            })

        # Sort by MC closeness (if MCs known), then trim
        def _score(v):
            mc = v.get("market_cap")
            if base_mc is None or mc is None:
                return float('inf')
            return abs((mc / base_mc) - 1.0)

        valid.sort(key=_score)
        peers_out = [{"ticker": v["ticker"], "name": v["name"]} for v in valid[:PEER_LIMIT]]

        return jsonify({
            "primary_company": base_prof["ticker"],
            "industry": base_ind or "N/A",
            "peers": peers_out
        })

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
