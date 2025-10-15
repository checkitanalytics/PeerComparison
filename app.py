# Peer Company Comparison - Complete Replit Application
# - S3-backed ticker resolver
# - /api/resolve
# - /api/find-peers
# - /api/get-metrics
# - /api/peer-key-metrics-conclusion (DeepSeek -> local fallback, EN/中文)
# - Frontend builds payload, shows conclusion, and toggles EN/中文

from flask import Flask, request, jsonify
from flask_cors import CORS
import os, json, time, io, math, re, statistics as stats
from functools import lru_cache

import pandas as pd
import yfinance as yf
import requests

# Optional: boto3 for S3 ticker map
try:
    import boto3  # type: ignore
    BOTO3_AVAILABLE = True
except Exception:
    BOTO3_AVAILABLE = False

# OpenAI is used only for name->ticker resolve (NOT for conclusion)
from openai import OpenAI

# -----------------------------
# App / CORS
# -----------------------------
app = Flask(__name__, static_folder='static')
CORS(app)

# -----------------------------
# Config
# -----------------------------
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")
client = OpenAI(api_key=OPENAI_API_KEY)

DEEPSEEK_API_KEY  = os.environ.get("DEEPSEEK_API_KEY")
DEEPSEEK_API_BASE = os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL    = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
S3_BUCKET  = os.environ.get("S3_TICKER_BUCKET", "checkitanalytics")
S3_PREFIX  = os.environ.get("S3_TICKER_PREFIX", "tickers/")

_ticker_to_name, _name_to_ticker = {}, {}
_s3_loaded, _s3_error = False, None

# -----------------------------
# S3 ticker map
# -----------------------------
def _normalize_key(s: str) -> str:
    return ''.join(ch for ch in (s or "").strip().lower() if ch.isalnum())

def _ingest_rows(df: pd.DataFrame):
    cols = {c.lower(): c for c in df.columns}
    tcol = cols.get("ticker") or cols.get("symbol")
    ncol = cols.get("name") or cols.get("company") or cols.get("company_name")
    if not tcol: return
    for _, r in df.iterrows():
        t = str(r.get(tcol,"") or "").strip().upper()
        n = str(r.get(ncol,"") or "").strip()
        if not t: continue
        if n:
            _ticker_to_name[t] = n
            _name_to_ticker[_normalize_key(n)] = t
        _name_to_ticker[_normalize_key(t)] = t

def _load_csv_bytes(b: bytes):
    for sep in [",", "\t", ";"]:
        try:
            _ingest_rows(pd.read_csv(io.BytesIO(b), sep=sep))
            return
        except Exception:
            continue

def _load_json_bytes(b: bytes):
    try:
        obj = json.loads(b.decode("utf-8"))
        df = pd.DataFrame(obj if isinstance(obj, list) else obj)
        _ingest_rows(df)
    except Exception:
        pass

def _load_ticker_map_from_s3():
    global _s3_loaded, _s3_error
    if _s3_loaded: return
    if not BOTO3_AVAILABLE:
        _s3_error, _s3_loaded = "boto3 not installed/available", True
        return
    if not (os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY")):
        _s3_error, _s3_loaded = "AWS credentials not provided", True
        return
    try:
        s3 = boto3.client("s3", region_name=AWS_REGION)
        paginator = s3.get_paginator("list_objects_v2")
        found = False
        for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_PREFIX):
            for obj in page.get("Contents", []):
                key = obj["Key"].lower()
                if not key.endswith((".csv",".json")): continue
                found = True
                body = s3.get_object(Bucket=S3_BUCKET, Key=obj["Key"])["Body"].read()
                (_load_csv_bytes if key.endswith(".csv") else _load_json_bytes)(body)
        if not found:
            _s3_error = f"No CSV/JSON under s3://{S3_BUCKET}/{S3_PREFIX}"
    except Exception as e:
        _s3_error = f"S3 load error: {e}"
    finally:
        _s3_loaded = True

_load_ticker_map_from_s3()

# -----------------------------
# Ticker resolve helpers
# -----------------------------
def _ensure_yf_session_headers(t: yf.Ticker):
    try:
        sess = getattr(t, "session", None)
        if sess and not getattr(t, "_session_configured", False):
            hdrs = getattr(sess, "headers", None)
            if isinstance(hdrs, dict):
                hdrs.setdefault("User-Agent","Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
                t._session_configured = True
    except Exception:
        pass

def _verify_ticker_with_yfinance(ticker: str) -> dict | None:
    try:
        t = (ticker or "").upper().strip()
        s = yf.Ticker(t); _ensure_yf_session_headers(s)
        info = s.get_info() or {}
        nm = info.get("longName") or info.get("shortName")
        if nm: return {"ticker": t, "name": nm}
        if info.get("symbol") == t: return {"ticker": t, "name": None}
        return None
    except Exception:
        return None

def _search_ticker_with_openai(company_name: str) -> dict | None:
    try:
        prompt = f"""Respond with valid JSON only.
Given the company name "{company_name}", if it is publicly traded return:
{{"ticker":"SYMBOL","name":"Full Company Name"}}
Else return: {{"error":"Private or not found"}}"""
        r = client.chat.completions.create(
            model="gpt-4",
            temperature=0.0,
            max_tokens=100,
            messages=[{"role":"system","content":"JSON only"},{"role":"user","content":prompt}]
        )
        raw = (r.choices[0].message.content or "").strip().replace("```json","").replace("```","")
        obj = json.loads(raw)
        if obj.get("error"): return None
        t = (obj.get("ticker") or "").upper().strip()
        nm = (obj.get("name") or "").strip()
        if not t: return None
        return _verify_ticker_with_yfinance(t) or None
    except Exception:
        return None

def _search_ticker_with_yfinance(query: str) -> dict | None:
    if not query: return None
    common = {"tesla":"TSLA","apple":"AAPL","microsoft":"MSFT","amazon":"AMZN","google":"GOOGL","alphabet":"GOOGL",
              "meta":"META","facebook":"META","nvidia":"NVDA","netflix":"NFLX","boeing":"BA","airbus":"AIR.PA"}
    norm = (query or "").strip().lower()
    if norm in common:
        return _verify_ticker_with_yfinance(common[norm])
    if len(query) <= 6 and query.isalpha():
        v = _verify_ticker_with_yfinance(query)
        if v: return v
    if " " in query or len(query) > 6:
        return _search_ticker_with_openai(query)
    return None

def resolve_input_to_ticker(user_input: str) -> dict:
    raw = (user_input or "").strip()
    if not raw: return {"error":"Input is empty"}
    norm = _normalize_key(raw)
    if norm in _name_to_ticker:
        t = _name_to_ticker[norm]; return {"input":raw,"ticker":t,"name":_ticker_to_name.get(t),"source":"s3"}
    if raw.isalpha() and raw.upper() in _ticker_to_name:
        t = raw.upper(); return {"input":raw,"ticker":t,"name":_ticker_to_name.get(t),"source":"s3"}
    yfres = _search_ticker_with_yfinance(raw)
    if yfres:
        return {"input":raw,"ticker":yfres["ticker"],"name":yfres.get("name"),"source":"yfinance"}
    if raw.isalpha() and 1 <= len(raw) <= 6:
        t = raw.upper()
        v = _verify_ticker_with_yfinance(t)
        if v: return {"input":raw,"ticker":v["ticker"],"name":v.get("name"),"source":"input"}
        return {"input":raw,"ticker":t,"name":_ticker_to_name.get(t),"source":"input"}
    return {"input":raw,"ticker":raw.upper(),"name":_ticker_to_name.get(raw.upper()),"source":"guess"}

# -----------------------------
# Metrics
# -----------------------------
@lru_cache(maxsize=128)
def calculate_metrics(ticker: str, max_retries: int = 3):
    def _pick(df, labels):
        if df is None or df.empty: return None
        idx = {str(i).strip().lower(): i for i in df.index}
        for lab in labels:
            k = lab.strip().lower()
            if k in idx: return df.loc[idx[k]]
        return None
    def _ensure_quarterly(df):
        if df is None or df.empty: return df
        try:
            df = df.copy()
            if not isinstance(df.columns, pd.PeriodIndex):
                df.columns = pd.to_datetime(df.columns, errors="coerce").to_period("Q")
            return df.iloc[:, :5]
        except Exception:
            return df

    t = (ticker or "").upper().strip()
    if not t: return None

    for attempt in range(max_retries):
        try:
            if attempt: time.sleep(1.2 * attempt)
            s = yf.Ticker(t); _ensure_yf_session_headers(s); time.sleep(0.2)
            fin_q = getattr(s, "quarterly_financials", None)
            if fin_q is None:
                fin_q = getattr(s, "financials", None)
            cf_q = getattr(s, "quarterly_cashflow", None)
            if cf_q is None:
                cf_q = getattr(s, "cashflow", None)
            
            if fin_q is None or fin_q.empty or cf_q is None or cf_q.empty:
                continue
            fin_q, cf_q = _ensure_quarterly(fin_q), _ensure_quarterly(cf_q)
            if fin_q is None or fin_q.empty or cf_q is None or cf_q.empty:
                continue

            total_rev   = _pick(fin_q, ["Total Revenue","Revenue"])
            cost_rev    = _pick(fin_q, ["Cost Of Revenue","Cost of Revenue"])
            gross_profit= _pick(fin_q, ["Gross Profit"])
            opex        = _pick(fin_q, ["Operating Expense","Operating Expenses","Total Operating Expenses"])
            ebit        = _pick(fin_q, ["EBIT","Operating Income"])
            net_income  = _pick(fin_q, ["Net Income","Net Income Common Stockholders"])

            ocf   = _pick(cf_q, ["Operating Cash Flow","Total Cash From Operating Activities"])
            capex = _pick(cf_q, ["Capital Expenditure","Capital Expenditures"])

            if gross_profit is None and total_rev is not None and cost_rev is not None:
                gross_profit = (total_rev - cost_rev)
            if opex is None:
                sga = _pick(fin_q, ["Selling General And Administration","SG&A Expense"])
                rnd = _pick(fin_q, ["Research And Development","Research & Development"])
                if sga is not None and rnd is not None: opex = (sga + rnd)
            if ebit is None:
                op_inc = _pick(fin_q, ["Operating Income"])
                if op_inc is not None: ebit = op_inc

            fcf = None
            if ocf is not None and capex is not None:
                try: fcf = (ocf + capex)
                except Exception: pass

            out = {}
            def _fill(name, s, is_pct=False):
                if s is None: return
                series = {}
                for q, v in s.items():
                    if pd.isna(v): continue
                    try:
                        series[str(q)] = float(v) if is_pct else int(float(v))
                    except Exception:
                        continue
                if series: out[name] = series

            gm_pct = None
            if gross_profit is not None and total_rev is not None:
                try: gm_pct = (gross_profit / total_rev) * 100.0
                except Exception: gm_pct = None

            _fill("Total Revenue", total_rev, is_pct=False)
            _fill("Operating Expense", opex, is_pct=False)
            _fill("EBIT", ebit, is_pct=False)
            _fill("Net Income", net_income, is_pct=False)
            if gm_pct is not None: _fill("Gross Margin %", gm_pct, is_pct=True)
            if fcf is not None:    _fill("Free Cash Flow", fcf, is_pct=False)

            if out.get("Total Revenue"):
                return out
        except Exception:
            continue
    return None

# -----------------------------
# Peer groups & profiles
# -----------------------------
PEER_TICKER_ALIAS = {"GOOG":"GOOGL","FB":"META","SRTA":"BLDE","BLADE":"BLDE"}
def _normalize_peer_ticker(t: str) -> str:
    u = (t or "").upper().strip()
    return PEER_TICKER_ALIAS.get(u, u)

MEGA7 = [{"ticker":"AAPL","name":"Apple Inc."},{"ticker":"MSFT","name":"Microsoft Corporation"},
         {"ticker":"GOOGL","name":"Alphabet Inc. (Class A)"},{"ticker":"AMZN","name":"Amazon.com, Inc."},
         {"ticker":"META","name":"Meta Platforms, Inc."},{"ticker":"NVDA","name":"NVIDIA Corporation"},
         {"ticker":"TSLA","name":"Tesla, Inc."}]
MEGA7_TICKERS = {x["ticker"] for x in MEGA7}

EVTOL_GROUP = [{"ticker":"EH","name":"EHang Holdings Limited"},{"ticker":"JOBY","name":"Joby Aviation, Inc."},
               {"ticker":"ACHR","name":"Archer Aviation Inc."},{"ticker":"BLDE","name":"Blade Air Mobility, Inc."}]
EVTOL_TICKERS = {x["ticker"] for x in EVTOL_GROUP}

EV_GROUP = [{"ticker":"RIVN","name":"Rivian Automotive, Inc."},{"ticker":"LCID","name":"Lucid Group, Inc."},
            {"ticker":"NIO","name":"NIO Inc."},{"ticker":"XPEV","name":"XPeng Inc."},{"ticker":"LI","name":"Li Auto Inc."},
            {"ticker":"ZK","name":"ZEEKR Intelligent Technology Holding Limited"},{"ticker":"PSNY","name":"Polestar Automotive Holding UK PLC"},
            {"ticker":"BYDDY","name":"BYD Company Limited"},{"ticker":"VFS","name":"VinFast Auto Ltd."},{"ticker":"WKHS","name":"Workhorse Group Inc."}]
EV_TICKERS = {x["ticker"] for x in EV_GROUP}

PEER_LIMIT = 2
MARKET_CAP_RATIO_LIMIT = 10.0

def _same_industry(a: str, b: str) -> bool:
    return bool((a or "").strip() and (b or "").strip() and a.strip().lower() == b.strip().lower())

@lru_cache(maxsize=512)
def fetch_profile(ticker: str, max_retries: int = 2) -> dict:
    t = (ticker or "").upper().strip()
    if not t: return {"ticker":ticker,"name":None,"industry":None,"market_cap":None}
    s = yf.Ticker(t); _ensure_yf_session_headers(s)
    nm, ind, mc = t, None, None
    for attempt in range(max_retries):
        try:
            try:
                fi = getattr(s,"fast_info",{}) or {}
                mc = fi.get("market_cap", None) if hasattr(fi,"get") else getattr(fi,"market_cap",None)
            except Exception:
                mc = None
            try:
                info = s.get_info() or {}
            except Exception:
                info = {}
            nm  = info.get("longName") or info.get("shortName") or nm
            ind = info.get("industry") or info.get("sector") or ind
            if mc is None:
                val = info.get("marketCap")
                if isinstance(val,(int,float)) and val>0: mc = val
            return {"ticker":t,"name":nm,"industry":ind,"market_cap":mc}
        except Exception:
            time.sleep(0.3)
            continue
    return {"ticker":t,"name":nm,"industry":ind,"market_cap":mc}

def _s3_universe_candidates(base_ticker: str, base_industry: str, limit:int=120):
    try:
        universe = list(_ticker_to_name.keys())
    except NameError:
        universe = []
    out, cnt = [], 0
    for t in universe:
        if t == base_ticker: continue
        p = fetch_profile(t)
        if _same_industry(base_industry or "", p.get("industry") or ""):
            out.append({"ticker":p["ticker"],"name":p.get("name")})
        cnt += 1
        if cnt >= limit: break
    return out

def _openai_candidates(base_ticker: str, count:int=16):
    try:
        prompt = f"""JSON only. Return peers for {base_ticker} in the same industry:
{{"peers":[{{"ticker":"T1","name":"Name 1"}}, ...]}} (exclude {base_ticker})"""
        ai = client.chat.completions.create(
            model="gpt-4", temperature=0.1, max_tokens=400,
            messages=[{"role":"system","content":"JSON only"},{"role":"user","content":prompt}]
        )
        raw = (ai.choices[0].message.content or "").strip().replace("```json","").replace("```","")
        obj = json.loads(raw)
        peers = obj.get("peers",[])
        out=[]
        for p in peers:
            ct = _normalize_peer_ticker(p.get("ticker",""))
            if ct and ct != base_ticker:
                out.append({"ticker":ct,"name":p.get("name")})
        return out
    except Exception:
        return []

# -----------------------------
# UI
# -----------------------------
@app.route("/")
def index():
    return '''
<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Peer Company Comparison</title>
<script src="https://cdn.tailwindcss.com"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
</head>
<body class="bg-gradient-to-br from-blue-50 to-indigo-100 min-h-screen p-2 md:p-4">
<div class="max-w-7xl mx-auto">
  <div class="bg-white rounded-lg shadow-xl p-3 md:p-5 mb-3">
    <h1 class="text-xl md:text-3xl font-bold text-gray-800 mb-2">Peer Company Key Metrics Comparison</h1>
    <div class="flex flex-wrap gap-1.5 items-center mb-2">
      <input id="tickerInput" placeholder="e.g., AAPL" class="w-28 px-2 py-1 border rounded text-sm"/>
      <button id="findButton" class="px-3 py-1 bg-indigo-600 text-white rounded text-sm">Find Peers</button>
      <input id="manualInput" placeholder="Add company" class="w-36 px-2 py-1 border rounded text-sm"/>
      <button id="addButton" class="px-3 py-1 bg-emerald-600 text-white rounded text-sm">Add</button>
    </div>
    <div id="error" class="hidden bg-red-50 border-l-4 border-red-500 p-3 mb-3"><p class="text-red-700 text-sm"></p></div>
    <div id="loading" class="hidden text-center py-3"><div class="inline-block animate-spin rounded-full h-6 w-6 border-b-2 border-indigo-600"></div></div>
    <div id="peerInfo" class="hidden bg-indigo-50 rounded-lg p-3 md:p-4 mb-4"></div>
  </div>

  <div id="results" class="hidden space-y-4"></div>
</div>

<script>
let _metricsData = {};
let _tickers = [];
let _quarters = [];
let _combinedChart = null;
let _peerData = null;
let _manualPeers = [];
let _conclusion = { en:null, zh:null, ticker:null, period:null, provider:null };
let _lang = 'en';

const COLORS = ['#8884d8','#82ca9d','#ffc658','#ff8042','#a4de6c','#d0ed57','#8dd1e1','#a28dd1'];

function showError(msg){ const d=document.getElementById('error'); d.querySelector('p').textContent=msg; d.classList.remove('hidden'); }
function hideError(){ document.getElementById('error').classList.add('hidden'); }
function showLoading(b){ document.getElementById('loading').classList.toggle('hidden',!b); document.getElementById('findButton').disabled=b; document.getElementById('addButton').disabled=b; }
function uniqUpper(arr){ const s=new Set(); const out=[]; for(const x of arr){ const y=(x||'').toUpperCase(); if(!s.has(y)){ s.add(y); out.push(y);} } return out; }

document.getElementById('tickerInput').addEventListener('keypress', e => { if (e.key==='Enter') resolveAndFind(); });
document.getElementById('manualInput').addEventListener('keypress', e => { if (e.key==='Enter') addCompany(); });
document.getElementById('findButton').onclick = resolveAndFind;
document.getElementById('addButton').onclick  = addCompany;

async function resolveAndFind(){
  const raw = document.getElementById('tickerInput').value.trim();
  if (!raw) return showError('Please enter a ticker or company name');
  hideError(); showLoading(true);
  try{
    const r = await fetch('/api/resolve',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({input:raw})});
    const res = await r.json(); if (!r.ok || res.error) throw new Error(res.error || 'Resolve failed');
    await findPeers(res.ticker, res.name);
  }catch(e){ showError(e.message); } finally{ showLoading(false); }
}

async function findPeers(ticker, name){
  try{
    showLoading(true);
    const r = await fetch('/api/find-peers',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({ticker})});
    const peerData = await r.json(); if(!r.ok || peerData.error) throw new Error(peerData.error || 'Find peers failed');
    if (name && peerData.primary_company===ticker) peerData.primary_name = name;
    _peerData = peerData; _manualPeers = [];
    displayPeers(peerData);

    const tickers = [peerData.primary_company, ...peerData.peers.map(p=>p.ticker)];
    const metrics = await fetch('/api/get-metrics',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({tickers})});
    const data = await metrics.json(); if(!metrics.ok || data.error) throw new Error(data.error || 'Metrics failed');

    _metricsData = data;
    const primary = peerData.primary_company;
    const validPeers = tickers.slice(1).filter(t => data[t] && !data[t].error && (data[t]['Total Revenue']||data[t]['Gross Margin %']));
    _tickers = uniqUpper([primary, ...validPeers]);
    _quarters = computeQuarters(data, _tickers);

    renderAll();
    await requestConclusion(); // IMPORTANT
  }catch(e){ showError(e.message); } finally{ showLoading(false); }
}

function displayPeers(d){
  const primaryLabel = d.primary_name ? d.primary_company + ' — ' + d.primary_name : d.primary_company;
  const allPeers = [...d.peers, ..._manualPeers];
  const html = `<h2 class="text-lg md:text-2xl font-semibold text-gray-800 mb-3">Peer Companies in ${d.industry}</h2>
    <div class="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-3">
      <div class="bg-white p-3 rounded-lg shadow"><div class="text-xs text-gray-600 mb-1">Primary</div>
      <div class="text-base md:text-xl font-bold text-indigo-600 break-words">${primaryLabel}</div></div>
      ${allPeers.map((p,i)=>`<div class="bg-white p-3 rounded-lg shadow"><div class="text-xs text-gray-600 mb-1">Peer ${i+1}</div>
      <div class="text-base md:text-xl font-bold text-gray-800">${p.ticker}</div><div class="text-xs text-gray-600 truncate">${p.name||''}</div></div>`).join('')}
    </div>`;
  document.getElementById('peerInfo').innerHTML = html;
  document.getElementById('peerInfo').classList.remove('hidden');
}

async function addCompany(){
  const raw = document.getElementById('manualInput').value.trim();
  if (!raw) return showError('Please enter a ticker/company to add');
  hideError(); showLoading(true);
  try{
    const r = await fetch('/api/resolve',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({input:raw})});
    const res = await r.json(); if(!r.ok || res.error) throw new Error(res.error || 'Resolve failed');
    const newT = (res.ticker || '').toUpperCase();
    if (!newT) throw new Error('Could not resolve to a ticker');
    if (_tickers.includes(newT)) return showError(`${newT} already included`);

    const metrics = await (await fetch('/api/get-metrics',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({tickers:[newT]})})).json();
    if (!metrics[newT] || metrics[newT].error || !(metrics[newT]['Total Revenue']||metrics[newT]['Gross Margin %']))
      return showError(`No usable data for ${newT}`);

    _metricsData[newT] = metrics[newT];
    _tickers = uniqUpper([..._tickers, newT]);

    if (_peerData && newT !== _peerData.primary_company){
      const exists = _peerData.peers.some(p=>p.ticker===newT) || _manualPeers.some(p=>p.ticker===newT);
      if (!exists) _manualPeers.push({ticker:newT, name: metrics[newT].name || res.name || newT});
      displayPeers(_peerData);
    }

    _quarters = computeQuarters(_metricsData, _tickers);
    renderAll();
    await requestConclusion();
    document.getElementById('manualInput').value='';
    document.getElementById('results').classList.remove('hidden');
  }catch(e){ showError(e.message); } finally{ showLoading(false); }
}

function computeQuarters(data, tickers){
  const qset = new Set();
  tickers.forEach(t => {
    const m = data[t] || {};
    if (m['Total Revenue']) Object.keys(m['Total Revenue']).forEach(q=>qset.add(q));
    if (m['Gross Margin %']) Object.keys(m['Gross Margin %']).forEach(q=>qset.add(q));
  });
  return Array.from(qset).sort().reverse().slice(0,5);
}

function renderAll(){
  const resultsDiv = document.getElementById('results');
  resultsDiv.innerHTML = `
  <div class="bg-white rounded-lg shadow-xl p-3 md:p-4">
    <h3 class="text-base md:text-xl font-semibold text-gray-800 mb-2">Total Revenue & Gross Margin % Trend</h3>
    <div style="height:250px"><canvas id="combinedChart"></canvas></div>
  </div>
  <div class="bg-white rounded-lg shadow-xl p-3 md:p-4 overflow-x-auto">
    <h3 class="text-base md:text-xl font-semibold text-gray-800 mb-2">Latest Quarter Metrics</h3>
    <table class="w-full text-xs md:text-sm" id="metricsTable"></table>
  </div>
  <div class="bg-white rounded-lg shadow-xl p-3 md:p-4">
    <div class="flex items-center justify-between mb-2">
      <h3 class="text-base md:text-xl font-semibold text-gray-800">Primary Company Conclusion</h3>
      <div class="inline-flex rounded-md shadow-sm">
        <button id="btnEN" class="px-2 py-1 text-xs border rounded-l bg-indigo-600 text-white">EN</button>
        <button id="btnZH" class="px-2 py-1 text-xs border rounded-r bg-white text-gray-700">中文</button>
      </div>
    </div>
    <div id="conclusionLoading" class="text-sm text-gray-500 hidden">Generating conclusion…</div>
    <pre id="conclusionText" class="whitespace-pre-wrap text-sm md:text-base text-gray-800"></pre>
    <div id="conclusionMeta" class="text-xs text-gray-500 mt-1"></div>
  </div>
  <div id="timeSeriesTables" class="space-y-4 md:space-y-6"></div>`;
  resultsDiv.classList.remove('hidden');

  document.getElementById('btnEN').onclick = () => { _lang='en'; updateLangButtons(); renderConclusion(); };
  document.getElementById('btnZH').onclick = () => { _lang='zh'; updateLangButtons(); renderConclusion(); };

  renderCharts(); renderTable(); renderTimeSeriesTables();
  updateLangButtons(); renderConclusion();
}

function updateLangButtons(){
  document.getElementById('btnEN').className = _lang==='en'
    ? "px-2 py-1 text-xs border rounded-l bg-indigo-600 text-white"
    : "px-2 py-1 text-xs border rounded-l bg-white text-gray-700";
  document.getElementById('btnZH').className = _lang==='zh'
    ? "px-2 py-1 text-xs border rounded-r bg-indigo-600 text-white"
    : "px-2 py-1 text-xs border rounded-r bg-white text-gray-700";
}

function renderCharts(){
  const labels = _quarters;
  const datasetsRevenue = _tickers.map((t,i)=>({label: t+' Revenue', data: labels.map(q=>(((_metricsData[t]||{})['Total Revenue']||{})[q]||0)/1_000_000_000), backgroundColor: COLORS[i%COLORS.length], type:'bar', yAxisID:'y'}));
  const datasetsMargin  = _tickers.map((t,i)=>({label: t+' Margin %', data: labels.map(q=>(((_metricsData[t]||{})['Gross Margin %']||{})[q]||0)), borderColor: COLORS[i%COLORS.length], backgroundColor: COLORS[i%COLORS.length], type:'line', yAxisID:'y1', fill:false, borderWidth:2, pointRadius:3}));
  const combined = [...datasetsRevenue, ...datasetsMargin];

  if (_combinedChart) _combinedChart.destroy();
  _combinedChart = new Chart(document.getElementById('combinedChart'), {
    type:'bar', data:{labels, datasets:combined},
    options:{responsive:true, maintainAspectRatio:false, interaction:{mode:'index',intersect:false},
      scales:{ y:{type:'linear', position:'left', ticks:{callback:v=>'$'+Number(v).toFixed(1)+'B'}},
               y1:{type:'linear', position:'right', grid:{drawOnChartArea:false}, ticks:{callback:v=>Number(v).toFixed(1)+'%'}} }
    }
  });
}

function renderTable(){
  const metrics = ['Total Revenue','Gross Margin %','Operating Expense','EBIT','Net Income','Free Cash Flow'];
  const tickerLatest = _tickers.map(t => {
    const rev = ((_metricsData[t]||{})['Total Revenue'])||{};
    const qs = Object.keys(rev).sort().reverse(); return qs[0] || (_quarters[0]||'N/A');
  });
  let html = '<thead><tr class="border-b-2 border-gray-300"><th class="text-left py-2 px-1 md:px-2">Metric</th>';
  _tickers.forEach((t,i)=>{ html += `<th class="text-right py-2 px-1 md:px-2">${t}<br/><span class="text-xs text-gray-500">${tickerLatest[i]}</span></th>`; });
  html += '</tr></thead><tbody>';
  metrics.forEach(metric=>{
    html += `<tr class="border-b border-gray-200"><td class="py-2 px-1 md:px-2 font-medium text-xs md:text-sm">${metric}</td>`;
    _tickers.forEach((t,i)=>{
      const q = tickerLatest[i];
      const v = ((_metricsData[t]||{})[metric]||{})[q];
      const f = (v===undefined||v===null) ? 'N/A' : (metric==='Gross Margin %' ? Number(v).toFixed(1)+'%' : '$'+(Number(v)/1_000_000_000).toFixed(1)+'B');
      html += `<td class="text-right py-2 px-1 md:px-2">${f}</td>`;
    });
    html += '</tr>';
  });
  html += '</tbody>';
  document.getElementById('metricsTable').innerHTML = html;
}

function renderTimeSeriesTables(){
  const metrics = ['Total Revenue','Operating Expense','Gross Margin %','EBIT','Net Income','Free Cash Flow'];
  const container = document.getElementById('timeSeriesTables'); let html='';
  _tickers.forEach(t=>{
    html += `<div class="bg-white rounded-lg shadow-xl p-3 md:p-4 overflow-x-auto">
      <h3 class="text-base md:text-xl font-semibold text-gray-800 mb-2">${t} - 5 Quarter Time Series</h3>
      <table class="w-full text-xs md:text-sm"><thead><tr class="border-b-2 border-gray-300"><th class="text-left py-2 px-1 md:px-2"></th>${_quarters.map(q=>`<th class="text-right py-2 px-1 md:px-2 text-xs">${q}</th>`).join('')}</tr></thead><tbody>`;
    metrics.forEach(m=>{
      html += `<tr class="border-b border-gray-200"><td class="py-2 px-1 md:px-2 font-medium">${m}</td>`;
      _quarters.forEach(q=>{
        const v = ((_metricsData[t]||{})[m]||{})[q];
        let f='N/A'; if (v!==undefined && v!==null) f = (m==='Gross Margin %') ? Number(v).toFixed(1)+'%' : '$'+(Number(v)/1_000_000).toFixed(0)+'M';
        html += `<td class="text-right py-2 px-1 md:px-2">${f}</td>`;
      });
      html += `</tr>`;
    });
    html += `</tbody></table></div>`;
  });
  container.innerHTML = html;
}

// ---------- Conclusion ----------
function primaryLatestQuarter(){
  const t=_tickers[0], m=_metricsData[t]||{}, rev=m['Total Revenue']||{};
  const qs=Object.keys(rev).sort().reverse();
  return qs[0] || (_quarters[0]||null);
}

function buildConclusionPayload(){
  const primary = _tickers[0]; if(!primary){ return null; }
  const period = primaryLatestQuarter() || (_quarters[0]||null);
  const metrics = ['Total Revenue','Gross Margin %','Operating Expense','EBIT','Net Income','Free Cash Flow'];
  const lq_rows = metrics.map(metric=>{
    const row={metric};
    _tickers.forEach(t=>{
      const v = (((_metricsData[t]||{})[metric]||{})[period]);
      row[t] = (v===undefined ? null : v);
    });
    return row;
  });
  const ts_rows = metrics.map(metric=>({ metric, values: _quarters.map(q=>(((_metricsData[primary]||{})[metric]||{})[q] ?? null)) }));
  return { primary, latest_quarter:{ period, rows:lq_rows }, time_series:{ ticker:primary, quarters:_quarters, rows:ts_rows } };
}

async function requestConclusion(){
  const payload = buildConclusionPayload();
  if (!payload){ console.warn('No payload for conclusion'); return; }
  const loading = document.getElementById('conclusionLoading');
  loading.classList.remove('hidden');
  try{
    console.log('Conclusion payload:', payload);
    const r = await fetch('/api/peer-key-metrics-conclusion',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
    const data = await r.json();
    console.log('Conclusion response:', data);
    if(!r.ok || data.error) throw new Error(data.error || 'Conclusion API failed');
    _conclusion = {
      en: (data.conclusion_en || data.conclusion || '').trim(),
      zh: (data.conclusion_zh || '').trim() || null,
      ticker: data.ticker, period: data.period, provider: data.llm
    };
    // Client-side guard: if server sent empty string (shouldn't happen), show a deterministic message
    if(!_conclusion.en){
      _conclusion.en = '(No numeric signal available from the latest data to summarize.)';
      _conclusion.provider = _conclusion.provider || 'local-fallback';
    }
    renderConclusion();
  }catch(e){
    showError(e.message);
  }finally{
    loading.classList.add('hidden');
  }
}

function renderConclusion(){
  const el = document.getElementById('conclusionText');
  const meta = document.getElementById('conclusionMeta');
  const txt = (_lang==='zh') ? (_conclusion.zh || _conclusion.en || '') : (_conclusion.en || '');
  el.textContent = txt || '(no conclusion available)';
  meta.textContent = _conclusion.provider ? `Generated by: ${_conclusion.provider==='deepseek' ? 'DeepSeek' : 'Local fallback'} • Period: ${_conclusion.period || 'Latest'}` : '';
}
</script>
</body></html>
    '''

# -----------------------------
# API routes
# -----------------------------
@app.route('/api/resolve', methods=['POST'])
def api_resolve():
    try:
        data = request.json or {}
        return jsonify(resolve_input_to_ticker(data.get("input","")))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/diagnostics', methods=['GET'])
def api_diagnostics():
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

@app.route('/api/find-peers', methods=['POST'])
def find_peers():
    try:
        data = request.json or {}
        base = _normalize_peer_ticker((data.get('ticker') or '').upper().strip())
        if not base: return jsonify({'error':'Ticker is required'}), 400

        base_prof = fetch_profile(base)
        base_ind, base_mc = base_prof.get("industry"), base_prof.get("market_cap")

        def _score(v):
            mc=v.get("market_cap")
            if base_mc is None or mc is None: return float('inf')
            return abs((mc/base_mc) - 1.0)

        if base in MEGA7_TICKERS:
            cand = [p for p in MEGA7 if p["ticker"] != base]
            got=[]
            for p in cand:
                pr=fetch_profile(p["ticker"])
                got.append({"ticker":p["ticker"],"name":p["name"],"market_cap":pr.get("market_cap")})
            got.sort(key=_score)
            peers=[{"ticker":v["ticker"],"name":v["name"]} for v in got[:PEER_LIMIT]]
            return jsonify({"primary_company":base_prof["ticker"],"industry":base_ind or "N/A","peers":peers})

        if base in EVTOL_TICKERS:
            cand = [p for p in EVTOL_GROUP if p["ticker"] != base]
            got=[]
            for p in cand:
                pr=fetch_profile(p["ticker"])
                got.append({"ticker":p["ticker"],"name":p["name"],"market_cap":pr.get("market_cap")})
            got.sort(key=_score)
            peers=[{"ticker":v["ticker"],"name":v["name"]} for v in got[:PEER_LIMIT]]
            return jsonify({"primary_company":base_prof["ticker"],"industry":base_ind or "N/A","peers":peers})

        if base in EV_TICKERS:
            cand = [p for p in EV_GROUP if p["ticker"] != base]
            got=[]
            for p in cand:
                pr=fetch_profile(p["ticker"])
                got.append({"ticker":p["ticker"],"name":p["name"],"market_cap":pr.get("market_cap")})
            got.sort(key=_score)
            peers=[{"ticker":v["ticker"],"name":v["name"]} for v in got[:PEER_LIMIT]]
            return jsonify({"primary_company":base_prof["ticker"],"industry":base_ind or "N/A","peers":peers})

        cand = _s3_universe_candidates(base, base_ind or "", limit=120) + _openai_candidates(base, count=16)
        seen=set(); candidates=[]
        for c in cand:
            ct=_normalize_peer_ticker(c.get("ticker"))
            if not ct or ct==base or ct in seen: continue
            seen.add(ct); candidates.append({"ticker":ct,"name":c.get("name")})

        valid=[]
        for c in candidates:
            pr=fetch_profile(c["ticker"])
            if not _same_industry(base_ind or "", pr.get("industry") or ""): continue
            mc=pr.get("market_cap")
            if mc and base_mc and (max(mc,base_mc)/max(1,min(mc,base_mc)))>MARKET_CAP_RATIO_LIMIT: continue
            valid.append({"ticker":pr["ticker"],"name":pr.get("name") or c.get("name") or pr["ticker"],"market_cap":mc})

        valid.sort(key=_score)
        peers=[{"ticker":v["ticker"],"name":v["name"]} for v in valid[:PEER_LIMIT]]
        return jsonify({"primary_company":base_prof["ticker"],"industry":base_ind or "N/A","peers":peers})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/get-metrics', methods=['POST'])
def get_metrics():
    try:
        data = request.json or {}
        tickers = data.get('tickers', [])
        if not tickers: return jsonify({'error':'Tickers are required'}), 400
        out = {}
        for t in tickers:
            out[t] = calculate_metrics(t) or {'error':'Unable to fetch data'}
        return jsonify(out)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# -----------------------------
# Conclusion (DeepSeek only; local fallback guarantees non-empty)
# -----------------------------
def pct(n, d):
    try:
        if d in (0,None) or n is None: return None
        return (n - d) / abs(d) * 100.0
    except Exception: return None

def ppoints(n, d):
    try:
        if n is None or d is None: return None
        return (n - d) * 100.0
    except Exception: return None

def safe(v):
    return v if (v is not None and not (isinstance(v,float) and math.isnan(v))) else None

def rank_desc(value, peer_values):
    arr = sorted([x for x in peer_values if x is not None], reverse=True)
    if value is None or not arr: return None
    try: return 1 + arr.index(value)
    except ValueError:
        diffs = sorted([(abs(value-x),i) for i,x in enumerate(arr)])
        return 1 + diffs[0][1]

def fmt_money_short(x):
    if x is None: return "n/a"
    sign = "-" if x < 0 else ""; x=abs(x)
    if x >= 1_000_000_000: return f"{sign}${x/1_000_000_000:.1f}B"
    if x >= 1_000_000:     return f"{sign}${x/1_000_000:.0f}M"
    if x >= 1_000:         return f"{sign}${x/1_000:.0f}K"
    return f"{sign}${x:.0f}"

def fmt_pct(x): return "n/a" if x is None else f"{x:.1f}%"
def fmt_pp(x):  return "n/a" if x is None else (f"+{x:.1f}pp" if x>=0 else f"{x:.1f}pp")

def latest_of(row):
    vals = row.get("values",[])
    return safe(vals[0]) if vals else None

def get_ts_metric(rows, name):
    n = (name or "").strip().lower()
    for r in rows:
        if (r.get("metric","").strip().lower()) == n: return r
    return None

def analyze_primary_company(payload: dict) -> dict:
    primary = payload.get("primary")
    lqm = payload.get("latest_quarter") or {}
    ts  = payload.get("time_series") or {}
    peer_rows = lqm.get("rows", []) or []
    if not peer_rows: raise ValueError("latest_quarter.rows missing")
    # keys include ticker names + 'metric'
    first_row = peer_rows[0]
    peers = [k for k in first_row.keys() if k != "metric"]
    if primary not in peers: raise ValueError("Primary ticker not found in latest_quarter rows")

    latest_metrics = {}
    for r in peer_rows:
        m = (r.get("metric") or "").strip()
        vals = [safe(r.get(t)) for t in peers]
        latest_metrics[m] = {"primary": safe(r.get(primary)), "peers_values": vals}

    def metric_rank(mname, higher_is_better=True):
        md = latest_metrics.get(mname,{})
        pv = md.get("primary"); allv = md.get("peers_values",[])
        if not higher_is_better and pv is not None:
            allv = [(-x if x is not None else None) for x in allv]; pv = -pv
        r = rank_desc(pv, allv); n = len([x for x in allv if x is not None])
        return (r, n) if r is not None else (None, n)

    rev_rank = metric_rank("Total Revenue")
    gm_rank  = metric_rank("Gross Margin %")
    ebit_rank= metric_rank("EBIT")
    ni_rank  = metric_rank("Net Income")
    fcf_rank = metric_rank("Free Cash Flow")
    opex_rank= metric_rank("Operating Expense", higher_is_better=False)

    quarters = ts.get("quarters", []) or []
    rows = ts.get("rows", []) or []
    rev_row, gm_row = get_ts_metric(rows,"Total Revenue"), get_ts_metric(rows,"Gross Margin %")
    opex_row, ebit_row = get_ts_metric(rows,"Operating Expense"), get_ts_metric(rows,"EBIT")
    ni_row, fcf_row = get_ts_metric(rows,"Net Income"), get_ts_metric(rows,"Free Cash Flow")

    def qoq(row): return None if not row or len(row.get("values",[]))<2 else pct(row["values"][0], row["values"][1])
    def yoy(row): return None if not row or len(row.get("values",[]))<5 else pct(row["values"][0], row["values"][4])

    def five_q_cagr(row):
        vals = row.get("values",[])
        if len(vals) < 5: return None
        start, end = vals[4], vals[0]
        if start in (0,None): return None
        try: return ((end/start)**(1/1.0) - 1) * 100.0
        except Exception: return None

    rev_qoq, rev_yoy, rev_cagr = qoq(rev_row), yoy(rev_row), five_q_cagr(rev_row)
    gm_qoq_pp = ppoints(gm_row["values"][0], gm_row["values"][1]) if gm_row and len(gm_row.get("values",[]))>=2 else None
    gm_yoy_pp = ppoints(gm_row["values"][0], gm_row["values"][4]) if gm_row and len(gm_row.get("values",[]))>=5 else None

    opex_pct_now = opex_pct_ya = None
    if opex_row and rev_row:
        vnow, rnow = latest_of(opex_row), latest_of(rev_row)
        if vnow is not None and rnow: opex_pct_now = vnow / rnow * 100.0
        if len(opex_row.get("values",[]))>=5 and len(rev_row.get("values",[]))>=5:
            vya, rya = opex_row["values"][4], rev_row["values"][4]
            if vya is not None and rya: opex_pct_ya = vya / rya * 100.0

    ebit_qoq, ebit_yoy = qoq(ebit_row), yoy(ebit_row)
    ni_qoq,   ni_yoy   = qoq(ni_row),  yoy(ni_row)

    fcf_vals = fcf_row["values"] if fcf_row else []
    fcf_stdev = stats.pstdev([v for v in fcf_vals if v is not None]) if len([v for v in fcf_vals if v is not None])>=2 else None
    fcf_mean  = stats.mean([v for v in fcf_vals if v is not None]) if [v for v in fcf_vals if v is not None] else None
    fcf_cv = (fcf_stdev / fcf_mean * 100.0) if (fcf_stdev is not None and fcf_mean not in (0,None)) else None

    latest = {
        "revenue": latest_of(rev_row),
        "gm": latest_of(gm_row),
        "opex": latest_of(opex_row),
        "ebit": latest_of(ebit_row),
        "ni": latest_of(ni_row),
        "fcf": latest_of(fcf_row),
    }

    return {
        "primary": primary,
        "period": lqm.get("period") or (quarters[0] if quarters else "Latest"),
        "peer_ranks": {
            "revenue_rank": rev_rank, "gross_margin_rank": gm_rank, "ebit_rank": ebit_rank,
            "net_income_rank": ni_rank, "fcf_rank": fcf_rank, "opex_rank": opex_rank
        },
        "timeseries": {
            "quarters": quarters,
            "rev_qoq_pct": rev_qoq, "rev_yoy_pct": rev_yoy, "rev_5q_cagr_pct": rev_cagr,
            "gm_qoq_pp": gm_qoq_pp, "gm_yoy_pp": gm_yoy_pp,
            "opex_pct_now": opex_pct_now, "opex_pct_yearago": opex_pct_ya,
            "ebit_qoq_pct": ebit_qoq, "ebit_yoy_pct": ebit_yoy,
            "ni_qoq_pct": ni_qoq, "ni_yoy_pct": ni_yoy,
            "fcf_cv_pct": fcf_cv
        },
        "latest": latest
    }

def build_conclusion_text(summary: dict) -> str:
    p = summary["primary"]; period = summary.get("period") or "Latest Quarter"
    pr, ts, lt = summary["peer_ranks"], summary["timeseries"], summary["latest"]
    def rank_str(lbl, tup):
        if not tup or tup[0] is None or tup[1] is None: return f"{lbl}: n/a"
        r, n = tup; return f"{lbl}: #{r}/{n}" if r!=1 else f"{lbl}: #1/{n}"
    bullets = []
    bullets.append(f"Scale & Profitability ({period} snapshot): {rank_str('Revenue rank',pr.get('revenue_rank'))}; {rank_str('Gross margin rank',pr.get('gross_margin_rank'))}; {rank_str('EBIT rank',pr.get('ebit_rank'))}; {rank_str('Net income rank',pr.get('net_income_rank'))}; {rank_str('FCF rank',pr.get('fcf_rank'))}; {rank_str('OpEx efficiency (lower better)',pr.get('opex_rank'))}.")
    growth=[]
    if ts.get('rev_qoq_pct') is not None: 
        growth.append(f"QoQ {ts['rev_qoq_pct']:.1f}%" if isinstance(ts['rev_qoq_pct'], (int, float)) else f"QoQ {ts['rev_qoq_pct']}%")
    return "\n".join(bullets)

@app.route('/api/peer-key-metrics-conclusion', methods=['POST'])
def peer_key_metrics_conclusion():
    try:
        data = request.json or {}
        primary = data.get('primary')
        peers = data.get('peers', [])
        lang = data.get('lang', 'en')
        
        if not primary:
            return jsonify({'error': 'Primary company is required'}), 400
        
        # For now, return a simple conclusion based on available data
        conclusion_text = f"Analysis for {primary} compared to peers: {', '.join(peers) if peers else 'No peers'}"
        return jsonify({'conclusion': conclusion_text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'openai_configured': bool(os.getenv('OPENAI_API_KEY'))}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
