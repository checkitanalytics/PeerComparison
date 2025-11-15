# main.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import time
import math
import statistics as stats
from functools import lru_cache
import re
from datetime import datetime

import pandas as pd
import yfinance as yf
import requests
import boto3
from botocore.exceptions import ClientError

# -----------------------------
# App / CORS
# -----------------------------
app = Flask(__name__, static_folder='static')
CORS(app)

# -----------------------------
# DeepSeek config (ONLY model used)
# -----------------------------
DEEPSEEK_API_KEY  = os.environ.get("DEEPSEEK_API_KEY")
DEEPSEEK_API_BASE = os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL    = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

# -----------------------------
# Perplexity config (fallback)
# -----------------------------
PERPLEXITY_API_KEY = os.environ.get("PERPLEXITY_API_KEY")
PERPLEXITY_MODEL   = os.environ.get("PERPLEXITY_MODEL", "llama-3.1-sonar-small-128k-online")

# -----------------------------
# S3 config for ticker mapping
# -----------------------------
BUCKET_NAME = os.environ.get("CHECKIT_BUCKET", "checkitanalytics")
TICKER_MAP_KEY = os.environ.get("TICKER_MAP_KEY", "tickers/ccm_link.xlsx")

s3_client = boto3.client("s3")

# ============================================================
# Helpers
# ============================================================
def _ensure_yf_session_headers(t: yf.Ticker):
    """
    Ensure yfinance session has a user-agent header set to avoid blocking.
    """
    try:
        sess = getattr(t, "session", None)
        if sess and not getattr(t, "_session_configured", False):
            hdrs = getattr(sess, "headers", None)
            if isinstance(hdrs, dict):
                hdrs.setdefault(
                    "User-Agent",
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                )
                t._session_configured = True
    except Exception:
        pass


def fmt_money_short(x):
    if x is None:
        return "n/a"
    sign = "-" if x < 0 else ""
    x = abs(x)
    if x >= 1_000_000_000:
        return f"{sign}${x/1_000_000_000:.1f}B"
    if x >= 1_000_000:
        return f"{sign}${x/1_000_000:.0f}M"
    if x >= 1_000:
        return f"{sign}${x/1_000:.0f}K"
    return f"{sign}${x:.0f}"


def deepseek_chat(messages, temperature=0.1, timeout=40, max_retries=2) -> str | None:
    """
    DeepSeek chat wrapper with retries. Returns text or None.
    """
    if not DEEPSEEK_API_KEY:
        return None

    payload = {
        "model": DEEPSEEK_MODEL,
        "temperature": temperature,
        "messages": messages,
    }

    for attempt in range(max_retries):
        try:
            r = requests.post(
                f"{DEEPSEEK_API_BASE}/chat/completions",
                headers={
                    "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=timeout,
            )
            r.raise_for_status()
            data = r.json()
            return (data.get("choices") or [{}])[0].get("message", {}).get("content", "").strip() or None
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            print(f"[ERROR] DeepSeek API failed after {max_retries} attempts: {e}")
            return None
    return None


def perplexity_chat(messages, temperature=0.2, timeout=30, max_retries=2) -> str | None:
    """
    Perplexity AI fallback wrapper with retries. Returns text or None.
    """
    if not PERPLEXITY_API_KEY:
        return None

    payload = {
        "model": PERPLEXITY_MODEL,
        "temperature": temperature,
        "messages": messages,
        "top_p": 0.9,
        "stream": False,
    }

    for attempt in range(max_retries):
        try:
            r = requests.post(
                "https://api.perplexity.ai/chat/completions",
                headers={
                    "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=timeout,
            )
            r.raise_for_status()
            data = r.json()
            return (data.get("choices") or [{}])[0].get("message", {}).get("content", "").strip() or None
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            print(f"[ERROR] Perplexity API failed after {max_retries} attempts: {e}")
            return None
    return None


# ============================================================
# Peer groups & profiles
# ============================================================
PEER_TICKER_ALIAS = {
    "GOOG": "GOOGL",
    "FB": "META",
    "SRTA": "BLDE",
    "BLADE": "BLDE",
}

def _normalize_peer_ticker(t: str) -> str:
    u = (t or "").upper().strip()
    return PEER_TICKER_ALIAS.get(u, u)

_norm_ticker = _normalize_peer_ticker  # alias for backward compat

# --- Magnificent 7 Tech Giants ---
MEGA7 = [
    {"ticker": "AAPL", "name": "Apple Inc."},
    {"ticker": "MSFT", "name": "Microsoft Corporation"},
    {"ticker": "GOOGL", "name": "Alphabet Inc."},
    {"ticker": "AMZN", "name": "Amazon.com, Inc."},
    {"ticker": "META", "name": "Meta Platforms, Inc."},
    {"ticker": "NVDA", "name": "NVIDIA Corporation"},
    {"ticker": "TSLA", "name": "Tesla, Inc."},
]
MEGA7_TICKERS = {x["ticker"] for x in MEGA7}

# --- eVTOL / Urban Air Mobility ---
EVTOL_GROUP = [
    {"ticker": "EH", "name": "EHang Holdings Limited"},
    {"ticker": "JOBY", "name": "Joby Aviation, Inc."},
    {"ticker": "ACHR", "name": "Archer Aviation Inc."},
    {"ticker": "BLDE", "name": "Blade Air Mobility, Inc."},
    {"ticker": "LILM", "name": "Lilium N.V."},
    {"ticker": "EVTL", "name": "Vertical Aerospace"},
    {"ticker": "HOVR", "name": "New Horizon Aircraft"},
    {"ticker": "SPR", "name": "Spirit Aerosystems Holdings"},
    {"ticker": "ESLT", "name": "Elbit Systems"},
    {"ticker": "AIR", "name": "AAR Corp."},
    {"ticker": "ERJ", "name": "Embraer S.A."},
]
EVTOL_TICKERS = {x["ticker"] for x in EVTOL_GROUP}

# --- Electric Vehicle Manufacturers ---
EV_GROUP = [
    {"ticker": "RIVN", "name": "Rivian Automotive, Inc."},
    {"ticker": "LCID", "name": "Lucid Group, Inc."},
    {"ticker": "NIO", "name": "NIO Inc."},
    {"ticker": "XPEV", "name": "XPeng Inc."},
    {"ticker": "LI", "name": "Li Auto Inc."},
    {"ticker": "ZK", "name": "ZEEKR Intelligent Technology Holding Limited"},
    {"ticker": "PSNY", "name": "Polestar Automotive Holding UK PLC"},
    {"ticker": "BYDDY", "name": "BYD Company Limited"},
    {"ticker": "VFS", "name": "VinFast Auto Ltd."},
    {"ticker": "LOT", "name": "Lotus Technology Inc."},
    {"ticker": "TSLA", "name": "Tesla, Inc."},
]
EV_TICKERS = {x["ticker"] for x in EV_GROUP}

# --- Semiconductors & Semiconductor Equipment ---
SEMICONDUCTORS_GROUP = [
    {"ticker": "AMD",  "name": "Advanced Micro Devices, Inc."},
    {"ticker": "INTC", "name": "Intel Corporation"},
    {"ticker": "AVGO", "name": "Broadcom Inc."},
    {"ticker": "NXPI", "name": "NXP Semiconductors N.V."},
    {"ticker": "QCOM", "name": "QUALCOMM Incorporated"},
    {"ticker": "MU",   "name": "Micron Technology, Inc."},
    {"ticker": "ARM",  "name": "Arm Holdings plc"},
    {"ticker": "TSM",  "name": "Taiwan Semiconductor Manufacturing Company Limited"},
    {"ticker": "ASX",  "name": "ASE Technology Holding Co., Ltd."},
    {"ticker": "NVDA", "name": "NVIDIA Corporation"},
    {"ticker": "ASML", "name": "ASML Holding N.V."},
    {"ticker": "ACMR", "name": "ACM Research, Inc."},
    {"ticker": "ASYS", "name": "Amtech Systems, Inc."},
    {"ticker": "MTSI", "name": "MACOM Technology Solutions Holdings, Inc."},
    {"ticker": "ADI",  "name": "Analog Devices, Inc."},
    {"ticker": "TXN",  "name": "Texas Instruments Incorporated"},
]
SEMICONDUCTORS_TICKERS = {x["ticker"] for x in SEMICONDUCTORS_GROUP}

# --- Digital Payments ---
Payment_GROUP = [
    {"ticker": "V", "name": "Visa, Inc."},
    {"ticker": "MA", "name": "Mastercard, Inc."},
    {"ticker": "PYPL", "name": "PayPal Holdings, Inc."},
    {"ticker": "AXP", "name": "American Express"},
    {"ticker": "SQ", "name": "Block, Inc."},
    {"ticker": "COF", "name": "Capital One Financial"},
    {"ticker": "STNE", "name": "StoneCo"},
]
Payment_TICKERS = {x["ticker"] for x in Payment_GROUP}

# --- Lending & Credit ---
Lending_GROUP = [
    {"ticker": "UPST", "name": "Upstart Holdings, Inc."},
    {"ticker": "LC", "name": "LendingClub Corporation"},
    {"ticker": "SOFI", "name": "SoFi Technologies, Inc."},
    {"ticker": "AFRM", "name": "Affirm Holdings, Inc."},
]
Lending_TICKERS = {x["ticker"] for x in Lending_GROUP}

# --- Online Brokers ---
Broker_GROUP = [
    {"ticker": "FUTU", "name": "Futu Holdings Limited"},
    {"ticker": "TIGR", "name": "UP Fintech Holding Limited"},
    {"ticker": "HOOD", "name": "Robinhood Markets, Inc."},
    {"ticker": "IBKR", "name": "Interactive Brokers Group, Inc."},
    {"ticker": "COIN", "name": "Coinbase Global, Inc."},
    {"ticker": "SCHW", "name": "The Charles Schwab Corporation"},
]
Broker_TICKERS = {x["ticker"] for x in Broker_GROUP}

# --- Banking & Financial Services ---
Banking_GROUP = [
    {"ticker": "USB", "name": "U.S. Bancorp"},
    {"ticker": "PNC", "name": "The PNC Financial Services Group, Inc."},
    {"ticker": "WFC", "name": "Wells Fargo & Company"},
    {"ticker": "BAC", "name": "Bank of America Corporation"},
    {"ticker": "JPM", "name": "JPMorgan Chase & Co."},
    {"ticker": "C", "name": "Citigroup Inc."},
    {"ticker": "COF", "name": "Capital One Financial Corporation"},
]
Banking_TICKERS = {x["ticker"] for x in Banking_GROUP}

# --- Airlines & Aviation ---
Airlines_GROUP = [
    {"ticker": "AAL", "name": "American Airlines Group Inc."},
    {"ticker": "DAL", "name": "Delta Air Lines, Inc."},
    {"ticker": "LUV", "name": "Southwest Airlines Co."},
    {"ticker": "UAL", "name": "United Airlines Holdings, Inc."},
    {"ticker": "ULCC", "name": "Frontier Group Holdings, Inc."},
    {"ticker": "ALK", "name": "Alaska Air Group, Inc."},
    {"ticker": "SAVE", "name": "Spirit Airlines, Inc."},
    {"ticker": "SNCY", "name": "Sun Country Airlines Holdings, Inc."},
]
Airlines_TICKERS = {x["ticker"] for x in Airlines_GROUP}

# ============================================================
# Ticker resolve (with S3 mapping + Perplexity fallback)
# ============================================================
@lru_cache(maxsize=1)
def load_s3_ticker_map() -> dict:
    """
    Loads pre-stored Excel sheet mapping of company names ↔ tickers.
    Expected columns: 'Ticker', 'Company'
    Returns dict: {lowercase company name: ticker}
    """
    import io

    try:
        obj = s3_client.get_object(Bucket=BUCKET_NAME, Key=TICKER_MAP_KEY)
        df = pd.read_excel(io.BytesIO(obj["Body"].read()))
        cols = [c.lower() for c in df.columns]
        if "ticker" not in cols or "company" not in cols:
            raise ValueError("Mapping sheet must contain 'Ticker' and 'Company' columns")
        df.columns = cols
        mapping = {}
        for _, r in df.iterrows():
            name = str(r.get("company", "")).strip().lower()
            tick = str(r.get("ticker", "")).strip().upper()
            if name and tick:
                mapping[name] = tick
        print(f"[INFO] Loaded {len(mapping)} tickers from S3 mapping sheet.")
        return mapping
    except ClientError as e:
        print(f"[WARN] Could not load ticker map from S3: {e}")
        return {}
    except Exception as e:
        print(f"[WARN] Error parsing ticker map: {e}")
        return {}


COMMON_NAME_MAP_FALLBACK = {
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
    "boeing": "BA",
    "airbus": "AIR.PA",
}


def _verify_ticker_with_yfinance(ticker: str) -> dict | None:
    try:
        t = _norm_ticker(ticker)
        s = yf.Ticker(t)
        _ensure_yf_session_headers(s)
        info = s.get_info() or {}
        nm = info.get("longName") or info.get("shortName")
        if nm:
            return {"ticker": t, "name": nm}
        if info.get("symbol") == t:
            return {"ticker": t, "name": None}
        return None
    except Exception:
        return None


def perplexity_lookup_company(input_text: str) -> dict | None:
    """
    Use Perplexity API to infer the correct ticker for a given company name.
    Example: "Palantir Technologies" -> "PLTR"
    """
    if not PERPLEXITY_API_KEY:
        return None
    try:
        query = (
            f"What is the stock ticker for {input_text} "
            f"(U.S.-listed or main exchange if multiple)? Return only the ticker symbol."
        )
        r = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers={
                "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": PERPLEXITY_MODEL,
                "temperature": 0.1,
                "messages": [
                    {"role": "system", "content": "You are a financial market assistant."},
                    {"role": "user", "content": query},
                ],
                "top_p": 0.9,
                "stream": False,
            },
            timeout=25,
        )
        r.raise_for_status()
        data = r.json()
        content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "").strip()
        if not content:
            return None
        m = re.findall(r"\b[A-Z]{1,6}\b", content)
        if not m:
            return None
        ticker = m[0]
        v = _verify_ticker_with_yfinance(ticker)
        if v:
            v["source"] = "perplexity"
            return v
        return {"ticker": ticker, "name": None, "source": "perplexity"}
    except Exception as e:
        print(f"[WARN] Perplexity fallback failed: {e}")
        return None


def resolve_input_to_ticker(user_input: str) -> dict:
    """
    Accepts ticker OR company name, returns {ticker, name, source} or {error}.
    Priority:
      1. Direct ticker (AAPL, TSLA)
      2. S3 mapping (exact or partial company name)
      3. Static fallback map
      4. Perplexity AI fallback
      5. yfinance heuristic verification
    """
    raw = (user_input or "").strip()
    if not raw:
        return {"error": "Input is empty"}

    # 1) Direct ticker
    if raw.isalpha() and 1 <= len(raw) <= 6:
        v = _verify_ticker_with_yfinance(raw)
        if v:
            return {"input": raw, "ticker": v["ticker"], "name": v.get("name"), "source": "input"}

    # 2) S3 mapping
    mapping = load_s3_ticker_map()
    norm = raw.lower()
    if norm in mapping:
        tkr = mapping[norm]
        v = _verify_ticker_with_yfinance(tkr)
        if v:
            return {"input": raw, "ticker": v["ticker"], "name": v.get("name"), "source": "s3-map"}

    for cname, tkr in mapping.items():
        if norm in cname:
            v = _verify_ticker_with_yfinance(tkr)
            if v:
                return {"input": raw, "ticker": v["ticker"], "name": v.get("name"), "source": "s3-partial"}

    # 3) Static fallback
    if norm in COMMON_NAME_MAP_FALLBACK:
        tkr = COMMON_NAME_MAP_FALLBACK[norm]
        v = _verify_ticker_with_yfinance(tkr)
        if v:
            return {"input": raw, "ticker": v["ticker"], "name": v.get("name"), "source": "fallback"}

    # 4) Perplexity fallback
    v = perplexity_lookup_company(raw)
    if v:
        return {"input": raw, "ticker": v["ticker"], "name": v.get("name"), "source": "perplexity"}

    # 5) Heuristic last try
    v = _verify_ticker_with_yfinance(raw)
    if v:
        return {"input": raw, "ticker": v["ticker"], "name": v.get("name"), "source": "guess"}

    return {"error": f"Could not resolve '{raw}' to a ticker"}


# ============================================================
# Profiles & metrics
# ============================================================
@lru_cache(maxsize=512)
def fetch_profile(ticker: str) -> dict:
    """Fetch company profile with error handling."""
    t = _norm_ticker(ticker)
    try:
        s = yf.Ticker(t)
        _ensure_yf_session_headers(s)
        info = s.get_info() or {}
        return {
            "ticker": t,
            "name": info.get("longName") or info.get("shortName") or t,
            "industry": info.get("industry"),
            "sector": info.get("sector"),
            "market_cap": info.get("marketCap"),
        }
    except Exception as e:
        print(f"[WARN] Could not fetch profile for {t}: {e}")
        return {
            "ticker": t,
            "name": t,
            "industry": None,
            "sector": None,
            "market_cap": None,
        }


@lru_cache(maxsize=128)
def calculate_metrics(ticker: str, max_retries: int = 3):
    """
    Fetch quarterly financial + cashflow metrics from yfinance and normalize
    into a dictionary of series:
      { metric_name: { quarter_label: value, ... }, ... }
    """
    def _pick(df, labels):
        if df is None or df.empty:
            return None
        idx = {str(i).strip().lower(): i for i in df.index}
        for lab in labels:
            k = lab.strip().lower()
            if k in idx:
                return df.loc[idx[k]]
        return None

    def _to_quarterly(df):
        if df is None or df.empty:
            return df
        try:
            df = df.copy()
            if not isinstance(df.columns, pd.PeriodIndex):
                df.columns = pd.to_datetime(df.columns, errors="coerce").to_period("Q")
            return df.iloc[:, :5]  # last 5 quarters
        except Exception:
            return df

    t = _norm_ticker(ticker)
    if not t:
        return None

    for attempt in range(max_retries):
        try:
            if attempt:
                time.sleep(0.8 * attempt)
            s = yf.Ticker(t)
            _ensure_yf_session_headers(s)

            fin_q = getattr(s, "quarterly_financials", None)
            if fin_q is None or fin_q.empty:
                fin_q = getattr(s, "financials", None)
            cf_q = getattr(s, "quarterly_cashflow", None)
            if cf_q is None or cf_q.empty:
                cf_q = getattr(s, "cashflow", None)

            if fin_q is None or fin_q.empty or cf_q is None or cf_q.empty:
                continue

            fin_q, cf_q = _to_quarterly(fin_q), _to_quarterly(cf_q)

            total_rev    = _pick(fin_q, ["Total Revenue", "Revenue", "Operating Revenue"])
            cost_rev     = _pick(fin_q, ["Cost Of Revenue", "Cost of Revenue", "Reconciled Cost Of Revenue"])
            gross_profit = _pick(fin_q, ["Gross Profit"])
            opex         = _pick(fin_q, ["Operating Expense", "Operating Expenses",
                                         "Total Operating Expenses", "Total Expenses"])
            ebit         = _pick(fin_q, ["EBIT", "Operating Income", "Total Operating Income As Reported"])
            net_income   = _pick(fin_q, ["Net Income", "Net Income Common Stockholders",
                                         "Net Income From Continuing Operation Net Minority Interest"])

            ocf   = _pick(cf_q, ["Operating Cash Flow", "Total Cash From Operating Activities"])
            capex = _pick(cf_q, ["Capital Expenditure", "Capital Expenditures"])

            # Derivations
            if gross_profit is None and total_rev is not None and cost_rev is not None:
                try:
                    gross_profit = (total_rev - cost_rev)
                except Exception:
                    gross_profit = None

            if opex is None:
                sga = _pick(fin_q, ["Selling General And Administration", "SG&A Expense",
                                    "General And Administrative Expense"])
                rnd = _pick(fin_q, ["Research And Development", "Research & Development"])
                selling = _pick(fin_q, ["Selling And Marketing Expense"])
                try:
                    if sga is not None and rnd is not None:
                        opex = (sga + rnd)
                    elif sga is not None and selling is not None:
                        opex = (sga + selling)
                    elif sga is not None:
                        opex = sga
                except Exception:
                    opex = None

            if ebit is None:
                try:
                    if gross_profit is not None and opex is not None:
                        ebit = (gross_profit - opex)
                    elif total_rev is not None and cost_rev is not None and opex is not None:
                        ebit = (total_rev - cost_rev - opex)
                    else:
                        pretax = _pick(fin_q, ["Pretax Income"])
                        if pretax is not None:
                            ebit = pretax
                except Exception:
                    ebit = None

            fcf = None
            try:
                if ocf is not None and capex is not None:
                    fcf = ocf + capex
            except Exception:
                fcf = None

            out = {}

            def _put(name, s, pct=False):
                if s is None:
                    return
                series = {}
                for q, v in s.items():
                    if pd.isna(v):
                        continue
                    try:
                        series[str(q)] = float(v) if pct else int(float(v))
                    except Exception:
                        continue
                if series:
                    out[name] = series

            gm_pct = None
            if gross_profit is not None and total_rev is not None:
                try:
                    gm_pct = (gross_profit / total_rev) * 100.0
                except Exception:
                    gm_pct = None

            _put("Total Revenue", total_rev, False)
            _put("Operating Expense", opex, False)
            _put("EBIT", ebit, False)
            _put("Net Income", net_income, False)
            if gm_pct is not None:
                _put("Gross Margin %", gm_pct, True)
            if fcf is not None:
                _put("Free Cash Flow", fcf, False)

            # Market Cap as current snapshot
            try:
                profile = fetch_profile(t)
                if profile.get("market_cap"):
                    out["Market Cap"] = {"Current": int(profile["market_cap"])}
            except Exception:
                pass

            if out.get("Total Revenue") or out.get("Gross Margin %"):
                return out
        except Exception as e:
            print(f"[WARN] Error calculating metrics for {t} (attempt {attempt+1}): {e}")
            continue

    return None


# ============================================================
# Peer selection with priority logic
# ============================================================
_UNIVERSE: list[str] = []
_UNIVERSE_BUILT = False


def _build_universe(max_size: int = 1800):
    """
    Combine fallback universe + S&P500 + NASDAQ tickers.
    """
    global _UNIVERSE, _UNIVERSE_BUILT
    if _UNIVERSE_BUILT and _UNIVERSE:
        return

    fallback_tickers = [
        # Mega Cap Tech
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
        # Tech & Software
        "ORCL", "CRM", "ADBE", "INTC", "AMD", "QCOM", "AVGO", "CSCO", "IBM", "NOW", "SNOW", "PLTR", "SHOP",
        # Consumer & Retail
        "WMT", "COST", "HD", "LOW", "TGT", "DG", "DLTR", "BBY", "GPS", "M", "KSS",
        # E-commerce & Digital
        "EBAY", "ETSY", "W", "CHWY",
        # Auto & EV
        "F", "GM", "RIVN", "LCID", "NIO", "XPEV", "LI",
        # Finance & Banks
        "JPM", "BAC", "WFC", "C", "GS", "MS", "BLK", "SCHW", "USB", "PNC", "TFC", "COF",
        # Fintech
        "V", "MA", "PYPL", "SQ", "COIN", "AFRM", "SOFI", "UPST",
        # Healthcare & Pharma
        "UNH", "JNJ", "PFE", "ABBV", "MRK", "TMO", "ABT", "DHR", "LLY", "BMY", "AMGN", "GILD", "CVS", "CI",
        # Biotech
        "MRNA", "BNTX", "REGN", "VRTX", "BIIB",
        # Energy
        "XOM", "CVX", "COP", "SLB", "EOG", "OXY", "MPC", "PSX", "VLO",
        # Industrials
        "BA", "CAT", "GE", "HON", "UPS", "LMT", "RTX", "DE", "MMM", "EMR",
        # Airlines
        "DAL", "UAL", "AAL", "LUV", "ALK", "JBLU",
        # Consumer Goods
        "PG", "KO", "PEP", "NKE", "SBUX", "MCD", "YUM", "CMG", "DPZ",
        # Telecom
        "T", "VZ", "TMUS",
        # Media & Entertainment
        "DIS", "NFLX", "CMCSA", "WBD", "PARA", "SPOT", "ROKU",
        # Semiconductors
        "TSM", "ASML", "MU", "LRCX", "AMAT", "KLAC", "MCHP", "ADI", "TXN", "NXPI",
        # Cloud & Cybersecurity
        "NET", "DDOG", "ZS", "OKTA", "CRWD", "S", "PANW", "FTNT",
        # Real Estate
        "AMT", "PLD", "CCI", "EQIX", "PSA", "DLR", "O", "VICI", "SPG", "AVB",
        # Utilities
        "NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "PEG",
        # Materials & Chemicals
        "LIN", "APD", "ECL", "SHW", "DD", "DOW", "PPG", "NUE", "FCX",
        # eVTOL
        "JOBY", "EH", "ACHR", "BLDE",
        # Other Notable
        "UBER", "LYFT", "ABNB", "DASH", "ZM", "DOCU", "TWLO", "TEAM", "SNAP", "PINS", "RBLX",
        # International ADRs
        "BABA", "JD", "PDD", "BIDU", "NIO", "XPEV", "LI", "TSM", "ASML", "SAP", "BYDDY",
    ]

    try:
        sp = yf.tickers_sp500() or []
    except Exception:
        sp = []
    try:
        ndq = yf.tickers_nasdaq() or []
    except Exception:
        ndq = []

    all_tickers = fallback_tickers + list(sp) + list(ndq)
    allu = list(dict.fromkeys([_norm_ticker(t) for t in all_tickers]))

    cleaned = [t for t in allu if t and len(t) <= 6 and t.replace(".", "").isalnum()]
    if max_size:
        cleaned = cleaned[:max_size]

    _UNIVERSE = cleaned
    _UNIVERSE_BUILT = True


def _ratio_score(base_mc, mc):
    if not base_mc or not mc or base_mc <= 0 or mc <= 0:
        return float("inf")
    big, small = (mc, base_mc) if mc >= base_mc else (base_mc, mc)
    return abs((big / small) - 1.0)


def select_peers_any_industry(base_ticker: str, peer_limit: int = 3):
    """
    Peer selection priority:
      1. Self-defined groups (if belongs to multiple groups, union)
      2. Perplexity AI suggested peers
      3. Industry match
      4. Sector match
      5. Market cap similarity

    Returns:
      {
        "primary_company": "TSLA",
        "industry": "Electric Vehicle Manufacturers / ...",
        "industry_list": [...],
        "peers": [{"ticker": "...", "name": "..."}, ...]
      }
    """
    base_ticker_norm = _normalize_peer_ticker(base_ticker)
    _build_universe()
    base_prof = fetch_profile(base_ticker_norm)
    base_ind = base_prof.get("industry")
    base_sec = base_prof.get("sector")
    base_mc  = base_prof.get("market_cap")

    SELF_DEFINED_GROUPS = {
        "Magnificent 7 Tech Giants": MEGA7_TICKERS,
        "eVTOL / Urban Air Mobility": EVTOL_TICKERS,
        "Electric Vehicle Manufacturers": EV_TICKERS,
        "Semiconductors & Semiconductor Equipment": SEMICONDUCTORS_TICKERS,
        "Fintech / Digital Payments": Payment_TICKERS,
        "Fintech / Lending": Lending_TICKERS,
        "Fintech / Broker": Broker_TICKERS,
        "Airlines & Aviation": Airlines_TICKERS,
        "Banking Services": Banking_TICKERS,
    }

    matched_groups = []
    combined_peers = []

    # 1) Self-defined groups
    for label, tickers in SELF_DEFINED_GROUPS.items():
        if base_ticker_norm in tickers:
            matched_groups.append(label)
            for group in [
                MEGA7, EVTOL_GROUP, EV_GROUP,
                SEMICONDUCTORS_GROUP, Lending_GROUP,
                Payment_GROUP, Broker_GROUP, Banking_GROUP,
                Airlines_GROUP,
            ]:
                for g in group:
                    if g["ticker"] != base_ticker_norm and g["ticker"] in tickers:
                        combined_peers.append(g)

    if matched_groups:
        unique_peers = {p["ticker"]: p for p in combined_peers}.values()
        peers_list = list(unique_peers)[:peer_limit]
        return {
            "primary_company": base_ticker_norm,
            "industry": " / ".join(sorted(set(matched_groups))),
            "industry_list": sorted(set(matched_groups)),
            "peers": peers_list,
        }

    # 2) Perplexity AI suggested peers
    print(f"[INFO] {base_ticker_norm} not in self-defined groups, trying Perplexity AI...")
    try:
        prompt = (
            f"Suggest {peer_limit} U.S.-listed peer companies for {base_ticker_norm} "
            "based on similar business model or competitive landscape. "
            "Return ONLY ticker symbols, comma-separated (e.g., AAPL, MSFT, GOOGL)."
        )
        ai_text = perplexity_chat(
            [
                {"role": "system", "content": "You are a financial analyst. Return only ticker symbols."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            timeout=20,
        )
        if ai_text:
            tickers = re.findall(r"\b[A-Z]{1,5}\b", ai_text)
            tickers = [t for t in tickers if t != base_ticker_norm][:peer_limit]
            peers = []
            for t in tickers:
                v = _verify_ticker_with_yfinance(t)
                if v:
                    peers.append({"ticker": v["ticker"], "name": v.get("name")})
            if peers:
                return {
                    "primary_company": base_ticker_norm,
                    "industry": "Perplexity AI Suggested Peers",
                    "industry_list": ["Perplexity AI Suggested Peers"],
                    "peers": peers,
                }
    except Exception as e:
        print(f"[WARN] Perplexity AI peer selection failed: {e}")

    # 3–5) Fallback: industry → sector → market cap similarity
    print(f"[INFO] Falling back to industry/sector/market cap matching for {base_ticker_norm}")

    def _fallback(selector):
        peers = []
        for t in _UNIVERSE:
            if t == base_ticker_norm:
                continue
            pr = fetch_profile(t)
            if selector(pr):
                peers.append((t, _ratio_score(base_mc, pr.get("market_cap"))))
        peers.sort(key=lambda x: x[1])
        peers_named = []
        for t, _score in peers[:peer_limit]:
            pr = fetch_profile(t)
            peers_named.append({"ticker": t, "name": pr.get("name")})
        return peers_named

    # by industry
    if base_ind:
        peers_named = _fallback(lambda pr: pr.get("industry") == base_ind)
        if peers_named:
            return {
                "primary_company": base_ticker_norm,
                "industry": base_ind,
                "industry_list": [base_ind],
                "peers": peers_named,
            }

    # by sector
    if base_sec:
        peers_named = _fallback(lambda pr: pr.get("sector") == base_sec)
        if peers_named:
            return {
                "primary_company": base_ticker_norm,
                "industry": base_sec,
                "industry_list": [base_sec],
                "peers": peers_named,
            }

    # by market cap similarity
    peers_named = _fallback(lambda pr: True)
    if not peers_named:
        return {
            "primary_company": base_ticker_norm,
            "industry": "No peers found",
            "industry_list": ["No peers found"],
            "peers": [],
        }

    return {
        "primary_company": base_ticker_norm,
        "industry": "Market Cap Similarity",
        "industry_list": ["Market Cap Similarity"],
        "peers": peers_named,
    }


# ============================================================
# Analysis math
# ============================================================
def safe(v):
    return v if (v is not None and not (isinstance(v, float) and math.isnan(v))) else None


def pct(n, d):
    try:
        if d in (0, None) or n is None:
            return None
        return (n - d) / abs(d) * 100.0
    except Exception:
        return None


def ppoints(n, d):
    try:
        if n is None or d is None:
            return None
        return (n - d) * 100.0
    except Exception:
        return None


def rank_desc(value, peer_values):
    arr = sorted([x for x in peer_values if x is not None], reverse=True)
    if value is None or not arr:
        return None
    try:
        return 1 + arr.index(value)
    except ValueError:
        diffs = sorted([(abs(value - x), i) for i, x in enumerate(arr)])
        return 1 + diffs[0][1]


def latest_of(row):
    vals = row.get("values", [])
    return safe(vals[0]) if vals else None


def get_ts_metric(rows, name):
    n = (name or "").strip().lower()
    for r in rows:
        if (r.get("metric") or "").strip().lower() == n:
            return r
    return None


def analyze_primary_company(payload: dict) -> dict:
    primary = _norm_ticker(payload.get("primary"))
    lqm = payload.get("latest_quarter") or {}
    ts  = payload.get("time_series") or {}
    peer_rows = lqm.get("rows", []) or []
    period = lqm.get("period") or "Latest"

    if not primary:
        raise ValueError("Primary ticker missing")
    if not peer_rows:
        raise ValueError("latest_quarter.rows missing")

    peer_keys = [k for k in peer_rows[0].keys() if k != "metric"]
    if primary not in peer_keys:
        raise ValueError("Primary not in latest_quarter rows")
    other_peers = [k for k in peer_keys if k != primary]

    metric_row_map = {(r.get("metric") or "").strip(): r for r in peer_rows}

    latest_metrics = {}
    for mname, row in metric_row_map.items():
        vals = [safe(row.get(t)) for t in peer_keys]
        latest_metrics[mname] = {"primary": safe(row.get(primary)), "peers_values": vals}

    def metric_rank(mname, higher_is_better=True):
        md = latest_metrics.get(mname, {})
        pv = md.get("primary")
        allv = md.get("peers_values", [])
        if not higher_is_better and pv is not None:
            allv = [(-x if x is not None else None) for x in allv]
            pv = -pv
        r = rank_desc(pv, allv)
        n = len([x for x in allv if x is not None])
        return (r, n) if r is not None else (None, n)

    rev_rank  = metric_rank("Total Revenue")
    gm_rank   = metric_rank("Gross Margin %")
    ebit_rank = metric_rank("EBIT")
    ni_rank   = metric_rank("Net Income")
    fcf_rank  = metric_rank("Free Cash Flow")
    opex_rank = metric_rank("Operating Expense", higher_is_better=False)

    # ---------- Valuation ratios ----------
    period_key = period
    tickers_all = [primary] + other_peers

    def _get_latest(metric_name: str, tkr: str):
        row = metric_row_map.get(metric_name)
        if not row:
            return None
        key = "Current" if metric_name == "Market Cap" else period_key
        return row.get(tkr)

    latest_vals = {}
    for tkr in tickers_all:
        mc  = _get_latest("Market Cap", tkr)
        rev = _get_latest("Total Revenue", tkr)
        ni  = _get_latest("Net Income", tkr)
        mc_rev = None
        mc_ni  = None
        try:
            if mc is not None and rev not in (None, 0):
                mc_rev = float(mc) / float(rev)
        except Exception:
            mc_rev = None
        try:
            if mc is not None and ni not in (None, 0):
                mc_ni = float(mc) / float(ni)
        except Exception:
            mc_ni = None
        latest_vals[tkr] = {"MC/Rev": mc_rev, "MC/NI": mc_ni}

    peer_deltas: dict[str, list] = {}
    peer_deltas["MC/Rev"] = []
    peer_deltas["MC/NI"]  = []
    pv_mc_rev = latest_vals[primary]["MC/Rev"]
    pv_mc_ni  = latest_vals[primary]["MC/NI"]

    def _pct(a, b):
        try:
            if a is None or b in (None, 0):
                return None
            return (a - b) / abs(b) * 100.0
        except Exception:
            return None

    for peer in other_peers:
        pr_mc_rev = latest_vals[peer]["MC/Rev"]
        pr_mc_ni  = latest_vals[peer]["MC/NI"]
        peer_deltas["MC/Rev"].append({"peer": peer, "pct": _pct(pv_mc_rev, pr_mc_rev)})
        peer_deltas["MC/NI"].append({"peer": peer, "pct": _pct(pv_mc_ni, pr_mc_ni)})

    def _median_clean(arr):
        arr = [x for x in arr if x is not None]
        return stats.median(arr) if arr else None

    peers_mc_rev_vals = [latest_vals[p]["MC/Rev"] for p in other_peers]
    peers_mc_ni_vals  = [latest_vals[p]["MC/NI"]  for p in other_peers]
    med_mc_rev = _median_clean(peers_mc_rev_vals)
    med_mc_ni  = _median_clean(peers_mc_ni_vals)

    def _pct_vs_med(a, m):
        try:
            if a is None or m in (None, 0):
                return None
            return (a - m) / abs(m) * 100.0
        except Exception:
            return None

    mc_rev_vs_med = _pct_vs_med(pv_mc_rev, med_mc_rev)
    mc_ni_vs_med  = _pct_vs_med(pv_mc_ni,  med_mc_ni)

    def _classify(x):
        if x is None:
            return None
        if x > 30:
            return "Overvalued"
        if x < -30:
            return "Undervalued"
        return "Inline"

    valuation_label = None
    lab1 = _classify(mc_rev_vs_med)
    lab2 = _classify(mc_ni_vs_med)

    if lab1 and lab2:
        if "Overvalued" in (lab1, lab2):
            valuation_label = "Overvalued"
        elif "Undervalued" in (lab1, lab2):
            valuation_label = "Undervalued"
        else:
            valuation_label = "Inline"
    else:
        valuation_label = lab1 or lab2 or None

    summary_out_extras = {
        "valuation_ratios": {
            "primary": {"MC/Rev": pv_mc_rev, "MC/NI": pv_mc_ni},
            "peers_median": {"MC/Rev": med_mc_rev, "MC/NI": med_mc_ni},
            "pct_vs_median": {"MC/Rev": mc_rev_vs_med, "MC/NI": mc_ni_vs_med},
            "label": valuation_label,
        }
    }

    def pct_change(a, b):
        try:
            if a is None or b in (None, 0):
                return None
            return (a - b) / abs(b) * 100.0
        except Exception:
            return None

    for metric, row in metric_row_map.items():
        if metric in ["MC/Rev", "MC/NI"]:
            continue
        primary_val = safe(row.get(primary))
        deltas = []
        for peer in other_peers:
            peer_val = safe(row.get(peer))
            if metric.strip().lower() == "gross margin %":
                diff = None if (primary_val is None or peer_val is None) else (primary_val - peer_val)
                deltas.append({"peer": peer, "pp": diff})
            else:
                deltas.append({"peer": peer, "pct": pct_change(primary_val, peer_val)})
        peer_deltas[metric] = deltas

    quarters = ts.get("quarters", []) or []
    rows = ts.get("rows", []) or []

    rev_row  = get_ts_metric(rows, "Total Revenue")
    gm_row   = get_ts_metric(rows, "Gross Margin %")
    opex_row = get_ts_metric(rows, "Operating Expense")
    ebit_row = get_ts_metric(rows, "EBIT")
    ni_row   = get_ts_metric(rows, "Net Income")
    fcf_row  = get_ts_metric(rows, "Free Cash Flow")

    def qoq(row):
        return None if not row or len(row.get("values", [])) < 2 else pct(row["values"][0], row["values"][1])

    def yoy(row):
        return None if not row or len(row.get("values", [])) < 5 else pct(row["values"][0], row["values"][4])

    rev_qoq, rev_yoy = qoq(rev_row), yoy(rev_row)
    gm_qoq_pp = ppoints(gm_row["values"][0], gm_row["values"][1]) if gm_row and len(gm_row.get("values", [])) >= 2 else None
    gm_yoy_pp = ppoints(gm_row["values"][0], gm_row["values"][4]) if gm_row and len(gm_row.get("values", [])) >= 5 else None

    opex_pct_now = opex_pct_ya = None
    if opex_row and rev_row:
        vnow, rnow = latest_of(opex_row), latest_of(rev_row)
        if vnow is not None and rnow:
            opex_pct_now = vnow / rnow * 100.0
        if len(opex_row.get("values", [])) >= 5 and len(rev_row.get("values", [])) >= 5:
            vya, rya = opex_row["values"][4], rev_row["values"][4]
            if vya is not None and rya:
                opex_pct_ya = vya / rya * 100.0

    ebit_qoq, ebit_yoy = qoq(ebit_row), yoy(ebit_row)
    ni_qoq,   ni_yoy   = qoq(ni_row),  yoy(ni_row)

    fcf_vals = (fcf_row or {}).get("values", [])
    clean_fcf = [v for v in fcf_vals if v is not None]
    fcf_stdev = stats.pstdev(clean_fcf) if len(clean_fcf) >= 2 else None
    fcf_mean  = stats.mean(clean_fcf) if clean_fcf else None
    fcf_cv    = (fcf_stdev / fcf_mean * 100.0) if (fcf_stdev is not None and fcf_mean not in (0, None)) else None

    latest = {
        "revenue": latest_of(rev_row),
        "gm": latest_of(gm_row),
        "opex": latest_of(opex_row),
        "ebit": latest_of(ebit_row),
        "ni": latest_of(ni_row),
        "fcf": latest_of(fcf_row),
    }

    ret = {
        "primary": primary,
        "period": period,
        "peer_ranks": {
            "revenue_rank": rev_rank,
            "gross_margin_rank": gm_rank,
            "ebit_rank": ebit_rank,
            "net_income_rank": ni_rank,
            "fcf_rank": fcf_rank,
            "opex_rank": opex_rank,
        },
        "peer_deltas": peer_deltas,
        "timeseries": {
            "quarters": quarters,
            "rev_qoq_pct": rev_qoq,
            "rev_yoy_pct": rev_yoy,
            "gm_qoq_pp": gm_qoq_pp,
            "gm_yoy_pp": gm_yoy_pp,
            "opex_pct_now": opex_pct_now,
            "opex_pct_yearago": opex_pct_ya,
            "ebit_qoq_pct": ebit_qoq,
            "ebit_yoy_pct": ebit_yoy,
            "ni_qoq_pct": ni_qoq,
            "ni_yoy_pct": ni_yoy,
            "fcf_cv_pct": fcf_cv,
        },
        "latest": latest,
    }
    ret.update(summary_out_extras)
    return ret


def build_conclusion_text(summary: dict) -> str:
    """
    Local fallback conclusion builder with enhanced valuation section.
    """
    p = summary["primary"]
    period = summary.get("period") or "Latest Quarter"
    ts = summary.get("timeseries", {})
    lt = summary.get("latest", {})
    deltas = summary.get("peer_deltas", {})

    def peer_line(metric, *, label=None, pp=False):
        label = label or metric
        arr = deltas.get(metric, [])
        if not arr:
            return f"{label}: n/a"
        parts = []
        for d in arr:
            v = d.get("pp") if pp else d.get("pct")
            parts.append(
                f"{d['peer']} "
                + (
                    "n/a"
                    if v is None
                    else (f"{v:+.1f}pp" if pp else f"{v:+.1f}%")
                )
            )
        return f"{label}: " + ", ".join(parts)

    def v(x, suf="%"):
        return "n/a" if x is None else f"{x:.1f}{suf}"

    bullets = [f"{p}: Past-performance takeaway -- {period}."]

    bullets.append("- ►► PEER COMPARISON:")
    bullets.append("  • " + peer_line("Total Revenue", label="Revenue"))
    bullets.append("  • " + peer_line("Operating Expense", label="OpEx (lower better)"))
    bullets.append("  • " + peer_line("EBIT"))
    bullets.append("  • " + peer_line("Net Income", label="Net income"))
    bullets.append("  • " + peer_line("Free Cash Flow", label="Free cash flow"))
    bullets.append("  • " + peer_line("Gross Margin %", label="GM (pp)", pp=True))
    bullets.append("  • " + peer_line("MC/Rev", label="MC/Revenue"))
    bullets.append("  • " + peer_line("MC/NI",  label="MC/Net income"))

    bullets.append("\n- ►► LATEST QUARTER METRICS:")
    bullets.append(
        "  • Revenue: "
        f"QoQ {v(ts.get('rev_qoq_pct'))}; "
        f"YoY {v(ts.get('rev_yoy_pct'))} → {fmt_money_short(lt.get('revenue'))}."
    )

    if ts.get("opex_pct_now") is not None:
        yoy_pp = None
        if ts.get("opex_pct_yearago") is not None:
            yoy_pp = ts["opex_pct_now"] - ts["opex_pct_yearago"]
        bullets.append(
            "  • OpEx ratio: "
            f"{ts['opex_pct_now']:.1f}%"
            + (f" (YoY {yoy_pp:+.1f}pp)." if yoy_pp is not None else ".")
        )
    else:
        bullets.append("  • OpEx ratio: n/a.")

    bullets.append(
        "  • EBIT: "
        f"QoQ {v(ts.get('ebit_qoq_pct'))}; "
        f"YoY {v(ts.get('ebit_yoy_pct'))} → {fmt_money_short(lt.get('ebit'))}."
    )

    bullets.append(
        "  • Net income: "
        f"QoQ {v(ts.get('ni_qoq_pct'))}; "
        f"YoY {v(ts.get('ni_yoy_pct'))} → {fmt_money_short(lt.get('ni'))}."
    )

    if ts.get("fcf_cv_pct") is not None or lt.get("fcf") is not None:
        stability = None
        if ts.get("fcf_cv_pct") is not None:
            stability = "stable" if ts["fcf_cv_pct"] < 25 else "volatile"
        tail = f" ({stability}, CV {ts['fcf_cv_pct']:.0f}%)." if stability else "."
        bullets.append("  • Free cash flow: " + fmt_money_short(lt.get("fcf")) + tail)
    else:
        bullets.append("  • Free cash flow: n/a.")

    vr = summary.get("valuation_ratios", {}) or {}
    primary_ratios = vr.get("primary", {}) or {}
    peer_median = vr.get("peers_median", {}) or {}
    pct_vm = vr.get("pct_vs_median", {}) or {}
    lab = vr.get("label")

    def _fmt_ratio(x): return "n/a" if x is None else f"{x:.2f}"
    def _fmt_pct(x): return "n/a" if x is None else f"{x:+.0f}%"

    mc_rev_str = (
        f"MC/Rev {_fmt_ratio(primary_ratios.get('MC/Rev'))} "
        f"vs peer median {_fmt_ratio(peer_median.get('MC/Rev'))} "
        f"({_fmt_pct(pct_vm.get('MC/Rev'))})"
    )
    mc_ni_str = (
        f"MC/NI {_fmt_ratio(primary_ratios.get('MC/NI'))} "
        f"vs median {_fmt_ratio(peer_median.get('MC/NI'))} "
        f"({_fmt_pct(pct_vm.get('MC/NI'))})"
    )
    verdict_str = f" → {lab}" if lab else ""

    bullets.append(
        f"\n- ►► VALUATION ANALYSIS: {mc_rev_str}; {mc_ni_str}{verdict_str}."
    )

    return "\n".join(bullets)


def llm_conclusion_with_deepseek(summary: dict) -> tuple[str, str]:
    """
    Use DeepSeek (or Perplexity as fallback) to phrase the summary.
    Returns: (conclusion_text, llm_used)
    """
    system = {
        "role": "system",
        "content": (
            "You are a senior equity research analyst. "
            "Use ONLY the provided data. Be concise, numerical, and objective. "
            "Focus on delivering clear valuation verdicts."
        ),
    }
    user = {
        "role": "user",
        "content": json.dumps(
            {
                "task": "Write analyst-style past-performance summary with valuation verdict for PRIMARY company only.",
                "style": "6-9 bullets; data-driven; concise; numeric-first; no speculation.",
                "format": [
                    "Start: '<TICKER>: Past-performance takeaway -- <period>.'",
                    "Peer comparison: Show % diffs for Revenue, OpEx, EBIT, Net Income, FCF vs EACH peer. "
                    "Format: 'Revenue vs PEER1 +X%, vs PEER2 +Y%; OpEx (lower better) vs PEER1 -X%, vs PEER2 -Y%; ...'.",
                    "Gross Margin: Show pp changes vs peers.",
                    "Latest quarter: Revenue $XB, GM X%, EBIT $XB, NI $XB, FCF $XB.",
                    "Growth metrics: QoQ/YoY % changes for key metrics.",
                    "CRITICAL - Valuation verdict: Compare MC/Revenue and MC/NI vs peer median. "
                    "State verdict clearly and prominently with Overvalued/Undervalued/Inline based on ±30% threshold.",
                    "Each bullet <200 chars.",
                ],
                "data": summary,
            },
            ensure_ascii=False,
        ),
    }

    out = deepseek_chat([system, user], temperature=0.1, timeout=40)
    if out:
        return out, "deepseek"

    out = perplexity_chat([system, user], temperature=0.2, timeout=30)
    if out:
        return out, "perplexity"

    # Local fallback
    return build_conclusion_text(summary), "local-fallback"


def translate_to_zh(text: str) -> str | None:
    """
    Translate English analysis to Simplified Chinese using DeepSeek.
    """
    if not text:
        return None
    system = {
        "role": "system",
        "content": (
            "You are a professional bilingual equity research translator "
            "specializing in financial analysis."
        ),
    }
    user = {
        "role": "user",
        "content": json.dumps(
            {
                "task": (
                    "Translate the following English financial analysis "
                    "to Simplified Chinese. Maintain professional finance "
                    "terminology and concise tone."
                ),
                "text": text,
            },
            ensure_ascii=False,
        ),
    }
    return deepseek_chat([system, user], temperature=0.3, timeout=40)


# ============================================================
# HTML Frontend
# ============================================================
INDEX_HTML = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width,initial-scale=1" />
    <title>Peer Company Comparison</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  </head>
  <body class="bg-gradient-to-br from-blue-50 to-indigo-100 min-h-screen p-2 md:p-4">
    <div class="max-w-7xl mx-auto">
      <div class="bg-white rounded-lg shadow-xl p-3 md:p-5 mb-3">
        <div class="flex items-center justify-between mb-2">
          <h1 class="text-xl md:text-3xl font-bold text-gray-800">
            Peer Company Key Metrics Comparison
          </h1>
          <div class="flex items-center gap-1.5">
            <div class="inline-flex rounded-md shadow-sm">
              <button id="btnEN" class="px-1.5 py-0.5 text-xs border rounded-l bg-indigo-600 text-white">EN</button>
              <button id="btnZH" class="px-1.5 py-0.5 text-xs border rounded-r bg-white text-gray-700">中文</button>
            </div>
            <a
              href="https://equityresearch.checkitanalytics.com/"
              class="flex items-center justify-center w-7 h-7 md:w-8 md:h-8 bg-white border-2 border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
              target="_blank"
              rel="noopener noreferrer"
            >
              <svg class="w-4 h-4 md:w-5 md:h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path
                  stroke-linecap="round"
                  stroke-linejoin="round"
                  stroke-width="2"
                  d="M10 19l-7-7m0 0l7-7m-7 7h18"
                />
              </svg>
            </a>
          </div>
        </div>

        <div class="flex flex-col gap-2 mb-2">
          <div class="flex items-center gap-1.5 flex-nowrap">
            <input
              id="tickerInput"
              class="px-2 py-1 border rounded text-sm w-auto max-w-full"
              placeholder="e.g., TSLA or Tesla"
            />
            <button
              id="findButton"
              class="shrink-0 px-3 py-1 bg-indigo-600 text-white rounded text-sm"
            >
              Find Peers
            </button>
          </div>

          <div class="flex items-center gap-1.5 flex-nowrap">
            <input
              id="manualInput"
              class="px-2 py-1 border rounded text-sm w-auto max-w-full"
              placeholder="Add/Remove company (ticker or name)"
            />
            <button
              id="addButton"
              class="shrink-0 px-3 py-1 bg-emerald-600 text-white rounded text-sm"
            >
              Add
            </button>
            <button
              id="removeButton"
              class="shrink-0 px-3 py-1 bg-red-600 text-white rounded text-sm opacity-50 cursor-not-allowed"
              disabled
            >
              Remove
            </button>
          </div>
        </div>

        <div
          id="error"
          class="hidden bg-red-50 border-l-4 border-red-500 p-3 mb-3"
        >
          <p class="text-red-700 text-sm"></p>
        </div>

        <div id="loading" class="hidden text-center py-3">
          <div
            class="inline-block animate-spin rounded-full h-6 w-6 border-b-2 border-indigo-600"
          ></div>
        </div>

        <div
          id="peerInfo"
          class="hidden bg-indigo-50 rounded-lg p-3 md:p-4 mb-4"
        ></div>
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
      let _conclusion = { en: null, zh: null, ticker: null, period: null, provider: null };
      let _lang = 'en';

      const COLORS = [
        '#8884d8',
        '#82ca9d',
        '#ffc658',
        '#ff8042',
        '#a4de6c',
        '#d0ed57',
        '#8dd1e1',
        '#a28dd1'
      ];

      function showError(msg) {
        const d = document.getElementById('error');
        d.querySelector('p').textContent = msg;
        d.classList.remove('hidden');
      }

      function hideError() {
        document.getElementById('error').classList.add('hidden');
      }

      function showLoading(b) {
        document.getElementById('loading').classList.toggle('hidden', !b);
        document.getElementById('findButton').disabled = b;
        document.getElementById('addButton').disabled = b;
        document.getElementById('removeButton').disabled = b;
      }

      function uniqUpper(arr) {
        const s = new Set();
        const out = [];
        for (const x of arr) {
          const y = (x || '').toUpperCase();
          if (!s.has(y)) {
            s.add(y);
            out.push(y);
          }
        }
        return out;
      }

      document.getElementById('tickerInput').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') resolveAndFind();
      });
      document.getElementById('manualInput').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') addCompany();
      });
      document.getElementById('findButton').onclick = resolveAndFind;
      document.getElementById('addButton').onclick = addCompany;
      document.getElementById('removeButton').onclick = removeCompany;

      document.getElementById('btnEN').onclick = () => {
        _lang = 'en';
        updateLangButtons();
        if (_tickers.length > 0) {
          displayPeers(_peerData);
          renderTable();
          renderTimeSeriesTables();
        }
        renderConclusion();
      };
      document.getElementById('btnZH').onclick = () => {
        _lang = 'zh';
        updateLangButtons();
        if (_tickers.length > 0) {
          displayPeers(_peerData);
          renderTable();
          renderTimeSeriesTables();
        }
        renderConclusion();
      };

      async function resolveAndFind() {
        const raw = document.getElementById('tickerInput').value.trim();
        if (!raw) return showError('Please enter a ticker or company name');
        hideError();
        showLoading(true);
        try {
          const r = await fetch('/api/resolve', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ input: raw })
          });
          const res = await r.json();
          if (!r.ok || res.error) throw new Error(res.error || 'Resolve failed');
          await findPeers(res.ticker, res.name);
        } catch (e) {
          showError(e.message);
        } finally {
          showLoading(false);
        }
      }

      async function findPeers(ticker, name) {
        try {
          showLoading(true);
          const r = await fetch('/api/find-peers', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ticker })
          });
          const peerData = await r.json();
          if (!r.ok || peerData.error)
            throw new Error(peerData.error || 'Find peers failed');
          if (name && peerData.primary_company === ticker)
            peerData.primary_name = name;
          _peerData = peerData;
          _manualPeers = [];
          displayPeers(peerData);

          const tickers = [
            peerData.primary_company,
            ...peerData.peers.map((p) => p.ticker)
          ];

          const metricsResp = await fetch('/api/get-metrics', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ tickers })
          });
          const data = await metricsResp.json();
          if (!metricsResp.ok || data.error)
            throw new Error(data.error || 'Metrics failed');

          _metricsData = data;
          const primary = peerData.primary_company;
          const validPeers = tickers
            .slice(1)
            .filter(
              (t) =>
                data[t] &&
                !data[t].error &&
                (data[t]['Total Revenue'] || data[t]['Gross Margin %'])
            );
          _tickers = uniqUpper([primary, ...validPeers]);
          _quarters = computeQuarters(data, _tickers);
          renderAll();
          await requestConclusion();
        } catch (e) {
          showError(e.message);
        } finally {
          showLoading(false);
        }
      }

      function displayPeers(d) {
        if (!d) return;
        const primaryLabel = d.primary_name
          ? d.primary_company + ' - ' + d.primary_name
          : d.primary_company;

        const allPeers = [...(d.peers || []), ..._manualPeers];

        const industries = d.industry_list || (d.industry ? d.industry.split(' / ') : []);
        const tagsHtml = industries
          .map((i) => {
            const colors = {
              'Magnificent 7 Tech Giants': 'bg-purple-100 text-purple-700',
              'eVTOL / Urban Air Mobility': 'bg-pink-100 text-pink-700',
              'Electric Vehicle Manufacturers': 'bg-green-100 text-green-700',
              'Semiconductors & Semiconductor Equipment':
                'bg-orange-100 text-orange-700',
              'Fintech / Digital Payments': 'bg-teal-100 text-teal-700',
              'Airlines & Aviation': 'bg-sky-100 text-sky-700',
              'Credit Services': 'bg-indigo-100 text-indigo-700',
              'Perplexity AI Suggested Peers': 'bg-yellow-100 text-yellow-700'
            };
            const style = colors[i] || 'bg-gray-100 text-gray-700';
            return (
              '<span class="inline-block ' +
              style +
              ' px-2 py-1 rounded-md mr-1 mb-1 text-xs">' +
              i +
              '</span>'
            );
          })
          .join(' ');

        const peersCards = allPeers
          .map(
            (p, i) =>
              '<div class="bg-white p-3 rounded-lg shadow">' +
              '<div class="text-xs text-gray-600 mb-1">Peer ' +
              (i + 1) +
              '</div>' +
              '<div class="text-base md:text-xl font-bold text-gray-800">' +
              p.ticker +
              '</div>' +
              '<div class="text-xs text-gray-600 truncate">' +
              (p.name || '') +
              '</div>' +
              '</div>'
          )
          .join('');

        const html =
          '<h2 class="text-lg font-semibold mb-2">Peer Companies in ' +
          tagsHtml +
          '</h2>' +
          '<div class="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-3">' +
          '<div class="bg-white p-3 rounded-lg shadow">' +
          '<div class="text-xs text-gray-600 mb-1">Primary</div>' +
          '<div class="text-base md:text-xl font-bold text-indigo-600 break-words">' +
          primaryLabel +
          '</div>' +
          '</div>' +
          peersCards +
          '</div>';

        const peerInfo = document.getElementById('peerInfo');
        peerInfo.innerHTML = html;
        peerInfo.classList.remove('hidden');

        const removeButton = document.getElementById('removeButton');
        const hasPeers = allPeers.length > 0;
        removeButton.disabled = !hasPeers;
        removeButton.classList.toggle('opacity-50', !hasPeers);
        removeButton.classList.toggle('cursor-not-allowed', !hasPeers);
      }

      async function addCompany() {
        const raw = document.getElementById('manualInput').value.trim();
        if (!raw) return showError('Please enter a ticker/company to add');
        hideError();
        showLoading(true);
        try {
          const r = await fetch('/api/resolve', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ input: raw })
          });
          const res = await r.json();
          if (!r.ok || res.error) throw new Error(res.error || 'Resolve failed');

          const newT = (res.ticker || '').toUpperCase();
          if (!newT) throw new Error('Could not resolve to a ticker');
          if (_tickers.includes(newT)) return showError(newT + ' already included');

          const metricsResp = await fetch('/api/get-metrics', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ tickers: [newT] })
          });
          const metrics = await metricsResp.json();
          if (
            !metrics[newT] ||
            metrics[newT].error ||
            !(metrics[newT]['Total Revenue'] || metrics[newT]['Gross Margin %'])
          )
            return showError('No usable data for ' + newT);

          _metricsData[newT] = metrics[newT];
          _tickers = uniqUpper([..._tickers, newT]);

          if (_peerData && newT !== _peerData.primary_company) {
            const exists =
              _peerData.peers.some((p) => p.ticker === newT) ||
              _manualPeers.some((p) => p.ticker === newT);
            if (!exists)
              _manualPeers.push({
                ticker: newT,
                name: metrics[newT].name || res.name || newT
              });
            displayPeers(_peerData);
          }

          _quarters = computeQuarters(_metricsData, _tickers);
          renderAll();
          await requestConclusion();

          document.getElementById('manualInput').value = '';
          document.getElementById('results').classList.remove('hidden');
        } catch (e) {
          showError(e.message);
        } finally {
          showLoading(false);
        }
      }

      async function removeCompany() {
        const raw = document.getElementById('manualInput').value.trim();
        if (!raw) return showError('Please enter a ticker/company to remove');
        if (!_peerData) return showError('No peer data available');
        hideError();
        showLoading(true);
        try {
          const r = await fetch('/api/resolve', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ input: raw })
          });
          const res = await r.json();
          if (!r.ok || res.error) throw new Error(res.error || 'Resolve failed');

          const tickerToRemove = (res.ticker || '').toUpperCase();
          if (!tickerToRemove) throw new Error('Could not resolve to a ticker');

          const primary = _peerData.primary_company;
          if (tickerToRemove === primary)
            throw new Error('Cannot remove the primary company');

          if (!_tickers.includes(tickerToRemove))
            throw new Error(tickerToRemove + ' is not in the comparison');

          _tickers = _tickers.filter((t) => t !== tickerToRemove);
          _peerData.peers = (_peerData.peers || []).filter(
            (p) => p.ticker !== tickerToRemove
          );
          _manualPeers = _manualPeers.filter((p) => p.ticker !== tickerToRemove);
          delete _metricsData[tickerToRemove];

          displayPeers(_peerData);
          _quarters = computeQuarters(_metricsData, _tickers);
          renderAll();
          await requestConclusion();

          document.getElementById('manualInput').value = '';
        } catch (e) {
          showError(e.message);
        } finally {
          showLoading(false);
        }
      }

      function computeQuarters(data, tickers) {
        const qset = new Set();
        tickers.forEach((t) => {
          const m = data[t] || {};
          if (m['Total Revenue']) {
            Object.keys(m['Total Revenue']).forEach((q) => qset.add(q));
          }
          if (m['Gross Margin %']) {
            Object.keys(m['Gross Margin %']).forEach((q) => qset.add(q));
          }
        });
        return Array.from(qset).sort().reverse().slice(0, 5);
      }

      function renderAll() {
        const resultsDiv = document.getElementById('results');
        resultsDiv.innerHTML =
          '<div class="bg-white rounded-lg shadow-xl p-3 md:p-4">' +
          '<h3 class="text-base md:text-xl font-semibold text-gray-800 mb-2">Primary Company Analysis</h3>' +
          '<div id="conclusionLoading" class="text-sm text-gray-500 hidden">Generating analysis...</div>' +
          '<pre id="conclusionText" class="whitespace-pre-wrap text-sm md:text-base text-gray-800" style="font-family: inherit;">(no analysis yet)</pre>' +
          '</div>' +
          '<div class="bg-white rounded-lg shadow-xl p-3 md:p-4">' +
          '<h3 class="text-base md:text-xl font-semibold text-gray-800 mb-2">Total Revenue & Gross Margin % Trend</h3>' +
          '<div style="height:250px"><canvas id="combinedChart"></canvas></div>' +
          '</div>' +
          '<div class="bg-white rounded-lg shadow-xl p-3 md:p-4 overflow-x-auto">' +
          '<h3 class="text-base md:text-xl font-semibold text-gray-800 mb-2">Latest Quarter Metrics</h3>' +
          '<table class="w-full text-xs md:text-sm" id="metricsTable"></table>' +
          '</div>' +
          '<div id="timeSeriesTables" class="space-y-4 md:space-y-6"></div>';

        resultsDiv.classList.remove('hidden');
        renderCharts();
        renderTable();
        renderTimeSeriesTables();
        renderConclusion();
      }

      function updateLangButtons() {
        document.getElementById('btnEN').className =
          _lang === 'en'
            ? 'px-1.5 py-0.5 text-xs border rounded-l bg-indigo-600 text-white'
            : 'px-1.5 py-0.5 text-xs border rounded-l bg-white text-gray-700';
        document.getElementById('btnZH').className =
          _lang === 'zh'
            ? 'px-1.5 py-0.5 text-xs border rounded-r bg-indigo-600 text-white'
            : 'px-1.5 py-0.5 text-xs border rounded-r bg-white text-gray-700';
      }

      function renderCharts() {
        const labels = _quarters;
        const datasetsRevenue = _tickers.map((t, i) => ({
          label: t + ' Revenue',
          data: labels.map(
            (q) =>
              (((_metricsData[t] || {})['Total Revenue'] || {})[q] || 0) /
              1_000_000_000
          ),
          backgroundColor: COLORS[i % COLORS.length],
          type: 'bar',
          yAxisID: 'y'
        }));

        const datasetsMargin = _tickers.map((t, i) => ({
          label: t + ' Margin %',
          data: labels.map(
            (q) => (((_metricsData[t] || {})['Gross Margin %'] || {})[q] || 0)
          ),
          borderColor: COLORS[i % COLORS.length],
          backgroundColor: COLORS[i % COLORS.length],
          type: 'line',
          yAxisID: 'y1',
          fill: false,
          borderWidth: 2,
          pointRadius: 3
        }));

        const combined = [...datasetsRevenue, ...datasetsMargin];

        if (_combinedChart) _combinedChart.destroy();
        _combinedChart = new Chart(document.getElementById('combinedChart'), {
          type: 'bar',
          data: { labels, datasets: combined },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            scales: {
              y: {
                type: 'linear',
                position: 'left',
                ticks: {
                  callback: (v) => '$' + Number(v).toFixed(1) + 'B'
                }
              },
              y1: {
                type: 'linear',
                position: 'right',
                grid: { drawOnChartArea: false },
                ticks: {
                  callback: (v) => Number(v).toFixed(1) + '%'
                }
              }
            }
          }
        });
      }

      function renderTable() {
        const metrics = [
          'Market Cap',
          'Total Revenue',
          'Gross Margin %',
          'Operating Expense',
          'EBIT',
          'Net Income',
          'Free Cash Flow',
          'MC/Revenue',
          'MC/Net Income'
        ];

        const tickerLatest = _tickers.map((t) => {
          const rev = ((_metricsData[t] || {})['Total Revenue']) || {};
          const qs = Object.keys(rev).sort().reverse();
          return qs[0] || (_quarters[0] || 'N/A');
        });

        let html =
          '<thead><tr class="border-b-2 border-gray-300"><th class="text-left py-2 px-1 md:px-2">Metric</th>';
        _tickers.forEach((t, i) => {
          html +=
            '<th class="text-right py-2 px-1 md:px-2">' +
            t +
            '<br/><span class="text-xs text-gray-500">' +
            tickerLatest[i] +
            '</span></th>';
        });
        html += '</tr></thead><tbody>';

        metrics.forEach((metric) => {
          html +=
            '<tr class="border-b border-gray-200"><td class="py-2 px-1 md:px-2 font-medium text-xs md:text-sm">' +
            metric +
            '</td>';
          _tickers.forEach((t, i) => {
            let q = tickerLatest[i];
            let f = 'N/A';
            if (metric === 'MC/Revenue' || metric === 'MC/Net Income') {
              const mc = (((_metricsData[t] || {})['Market Cap'] || {})['Current']);
              let denominator;
              if (metric === 'MC/Revenue') {
                denominator = (((_metricsData[t] || {})['Total Revenue'] || {})[q]);
              } else {
                denominator = (((_metricsData[t] || {})['Net Income'] || {})[q]);
              }
              if (mc && denominator && denominator !== 0) {
                const ratio = mc / denominator;
                f = ratio.toFixed(2);
              }
            } else {
              if (metric === 'Market Cap') q = 'Current';
              const v = (((_metricsData[t] || {})[metric] || {})[q]);
              if (v !== undefined && v !== null) {
                if (metric === 'Gross Margin %') {
                  f = Number(v).toFixed(1) + '%';
                } else {
                  f = '$' + (Number(v) / 1_000_000_000).toFixed(1) + 'B';
                }
              }
            }
            html += '<td class="text-right py-2 px-1 md:px-2">' + f + '</td>';
          });
          html += '</tr>';
        });

        html += '</tbody>';
        document.getElementById('metricsTable').innerHTML = html;
      }

      function renderTimeSeriesTables() {
        const metrics = [
          'Total Revenue',
          'Operating Expense',
          'Gross Margin %',
          'EBIT',
          'Net Income',
          'Free Cash Flow'
        ];
        const container = document.getElementById('timeSeriesTables');
        let html = '';

        _tickers.forEach((ticker) => {
          html +=
            '<div class="bg-white rounded-lg shadow-xl p-3 md:p-4 overflow-x-auto">' +
            '<h3 class="text-base md:text-xl font-semibold text-gray-800 mb-2">' +
            ticker +
            ' - 5 Quarter Time Series</h3>' +
            '<table class="w-full text-xs md:text-sm"><thead>' +
            '<tr class="border-b-2 border-gray-300"><th class="text-left py-2 px-1 md:px-2"></th>' +
            _quarters
              .map(
                (q) =>
                  '<th class="text-right py-2 px-1 md:px-2 text-xs">' + q + '</th>'
              )
              .join('') +
            '</tr></thead><tbody>';

          metrics.forEach((m) => {
            html +=
              '<tr class="border-b border-gray-200"><td class="py-2 px-1 md:px-2 font-medium">' +
              m +
              '</td>';
            _quarters.forEach((q) => {
              const v = (((_metricsData[ticker] || {})[m] || {})[q]);
              let f = 'N/A';
              if (v !== undefined && v !== null) {
                if (m === 'Gross Margin %') {
                  f = Number(v).toFixed(1) + '%';
                } else {
                  f = '$' + (Number(v) / 1_000_000).toFixed(0) + 'M';
                }
              }
              html += '<td class="text-right py-2 px-1 md:px-2">' + f + '</td>';
            });
            html += '</tr>';
          });

          html += '</tbody></table></div>';
        });

        container.innerHTML = html;
      }

      function primaryLatestQuarter() {
        const t = _tickers[0];
        const m = _metricsData[t] || {};
        const rev = m['Total Revenue'] || {};
        const qs = Object.keys(rev).sort().reverse();
        return qs[0] || (_quarters[0] || null);
      }

      function buildConclusionPayload() {
        const primary = _tickers[0];
        if (!primary) return null;
        const period = primaryLatestQuarter() || (_quarters[0] || null);
        const metrics = [
          'Market Cap',
          'Total Revenue',
          'Gross Margin %',
          'Operating Expense',
          'EBIT',
          'Net Income',
          'Free Cash Flow'
        ];

        const lq_rows = metrics.map((metric) => {
          const row = { metric };
          _tickers.forEach((t) => {
            const key = metric === 'Market Cap' ? 'Current' : period;
            const v = (((_metricsData[t] || {})[metric] || {})[key]);
            row[t] = v === undefined ? null : v;
          });
          return row;
        });

        const ts_rows = metrics.map((metric) => {
          const values = _quarters.map((q) => {
            const val = (((_metricsData[primary] || {})[metric] || {})[q]);
            return val === undefined ? null : val;
          });
          return { metric, values };
        });

        return {
          primary,
          latest_quarter: { period, rows: lq_rows },
          time_series: { ticker: primary, quarters: _quarters, rows: ts_rows }
        };
      }

      async function requestConclusion() {
        const payload = buildConclusionPayload();
        const loading = document.getElementById('conclusionLoading');
        if (!payload) {
          document.getElementById('conclusionText').textContent =
            '(no analysis available)';
          return;
        }
        loading.classList.remove('hidden');
        try {
          const r = await fetch('/api/peer-key-metrics-conclusion', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
          });
          const data = await r.json();
          if (!r.ok || data.error)
            throw new Error(data.error || 'Analysis API failed');
          _conclusion = {
            en: (data.conclusion_en || data.conclusion || '').trim() ||
              '(No numeric signal available to summarize.)',
            zh: (data.conclusion_zh || '').trim() || null,
            ticker: data.ticker,
            period: data.period,
            provider: data.llm
          };
          renderConclusion();
        } catch (e) {
          showError(e.message);
          document.getElementById('conclusionText').textContent =
            '(no analysis available)';
        } finally {
          loading.classList.add('hidden');
        }
      }

      function renderConclusion() {
        const el = document.getElementById('conclusionText');
        const txt =
          _lang === 'zh'
            ? _conclusion.zh || _conclusion.en || ''
            : _conclusion.en || '';
        el.textContent = txt || '(no analysis available)';
      }
    </script>
  </body>
</html>
"""

# ============================================================
# Routes
# ============================================================
@app.route("/")
def index():
    return INDEX_HTML


@app.route("/api/resolve", methods=["POST"])
def api_resolve():
    try:
        data = request.json or ""
        return jsonify(resolve_input_to_ticker(data.get("input", "")))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/find-peers", methods=["POST"])
def api_find_peers():
    try:
        data = request.json or {}
        base = _norm_ticker((data.get("ticker") or "").upper().strip())
        if not base:
            return jsonify({"error": "Ticker is required"}), 400
        result = select_peers_any_industry(base, peer_limit=3)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/get-metrics", methods=["POST"])
def api_get_metrics():
    try:
        data = request.json or {}
        tickers = data.get("tickers", [])
        if not tickers:
            return jsonify({"error": "Tickers are required"}), 400
        out = {}
        for t in tickers:
            out[t] = calculate_metrics(t) or {"error": "Unable to fetch data"}
        return jsonify(out)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/peer-key-metrics-conclusion", methods=["POST"])
def api_peer_key_metrics_conclusion():
    payload = None
    try:
        payload = request.get_json(force=True, silent=False)
        summary = analyze_primary_company(payload)
        en, llm_used = llm_conclusion_with_deepseek(summary)
        zh = translate_to_zh(en) if en else None

        return jsonify(
            {
                "ticker": summary["primary"],
                "period": summary.get("period"),
                "conclusion_en": en,
                "conclusion_zh": zh or None,
                "llm": llm_used,
            }
        )
    except Exception as e:
        # Fallback: local summary even if LLM failed
        if payload:
            try:
                summary = analyze_primary_company(payload)
                fallback = build_conclusion_text(summary)
                return jsonify(
                    {
                        "ticker": summary.get("primary"),
                        "period": summary.get("period"),
                        "conclusion_en": fallback,
                        "conclusion_zh": None,
                        "llm": "local-fallback",
                    }
                )
            except Exception:
                pass
        return jsonify({"error": str(e)}), 400


@app.route("/api/primary-company-analysis", methods=["GET"])
def api_primary_company_analysis():
    """
    Convenience endpoint: one-shot analysis for a given primary ticker (and optional peers),
    returning English or Chinese summary text.
    """
    ticker = (request.args.get("ticker") or "").strip().upper()
    peers_str = (request.args.get("peers") or "").strip()
    lang = (request.args.get("lang") or "en").lower()

    if not ticker:
        return jsonify({"error": "Missing required parameter: ticker"}), 400

    ticker = _norm_ticker(ticker)

    # Validate ticker by checking current price
    try:
        time.sleep(0.3)
        s = yf.Ticker(ticker)
        _ensure_yf_session_headers(s)
        info = s.get_info() or {}
        if not info or info.get("regularMarketPrice") is None:
            return jsonify({"error": f"Invalid or unsupported ticker: {ticker}"}), 404
    except Exception as e:
        return jsonify({"error": f"Cannot fetch data for ticker {ticker}: {str(e)}"}), 404

    try:
        if not peers_str:
            peer_result = select_peers_any_industry(ticker, peer_limit=3)
            if not peer_result or peer_result.get("error"):
                return jsonify({"error": f"Could not find peers for {ticker}"}), 404
            peers = [p["ticker"] for p in peer_result.get("peers", [])]
        else:
            peers = [
                _norm_ticker(p.strip().upper())
                for p in peers_str.split(",")
                if p.strip()
            ]

        peers = [p for p in peers if p]

        # Fetch metrics for primary + peers
        all_tickers = [ticker] + peers
        metrics: dict[str, dict] = {}
        for t in all_tickers:
            metrics[t] = calculate_metrics(t) or {"error": "Unable to fetch data"}

        if not metrics.get(ticker) or metrics[ticker].get("error"):
            return jsonify({"error": f"No financial data available for {ticker}"}), 404

        valid_peers = [
            p
            for p in peers
            if metrics.get(p)
            and not metrics[p].get("error")
            and (metrics[p].get("Total Revenue") or metrics[p].get("Gross Margin %"))
        ]

        quarters: list[str] = []
        for t in [ticker] + valid_peers:
            t_data = metrics.get(t, {})
            rev = t_data.get("Total Revenue", {})
            if isinstance(rev, dict):
                quarters.extend(rev.keys())
        quarters = sorted(list(set(quarters)), reverse=True)[:5]

        latest_period = quarters[0] if quarters else None
        metric_names = [
            "Market Cap",
            "Total Revenue",
            "Gross Margin %",
            "Operating Expense",
            "EBIT",
            "Net Income",
            "Free Cash Flow",
        ]

        lq_rows = []
        for metric in metric_names:
            row = {"metric": metric}
            for t in [ticker] + valid_peers:
                key = "Current" if metric == "Market Cap" else latest_period
                value = metrics.get(t, {}).get(metric, {}).get(key)
                row[t] = value if value is not None else None
            lq_rows.append(row)

        ts_rows = []
        for metric in metric_names:
            values = []
            for q in quarters:
                val = metrics.get(ticker, {}).get(metric, {}).get(q)
                values.append(val if val is not None else None)
            ts_rows.append({"metric": metric, "values": values})

        payload = {
            "primary": ticker,
            "latest_quarter": {"period": latest_period, "rows": lq_rows},
            "time_series": {"ticker": ticker, "quarters": quarters, "rows": ts_rows},
        }

        summary = analyze_primary_company(payload)
        analysis_text, llm_used = llm_conclusion_with_deepseek(summary)

        if lang == "zh":
            zh = translate_to_zh(analysis_text)
            final_text = zh if zh else analysis_text
        else:
            final_text = analysis_text

        return jsonify(
            {
                "ticker": ticker,
                "period": latest_period,
                "analysis": final_text,
                "language": lang,
                "llm_provider": llm_used,
                "peers": valid_peers,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/health")
def api_health():
    return jsonify(
        {
            "status": "healthy",
            "deepseek_configured": bool(DEEPSEEK_API_KEY),
            "deepseek_model": DEEPSEEK_MODEL if DEEPSEEK_API_KEY else None,
            "perplexity_configured": bool(PERPLEXITY_API_KEY),
            "perplexity_model": PERPLEXITY_MODEL if PERPLEXITY_API_KEY else None,
            "version": "2.1-production-rewrite",
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # debug=False for production; Replit will show logs regardless
    app.run(host="0.0.0.0", port=port, debug=False)
