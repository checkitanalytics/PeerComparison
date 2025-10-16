"""
Peer Company Comparison - DeepSeek-Only App
- Uses DeepSeek for analysis AND translation (local deterministic fallback if unavailable)
- Simple, robust ticker resolver (yfinance + small common-name map)
- Peer selection works across ALL sectors/industries:
    * Build a broad US-listed universe (S&P 500 + NASDAQ) via yfinance
    * Choose peers in the SAME INDUSTRY (fallback: same sector) with closest market cap
- Endpoints: /, /api/resolve, /api/find-peers, /api/get-metrics, /api/peer-key-metrics-conclusion, /api/health
- UI: "Primary Company Analysis" section ABOVE chart + EN/中文 toggle
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os, json, time, math, statistics as stats
from functools import lru_cache

import pandas as pd
import yfinance as yf
import requests

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
DEEPSEEK_MODEL    = os.environ.get("DEEPSEEK_MODEL", "deepseek-v3.2-exp")

# -----------------------------
# Perplexity config (fallback)
# -----------------------------
PERPLEXITY_API_KEY = os.environ.get("PERPLEXITY_API_KEY")
PERPLEXITY_MODEL   = os.environ.get("PERPLEXITY_MODEL", "llama-3.1-sonar-small-128k-online")

# ============================================================
# Helpers
# ============================================================
def _ensure_yf_session_headers(t: yf.Ticker):
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
    if x >= 1_000_000_000: return f"{sign}${x/1_000_000_000:.1f}B"
    if x >= 1_000_000:     return f"{sign}${x/1_000_000:.0f}M"
    if x >= 1_000:         return f"{sign}${x/1_000:.0f}K"
    return f"{sign}${x:.0f}"

def deepseek_chat(messages, temperature=0.1, timeout=30) -> str | None:
    """Minimal DeepSeek chat wrapper. Returns text or None."""
    if not DEEPSEEK_API_KEY:
        return None
    try:
        r = requests.post(
            f"{DEEPSEEK_API_BASE}/chat/completions",
            headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"},
            json={"model": DEEPSEEK_MODEL, "temperature": temperature, "messages": messages},
            timeout=timeout
        )
        r.raise_for_status()
        data = r.json()
        return (data.get("choices") or [{}])[0].get("message", {}).get("content", "").strip() or None
    except Exception:
        return None

def perplexity_chat(messages, temperature=0.2, timeout=30) -> str | None:
    """Perplexity AI fallback wrapper. Returns text or None."""
    if not PERPLEXITY_API_KEY:
        return None
    try:
        r = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers={"Authorization": f"Bearer {PERPLEXITY_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": PERPLEXITY_MODEL,
                "temperature": temperature,
                "messages": messages,
                "top_p": 0.9,
                "stream": False
            },
            timeout=timeout
        )
        r.raise_for_status()
        data = r.json()
        return (data.get("choices") or [{}])[0].get("message", {}).get("content", "").strip() or None
    except Exception:
        return None

# ============================================================
# Ticker resolve (no OpenAI, no S3)
# ============================================================
COMMON_NAME_MAP = {
    "tesla": "TSLA", "apple": "AAPL", "microsoft": "MSFT", "amazon": "AMZN",
    "google": "GOOGL", "alphabet": "GOOGL", "meta": "META", "facebook": "META",
    "nvidia": "NVDA", "netflix": "NFLX", "boeing": "BA", "airbus": "AIR.PA"
}
PEER_TICKER_ALIAS = {"GOOG": "GOOGL", "FB": "META"}
def _norm_ticker(t: str) -> str:
    u = (t or "").upper().strip()
    return PEER_TICKER_ALIAS.get(u, u)

def _verify_ticker_with_yfinance(ticker: str) -> dict | None:
    try:
        t = _norm_ticker(ticker)
        s = yf.Ticker(t); _ensure_yf_session_headers(s)
        info = s.get_info() or {}
        nm = info.get("longName") or info.get("shortName")
        if nm: return {"ticker": t, "name": nm}
        if info.get("symbol") == t: return {"ticker": t, "name": None}
        return None
    except Exception:
        return None

def resolve_input_to_ticker(user_input: str) -> dict:
    raw = (user_input or "").strip()
    if not raw:
        return {"error":"Input is empty"}

    # Try direct ticker first
    if raw.isalpha() and 1 <= len(raw) <= 6:
        v = _verify_ticker_with_yfinance(raw)
        if v: return {"input": raw, "ticker": v["ticker"], "name": v.get("name"), "source": "input"}

    # Try common names
    norm = raw.lower()
    if norm in COMMON_NAME_MAP:
        v = _verify_ticker_with_yfinance(COMMON_NAME_MAP[norm])
        if v: return {"input": raw, "ticker": v["ticker"], "name": v.get("name"), "source": "common"}

    # Last: treat as ticker again
    v = _verify_ticker_with_yfinance(raw)
    if v: return {"input": raw, "ticker": v["ticker"], "name": v.get("name"), "source": "guess"}

    return {"error": f"Could not resolve '{raw}' to a ticker"}

# ============================================================
# Profiles & metrics
# ============================================================
@lru_cache(maxsize=512)
def fetch_profile(ticker: str) -> dict:
    t = _norm_ticker(ticker)
    s = yf.Ticker(t); _ensure_yf_session_headers(s)
    info = {}
    try: info = s.get_info() or {}
    except Exception: pass
    return {
        "ticker": t,
        "name": info.get("longName") or info.get("shortName") or t,
        "industry": info.get("industry"),
        "sector": info.get("sector"),
        "market_cap": info.get("marketCap"),
    }

@lru_cache(maxsize=128)
def calculate_metrics(ticker: str, max_retries: int = 3):
    def _pick(df, labels):
        if df is None or df.empty: return None
        idx = {str(i).strip().lower(): i for i in df.index}
        for lab in labels:
            k = lab.strip().lower()
            if k in idx: return df.loc[idx[k]]
        return None

    def _to_quarterly(df):
        if df is None or df.empty: return df
        try:
            df = df.copy()
            if not isinstance(df.columns, pd.PeriodIndex):
                df.columns = pd.to_datetime(df.columns, errors="coerce").to_period("Q")
            return df.iloc[:, :5]
        except Exception:
            return df

    t = _norm_ticker(ticker)
    if not t: return None

    for attempt in range(max_retries):
        try:
            if attempt: time.sleep(0.8 * attempt)
            s = yf.Ticker(t); _ensure_yf_session_headers(s)
            fin_q = getattr(s, "quarterly_financials", None)
            if fin_q is None:
                fin_q = getattr(s, "financials", None)
            cf_q = getattr(s, "quarterly_cashflow", None)
            if cf_q is None:
                cf_q = getattr(s, "cashflow", None)
            if fin_q is None or fin_q.empty or cf_q is None or cf_q.empty: continue
            fin_q, cf_q = _to_quarterly(fin_q), _to_quarterly(cf_q)

            total_rev    = _pick(fin_q, ["Total Revenue","Revenue","Operating Revenue"])
            cost_rev     = _pick(fin_q, ["Cost Of Revenue","Cost of Revenue","Reconciled Cost Of Revenue"])
            gross_profit = _pick(fin_q, ["Gross Profit"])
            opex         = _pick(fin_q, ["Operating Expense","Operating Expenses","Total Operating Expenses","Total Expenses"])
            ebit         = _pick(fin_q, ["EBIT","Operating Income","Total Operating Income As Reported"])
            net_income   = _pick(fin_q, ["Net Income","Net Income Common Stockholders","Net Income From Continuing Operation Net Minority Interest"])

            ocf   = _pick(cf_q, ["Operating Cash Flow","Total Cash From Operating Activities"])
            capex = _pick(cf_q, ["Capital Expenditure","Capital Expenditures"])

            # Calculate gross profit if not directly available
            if gross_profit is None and total_rev is not None and cost_rev is not None:
                gross_profit = (total_rev - cost_rev)
            
            # Calculate operating expense from components
            if opex is None:
                sga = _pick(fin_q, ["Selling General And Administration","SG&A Expense","General And Administrative Expense"])
                rnd = _pick(fin_q, ["Research And Development","Research & Development"])
                selling = _pick(fin_q, ["Selling And Marketing Expense"])
                
                if sga is not None and rnd is not None:
                    opex = (sga + rnd)
                elif sga is not None and selling is not None:
                    opex = (sga + selling)
                elif sga is not None:
                    opex = sga
            
            # Calculate EBIT from available fields
            if ebit is None:
                if gross_profit is not None and opex is not None:
                    try: ebit = (gross_profit - opex)
                    except Exception: pass
                
                if ebit is None and total_rev is not None and cost_rev is not None and opex is not None:
                    try: ebit = (total_rev - cost_rev - opex)
                    except Exception: pass
                
                if ebit is None:
                    pretax = _pick(fin_q, ["Pretax Income"])
                    if pretax is not None:
                        ebit = pretax

            fcf = (ocf + capex) if (ocf is not None and capex is not None) else None

            out = {}
            def _put(name, s, pct=False):
                if s is None: return
                series = {}
                for q, v in s.items():
                    if pd.isna(v): continue
                    try:
                        series[str(q)] = float(v) if pct else int(float(v))
                    except Exception: continue
                if series: out[name] = series

            gm_pct = None
            if gross_profit is not None and total_rev is not None:
                try: gm_pct = (gross_profit / total_rev) * 100.0
                except Exception: gm_pct = None

            _put("Total Revenue", total_rev, False)
            _put("Operating Expense", opex, False)
            _put("EBIT", ebit, False)
            _put("Net Income", net_income, False)
            if gm_pct is not None: _put("Gross Margin %", gm_pct, True)
            if fcf is not None:    _put("Free Cash Flow", fcf, False)

            return out if out.get("Total Revenue") or out.get("Gross Margin %") else None
        except Exception:
            continue
    return None

# ============================================================
# Peer selection that covers ALL sectors/industries
# ============================================================
_UNIVERSE: list[str] = []
_UNIVERSE_BUILT = False

def _build_universe(max_size: int = 1800):
    """Combine S&P 500 + NASDAQ tickers (broad coverage) and cache."""
    global _UNIVERSE, _UNIVERSE_BUILT
    if _UNIVERSE_BUILT and _UNIVERSE:
        return
    try:
        sp = yf.tickers_sp500() or []
    except Exception:
        sp = []
    try:
        ndq = yf.tickers_nasdaq() or []
    except Exception:
        ndq = []
    allu = list(dict.fromkeys([_norm_ticker(t) for t in (sp + ndq)]))
    # Filter out weird tickers (too long or have non-alpha plus-dot except BRK.B style handled upstream via aliasing)
    cleaned = [t for t in allu if t and len(t) <= 6 and t.isalpha()]
    _UNIVERSE = cleaned[:max_size] if max_size else cleaned
    _UNIVERSE_BUILT = True

def _ratio_score(base_mc, mc):
    if not base_mc or not mc or base_mc <= 0 or mc <= 0:
        return float('inf')
    big, small = (mc, base_mc) if mc >= base_mc else (base_mc, mc)
    return abs((big / small) - 1.0)

def select_peers_any_industry(base_ticker: str, peer_limit: int = 2):
    """
    Choose peers from a broad universe:
      1) Try SAME INDUSTRY → closest market cap
      2) If insufficient, use SAME SECTOR → closest market cap
      3) If still insufficient, pick any tickers closest in market cap
    """
    _build_universe()
    base_prof = fetch_profile(base_ticker)
    base_ind, base_sector, base_mc = base_prof.get("industry"), base_prof.get("sector"), base_prof.get("market_cap")

    # 1) Same industry
    same_ind = []
    for t in _UNIVERSE:
        if t == base_prof["ticker"]: continue
        pr = fetch_profile(t)
        if base_ind and pr.get("industry") and pr["industry"] == base_ind:
            same_ind.append((t, _ratio_score(base_mc, pr.get("market_cap"))))
    same_ind.sort(key=lambda x: x[1])
    peers = [ {"ticker": t} for t,_ in same_ind[:peer_limit] ]

    # 2) Fallback: same sector
    if len(peers) < peer_limit:
        same_sec = []
        for t in _UNIVERSE:
            if t == base_prof["ticker"]: continue
            pr = fetch_profile(t)
            if base_sector and pr.get("sector") and pr["sector"] == base_sector:
                same_sec.append((t, _ratio_score(base_mc, pr.get("market_cap"))))
        same_sec.sort(key=lambda x: x[1])
        needed = peer_limit - len(peers)
        peers.extend([{"ticker": t} for t,_ in same_sec[:needed]])

    # 3) Fallback: any closest market cap
    if len(peers) < peer_limit:
        anyc = []
        for t in _UNIVERSE:
            if t == base_prof["ticker"]: continue
            pr = fetch_profile(t)
            anyc.append((t, _ratio_score(base_mc, pr.get("market_cap"))))
        anyc.sort(key=lambda x: x[1])
        needed = peer_limit - len(peers)
        peers.extend([{"ticker": t} for t,_ in anyc[:needed]])

    # Attach names
    peers_named = [{"ticker": p["ticker"], "name": fetch_profile(p["ticker"]).get("name")} for p in peers[:peer_limit]]
    return {"primary_company": base_prof["ticker"], "industry": base_ind or (base_sector or "N/A"), "peers": peers_named}

# ============================================================
# Analysis math + DeepSeek phrasing
# ============================================================
def safe(v):
    return v if (v is not None and not (isinstance(v, float) and math.isnan(v))) else None

def pct(n, d):
    try:
        if d in (0, None) or n is None: return None
        return (n - d) / abs(d) * 100.0
    except Exception: return None

def ppoints(n, d):
    try:
        if n is None or d is None: return None
        return (n - d) * 100.0
    except Exception: return None

def rank_desc(value, peer_values):
    arr = sorted([x for x in peer_values if x is not None], reverse=True)
    if value is None or not arr: return None
    try: return 1 + arr.index(value)
    except ValueError:
        diffs = sorted([(abs(value-x), i) for i, x in enumerate(arr)])
        return 1 + diffs[0][1]

def latest_of(row):
    vals = row.get("values", [])
    return safe(vals[0]) if vals else None

def get_ts_metric(rows, name):
    n = (name or "").strip().lower()
    for r in rows:
        if (r.get("metric", "").strip().lower()) == n: return r
    return None

def analyze_primary_company(payload: dict) -> dict:
    primary = _norm_ticker(payload.get("primary"))
    lqm = payload.get("latest_quarter") or {}
    ts  = payload.get("time_series") or {}
    peer_rows = lqm.get("rows", []) or []
    period = lqm.get("period") or "Latest"

    if not peer_rows: raise ValueError("latest_quarter.rows missing")
    peer_keys = [k for k in peer_rows[0].keys() if k != "metric"]
    if primary not in peer_keys: raise ValueError("Primary not in latest_quarter rows")

    # latest snapshot ranks
    latest_metrics = {}
    for r in peer_rows:
        m = (r.get("metric") or "").strip()
        vals = [safe(r.get(t)) for t in peer_keys]
        latest_metrics[m] = {"primary": safe(r.get(primary)), "peers_values": vals}

    def metric_rank(mname, higher_is_better=True):
        md = latest_metrics.get(mname, {})
        pv = md.get("primary"); allv = md.get("peers_values", [])
        if not higher_is_better and pv is not None:
            allv = [(-x if x is not None else None) for x in allv]; pv = -pv
        r = rank_desc(pv, allv)
        n = len([x for x in allv if x is not None])
        return (r, n) if r is not None else (None, n)

    rev_rank = metric_rank("Total Revenue")
    gm_rank  = metric_rank("Gross Margin %")
    ebit_rank= metric_rank("EBIT")
    ni_rank  = metric_rank("Net Income")
    fcf_rank = metric_rank("Free Cash Flow")
    opex_rank= metric_rank("Operating Expense", higher_is_better=False)

    # time series deltas
    quarters = ts.get("quarters", []) or []
    rows = ts.get("rows", []) or []
    rev_row, gm_row = get_ts_metric(rows,"Total Revenue"), get_ts_metric(rows,"Gross Margin %")
    opex_row, ebit_row = get_ts_metric(rows,"Operating Expense"), get_ts_metric(rows,"EBIT")
    ni_row, fcf_row = get_ts_metric(rows,"Net Income"), get_ts_metric(rows,"Free Cash Flow")

    def qoq(row): return None if not row or len(row.get("values",[]))<2 else pct(row["values"][0], row["values"][1])
    def yoy(row): return None if not row or len(row.get("values",[]))<5 else pct(row["values"][0], row["values"][4])

    rev_qoq, rev_yoy = qoq(rev_row), yoy(rev_row)
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

    return {
        "primary": primary, "period": period,
        "peer_ranks": {
            "revenue_rank": rev_rank, "gross_margin_rank": gm_rank,
            "ebit_rank": ebit_rank, "net_income_rank": ni_rank,
            "fcf_rank": fcf_rank, "opex_rank": opex_rank
        },
        "timeseries": {
            "quarters": quarters,
            "rev_qoq_pct": rev_qoq, "rev_yoy_pct": rev_yoy,
            "gm_qoq_pp": gm_qoq_pp, "gm_yoy_pp": gm_yoy_pp,
            "opex_pct_now": opex_pct_now, "opex_pct_yearago": opex_pct_ya,
            "ebit_qoq_pct": ebit_qoq, "ebit_yoy_pct": ebit_yoy,
            "ni_qoq_pct": ni_qoq, "ni_yoy_pct": ni_yoy,
            "fcf_cv_pct": fcf_cv
        },
        "latest": latest
    }

def build_conclusion_text(summary: dict) -> str:
    """Deterministic local fallback (English)."""
    p = summary["primary"]; period = summary.get("period") or "Latest Quarter"
    pr, ts, lt = summary["peer_ranks"], summary["timeseries"], summary["latest"]

    def rank_str(lbl, tup):
        if not tup or tup[0] is None or tup[1] is None: return f"{lbl}: n/a"
        r, n = tup; return f"{lbl}: #{r}/{n}" if r!=1 else f"{lbl}: #1/{n}"

    bullets = [
        f"{p}: Past-performance takeaway — {period}.",
        f"- Peer snapshot: {rank_str('Revenue',pr.get('revenue_rank'))}; {rank_str('GM',pr.get('gross_margin_rank'))}; "
        f"{rank_str('EBIT',pr.get('ebit_rank'))}; {rank_str('Net income',pr.get('net_income_rank'))}; "
        f"{rank_str('FCF',pr.get('fcf_rank'))}; {rank_str('OpEx (lower better)',pr.get('opex_rank'))}.",
    ]

    if ts.get("rev_qoq_pct") is not None or ts.get("rev_yoy_pct") is not None:
        qoq = f"{ts['rev_qoq_pct']:.1f}%" if ts.get('rev_qoq_pct') is not None else "n/a"
        yoy = f"{ts['rev_yoy_pct']:.1f}%" if ts.get('rev_yoy_pct') is not None else "n/a"
        bullets.append(f"- Revenue: QoQ {qoq}; YoY {yoy} → {fmt_money_short(lt.get('revenue'))}.")
    if ts.get("gm_qoq_pp") is not None or ts.get("gm_yoy_pp") is not None or lt.get("gm") is not None:
        gm_now = f'{lt.get("gm"):.1f}%' if isinstance(lt.get("gm"), (int, float)) else "n/a"
        parts = []
        if ts.get("gm_qoq_pp") is not None: parts.append(f"QoQ {ts['gm_qoq_pp']:+.1f}pp")
        if ts.get("gm_yoy_pp") is not None: parts.append(f"YoY {ts['gm_yoy_pp']:+.1f}pp")
        bullets.append(f"- Gross margin: {gm_now}" + (f" ({'; '.join(parts)})" if parts else "") + ".")
    if ts.get("opex_pct_now") is not None:
        if ts.get("opex_pct_yearago") is not None:
            d = ts["opex_pct_now"] - ts["opex_pct_yearago"]
            bullets.append(f"- OpEx ratio: {ts['opex_pct_now']:.1f}% (YoY {d:+.1f}pp).")
        else:
            bullets.append(f"- OpEx ratio: {ts['opex_pct_now']:.1f}%.")

    if ts.get("ebit_qoq_pct") is not None or ts.get("ebit_yoy_pct") is not None:
        qoq = f"{ts['ebit_qoq_pct']:.1f}%" if ts.get('ebit_qoq_pct') is not None else "n/a"
        yoy = f"{ts['ebit_yoy_pct']:.1f}%" if ts.get('ebit_yoy_pct') is not None else "n/a"
        bullets.append(f"- EBIT: QoQ {qoq}; YoY {yoy} → {fmt_money_short(lt.get('ebit'))}.")
    if ts.get("ni_qoq_pct") is not None or ts.get("ni_yoy_pct") is not None:
        qoq = f"{ts['ni_qoq_pct']:.1f}%" if ts.get('ni_qoq_pct') is not None else "n/a"
        yoy = f"{ts['ni_yoy_pct']:.1f}%" if ts.get('ni_yoy_pct') is not None else "n/a"
        bullets.append(f"- Net income: QoQ {qoq}; YoY {yoy} → {fmt_money_short(lt.get('ni'))}.")
    if ts.get("fcf_cv_pct") is not None or lt.get("fcf") is not None:
        stability = None
        if ts.get("fcf_cv_pct") is not None:
            stability = "stable" if ts["fcf_cv_pct"] < 25 else "volatile"
        tail = f" ({stability}, CV {ts['fcf_cv_pct']:.0f}%)." if stability else "."
        bullets.append(f"- Free cash flow: {fmt_money_short(lt.get('fcf'))}{tail}")

    return "\n".join(bullets)

def llm_conclusion_with_deepseek(summary: dict) -> tuple[str, str]:
    """DeepSeek phrasing with Perplexity fallback; final fallback to local builder.
    Returns: (conclusion_text, llm_used)
    """
    system = {"role":"system","content":"You are a finance analyst. Use ONLY numbers provided in the JSON. Be concise and factual."}
    user = {"role":"user","content": json.dumps({
        "task": "Write analyst-style past-performance summary for PRIMARY only.",
        "style": "4–7 bullets; include ranks vs peers, growth, margin pp deltas, OpEx ratio, EBIT/NI, FCF stability.",
        "format": [
            "Start with '<TICKER>: Past-performance takeaway — <period>.'",
            "Bullets use '-' prefix; keep each under ~200 chars.",
            "1 decimal for %; 'pp' for margin deltas."
        ],
        "data": summary
    }, ensure_ascii=False)}
    
    out = deepseek_chat([system, user], temperature=0.1)
    if out:
        return (out, "deepseek")
    
    out = perplexity_chat([system, user], temperature=0.2)
    if out:
        return (out, "perplexity")
    
    return (build_conclusion_text(summary), "local-fallback")

def translate_to_zh(text: str) -> str | None:
    if not text: return None
    system = {"role":"system","content":"You are a professional bilingual equity research translator."}
    user = {"role":"user","content": json.dumps({"task":"Translate to Simplified Chinese with concise finance tone","text":text}, ensure_ascii=False)}
    return deepseek_chat([system, user], temperature=1.3)

# ============================================================
# Routes
# ============================================================
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
      <input id="tickerInput" placeholder="e.g., TSLA" class="w-28 px-2 py-1 border rounded text-sm"/>
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
    await requestConclusion(); // generate analysis after rendering
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
  <!-- Primary Company Analysis ABOVE the chart -->
  <div class="bg-white rounded-lg shadow-xl p-3 md:p-4">
    <div class="flex items-center justify-between mb-2">
      <h3 class="text-base md:text-xl font-semibold text-gray-800">Primary Company Analysis</h3>
      <div class="inline-flex rounded-md shadow-sm">
        <button id="btnEN" class="px-2 py-1 text-xs border rounded-l bg-indigo-600 text-white">EN</button>
        <button id="btnZH" class="px-2 py-1 text-xs border rounded-r bg-white text-gray-700">中文</button>
      </div>
    </div>
    <div id="conclusionLoading" class="text-sm text-gray-500 hidden">Generating analysis…</div>
    <pre id="conclusionText" class="whitespace-pre-wrap text-sm md:text-base text-gray-800">(no analysis yet)</pre>
    <div id="conclusionMeta" class="text-xs text-gray-500 mt-1"></div>
  </div>

  <div class="bg-white rounded-lg shadow-xl p-3 md:p-4">
    <h3 class="text-base md:text-xl font-semibold text-gray-800 mb-2">Total Revenue & Gross Margin % Trend</h3>
    <div style="height:250px"><canvas id="combinedChart"></canvas></div>
  </div>

  <div class="bg-white rounded-lg shadow-xl p-3 md:p-4 overflow-x-auto">
    <h3 class="text-base md:text-xl font-semibold text-gray-800 mb-2">Latest Quarter Metrics</h3>
    <table class="w-full text-xs md:text-sm" id="metricsTable"></table>
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

// ---------- Analysis payload ----------
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
  const loading = document.getElementById('conclusionLoading');
  if (!payload){ document.getElementById('conclusionText').textContent='(no analysis available)'; return; }
  loading.classList.remove('hidden');
  try{
    const r = await fetch('/api/peer-key-metrics-conclusion',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
    const data = await r.json();
    if(!r.ok || data.error) throw new Error(data.error || 'Analysis API failed');
    _conclusion = {
      en: (data.conclusion_en || data.conclusion || '').trim() || '(No numeric signal available to summarize.)',
      zh: (data.conclusion_zh || '').trim() || null,
      ticker: data.ticker, period: data.period, provider: data.llm
    };
    renderConclusion();
  }catch(e){
    showError(e.message);
    document.getElementById('conclusionText').textContent='(no analysis available)';
  }finally{
    loading.classList.add('hidden');
  }
}

function renderConclusion(){
  const el = document.getElementById('conclusionText');
  const meta = document.getElementById('conclusionMeta');
  const txt = (_lang==='zh') ? (_conclusion.zh || _conclusion.en || '') : (_conclusion.en || '');
  el.textContent = txt || '(no analysis available)';
  meta.textContent = _conclusion.provider ? `Generated by: ${_conclusion.provider==='deepseek' ? 'DeepSeek' : 'Local fallback'} • Period: ${_conclusion.period || 'Latest'}` : '';
}
</script>
</body></html>
    '''

@app.route('/api/resolve', methods=['POST'])
def api_resolve():
    try:
        data = request.json or {}
        return jsonify(resolve_input_to_ticker(data.get("input", "")))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/find-peers', methods=['POST'])
def find_peers():
    try:
        data = request.json or {}
        base = _norm_ticker((data.get('ticker') or '').upper().strip())
        if not base: return jsonify({'error': 'Ticker is required'}), 400
        result = select_peers_any_industry(base, peer_limit=2)
        return jsonify(result)
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

@app.route('/api/peer-key-metrics-conclusion', methods=['POST'])
def peer_key_metrics_conclusion():
    """
    Input: { "primary": "...", "latest_quarter": {...}, "time_series": {...} }
    Output: { "ticker": "...", "period": "...", "conclusion_en": "...", "conclusion_zh": "...|null", "llm": "deepseek|perplexity|local-fallback" }
    """
    payload = None
    try:
        payload = request.get_json(force=True, silent=False)
        summary = analyze_primary_company(payload)
        en, llm_used = llm_conclusion_with_deepseek(summary)
        zh = translate_to_zh(en) if en else None
        
        return jsonify({
            "ticker": summary["primary"],
            "period": summary.get("period"),
            "conclusion_en": en,
            "conclusion_zh": zh or None,
            "llm": llm_used
        })
    except Exception as e:
        # Deterministic fallback even on errors
        if payload:
            try:
                summary = analyze_primary_company(payload)
                fallback = build_conclusion_text(summary)
                return jsonify({
                    "ticker": summary.get("primary"),
                    "period": summary.get("period"),
                    "conclusion_en": fallback,
                    "conclusion_zh": None,
                    "llm": "local-fallback"
                })
            except Exception:
                pass
        return jsonify({"error": str(e)}), 400

@app.route('/api/health')
def health():
    return jsonify({
        "status": "healthy",
        "deepseek_configured": bool(DEEPSEEK_API_KEY),
        "deepseek_model": DEEPSEEK_MODEL if DEEPSEEK_API_KEY else None,
        "perplexity_configured": bool(PERPLEXITY_API_KEY),
        "perplexity_model": PERPLEXITY_MODEL if PERPLEXITY_API_KEY else None
    })

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
