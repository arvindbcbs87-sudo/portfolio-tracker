#!/usr/bin/env python3
"""
Portfolio Tracker Server
  /api/prices         - live LTP + day change  (cached 60s)
  /api/analysis       - RSI, MACD, trend, fundamentals, news (cached 1h, pre-computed on start)
  /api/chart          - 6-month price + indicator history for a single stock (cached 30min)
  /api/portfolio      - GET/POST portfolio data (persisted to portfolio.json)
  /api/analyse_stock  - analyse a single stock on demand, add to cache
"""

import json, time, threading, os, math
import urllib.parse
from http.server import HTTPServer, SimpleHTTPRequestHandler
from concurrent.futures import ThreadPoolExecutor, as_completed

PORT      = int(os.environ.get('PORT', 3000))
SERVE_DIR = os.path.dirname(os.path.abspath(__file__))
PORTFOLIO_FILE = os.path.join(SERVE_DIR, 'portfolio.json')
SGB_SET   = {'SGBJUL28IV-GB', 'SGBJUN28-GB', 'SGBMAY29I'}

import yfinance as yf
print("yfinance ready")

# ── Cache ──────────────────────────────────────────────────────────────────────
_pc  = {'data':{}, 'ts':0, 'error':None, 'lock': threading.Lock()}
_ac  = {'data':{}, 'ts':0, 'error':None, 'lock': threading.Lock()}
_cc  = {}
_ccl = threading.Lock()

PRICE_TTL    = 60
ANALYSIS_TTL = 3600
CHART_TTL    = 1800

ALL_SYMS = [
    'BAJFINANCE','BAJAJFINSV','HDFCBANK','HINDUNILVR','JIOFIN','M&M',
    'NTPC','ONGC','POWERGRID','RADIOCITY','RECLTD','RELIANCE',
    'SUNPHARMA','SUVEN','TATAMOTORS','WIPRO','YESBANK',
    'ALKYLAMINE','BATAINDIA','CAMPUS','DMART','HAPPSTMNDS','IOLCP',
    'KOTAKBANK','LT','NSLNISP','POLYPLEX','RELAXO','SBICARD',
    'SHREECEM','TATAPOWER','TCS'
]

def nse(sym):
    return None if sym in SGB_SET else urllib.parse.quote(sym, safe='') + '.NS'

def rsi14(s):
    s = s.dropna()
    if len(s) < 15:
        import pandas as pd
        return pd.Series([float('nan')] * len(s), index=s.index)
    d = s.diff()
    g = d.clip(lower=0).rolling(14, min_periods=14).mean()
    l = (-d.clip(upper=0)).rolling(14, min_periods=14).mean()
    # avoid 0/0 and x/0: replace 0 loss with NaN so we get NaN (not inf) RSI
    l_safe = l.replace(0, float('nan'))
    return (100 - 100 / (1 + g / l_safe)).round(2)

def sf(v):
    try:
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else round(f, 4)
    except: return None

def clean(series):
    return [sf(v) for v in series]

def clean_json(obj):
    """Recursively replace NaN/Infinity floats with None so json.dumps produces valid JSON."""
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_json(v) for v in obj]
    return obj

# ── PRICES ─────────────────────────────────────────────────────────────────────
def fetch_prices(symbols):
    sm = {nse(s): s for s in symbols if nse(s)}
    if not sm: return {}
    ys, multi = list(sm.keys()), len(sm) > 1
    g = 'ticker' if multi else 'column'
    result = {}
    try:
        df = yf.download(ys, period='1d', interval='1m', progress=False,
                         auto_adjust=True, group_by=g, threads=True, timeout=30)
        if df is not None and not df.empty:
            for y in ys:
                try:
                    c = (df['Close'] if not multi else df[y]['Close']).dropna()
                    if not c.empty: result[sm[y]] = {'ltp': round(float(c.iloc[-1]), 2)}
                except: pass
    except Exception as e: print(f"Prices 1m error: {e}")
    try:
        df2 = yf.download(ys, period='5d', interval='1d', progress=False,
                          auto_adjust=True, group_by=g, threads=True, timeout=30)
        if df2 is not None and not df2.empty:
            for y in ys:
                orig = sm[y]
                if orig not in result: continue
                try:
                    c = (df2['Close'] if not multi else df2[y]['Close']).dropna()
                    if len(c) >= 2:
                        prev, ltp = float(c.iloc[-2]), result[orig]['ltp']
                        chg = ltp - prev
                        result[orig].update({'change': round(chg,2), 'changePct': round(chg/prev*100,2) if prev else 0})
                    else: result[orig].update({'change':0,'changePct':0})
                except: result[orig].update({'change':0,'changePct':0})
    except Exception as e: print(f"Prices 1d error: {e}")
    return result

# ── ANALYSIS (per-stock) ───────────────────────────────────────────────────────
def analyse_one(sym):
    y = nse(sym)
    if not y: return sym, {'call':'HOLD','trend':'N/A','rsi':None,'news':[]}
    try:
        t = yf.Ticker(y)
        h = t.history(period='1y', interval='1d')
        if h.empty: return sym, {'call':'HOLD','trend':'N/A','rsi':None,'news':[]}
        c   = h['Close'].dropna()   # clean series — removes any NaN price rows
        vol = h['Volume'].dropna()
        if len(c) < 20:
            return sym, {'call':'HOLD','trend':'N/A','rsi':None,'news':[],'error':'insufficient data'}
        ltp = float(c.iloc[-1])

        # ── Moving averages (use sf() so NaN → None, not a crash) ────────────────
        ma20  = sf(c.rolling(20, min_periods=20).mean().iloc[-1])
        ma50  = sf(c.rolling(50, min_periods=50).mean().iloc[-1])
        ma200 = sf(c.rolling(200, min_periods=200).mean().iloc[-1])

        # ── RSI ───────────────────────────────────────────────────────────────────
        rsi_v = sf(rsi14(c).iloc[-1])   # sf() converts NaN → None

        # ── MACD histogram + direction ────────────────────────────────────────────
        ema12     = c.ewm(span=12, min_periods=12).mean()
        ema26     = c.ewm(span=26, min_periods=26).mean()
        macd_line = ema12 - ema26
        sig_line  = macd_line.ewm(span=9, min_periods=9).mean()
        hist      = macd_line - sig_line
        mh_cur    = sf(hist.iloc[-1])
        mh_prev   = sf(hist.iloc[-2]) if len(hist) > 1 else mh_cur

        # ── Bollinger Bands ───────────────────────────────────────────────────────
        bb_m = c.rolling(20, min_periods=20).mean()
        bb_s = c.rolling(20, min_periods=20).std()
        bb_u = sf((bb_m + 2*bb_s).iloc[-1])
        bb_l = sf((bb_m - 2*bb_s).iloc[-1])

        # ── 52-week range ─────────────────────────────────────────────────────────
        w52h = float(c.max()); w52l = float(c.min())
        pct_from_high = (ltp - w52h) / w52h * 100

        # ── Volume (5-day avg vs 20-day avg) ──────────────────────────────────────
        vol_ma20  = sf(vol.rolling(20, min_periods=20).mean().iloc[-1]) if len(vol) >= 20 else None
        vol_5d    = sf(vol.tail(5).mean()) if len(vol) >= 5 else None
        vol_ratio = round(vol_5d / vol_ma20, 2) if (vol_ma20 and vol_5d and vol_ma20 > 0) else None
        price_5d_ago = float(c.iloc[-6]) if len(c) > 5 else None

        # ── Trend (only if MAs are valid) ─────────────────────────────────────────
        if ma50 is not None and ma200 is not None:
            trend = ('UPTREND'   if ltp > ma50 > ma200 else
                     'DOWNTREND' if ltp < ma50 < ma200 else 'SIDEWAYS')
        elif ma50 is not None:
            trend = 'UPTREND' if ltp > ma50 else 'DOWNTREND'
        else:
            trend = 'SIDEWAYS'

        # ── Fundamentals ─────────────────────────────────────────────────────────
        try:   info = t.info
        except: info = {}
        pe_val  = sf(info.get('trailingPE'))
        roe_val = sf(info.get('returnOnEquity'))
        rev_g   = sf(info.get('revenueGrowth'))

        # ── WEIGHTED SCORE ────────────────────────────────────────────────────────
        # Range: approx -12 to +12
        # BUY >= +4  |  HOLD -3 to +3  |  REDUCE <= -4
        score = 0

        # 1. RSI — momentum / oversold-overbought (-3 to +3)
        if rsi_v is not None:
            if   rsi_v < 25: score += 3
            elif rsi_v < 35: score += 2
            elif rsi_v < 45: score += 1
            elif rsi_v < 55: pass
            elif rsi_v < 65: score -= 1
            elif rsi_v < 75: score -= 2
            else:            score -= 3

        # 2. MACD histogram — momentum direction (-2 to +2)
        if mh_cur is not None and mh_prev is not None:
            if   mh_cur > 0 and mh_cur >= mh_prev: score += 2
            elif mh_cur > 0:                        score += 1
            elif mh_cur < 0 and mh_cur > mh_prev:  score -= 1
            else:                                   score -= 2

        # 3. Trend via moving averages (-2 to +2)
        if ma50 is not None and ma200 is not None:
            if   ltp > ma50 > ma200: score += 2
            elif ltp > ma200:        score += 1
            elif ltp > ma50:         score -= 1
            else:                    score -= 2
        elif ma50 is not None:
            if   ltp > ma50: score += 1
            else:            score -= 1

        # 4. Bollinger Bands position (-1 to +1)
        if bb_l is not None and bb_u is not None:
            if   ltp <= bb_l * 1.02: score += 1
            elif ltp >= bb_u * 0.98: score -= 1

        # 5. Volume confirmation (-1 to +1)
        if vol_ratio is not None and price_5d_ago is not None:
            if   vol_ratio > 1.2 and ltp >= price_5d_ago: score += 1
            elif vol_ratio > 1.2 and ltp <  price_5d_ago: score -= 1

        # 6. 52-week position (-1 to +1)
        if   pct_from_high < -40: score += 1
        elif pct_from_high > -5:  score -= 1

        # 7. P/E ratio (-1 to +1)
        if   pe_val and 0 < pe_val < 15: score += 1
        elif pe_val and pe_val > 50:     score -= 1

        # 8. Return on Equity (0 to +1)
        if roe_val and roe_val > 0.15: score += 1

        # 9. Revenue growth (-1 to 0)
        if rev_g and rev_g < 0: score -= 1

        # ── FINAL SIGNAL ──────────────────────────────────────────────────────────
        if   score >= 4:  call = 'BUY'
        elif score <= -4: call = 'REDUCE'
        else:             call = 'HOLD'

        def ret(n): return round((ltp/float(c.iloc[-n])-1)*100,1) if len(c)>n else None

        try:   news = [{'title':n.get('title',''),'link':n.get('link',''),'publisher':n.get('publisher','')} for n in (t.news or [])[:5]]
        except: news = []

        return sym, {
            'rsi': rsi_v,
            'macd_hist': mh_cur, 'macd_bullish': bool(mh_cur > 0) if mh_cur is not None else None,
            'macd_strengthening': bool(mh_cur > mh_prev) if (mh_cur is not None and mh_prev is not None) else None,
            'trend': trend, 'call': call, 'score': score,
            'ma20': ma20, 'ma50': ma50, 'ma200': ma200,
            'bb_upper': bb_u, 'bb_lower': bb_l,
            'w52_high': round(w52h,2), 'w52_low': round(w52l,2),
            'from_high': round(pct_from_high,1),
            'vol_ratio': vol_ratio,
            'ret_1m': ret(21), 'ret_3m': ret(63), 'ret_6m': ret(126),
            'pe': pe_val, 'pb': sf(info.get('priceToBook')),
            'roe': roe_val, 'eps': sf(info.get('trailingEps')),
            'div_yield': sf(info.get('dividendYield')),
            'rev_growth': rev_g,
            'profit_margin': sf(info.get('profitMargins')),
            'mkt_cap': info.get('marketCap'),
            'sector': info.get('sector',''), 'industry': info.get('industry',''),
            'news': news,
        }
    except Exception as e:
        print(f"  analyse_one {sym}: {e}")
        return sym, {'call':'HOLD','trend':'N/A','rsi':None,'news':[],'error':str(e)}

def fetch_analysis(symbols):
    results = {}
    with ThreadPoolExecutor(max_workers=6) as ex:
        futures = {ex.submit(analyse_one, s): s for s in symbols}
        for future in as_completed(futures):
            try:
                sym, data = future.result()
                results[sym] = data
                print(f"  {sym}: {data.get('trend','?')} RSI={data.get('rsi','?')} {data.get('call','?')}")
            except Exception as e:
                s = futures[future]
                print(f"  {s} error: {e}")
                results[s] = {'call':'HOLD','trend':'N/A','rsi':None,'news':[]}
    return results

# ── CHART (single stock) ───────────────────────────────────────────────────────
def fetch_chart(sym):
    y = nse(sym)
    if not y: return {}
    h = yf.Ticker(y).history(period='6mo', interval='1d')
    if h.empty: return {}
    c = h['Close']; v = h['Volume']
    ma20 = c.rolling(20).mean()
    ma50 = c.rolling(50).mean()
    bb_m = c.rolling(20).mean()
    bb_u = bb_m + 2*c.rolling(20).std()
    bb_l = bb_m - 2*c.rolling(20).std()
    rs   = rsi14(c)
    e12  = c.ewm(span=12).mean(); e26 = c.ewm(span=26).mean()
    macd = e12 - e26; sig = macd.ewm(span=9).mean(); mh = macd - sig
    return {
        'dates':       [d.strftime('%d %b') for d in h.index],
        'close':       clean(c),
        'ma20':        clean(ma20),  'ma50':        clean(ma50),
        'bb_upper':    clean(bb_u),  'bb_lower':    clean(bb_l),
        'rsi':         clean(rs),
        'macd':        clean(macd),  'macd_signal': clean(sig),
        'macd_hist':   clean(mh),
        'volume':      [int(x) for x in v],
    }

# ── PORTFOLIO FILE ─────────────────────────────────────────────────────────────
_pf_lock = threading.Lock()

def load_portfolio():
    try:
        if os.path.exists(PORTFOLIO_FILE):
            with open(PORTFOLIO_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"Portfolio load error: {e}")
    return None

def save_portfolio(data):
    try:
        with _pf_lock:
            with open(PORTFOLIO_FILE, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
        return True
    except Exception as e:
        print(f"Portfolio save error: {e}")
        return False

# ── HTTP HANDLER ───────────────────────────────────────────────────────────────
class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k): super().__init__(*a, directory=SERVE_DIR, **k)

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        p = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(p.query)
        if   p.path == '/' or p.path == '':
            self.send_response(302)
            self.send_header('Location', '/portfolio-tracker.html')
            self.end_headers()
        elif p.path == '/api/prices':         self._prices(q)
        elif p.path == '/api/analysis':        self._analysis(q)
        elif p.path == '/api/chart':           self._chart(q)
        elif p.path == '/api/portfolio':       self._get_portfolio()
        elif p.path == '/api/analyse_stock':   self._analyse_stock(q)
        else: super().do_GET()

    def do_POST(self):
        p = urllib.parse.urlparse(self.path)
        if p.path == '/api/portfolio':
            self._post_portfolio()
        else:
            self.send_response(404); self.end_headers()

    def _get_portfolio(self):
        data = load_portfolio()
        if data:
            self._json({'ok': True, 'portfolio': data})
        else:
            self._json({'ok': False, 'portfolio': None})

    def _post_portfolio(self):
        try:
            length = int(self.headers.get('Content-Length', 0))
            body   = self.rfile.read(length)
            data   = json.loads(body.decode('utf-8'))
            ok = save_portfolio(data)
            self._json({'ok': ok})
        except Exception as e:
            print(f"POST /api/portfolio error: {e}")
            self._json({'ok': False, 'error': str(e)})

    def _analyse_stock(self, q):
        sym = q.get('symbol',[''])[0].strip().upper()
        if not sym:
            self._json({'error': 'no symbol'}); return
        # Check cache first (skip if older than 1h)
        with _ac['lock']:
            cached = _ac['data'].get(sym)
            age    = time.time() - _ac['ts']
        if cached and age < ANALYSIS_TTL:
            self._json({'symbol': sym, 'analysis': cached, 'cached': True})
            return
        # Run fresh analysis in-thread (blocking but quick for 1 stock)
        print(f"On-demand analysis: {sym}")
        _, result = analyse_one(sym)
        with _ac['lock']:
            _ac['data'][sym] = result
        self._json({'symbol': sym, 'analysis': result, 'cached': False})

    def _prices(self, q):
        syms = [s.strip() for s in urllib.parse.unquote(q.get('symbols',[''])[0]).split(',') if s.strip()]
        now  = time.time()
        with _pc['lock']:
            if now - _pc['ts'] >= PRICE_TTL:
                try:
                    print(f"Fetching prices ({len(syms)} symbols)...")
                    _pc['data'] = fetch_prices(syms) if syms else {}
                    _pc['ts'] = now; _pc['error'] = None
                    print(f"  Got {len(_pc['data'])} prices")
                except Exception as e:
                    _pc['error'] = str(e); print(f"  Error: {e}")
        self._json({'prices': _pc['data'], 'ts': _pc['ts'], 'error': _pc['error'],
                    'next_refresh_in': max(0, int(PRICE_TTL - (now - _pc['ts'])))})

    def _analysis(self, q):
        syms      = [s.strip() for s in urllib.parse.unquote(q.get('symbols',[''])[0]).split(',') if s.strip()]
        now       = time.time()
        computing = _ac['ts'] == 0
        stale     = not computing and (now - _ac['ts'] >= ANALYSIS_TTL)
        if stale:
            def bg():
                d = fetch_analysis(syms or ALL_SYMS)
                with _ac['lock']: _ac['data'] = d; _ac['ts'] = time.time()
            threading.Thread(target=bg, daemon=True).start()
        self._json({'analysis': _ac['data'], 'ts': _ac['ts'],
                    'computing': computing, 'stale': stale,
                    'next_refresh_in': max(0, int(ANALYSIS_TTL - (now - _ac['ts']))) if not computing else 0})

    def _chart(self, q):
        sym = q.get('symbol',[''])[0].strip()
        if not sym: self._json({'error':'no symbol'}); return
        now = time.time()
        with _ccl:
            if now - _cc.get(sym,{}).get('ts',0) >= CHART_TTL:
                try:
                    print(f"Fetching chart: {sym}")
                    _cc[sym] = {'data': fetch_chart(sym), 'ts': now}
                except Exception as e:
                    print(f"Chart error {sym}: {e}")
                    _cc[sym] = {'data': {}, 'ts': now}
        self._json({'chart': _cc.get(sym,{}).get('data',{}), 'symbol': sym})

    def _cors(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')

    def _json(self, obj):
        b = json.dumps(clean_json(obj), default=str).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(b)))
        self._cors()
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        self.wfile.write(b)

    def log_message(self, *_): pass

# ── ENTRY ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    os.chdir(SERVE_DIR)
    print(f"\n{'='*54}")
    print(f"  Portfolio Tracker  --  live + analysis + charts")
    print(f"  http://localhost:{PORT}/portfolio-tracker.html")
    print(f"{'='*54}\n")

    def precompute():
        print("Pre-computing analysis in background (takes ~60s)...")
        d = fetch_analysis(ALL_SYMS)
        with _ac['lock']: _ac['data'] = d; _ac['ts'] = time.time()
        print(f"Analysis ready: {len(d)} stocks")
    threading.Thread(target=precompute, daemon=True).start()

    HTTPServer(('', PORT), Handler).serve_forever()
