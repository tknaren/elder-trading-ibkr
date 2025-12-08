"""
Elder Trading System - IBKR Client Portal API Data Provider
============================================================

Provides market data from IBKR Client Portal REST API.

PREREQUISITES:
1. Download Client Portal Gateway from IBKR
2. Extract and run: bin\run.bat root\conf.yaml
3. Authenticate at https://localhost:5000
4. Then run the Elder Trading System
"""

import requests
import urllib3
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple, Any

# Disable SSL warnings (Gateway uses self-signed cert)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def convert_to_native(obj: Any) -> Any:
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(item) for item in obj]
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj) if not np.isnan(obj) else None
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    return obj


class IBKRClient:
    """
    IBKR Client Portal API Client

    Handles all communication with the IBKR Gateway for market data.
    """

    def __init__(self, host: str = "localhost", port: int = 5000):
        self.base_url = f"https://{host}:{port}/v1/api"
        self.session = requests.Session()
        self.session.verify = False
        self.timeout = 15
        self._conid_cache: Dict[str, int] = {}
        self._authenticated = False

    def _request(self, method: str, endpoint: str, **kwargs) -> Optional[Any]:
        """Make API request with error handling"""
        try:
            url = f"{self.base_url}{endpoint}"
            kwargs.setdefault('timeout', self.timeout)

            response = self.session.request(method, url, **kwargs)

            if response.status_code == 200:
                try:
                    return response.json()
                except:
                    return None
            elif response.status_code == 401:
                self._authenticated = False
                return None
            elif response.status_code == 429:
                time.sleep(1)
                return self._request(method, endpoint, **kwargs)
            else:
                return None

        except requests.exceptions.ConnectionError:
            return None
        except Exception as e:
            print(f"Request error: {e}")
            return None

    def check_auth(self) -> bool:
        """Check if authenticated with Gateway"""
        try:
            data = self._request('GET', '/iserver/auth/status')
            if data:
                self._authenticated = data.get(
                    'authenticated', False) and data.get('connected', False)
                if not self._authenticated and data.get('connected', False):
                    self._request('POST', '/iserver/reauthenticate')
                    time.sleep(1)
                    data = self._request('GET', '/iserver/auth/status')
                    self._authenticated = data.get(
                        'authenticated', False) if data else False
                return self._authenticated
        except:
            pass
        self._authenticated = False
        return False

    def tickle(self) -> bool:
        """Keep session alive"""
        data = self._request('POST', '/tickle')
        return data is not None

    def get_conid(self, symbol: str) -> Optional[int]:
        """Get contract ID for a symbol"""
        if symbol in self._conid_cache:
            return self._conid_cache[symbol]

        # Clean symbol
        clean_symbol = symbol.replace('.NS', '').replace('.BSE', '').upper()

        # Search for contract
        data = self._request('GET', '/iserver/secdef/search',
                             params={'symbol': clean_symbol})

        if not data or not isinstance(data, list) or len(data) == 0:
            return None

        # Try to find stock conid
        for item in data:
            conid = item.get('conid')
            if conid:
                self._conid_cache[symbol] = int(conid)
                return int(conid)

            # Check sections for STK type
            sections = item.get('sections', [])
            for section in sections:
                if section.get('secType') == 'STK':
                    # Need to get conid from secdef/info
                    info_data = self._request('GET', '/iserver/secdef/info',
                                              params={'symbol': clean_symbol, 'secType': 'STK'})
                    if info_data and isinstance(info_data, list) and len(info_data) > 0:
                        conid = info_data[0].get('conid')
                        if conid:
                            self._conid_cache[symbol] = int(conid)
                            return int(conid)

        return None

    def get_historical_data(self, conid: int, period: str = "6m", bar: str = "1d") -> Optional[pd.DataFrame]:
        """Get historical OHLCV data"""
        if not conid:
            return None

        data = self._request('GET', '/iserver/marketdata/history',
                             params={'conid': conid, 'period': period, 'bar': bar})

        if not data or 'data' not in data or not data['data']:
            return None

        try:
            df = pd.DataFrame(data['data'])

            # Rename columns
            column_map = {'o': 'Open', 'h': 'High', 'l': 'Low',
                          'c': 'Close', 'v': 'Volume', 't': 'Timestamp'}
            df = df.rename(columns=column_map)

            # Convert timestamp to datetime
            if 'Timestamp' in df.columns:
                df['Date'] = pd.to_datetime(df['Timestamp'], unit='ms')
                df.set_index('Date', inplace=True)
                df.drop('Timestamp', axis=1, inplace=True, errors='ignore')

            # Ensure numeric types
            for col in ['Open', 'High', 'Low', 'Close']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            if 'Volume' in df.columns:
                df['Volume'] = pd.to_numeric(
                    df['Volume'], errors='coerce').fillna(0).astype(int)

            return df

        except Exception as e:
            print(f"Error parsing historical data: {e}")
            return None

    def get_contract_info(self, conid: int) -> Optional[Dict]:
        """Get contract details"""
        if not conid:
            return None
        return self._request('GET', f'/iserver/contract/{conid}/info')

    def get_market_snapshot(self, conid: int) -> Optional[Dict]:
        """Get current market data snapshot"""
        if not conid:
            return None

        fields = "31,84,86,85,87,88,7295"

        # First request initiates subscription
        self._request('GET', '/iserver/marketdata/snapshot',
                      params={'conids': str(conid), 'fields': fields})
        time.sleep(0.3)

        # Second request gets data
        data = self._request('GET', '/iserver/marketdata/snapshot',
                             params={'conids': str(conid), 'fields': fields})

        if data and isinstance(data, list) and len(data) > 0:
            item = data[0]
            return {
                'last': self._parse_float(item.get('31')),
                'bid': self._parse_float(item.get('84')),
                'ask': self._parse_float(item.get('86')),
                'high': self._parse_float(item.get('85')),
                'low': self._parse_float(item.get('87')),
                'volume': self._parse_int(item.get('88')),
                'open': self._parse_float(item.get('7295')),
            }
        return None

    def _parse_float(self, value) -> Optional[float]:
        """Parse float value safely"""
        if value is None:
            return None
        try:
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                clean = value.replace(',', '').replace(
                    'C', '').replace('H', '').strip()
                return float(clean) if clean else None
        except:
            pass
        return None

    def _parse_int(self, value) -> int:
        """Parse integer value safely"""
        if value is None:
            return 0
        try:
            if isinstance(value, int):
                return value
            if isinstance(value, float):
                return int(value)
            if isinstance(value, str):
                clean = value.replace(',', '').replace(
                    'M', '000000').replace('K', '000').strip()
                return int(float(clean)) if clean else 0
        except:
            pass
        return 0


# Global client instance
_client: Optional[IBKRClient] = None


def get_client() -> IBKRClient:
    """Get or create IBKR client instance"""
    global _client
    if _client is None:
        _client = IBKRClient()
    return _client


def check_connection() -> Tuple[bool, str]:
    """Check IBKR Gateway connection status"""
    client = get_client()
    try:
        if client.check_auth():
            return True, "Connected and authenticated to IBKR Gateway"
        else:
            return False, "Gateway running but not authenticated. Please login at https://localhost:5000"
    except:
        return False, "Cannot connect to IBKR Gateway. Please start it first."


def fetch_stock_data(symbol: str, period: str = '2y') -> Optional[Dict]:
    """
    Fetch stock data with intelligent indicator caching

    Strategy:
    1. Check cached indicators - if fresh (< 1 day old), load them
    2. Find last date in cached indicators
    3. Fetch OHLCV data only for dates AFTER last cached date
    4. Recalculate indicators incrementally from last cached values
    5. Store new indicator values in cache

    Returns dictionary with 'history' DataFrame, indicators, and metadata, or None if failed.
    """
    from models.database import get_database
    import pandas as pd
    from datetime import datetime, timedelta

    client = get_client()

    # Check authentication
    if not client.check_auth():
        print(f"❌ {symbol}: Not authenticated with IBKR Gateway")
        return None

    # Keep session alive
    client.tickle()

    # Get contract ID
    conid = client.get_conid(symbol)
    if not conid:
        print(f"❌ {symbol}: Could not find contract")
        return None

    db = get_database().get_connection()

    # Check if we have fresh cached indicators (< 1 day old)
    indicator_sync = db.execute(
        'SELECT * FROM stock_indicator_sync WHERE symbol = ?',
        (symbol,)
    ).fetchone()

    last_daily_date = None
    cached_indicators = {}
    use_indicator_cache = False

    if indicator_sync:
        last_updated = datetime.fromisoformat(indicator_sync['last_updated'])
        if datetime.now() - last_updated < timedelta(hours=24):
            last_daily_date = indicator_sync['last_daily_date']
            use_indicator_cache = True
            print(
                f"✅ {symbol}: Cached indicators are fresh (last updated: {indicator_sync['last_updated']})")

    # Determine data to fetch
    hist = None
    needs_full_calc = False

    if use_indicator_cache and last_daily_date:
        print(f"📊 {symbol}: Fetching data AFTER {last_daily_date} from IBKR...")
        # Fetch only new data since last cached date
        hist = client.get_historical_data(conid, period=period, bar='1d')

        if hist is not None and not hist.empty:
            # Filter to only dates after last cached date
            last_cached = pd.to_datetime(last_daily_date)
            hist = hist[hist.index > last_cached]

            if len(hist) > 0:
                print(f"📈 {symbol}: Got {len(hist)} new data points to process")
            else:
                print(f"✓ {symbol}: No new data since {last_daily_date}")
                # Return cached data + indicators
                cached_rows = db.execute('''
                    SELECT date, close, ema_22, ema_50, ema_100, ema_200, 
                           macd_line, macd_signal, macd_histogram, rsi, 
                           stochastic, stoch_d, atr, force_index, kc_upper, kc_middle, kc_lower
                    FROM stock_indicators_daily 
                    WHERE symbol = ? 
                    ORDER BY date ASC
                ''', (symbol,)).fetchall()

                if cached_rows:
                    hist = pd.DataFrame([dict(row) for row in cached_rows])
                    hist['Date'] = pd.to_datetime(hist['date'])
                    hist.set_index('Date', inplace=True)
                    hist = hist.drop('date', axis=1)
                    hist = hist[['close', 'ema_22', 'ema_50', 'ema_100', 'ema_200',
                                 'macd_line', 'macd_signal', 'macd_histogram', 'rsi',
                                 'stochastic', 'stoch_d', 'atr', 'force_index',
                                 'kc_upper', 'kc_middle', 'kc_lower']]
                    db.close()

                    # Get contract info
                    info = client.get_contract_info(conid)
                    name = info.get('company_name', symbol) if info else symbol
                    sector = info.get(
                        'industry', 'Unknown') if info else 'Unknown'

                    return {
                        'symbol': symbol,
                        'name': name,
                        'sector': sector,
                        'history': hist,
                        'info': info or {},
                        'conid': conid,
                        'from_cache': True
                    }
    else:
        print(f"🔄 {symbol}: No valid indicator cache, fetching full 2-year data...")
        hist = client.get_historical_data(conid, period=period, bar='1d')
        needs_full_calc = True

    # Handle fetch failure
    if hist is None or hist.empty or len(hist) < 30:
        print(
            f"❌ {symbol}: Insufficient historical data ({len(hist) if hist is not None else 0} bars)")
        db.close()
        return None

    # At this point, we have new data to process
    # Load previous indicator values for incremental calculation
    if use_indicator_cache and not needs_full_calc:
        print(
            f"📋 {symbol}: Loading previous indicators for incremental calculation...")
        last_row = db.execute('''
            SELECT ema_22, ema_50, ema_100, ema_200, macd_line, macd_signal 
            FROM stock_indicators_daily 
            WHERE symbol = ? 
            ORDER BY date DESC 
            LIMIT 1
        ''', (symbol,)).fetchone()

        if last_row:
            cached_indicators = dict(last_row)
            print(f"✓ {symbol}: Loaded previous indicator values")

    # If full calculation needed, get all OHLCV first
    if needs_full_calc:
        print(f"💾 {symbol}: Caching {len(hist)} OHLCV records to database...")
        for date, row in hist.iterrows():
            db.execute('''
                INSERT OR REPLACE INTO stock_historical_data 
                (symbol, date, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                symbol,
                date.strftime('%Y-%m-%d'),
                float(row['Open']),
                float(row['High']),
                float(row['Low']),
                float(row['Close']),
                int(row['Volume'])
            ))

        earliest_date = hist.index.min().strftime('%Y-%m-%d')
        latest_date = hist.index.max().strftime('%Y-%m-%d')

        db.execute('''
            INSERT OR REPLACE INTO stock_data_sync 
            (symbol, last_updated, earliest_date, latest_date, record_count)
            VALUES (?, ?, ?, ?, ?)
        ''', (symbol, datetime.now().isoformat(), earliest_date, latest_date, len(hist)))
        db.commit()
        print(f"✓ {symbol}: OHLCV data cached successfully")
    else:
        # Save new OHLCV data
        if len(hist) > 0:
            print(f"💾 {symbol}: Caching {len(hist)} new OHLCV records...")
            for date, row in hist.iterrows():
                db.execute('''
                    INSERT OR REPLACE INTO stock_historical_data 
                    (symbol, date, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    symbol,
                    date.strftime('%Y-%m-%d'),
                    float(row['Open']),
                    float(row['High']),
                    float(row['Low']),
                    float(row['Close']),
                    int(row['Volume'])
                ))

            db.execute('''
                UPDATE stock_data_sync 
                SET latest_date = ?, record_count = (
                    SELECT COUNT(*) FROM stock_historical_data WHERE symbol = ?
                ), last_updated = ?
                WHERE symbol = ?
            ''', (
                hist.index.max().strftime('%Y-%m-%d'),
                symbol,
                datetime.now().isoformat(),
                symbol
            ))
            db.commit()

    # Prepare full OHLCV history for indicator calculation
    # Load all cached OHLCV if we're doing incremental calculation
    if not needs_full_calc:
        all_rows = db.execute('''
            SELECT date, open, high, low, close, volume 
            FROM stock_historical_data 
            WHERE symbol = ? 
            ORDER BY date ASC
        ''', (symbol,)).fetchall()

        if all_rows:
            hist_full = pd.DataFrame([
                {
                    'Date': row['date'],
                    'Open': row['open'],
                    'High': row['high'],
                    'Low': row['low'],
                    'Close': row['close'],
                    'Volume': row['volume']
                }
                for row in all_rows
            ])
            hist_full['Date'] = pd.to_datetime(hist_full['Date'])
            hist_full.set_index('Date', inplace=True)
            hist = hist_full
            print(
                f"📊 {symbol}: Using {len(hist)} total records for indicator calculation")

    # Get contract info
    info = client.get_contract_info(conid)
    name = convert_to_native(
        info.get('company_name', symbol)) if info else symbol
    sector = convert_to_native(
        info.get('industry', 'Unknown')) if info else 'Unknown'

    # Get current snapshot
    snapshot = client.get_market_snapshot(conid)

    db.close()

    return {
        'symbol': symbol,
        'name': name,
        'sector': sector,
        'history': hist,
        'info': info or {},
        'snapshot': snapshot,
        'conid': conid,
        'cached_indicators': cached_indicators,
        'is_incremental': not needs_full_calc and use_indicator_cache
    }


# Test function
if __name__ == "__main__":
    print("=" * 60)
    print("  IBKR Client Portal API - Test")
    print("=" * 60)

    connected, message = check_connection()
    print(f"\n{message}")

    if not connected:
        print("\nPlease start Gateway and login at https://localhost:5000")
        exit(1)

    print("\nTesting AAPL...")
    data = fetch_stock_data('AAPL')

    if data:
        print(f"✓ Symbol: {data['symbol']}")
        print(f"✓ Name: {data['name']}")
        print(f"✓ Bars: {len(data['history'])}")
        print(f"\nLast 3 bars:")
        print(data['history'].tail(3))
    else:
        print("❌ Failed to fetch AAPL data")
