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
    Fetch stock data from IBKR with database caching

    Strategy:
    1. Check if we have cached data in database
    2. If we have recent data (< 1 day old), use it
    3. If not, fetch from IBKR and cache it
    4. Always ensure we have at least 2 years of data

    Returns dictionary with 'history' DataFrame and metadata, or None if failed.
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

    # Try to load from database first
    db = get_database().get_connection()

    # Check if we have cached data that's fresh (less than 1 day old)
    sync_record = db.execute(
        'SELECT * FROM stock_data_sync WHERE symbol = ?',
        (symbol,)
    ).fetchone()

    use_cache = False
    if sync_record:
        last_updated = datetime.fromisoformat(sync_record['last_updated'])
        if datetime.now() - last_updated < timedelta(hours=24):
            # Cache is fresh, use it
            use_cache = True
            print(
                f"✅ {symbol}: Using cached data from {sync_record['last_updated']}")

    hist = None

    if use_cache:
        # Load from database
        rows = db.execute('''
            SELECT date, open, high, low, close, volume 
            FROM stock_historical_data 
            WHERE symbol = ? 
            ORDER BY date ASC
        ''', (symbol,)).fetchall()

        if rows and len(rows) >= 30:
            hist = pd.DataFrame([
                {
                    'Date': row['date'],
                    'Open': row['open'],
                    'High': row['high'],
                    'Low': row['low'],
                    'Close': row['close'],
                    'Volume': row['volume']
                }
                for row in rows
            ])
            hist['Date'] = pd.to_datetime(hist['Date'])
            hist.set_index('Date', inplace=True)
            print(f"📊 {symbol}: Loaded {len(hist)} cached records from database")

    # If cache miss or insufficient data, fetch from IBKR
    if hist is None or len(hist) < 100:
        print(f"🔄 {symbol}: Fetching data from IBKR...")
        hist = client.get_historical_data(conid, period=period, bar='1d')

        if hist is None or hist.empty or len(hist) < 30:
            print(
                f"❌ {symbol}: Insufficient historical data ({len(hist) if hist is not None else 0} bars)")
            db.close()
            return None

        # Save to database (upsert)
        print(f"💾 {symbol}: Caching {len(hist)} records to database...")
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

        # Update sync record
        earliest_date = hist.index.min().strftime('%Y-%m-%d')
        latest_date = hist.index.max().strftime('%Y-%m-%d')

        db.execute('''
            INSERT OR REPLACE INTO stock_data_sync 
            (symbol, last_updated, earliest_date, latest_date, record_count)
            VALUES (?, ?, ?, ?, ?)
        ''', (symbol, datetime.now().isoformat(), earliest_date, latest_date, len(hist)))

        db.commit()
        print(f"✅ {symbol}: Data cached successfully")

    # Get contract info for name
    info = client.get_contract_info(conid)
    name = symbol
    sector = 'Unknown'
    if info:
        name = info.get('company_name', info.get('con_id', symbol))
        sector = info.get('industry', 'Unknown')

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
        'conid': conid
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
