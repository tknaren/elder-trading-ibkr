"""
Elder Trading System - Additional Screener API Routes
Routes for Candlestick Pattern and RSI+MACD Screeners

Endpoints:
1. Candlestick Pattern Screener
   - POST /api/v2/screener/candlestick/run
   - GET /api/v2/screener/candlestick/single/<symbol>
   - GET /api/v2/screener/candlestick/stocks

2. RSI + MACD Indicator Screener
   - POST /api/v2/screener/rsi-macd/run
   - GET /api/v2/screener/rsi-macd/single/<symbol>
   - GET /api/v2/screener/rsi-macd/stocks
"""

from flask import Blueprint, request, jsonify
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List

# Create blueprint
screener_routes = Blueprint('screener_routes', __name__)


def fetch_historical_data(symbols: List[str], lookback_days: int = 180) -> Dict[str, pd.DataFrame]:
    """
    Fetch historical OHLCV data for multiple symbols from IBKR or cache
    Uses the same data source as the live screener

    Args:
        symbols: List of stock tickers
        lookback_days: Number of days of history to fetch

    Returns:
        Dict mapping symbol to DataFrame
    """
    from services.ibkr_client import fetch_stock_data

    hist_data = {}

    for symbol in symbols:
        try:
            # Fetch from IBKR (handles caching internally)
            data = fetch_stock_data(symbol, period='2y')

            if data is not None and 'history' in data:
                hist = data['history']
                if hist is not None and len(hist) >= 50:
                    print(f"✅ {symbol}: Got {len(hist)} bars")
                    hist_data[symbol] = hist
                else:
                    print(
                        f"⚠️ {symbol}: Insufficient data ({len(hist) if hist is not None else 0} bars)")
            else:
                print(f"⚠️ {symbol}: No data returned")

        except Exception as e:
            print(f"❌ Error fetching {symbol}: {e}")
            import traceback
            traceback.print_exc()
            continue

    return hist_data


# ══════════════════════════════════════════════════════════════════════════════
# CANDLESTICK PATTERN SCREENER ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@screener_routes.route('/candlestick/stocks', methods=['GET'])
def get_candlestick_stocks():
    """
    Get list of available stocks for candlestick screening
    """
    market = request.args.get('market', 'US')

    from services.candlestick_screener import get_stock_list
    stocks = get_stock_list(market)

    return jsonify({
        'market': market,
        'stocks': stocks,
        'count': len(stocks)
    })


@screener_routes.route('/candlestick/run', methods=['POST'])
def run_candlestick_screener_endpoint():
    """
    Run candlestick pattern screener

    Request body:
    {
        "symbols": ["AAPL", "MSFT", ...] or "all",
        "lookback_days": 180,
        "market": "US",
        "filter_mode": "all" | "filtered_only" | "patterns_only"
    }

    filter_mode:
        - "all": Show all patterns with indicator values (default)
        - "filtered_only": Only show patterns matching KC and RSI filters
        - "patterns_only": Only show patterns, no filter requirement

    Returns:
        Signals sorted by date with pattern and indicator info
    """
    data = request.get_json() or {}

    symbols = data.get('symbols', [])
    lookback_days = min(max(data.get('lookback_days', 180), 30), 365)
    market = data.get('market', 'US')
    filter_mode = data.get('filter_mode', 'all')

    from services.candlestick_screener import (
        run_candlestick_screener,
        get_stock_list
    )

    # Handle "all" or empty symbols
    if not symbols or symbols == 'all' or (isinstance(symbols, list) and 'all' in symbols):
        symbols = get_stock_list(market)

    # Validate filter_mode
    if filter_mode not in ['all', 'filtered_only', 'patterns_only']:
        filter_mode = 'all'

    try:
        # Fetch historical data
        hist_data = fetch_historical_data(symbols, lookback_days)

        if not hist_data:
            return jsonify({
                'error': 'Could not fetch historical data for any symbols',
                'symbols_requested': len(symbols)
            }), 500

        # Run screener
        result = run_candlestick_screener(
            symbols=list(hist_data.keys()),
            hist_data=hist_data,
            lookback_days=lookback_days,
            filter_mode=filter_mode
        )

        return jsonify(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


@screener_routes.route('/candlestick/single/<symbol>', methods=['GET'])
def scan_single_candlestick(symbol):
    """
    Scan a single stock's history for candlestick patterns

    Query params:
        - lookback_days: Number of days to scan (default: 180)
        - filter_mode: "all" | "filtered_only" | "patterns_only" (default: "all")
    """
    lookback_days = request.args.get('lookback_days', 180, type=int)
    filter_mode = request.args.get('filter_mode', 'all')

    lookback_days = min(max(lookback_days, 30), 365)

    from services.candlestick_screener import scan_stock_candlestick_historical

    try:
        # Fetch historical data
        hist_data = fetch_historical_data([symbol.upper()], lookback_days)

        if not hist_data or symbol.upper() not in hist_data:
            return jsonify({
                'error': f'Could not fetch historical data for {symbol}',
                'symbol': symbol.upper()
            }), 404

        hist = hist_data[symbol.upper()]

        # Scan stock
        signals = scan_stock_candlestick_historical(
            symbol=symbol.upper(),
            hist=hist,
            lookback_days=lookback_days
        )

        # Apply filter mode
        if filter_mode == 'filtered_only':
            signals = [s for s in signals if s.get('filters_match')]

        return jsonify({
            'symbol': symbol.upper(),
            'signals': signals,
            'count': len(signals),
            'lookback_days': lookback_days,
            'filter_mode': filter_mode,
            'patterns_found': list(set(
                p for s in signals for p in s.get('patterns', [])
            ))
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ══════════════════════════════════════════════════════════════════════════════
# RSI + MACD INDICATOR SCREENER ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@screener_routes.route('/rsi-macd/stocks', methods=['GET'])
def get_rsi_macd_stocks():
    """
    Get list of available stocks for RSI+MACD screening
    """
    market = request.args.get('market', 'US')

    from services.rsi_macd_screener import get_stock_list
    stocks = get_stock_list(market)

    return jsonify({
        'market': market,
        'stocks': stocks,
        'count': len(stocks)
    })


@screener_routes.route('/rsi-macd/run', methods=['POST'])
def run_rsi_macd_screener_endpoint():
    """
    Run RSI + MACD indicator screener

    Request body:
    {
        "symbols": ["AAPL", "MSFT", ...] or "all",
        "lookback_days": 180,
        "market": "US"
    }

    Filter conditions (ALL must be TRUE):
    1. RSI(14) < 30 (Oversold)
    2. RSI is Increasing (today > yesterday)
    3. MACD pointing up OR crossing up

    Returns:
        Signals sorted by date with all indicator values
    """
    data = request.get_json() or {}

    symbols = data.get('symbols', [])
    lookback_days = min(max(data.get('lookback_days', 180), 30), 365)
    market = data.get('market', 'US')

    from services.rsi_macd_screener import (
        run_rsi_macd_screener,
        get_stock_list
    )

    # Handle "all" or empty symbols
    if not symbols or symbols == 'all' or (isinstance(symbols, list) and 'all' in symbols):
        symbols = get_stock_list(market)

    try:
        # Fetch historical data
        hist_data = fetch_historical_data(symbols, lookback_days)

        if not hist_data:
            return jsonify({
                'error': 'Could not fetch historical data for any symbols',
                'symbols_requested': len(symbols)
            }), 500

        # Run screener
        result = run_rsi_macd_screener(
            symbols=list(hist_data.keys()),
            hist_data=hist_data,
            lookback_days=lookback_days
        )

        return jsonify(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


@screener_routes.route('/rsi-macd/single/<symbol>', methods=['GET'])
def scan_single_rsi_macd(symbol):
    """
    Scan a single stock's history for RSI + MACD signals

    Query params:
        - lookback_days: Number of days to scan (default: 180)
    """
    lookback_days = request.args.get('lookback_days', 180, type=int)
    lookback_days = min(max(lookback_days, 30), 365)

    from services.rsi_macd_screener import scan_stock_rsi_macd_historical

    try:
        # Fetch historical data
        hist_data = fetch_historical_data([symbol.upper()], lookback_days)

        if not hist_data or symbol.upper() not in hist_data:
            return jsonify({
                'error': f'Could not fetch historical data for {symbol}',
                'symbol': symbol.upper()
            }), 404

        hist = hist_data[symbol.upper()]

        # Scan stock
        signals = scan_stock_rsi_macd_historical(
            symbol=symbol.upper(),
            hist=hist,
            lookback_days=lookback_days
        )

        return jsonify({
            'symbol': symbol.upper(),
            'signals': signals,
            'count': len(signals),
            'lookback_days': lookback_days,
            'signal_types': {
                'crossover': len([s for s in signals if s['conditions']['macd_crossing_up']]),
                'rising': len([s for s in signals if s['conditions']['macd_pointing_up'] and not s['conditions']['macd_crossing_up']])
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ══════════════════════════════════════════════════════════════════════════════
# COMBINED SCREENER INFO
# ══════════════════════════════════════════════════════════════════════════════

@screener_routes.route('/info', methods=['GET'])
def get_screener_info():
    """
    Get information about available screeners and their criteria
    """
    return jsonify({
        'screeners': [
            {
                'id': 'candlestick',
                'name': 'Candlestick Pattern Screener',
                'description': 'Scans for bullish candlestick patterns (Hammer, Engulfing, Piercing, Tweezer Bottom)',
                'filters': [
                    'Pattern detected on daily candle',
                    'Price below Keltner Channel Lower (KC-1)',
                    'RSI(14) < 30 (Oversold)'
                ],
                'patterns': ['Hammer', 'Bullish Engulfing', 'Piercing Pattern', 'Tweezer Bottom'],
                'endpoints': {
                    'run': 'POST /api/v2/screener/candlestick/run',
                    'single': 'GET /api/v2/screener/candlestick/single/<symbol>',
                    'stocks': 'GET /api/v2/screener/candlestick/stocks'
                }
            },
            {
                'id': 'rsi-macd',
                'name': 'RSI + MACD Indicator Screener',
                'description': 'Scans for RSI oversold conditions with MACD confirmation',
                'filters': [
                    'RSI(14) < 30 (Oversold)',
                    'RSI is Increasing (today > yesterday)',
                    'MACD pointing up OR crossing up'
                ],
                'indicators': ['RSI(14)', 'MACD(12, 26, 9)', 'Stochastic', 'Keltner Channel', 'EMA 20'],
                'endpoints': {
                    'run': 'POST /api/v2/screener/rsi-macd/run',
                    'single': 'GET /api/v2/screener/rsi-macd/single/<symbol>',
                    'stocks': 'GET /api/v2/screener/rsi-macd/stocks'
                }
            }
        ],
        'markets': ['US', 'India'],
        'default_lookback_days': 180,
        'max_lookback_days': 365
    })
