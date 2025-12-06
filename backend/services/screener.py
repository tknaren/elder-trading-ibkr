"""
Elder Trading System - Screener Service
Implements the Triple Screen methodology for stock screening

Features:
- Weekly Screen (Screen 1): Trend identification
- Daily Screen (Screen 2): Entry timing
- Signal strength scoring with configurable indicators
- Candlestick pattern recognition
- Transparent grading with explanations

Data Source: IBKR Client Portal API
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from services.indicators import (
    calculate_all_indicators,
    calculate_ema,
    calculate_macd,
    get_grading_criteria
)
from services.candlestick_patterns import scan_patterns, get_bullish_patterns, get_pattern_score
from services.indicator_config import (
    INDICATOR_CATALOG,
    DEFAULT_INDICATOR_CONFIG,
    get_indicator_info,
    get_config_summary
)
from services.ibkr_client import fetch_stock_data, check_connection, get_client, convert_to_native


# Default watchlists - Full NASDAQ 100
NASDAQ_100_TOP = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AMD', 'AVGO', 'NFLX',
    'COST', 'PEP', 'ADBE', 'CSCO', 'INTC', 'QCOM', 'TXN', 'INTU', 'AMAT', 'MU',
    'LRCX', 'KLAC', 'SNPS', 'CDNS', 'MRVL', 'ON', 'NXPI', 'ADI', 'MCHP', 'FTNT',
    'VRTX', 'CHTR', 'ASML', 'CRWD', 'SPLK', 'ENPH', 'MNST', 'TEAM', 'PAYX', 'AEP',
    'DXCM', 'CPRT', 'PCAR', 'ALGN', 'AMGN', 'MRNA', 'XEL', 'WDAY', 'ABNB', 'MDLZ',
    'PDD', 'ANSS', 'DDOG', 'FOXA', 'MOBILEYE', 'ODFL', 'GOOG', 'IDXX', 'ISRG', 'JD',
    'TTD', 'DASH', 'OKTA', 'ORLY', 'NTES', 'BROS', 'MSTR', 'KKR', 'FFIV', 'SHAP',
    'VEEX', 'CRSP', 'VRSN', 'ACGL', 'SMAR', 'GTLB', 'FRSH', 'RVTY', 'BLDR', 'NTAP',
    'ULTA', 'TXRH', 'NWLI', 'CFLT', 'CVNA', 'ROKU', 'SMCI', 'LCID', 'LCID', 'COIN',
    'ANET', 'SOUN', 'LULU', 'HOOD', 'PANW', 'DELL', 'PTON', 'Z', 'TPG', 'BMRN'
]

NIFTY_50 = [
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
    'HINDUNILVR.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'ITC.NS', 'KOTAKBANK.NS',
    'LT.NS', 'AXISBANK.NS', 'ASIANPAINT.NS', 'MARUTI.NS', 'TITAN.NS',
    'SUNPHARMA.NS', 'ULTRACEMCO.NS', 'BAJFINANCE.NS', 'WIPRO.NS', 'HCLTECH.NS',
    'TATAMOTORS.NS', 'POWERGRID.NS', 'NTPC.NS', 'M&M.NS', 'JSWSTEEL.NS'
]


# fetch_stock_data is now imported from ibkr_client.py


def analyze_weekly_trend(hist: pd.DataFrame) -> Dict:
    """
    Screen 1: Analyze weekly trend using Elder's methodology

    Checks:
    - 22-period EMA slope (rising = bullish)
    - MACD Histogram slope (rising = bulls gaining)

    Args:
        hist: Historical price data (daily, will be resampled to weekly)

    Returns:
        Dictionary with weekly trend analysis
    """
    # Resample to weekly if we have daily data
    weekly = hist.resample('W').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()

    if len(weekly) < 26:  # Need enough data for MACD
        return {
            'weekly_ema_slope': 'INSUFFICIENT_DATA',
            'weekly_macd_rising': False,
            'weekly_trend': 'UNKNOWN',
            'weekly_bullish': False
        }

    closes = weekly['Close']

    # Calculate weekly indicators
    ema_22 = calculate_ema(closes, 22)
    macd = calculate_macd(closes)

    # Get current and previous values
    current_ema = ema_22.iloc[-1]
    prev_ema = ema_22.iloc[-2] if len(ema_22) > 1 else current_ema
    ema_slope = current_ema - prev_ema

    current_macd_h = macd['histogram'].iloc[-1]
    prev_macd_h = macd['histogram'].iloc[-2] if len(
        macd['histogram']) > 1 else current_macd_h
    macd_rising = current_macd_h > prev_macd_h

    # Determine weekly trend
    if ema_slope > 0 and macd_rising:
        weekly_trend = 'STRONG_BULLISH'
    elif ema_slope > 0 or macd_rising:
        weekly_trend = 'BULLISH'
    elif ema_slope < 0 and not macd_rising:
        weekly_trend = 'BEARISH'
    else:
        weekly_trend = 'NEUTRAL'

    return {
        'weekly_ema': round(current_ema, 2),
        'weekly_ema_prev': round(prev_ema, 2),
        'weekly_ema_slope': 'RISING' if ema_slope > 0 else 'FALLING' if ema_slope < 0 else 'FLAT',
        'weekly_ema_slope_value': round(ema_slope, 4),
        'weekly_macd_histogram': round(current_macd_h, 4),
        'weekly_macd_histogram_prev': round(prev_macd_h, 4),
        'weekly_macd_rising': macd_rising,
        'weekly_trend': weekly_trend,
        'weekly_bullish': weekly_trend in ['STRONG_BULLISH', 'BULLISH']
    }


def calculate_signal_strength(indicators: Dict, weekly: Dict, patterns: list = None) -> Dict:
    """
    Calculate signal strength score based on Elder criteria

    Scoring breakdown (max 10):
    +2: Weekly EMA rising strongly
    +1: Weekly MACD-H rising
    +2: Force Index < 0 (pullback in uptrend)
    +1: Force Index uptick from negative
    +2: Stochastic < 30 (oversold)
    +1: Stochastic 30-50
    +1: Price at/below 22-EMA
    +2: Bullish divergence
    +1: Impulse GREEN
    +1: Bullish candlestick pattern (bonus)
    -2: Impulse RED (penalty)

    Args:
        indicators: Daily indicator values
        weekly: Weekly trend analysis
        patterns: Detected candlestick patterns

    Returns:
        Dictionary with score, breakdown, and signals
    """
    if patterns is None:
        patterns = []

    score = 0
    signals = []
    breakdown = []

    # Weekly trend scoring
    if weekly.get('weekly_ema_slope') == 'RISING':
        if weekly.get('weekly_trend') == 'STRONG_BULLISH':
            score += 2
            breakdown.append('+2: Weekly EMA strongly rising')
            signals.append('Weekly uptrend confirmed')
        else:
            score += 1
            breakdown.append('+1: Weekly EMA rising')

    if weekly.get('weekly_macd_rising'):
        score += 1
        breakdown.append('+1: Weekly MACD-H rising')
        signals.append('Weekly momentum bullish')

    # Force Index scoring
    force_index = indicators.get('force_index_2', 0)
    if force_index < 0:
        score += 2
        breakdown.append(
            f'+2: Force Index < 0 ({force_index:.0f}) - Pullback zone')
        signals.append('Force Index negative = buying opportunity')

    # Stochastic scoring
    stochastic = indicators.get('stochastic_k', 50)
    if stochastic < 30:
        score += 2
        breakdown.append(f'+2: Stochastic < 30 ({stochastic:.1f}) - Oversold')
        signals.append('Stochastic oversold = entry zone')
    elif stochastic < 50:
        score += 1
        breakdown.append(f'+1: Stochastic 30-50 ({stochastic:.1f})')

    # Price vs EMA scoring
    price_vs_ema = indicators.get('price_vs_ema', 0)
    if price_vs_ema <= 0:
        score += 1
        breakdown.append(
            f'+1: Price at/below EMA ({price_vs_ema:.1f}%) - Value zone')
        signals.append('Buying value, not chasing')
    elif price_vs_ema > 5:
        breakdown.append(
            f'+0: Price far above EMA ({price_vs_ema:.1f}%) - Overpaying')
        signals.append('⚠️ Price extended above EMA')

    # Divergence scoring
    if indicators.get('bullish_divergence_macd') or indicators.get('bullish_divergence_rsi'):
        score += 2
        breakdown.append('+2: Bullish divergence detected')
        signals.append('⭐ Bullish divergence = strong signal')

    # Impulse system scoring
    impulse = indicators.get('impulse_color', 'BLUE')
    if impulse == 'GREEN':
        score += 1
        breakdown.append('+1: Impulse GREEN - Bulls in control')
        signals.append('Impulse permits buying')
    elif impulse == 'RED':
        score -= 2
        breakdown.append('-2: Impulse RED - Bears in control')
        signals.append('⛔ Impulse RED = DO NOT BUY')
    else:  # BLUE
        breakdown.append('+0: Impulse BLUE - Neutral')
        signals.append('Impulse neutral - caution')

    # Candlestick pattern bonus
    bullish_patterns = [p for p in patterns if 'bullish' in p.get('type', '')]
    if bullish_patterns:
        pattern_names = [p['name'] for p in bullish_patterns]
        best_reliability = max(p.get('reliability', 1)
                               for p in bullish_patterns)
        if best_reliability >= 4:
            score += 2
            breakdown.append(
                f'+2: Strong bullish pattern ({", ".join(pattern_names)})')
        else:
            score += 1
            breakdown.append(
                f'+1: Bullish pattern ({", ".join(pattern_names)})')
        signals.append(f'🕯️ Candlestick: {", ".join(pattern_names)}')

    # Ensure score is within bounds
    score = max(0, min(10, score))

    # Determine grade
    if impulse == 'RED':
        grade = 'AVOID'
        action = 'STAY OUT'
    elif score >= 5:
        grade = 'A'
        action = 'BUY'
    elif score >= 3:
        grade = 'B'
        action = 'WATCH'
    else:
        grade = 'C'
        action = 'WATCH'

    return {
        'signal_strength': score,
        'grade': grade,
        'action': action,
        'is_a_trade': grade == 'A' and impulse != 'RED',
        'breakdown': breakdown,
        'signals': signals
    }


def calculate_trade_levels(price: float, atr: float, target_rr: float = 2.0) -> Dict:
    """
    Calculate entry, stop-loss, and target levels

    Elder's SafeZone method:
    - Stop: 2× ATR below entry (outside normal noise)
    - Target: Based on risk:reward ratio

    Args:
        price: Current price (entry)
        atr: Average True Range
        target_rr: Target risk:reward ratio (default 2.0)

    Returns:
        Dictionary with entry, stop, targets, and position info
    """
    stop_distance = atr * 2
    entry = round(price, 2)
    stop_loss = round(price - stop_distance, 2)
    risk = entry - stop_loss

    target_1 = round(entry + (risk * 1.5), 2)  # 1:1.5 R:R
    target_2 = round(entry + (risk * target_rr), 2)  # Full target
    target_3 = round(entry + (risk * 3), 2)  # Extended target

    return {
        'entry': entry,
        'stop_loss': stop_loss,
        'target_1': target_1,
        'target_2': target_2,
        'target_3': target_3,
        'risk_per_share': round(risk, 2),
        'risk_percent': round((risk / entry) * 100, 2),
        'risk_reward': target_rr
    }


def scan_stock(symbol: str, config: Dict = None) -> Optional[Dict]:
    """
    Complete analysis of a single stock

    Performs:
    1. Data fetch
    2. Weekly trend analysis (Screen 1)
    3. Daily indicator calculation (Screen 2)
    4. Candlestick pattern recognition
    5. Signal strength scoring
    6. Trade level calculation

    Args:
        symbol: Stock ticker
        config: Optional indicator configuration (uses default if None)

    Returns:
        Complete analysis dictionary or None if failed
    """
    if config is None:
        config = DEFAULT_INDICATOR_CONFIG

    # Fetch data
    data = fetch_stock_data(symbol)
    if not data:
        return None

    hist = data['history']

    # Weekly analysis (Screen 1)
    weekly = analyze_weekly_trend(hist)

    # Daily indicators (Screen 2)
    indicators = calculate_all_indicators(
        hist['High'],
        hist['Low'],
        hist['Close'],
        hist['Volume']
    )

    # Candlestick patterns
    patterns = scan_patterns(hist)
    bullish_patterns = get_bullish_patterns(patterns)
    pattern_score = get_pattern_score(patterns)

    # Calculate signal strength
    scoring = calculate_signal_strength(indicators, weekly, patterns)

    # Calculate trade levels
    levels = calculate_trade_levels(indicators['price'], indicators['atr'])

    # Get price change
    current_price = hist['Close'].iloc[-1]
    prev_price = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
    change = current_price - prev_price
    change_pct = (change / prev_price) * 100

    result = {
        'symbol': symbol,
        'name': data['name'],
        'sector': data['sector'],
        'price': round(current_price, 2),
        'change': round(change, 2),
        'change_percent': round(change_pct, 2),

        # Weekly Screen (Screen 1)
        'weekly_trend': weekly['weekly_trend'],
        'weekly_ema_slope': weekly['weekly_ema_slope'],
        'weekly_macd_rising': weekly['weekly_macd_rising'],
        'weekly_bullish': weekly['weekly_bullish'],

        # Daily Indicators (Screen 2)
        'ema_13': round(indicators['ema_13'], 2),
        'ema_22': round(indicators['ema_22'], 2),
        'macd_histogram': round(indicators['macd_histogram'], 4),
        'macd_rising': indicators['macd_rising'],
        'force_index': round(indicators['force_index_2'], 0),
        'stochastic': round(indicators['stochastic_k'], 1),
        'rsi': round(indicators['rsi'], 1),
        'atr': round(indicators['atr'], 2),
        'impulse_color': indicators['impulse_color'],
        'price_vs_ema': round(indicators['price_vs_ema'], 1),
        'channel_width': round(indicators['channel_width'], 1),

        # Keltner Channel (KC 20,10,1) for Trade Bill
        'kc_upper': round(indicators.get('kc_upper', current_price * 1.03), 2),
        'kc_lower': round(indicators.get('kc_lower', current_price * 0.97), 2),
        'kc_middle': round(indicators.get('kc_middle', current_price), 2),
        'kc_channel_height': round(indicators.get('kc_channel_height', indicators['atr'] * 2), 2),

        # Divergences
        'bullish_divergence_macd': indicators['bullish_divergence_macd'],
        'bullish_divergence_rsi': indicators['bullish_divergence_rsi'],

        # Candlestick Patterns
        'candlestick_patterns': patterns,
        'bullish_patterns': bullish_patterns,
        'pattern_names': [p['name'] for p in patterns],
        'bullish_pattern_names': [p['name'] for p in bullish_patterns],
        'pattern_score': pattern_score,

        # Scoring
        'signal_strength': scoring['signal_strength'],
        'grade': scoring['grade'],
        'action': scoring['action'],
        'is_a_trade': scoring['is_a_trade'],
        'score_breakdown': scoring['breakdown'],
        'signals': scoring['signals'],

        # Trade Levels
        'entry': levels['entry'],
        'stop_loss': levels['stop_loss'],
        'target_1': levels['target_1'],
        'target_2': levels['target_2'],
        'risk_percent': levels['risk_percent'],
        'risk_reward': levels['risk_reward'],

        # Indicator Config Used
        'indicator_config': config.get('name', 'Custom')
    }

    # Convert all numpy types to native Python types for JSON serialization
    return convert_to_native(result)


def run_weekly_screen(market: str = 'US', symbols: List[str] = None) -> Dict:
    """
    Run weekly screener on watchlist

    Args:
        market: 'US' or 'IN'
        symbols: Optional custom symbol list

    Returns:
        Dictionary with scan results and summary
    """
    if symbols is None:
        symbols = NASDAQ_100_TOP if market == 'US' else NIFTY_50

    results = []
    passed = []

    for symbol in symbols:
        analysis = scan_stock(symbol)
        if analysis:
            results.append(analysis)
            if analysis['weekly_bullish']:
                passed.append(analysis)

    # Sort by signal strength
    results.sort(key=lambda x: x['signal_strength'], reverse=True)
    passed.sort(key=lambda x: x['signal_strength'], reverse=True)

    # Categorize results
    a_trades = [r for r in results if r['is_a_trade']]
    b_trades = [r for r in results if r['grade'] == 'B']
    watch = [r for r in results if r['grade'] == 'C']
    avoid = [r for r in results if r['grade'] == 'AVOID']

    return convert_to_native({
        'scan_date': datetime.now().isoformat(),
        'market': market,
        'total_scanned': len(symbols),
        'total_analyzed': len(results),
        'weekly_bullish_count': len(passed),

        'summary': {
            'a_trades': len(a_trades),
            'b_trades': len(b_trades),
            'watch_list': len(watch),
            'avoid': len(avoid)
        },

        'a_trades': a_trades,
        'b_trades': b_trades,
        'watch_list': watch,
        'avoid': avoid,
        'all_results': results,

        'grading_criteria': get_grading_criteria()
    })


def run_daily_screen(weekly_results: List[Dict]) -> Dict:
    """
    Run daily screen on stocks that passed weekly screen

    Args:
        weekly_results: Results from weekly screen

    Returns:
        Dictionary with daily screen results
    """
    if not weekly_results:
        return {
            'error': 'No weekly results provided',
            'message': 'Run weekly screen first'
        }

    symbols = [r['symbol'] for r in weekly_results]

    results = []
    for symbol in symbols:
        analysis = scan_stock(symbol)
        if analysis:
            # Re-check daily conditions
            daily_ready = (
                analysis['force_index'] < 0 or
                analysis['stochastic'] < 50 or
                analysis['impulse_color'] != 'RED'
            )
            analysis['daily_ready'] = daily_ready
            results.append(analysis)

    # Sort by signal strength
    results.sort(key=lambda x: x['signal_strength'], reverse=True)

    a_trades = [r for r in results if r['is_a_trade']]

    return {
        'scan_date': datetime.now().isoformat(),
        'stocks_from_weekly': len(symbols),
        'daily_ready_count': len([r for r in results if r.get('daily_ready')]),
        'a_trades': a_trades,
        'all_results': results
    }
