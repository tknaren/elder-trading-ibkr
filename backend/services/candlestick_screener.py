"""
Elder Trading System - Candlestick Pattern Historical Screener
Scans for specific candlestick patterns with indicator filters

Patterns Scanned (Long Trades Only):
1. Hammer - Small body at top, long lower shadow (2x+ body)
2. Bullish Engulfing - Current bullish candle engulfs previous bearish
3. Piercing Pattern - White candle pushes >50% into prior black body
4. Tweezer Bottom - Two successive candles with same lows

Filter Conditions:
- Pattern detected on daily candle
- Price below Keltner Channel Lower (KC-1)
- RSI(14) < 30 (Oversold)

Shows all indicator values even if filters don't match.
"""

import os
import csv
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any


def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average"""
    return series.ewm(span=period, adjust=False).mean()


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Average True Range"""
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def calculate_keltner_channel(high: pd.Series, low: pd.Series, close: pd.Series,
                              ema_period: int = 20, atr_period: int = 10,
                              atr_mult: float = 1.0) -> Dict:
    """
    Calculate Keltner Channel
    KC(20, 10, 1): EMA(20) +/- 1*ATR(10)
    """
    middle = calculate_ema(close, ema_period)
    atr = calculate_atr(high, low, close, atr_period)

    upper = middle + (atr_mult * atr)
    lower = middle - (atr_mult * atr)

    return {
        'kc_upper': upper,
        'kc_middle': middle,
        'kc_lower': lower,
        'atr': atr
    }


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index using Close prices (Wilder's smoothing)"""
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # Use Wilder's smoothing (EMA with alpha = 1/period) for RSI calculation
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict:
    """Calculate MACD indicator"""
    ema_fast = calculate_ema(close, fast)
    ema_slow = calculate_ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line

    return {
        'macd_line': macd_line,
        'signal_line': signal_line,
        'histogram': histogram
    }


def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                         k_period: int = 14, d_period: int = 3) -> Dict:
    """Calculate Stochastic Oscillator"""
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()

    stoch_k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    stoch_d = stoch_k.rolling(window=d_period).mean()

    return {
        'stoch_k': stoch_k,
        'stoch_d': stoch_d
    }


# ============================================================
# CANDLESTICK PATTERN DETECTION (Based on Candle_Sticks.docx)
# ============================================================

def _body_size(open_price: float, close_price: float) -> float:
    """Calculate candle body size"""
    return abs(close_price - open_price)


def _is_bullish(open_price: float, close_price: float) -> bool:
    """Check if candle is bullish (green/white)"""
    return close_price > open_price


def _is_bearish(open_price: float, close_price: float) -> bool:
    """Check if candle is bearish (red/black)"""
    return close_price < open_price


def _upper_shadow(high: float, open_price: float, close_price: float) -> float:
    """Calculate upper shadow length"""
    return high - max(open_price, close_price)


def _lower_shadow(low: float, open_price: float, close_price: float) -> float:
    """Calculate lower shadow length"""
    return min(open_price, close_price) - low


def detect_hammer(row: pd.Series, trend: str = 'down') -> Dict:
    """
    Detect Hammer Pattern (from Candle_Sticks.docx)

    Rules:
    1. The real body is at the upper end of the trading range
    2. Color of real body is not important
    3. Long lower shadow >= 2x height of real body
    4. No or very short upper shadow
    5. Must come after a decline
    """
    o, h, l, c = row['Open'], row['High'], row['Low'], row['Close']

    body = _body_size(o, c)
    lower_shadow = _lower_shadow(l, o, c)
    upper_shadow = _upper_shadow(h, o, c)
    total_range = h - l

    if total_range == 0:
        return {'detected': False, 'pattern': None}

    # Hammer criteria from document
    body_at_top = max(o, c) > (l + total_range * 0.6)  # Body in upper 40%
    long_lower_shadow = lower_shadow >= body * 2  # Lower shadow >= 2x body
    short_upper_shadow = upper_shadow <= body * 0.3  # Very short upper shadow

    detected = body_at_top and long_lower_shadow and short_upper_shadow

    return {
        'detected': detected,
        'pattern': 'Hammer' if detected else None,
        'reliability': 3,  # Medium-high reliability
        'body_size': round(body, 2),
        'lower_shadow': round(lower_shadow, 2),
        'upper_shadow': round(upper_shadow, 2),
        'lower_shadow_ratio': round(lower_shadow / body if body > 0 else 0, 2)
    }


def detect_bullish_engulfing(current: pd.Series, prev: pd.Series, trend: str = 'down') -> Dict:
    """
    Detect Bullish Engulfing Pattern (from Candle_Sticks.docx)

    Rules:
    1. Market must be in a clearly definable downtrend
    2. Two candles comprise the pattern
    3. Second real body must engulf the prior real body (need not engulf shadows)
    4. Second real body should be opposite color (bullish after bearish)

    Exception: If first real body is a doji, it's valid
    """
    if prev is None:
        return {'detected': False, 'pattern': None}

    curr_o, curr_c = current['Open'], current['Close']
    prev_o, prev_c = prev['Open'], prev['Close']

    # Current candle must be bullish
    curr_bullish = _is_bullish(curr_o, curr_c)
    # Previous candle must be bearish (or doji)
    prev_bearish = _is_bearish(prev_o, prev_c) or abs(prev_c - prev_o) < 0.01

    # Current body must engulf previous body
    curr_body_high = max(curr_o, curr_c)
    curr_body_low = min(curr_o, curr_c)
    prev_body_high = max(prev_o, prev_c)
    prev_body_low = min(prev_o, prev_c)

    engulfs = curr_body_low <= prev_body_low and curr_body_high >= prev_body_high

    detected = curr_bullish and prev_bearish and engulfs

    return {
        'detected': detected,
        'pattern': 'Bullish Engulfing' if detected else None,
        'reliability': 4,  # High reliability
        'curr_body': round(abs(curr_c - curr_o), 2),
        'prev_body': round(abs(prev_c - prev_o), 2)
    }


def detect_piercing_pattern(current: pd.Series, prev: pd.Series, trend: str = 'down') -> Dict:
    """
    Detect Piercing Pattern (from Candle_Sticks.docx)

    Rules:
    1. White (bullish) real body pierces but does not wrap around prior black body
    2. Greater degree of penetration = more likely bottom reversal
    3. Ideal: White body pushes MORE than halfway into prior black body
    """
    if prev is None:
        return {'detected': False, 'pattern': None}

    curr_o, curr_c = current['Open'], current['Close']
    prev_o, prev_c = prev['Open'], prev['Close']

    # Current must be bullish, previous must be bearish
    curr_bullish = _is_bullish(curr_o, curr_c)
    prev_bearish = _is_bearish(prev_o, prev_c)

    if not (curr_bullish and prev_bearish):
        return {'detected': False, 'pattern': None}

    # Calculate penetration into previous black body
    prev_body_size = abs(prev_c - prev_o)
    if prev_body_size == 0:
        return {'detected': False, 'pattern': None}

    # Current close should be above midpoint of previous body
    prev_midpoint = prev_c + (prev_body_size / 2)  # prev_c is lower (bearish)

    # Current open should be below previous close
    opens_below = curr_o <= prev_c
    # Current close should be above previous midpoint but below previous open
    pierces_halfway = curr_c > prev_midpoint and curr_c < prev_o

    detected = opens_below and pierces_halfway

    penetration_pct = ((curr_c - prev_c) / prev_body_size *
                       100) if prev_body_size > 0 else 0

    return {
        'detected': detected,
        'pattern': 'Piercing Pattern' if detected else None,
        'reliability': 3,
        'penetration_pct': round(penetration_pct, 1)
    }


def detect_tweezer_bottom(df: pd.DataFrame, lookback: int = 5) -> Dict:
    """
    Detect Tweezer Bottom Pattern (from Candle_Sticks.docx)

    Rules:
    1. In a falling market
    2. Two or more successive lows are the same (within tolerance)
    3. Can be composed of real bodies, shadows, and/or doji
    4. Ideal: Long first candle, small real body as next session
    """
    if len(df) < 2:
        return {'detected': False, 'pattern': None}

    recent = df.tail(lookback)

    # Check last two candles
    current = recent.iloc[-1]
    prev = recent.iloc[-2]

    # Tolerance for "same" low (0.1% of price)
    tolerance = current['Low'] * 0.001

    same_lows = abs(current['Low'] - prev['Low']) <= tolerance

    # Ideally first candle is longer
    prev_body = abs(prev['Close'] - prev['Open'])
    curr_body = abs(current['Close'] - current['Open'])
    ideal_size = prev_body > curr_body

    detected = same_lows

    return {
        'detected': detected,
        'pattern': 'Tweezer Bottom' if detected else None,
        'reliability': 2 if not ideal_size else 3,
        'low_1': round(prev['Low'], 2),
        'low_2': round(current['Low'], 2),
        'ideal_size_ratio': ideal_size
    }


def scan_candlestick_patterns(df: pd.DataFrame) -> List[Dict]:
    """
    Scan for all candlestick patterns on the last candle
    Returns list of detected patterns
    """
    if len(df) < 3:
        return []

    patterns = []
    current = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else None

    # Determine trend (simple: compare to 10-period SMA)
    sma_10 = df['Close'].rolling(10).mean().iloc[-1]
    trend = 'down' if current['Close'] < sma_10 else 'up'

    # Single candle patterns
    hammer = detect_hammer(current, trend)
    if hammer['detected']:
        patterns.append(hammer)

    # Two candle patterns
    if prev is not None:
        engulfing = detect_bullish_engulfing(current, prev, trend)
        if engulfing['detected']:
            patterns.append(engulfing)

        piercing = detect_piercing_pattern(current, prev, trend)
        if piercing['detected']:
            patterns.append(piercing)

    # Multi-candle patterns
    tweezer = detect_tweezer_bottom(df)
    if tweezer['detected']:
        patterns.append(tweezer)

    return patterns


def calculate_all_indicators(df: pd.DataFrame) -> Dict:
    """Calculate all required indicators for a dataframe"""
    if len(df) < 30:
        return {}

    # RSI
    rsi = calculate_rsi(df['Close'], 14)

    # Keltner Channel (20, 10, 1)
    kc = calculate_keltner_channel(
        df['High'], df['Low'], df['Close'],
        ema_period=20, atr_period=10, atr_mult=1.0
    )

    # MACD
    macd = calculate_macd(df['Close'], 12, 26, 9)

    # Stochastic
    stoch = calculate_stochastic(df['High'], df['Low'], df['Close'], 14, 3)

    return {
        'rsi': rsi,
        'kc_upper': kc['kc_upper'],
        'kc_middle': kc['kc_middle'],
        'kc_lower': kc['kc_lower'],
        'atr': kc['atr'],
        'macd_line': macd['macd_line'],
        'signal_line': macd['signal_line'],
        'macd_hist': macd['histogram'],
        'stoch_k': stoch['stoch_k'],
        'stoch_d': stoch['stoch_d']
    }


def scan_stock_candlestick_historical(
    symbol: str,
    hist: pd.DataFrame,
    lookback_days: int = 180,
    stock_info: Dict = None
) -> List[Dict]:
    """
    Scan a single stock's history for candlestick patterns with filters

    Args:
        symbol: Stock ticker
        hist: Historical OHLCV dataframe
        lookback_days: Number of days to scan
        stock_info: Dict with Name, Market Cap, Sector, Industry

    Returns:
        List of signals with pattern info and indicator values
    """
    if hist is None or len(hist) < 50:
        return []

    signals = []

    # Calculate all indicators once
    indicators = calculate_all_indicators(hist)
    if not indicators:
        return []

    # Calculate cutoff date for lookback period
    latest_date = hist.index[-1]
    cutoff_date = latest_date - pd.Timedelta(days=lookback_days)

    # Scan each day starting from day 50 for indicator stability
    for i in range(50, len(hist)):
        df_slice = hist.iloc[:i+1].copy()
        current_row = df_slice.iloc[-1]
        date = df_slice.index[-1]

        # Skip if date is outside lookback period
        if date < cutoff_date:
            continue

        # Get indicator values for this date
        idx = i
        rsi_val = float(indicators['rsi'].iloc[idx]) if not pd.isna(
            indicators['rsi'].iloc[idx]) else None
        kc_lower = float(indicators['kc_lower'].iloc[idx]) if not pd.isna(
            indicators['kc_lower'].iloc[idx]) else None
        kc_middle = float(indicators['kc_middle'].iloc[idx]) if not pd.isna(
            indicators['kc_middle'].iloc[idx]) else None
        kc_upper = float(indicators['kc_upper'].iloc[idx]) if not pd.isna(
            indicators['kc_upper'].iloc[idx]) else None
        macd_val = float(indicators['macd_line'].iloc[idx]) if not pd.isna(
            indicators['macd_line'].iloc[idx]) else None
        signal_val = float(indicators['signal_line'].iloc[idx]) if not pd.isna(
            indicators['signal_line'].iloc[idx]) else None
        macd_hist = float(indicators['macd_hist'].iloc[idx]) if not pd.isna(
            indicators['macd_hist'].iloc[idx]) else None
        stoch_k = float(indicators['stoch_k'].iloc[idx]) if not pd.isna(
            indicators['stoch_k'].iloc[idx]) else None

        close_price = float(current_row['Close'])

        # Scan for patterns
        patterns = scan_candlestick_patterns(df_slice)

        if patterns:
            # Get pattern names
            pattern_names = [p['pattern']
                             for p in patterns if p.get('pattern')]

            # Filter by selected patterns if specified
            if selected_patterns:
                pattern_names = [
                    p for p in pattern_names if p in selected_patterns]
                patterns = [p for p in patterns if p.get(
                    'pattern') in selected_patterns]

            # Skip if no matching patterns after filtering
            if not pattern_names:
                continue

            # Calculate KC threshold based on kc_level
            # kc_level: 0 means price < kc_middle, -1 means price < kc_lower, -2 means price < (kc_lower - ATR)
            if kc_level == 0:
                kc_threshold = kc_middle
            elif kc_level == -1:
                kc_threshold = kc_lower
            elif kc_level == -2:
                atr_val = float(indicators['atr'].iloc[idx]) if not pd.isna(
                    indicators['atr'].iloc[idx]) else 0
                kc_threshold = kc_lower - atr_val if kc_lower and atr_val else kc_lower
            else:
                # For other levels, use kc_middle + (kc_level * ATR)
                atr_val = float(indicators['atr'].iloc[idx]) if not pd.isna(
                    indicators['atr'].iloc[idx]) else 0
                kc_threshold = kc_middle + \
                    (kc_level * atr_val) if kc_middle and atr_val else kc_middle

            # Calculate filter conditions
            below_kc_threshold = close_price < kc_threshold if kc_threshold else False
            rsi_oversold = rsi_val < 30 if rsi_val else False

            # All filters must match for the stock to be filtered
            filters_match = below_kc_threshold and rsi_oversold

            # Sanitize pattern_details to ensure JSON serializable
            pattern_details_clean = []
            for p in patterns:
                clean_p = {}
                for k, v in p.items():
                    if isinstance(v, (np.integer, np.floating)):
                        clean_p[k] = float(v)
                    elif isinstance(v, np.bool_):
                        clean_p[k] = bool(v)
                    elif isinstance(v, np.ndarray):
                        clean_p[k] = v.tolist()
                    elif pd.isna(v):
                        clean_p[k] = None
                    else:
                        clean_p[k] = v
                pattern_details_clean.append(clean_p)

            signal = {
                'symbol': symbol,
                'name': stock_info.get('Name', '') if stock_info else '',
                'market_cap': stock_info.get('Market Cap', '') if stock_info else '',
                'sector': stock_info.get('Sector', '') if stock_info else '',
                'industry': stock_info.get('Industry', '') if stock_info else '',
                'date': str(date)[:10] if hasattr(date, 'strftime') else str(date)[:10],
                'patterns': pattern_names,
                'pattern_count': len(patterns),
                'pattern_details': pattern_details_clean,
                'close': round(float(close_price), 2),
                # Indicator values
                'rsi': round(float(rsi_val), 1) if rsi_val else None,
                'kc_lower': round(float(kc_lower), 2) if kc_lower else None,
                'kc_middle': round(float(kc_middle), 2) if kc_middle else None,
                'kc_upper': round(float(kc_upper), 2) if kc_upper else None,
                'kc_threshold': round(float(kc_threshold), 2) if kc_threshold else None,
                'macd': round(float(macd_val), 3) if macd_val else None,
                'macd_signal': round(float(signal_val), 3) if signal_val else None,
                'macd_hist': round(float(macd_hist), 3) if macd_hist else None,
                'stoch_k': round(float(stoch_k), 1) if stoch_k else None,
                # Filter status
                'below_kc_threshold': bool(below_kc_threshold),
                'rsi_oversold': bool(rsi_oversold),
                'filters_match': bool(filters_match)
            }

            signals.append(signal)

    return signals


def run_candlestick_screener(
    symbols: List[str],
    hist_data: Dict[str, pd.DataFrame],
    lookback_days: int = 180,
    filter_mode: str = 'all',  # 'all', 'filtered_only', 'patterns_only'
    stock_info_map: Dict[str, Dict] = None
) -> Dict:
    """
    Run candlestick pattern screener across multiple symbols

    Args:
        symbols: List of stock tickers
        hist_data: Dict of symbol -> DataFrame with OHLCV data
        lookback_days: Number of days to scan
        filter_mode:
            'all' - Show all patterns with indicator values
            'filtered_only' - Only show patterns matching KC and RSI filters
            'patterns_only' - Only show patterns, no filter requirement
        stock_info_map: Dict of symbol -> company info (Name, Market Cap, Sector, Industry)

    Returns:
        Dict with signals, summary, and metadata
    """
    all_signals = []
    symbols_with_signals = 0

    for symbol in symbols:
        try:
            hist = hist_data.get(symbol)
            if hist is None or len(hist) < 50:
                continue

            stock_info = stock_info_map.get(symbol) if stock_info_map else None
            signals = scan_stock_candlestick_historical(
                symbol, hist, lookback_days, stock_info)

            if signals:
                # Apply filter mode
                if filter_mode == 'filtered_only':
                    signals = [s for s in signals if s.get('filters_match')]

                if signals:
                    all_signals.extend(signals)
                    symbols_with_signals += 1

        except Exception as e:
            print(f"Error scanning {symbol}: {e}")

    # Sort by date descending, then by pattern count
    all_signals.sort(key=lambda x: (
        x['date'], x['pattern_count']), reverse=True)

    # Summary stats
    patterns_count = {}
    for s in all_signals:
        for p in s.get('patterns', []):
            patterns_count[p] = patterns_count.get(p, 0) + 1

    summary = {
        'total_signals': len(all_signals),
        'filtered_signals': len([s for s in all_signals if s.get('filters_match')]),
        'patterns_breakdown': patterns_count,
        'symbols_with_patterns': symbols_with_signals
    }

    return {
        'signals': all_signals,
        'summary': summary,
        'symbols_scanned': len(symbols),
        'lookback_days': lookback_days,
        'filter_mode': filter_mode,
        'kc_level': kc_level,
        'selected_patterns': selected_patterns
    }


# Stock list loader


def load_nasdaq_stocks_from_csv():
    """
    Load NASDAQ stocks from CSV file with company details

    Returns:
        Tuple of (symbols: List[str], stock_info_map: Dict[str, Dict])
        where stock_info_map contains Name, Market Cap, Sector, Industry for each symbol
    """
    csv_path = os.path.join(os.path.dirname(__file__),
                            '..', '..', 'docs', 'nasdaq_screener.csv')
    symbols = []
    stock_info_map = {}

    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                symbol = row.get('Symbol', '').strip()
                if symbol:
                    symbols.append(symbol)
                    stock_info_map[symbol] = {
                        'Name': row.get('Name', '').strip(),
                        'Market Cap': row.get('Market Cap', '').strip(),
                        'Sector': row.get('Sector', '').strip(),
                        'Industry': row.get('Industry', '').strip()
                    }
    except Exception as e:
        print(f"Error loading NASDAQ stocks from CSV: {e}")
        # Return empty if file not found or error
        return [], {}

    return symbols, stock_info_map


def get_stock_list(market: str = 'US'):
    """
    Get available stock list with company information

    Returns:
        Tuple of (symbols: List[str], stock_info_map: Dict[str, Dict])
    """
    if market.upper() == 'US':
        return load_nasdaq_stocks_from_csv()
    else:
        return NIFTY_100
