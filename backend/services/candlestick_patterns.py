"""
Elder Trading System - Candlestick Pattern Recognition
Identifies key candlestick patterns for entry confirmation

PATTERN CATEGORIES:
1. SINGLE CANDLE - Hammer, Doji, Marubozu, etc.
2. TWO CANDLE - Engulfing, Harami, Piercing, etc.
3. THREE CANDLE - Morning Star, Three Soldiers, etc.

Each pattern includes:
- Name and description
- Bullish/Bearish classification
- Reliability score (1-5)
- Context requirements (uptrend/downtrend)
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional


# ============================================================
# PATTERN DEFINITIONS
# ============================================================

CANDLESTICK_PATTERNS = {
    # ========== SINGLE CANDLE PATTERNS ==========
    "HAMMER": {
        "name": "Hammer",
        "type": "bullish_reversal",
        "candles": 1,
        "reliability": 3,
        "description": "Small body at top, long lower shadow (2x+ body)",
        "context": "downtrend",
        "signal": "Potential bottom - bulls rejected lower prices"
    },
    "INVERTED_HAMMER": {
        "name": "Inverted Hammer",
        "type": "bullish_reversal",
        "candles": 1,
        "reliability": 2,
        "description": "Small body at bottom, long upper shadow",
        "context": "downtrend",
        "signal": "Potential reversal - needs confirmation"
    },
    "HANGING_MAN": {
        "name": "Hanging Man",
        "type": "bearish_reversal",
        "candles": 1,
        "reliability": 2,
        "description": "Same shape as hammer but in uptrend",
        "context": "uptrend",
        "signal": "Warning sign - potential top"
    },
    "SHOOTING_STAR": {
        "name": "Shooting Star",
        "type": "bearish_reversal",
        "candles": 1,
        "reliability": 3,
        "description": "Small body at bottom, long upper shadow in uptrend",
        "context": "uptrend",
        "signal": "Potential top - bulls lost control"
    },
    "DOJI": {
        "name": "Doji",
        "type": "neutral",
        "candles": 1,
        "reliability": 2,
        "description": "Open equals close - indecision",
        "context": "any",
        "signal": "Indecision - trend may change"
    },
    "DRAGONFLY_DOJI": {
        "name": "Dragonfly Doji",
        "type": "bullish_reversal",
        "candles": 1,
        "reliability": 3,
        "description": "Doji with long lower shadow, no upper shadow",
        "context": "downtrend",
        "signal": "Strong reversal signal at bottom"
    },
    "GRAVESTONE_DOJI": {
        "name": "Gravestone Doji",
        "type": "bearish_reversal",
        "candles": 1,
        "reliability": 3,
        "description": "Doji with long upper shadow, no lower shadow",
        "context": "uptrend",
        "signal": "Strong reversal signal at top"
    },
    "BULLISH_MARUBOZU": {
        "name": "Bullish Marubozu",
        "type": "bullish_continuation",
        "candles": 1,
        "reliability": 4,
        "description": "Large green candle, no shadows - pure buying",
        "context": "any",
        "signal": "Strong bullish momentum"
    },
    "BEARISH_MARUBOZU": {
        "name": "Bearish Marubozu",
        "type": "bearish_continuation",
        "candles": 1,
        "reliability": 4,
        "description": "Large red candle, no shadows - pure selling",
        "context": "any",
        "signal": "Strong bearish momentum"
    },
    "SPINNING_TOP": {
        "name": "Spinning Top",
        "type": "neutral",
        "candles": 1,
        "reliability": 1,
        "description": "Small body with shadows on both sides",
        "context": "any",
        "signal": "Indecision - wait for confirmation"
    },
    
    # ========== TWO CANDLE PATTERNS ==========
    "BULLISH_ENGULFING": {
        "name": "Bullish Engulfing",
        "type": "bullish_reversal",
        "candles": 2,
        "reliability": 4,
        "description": "Green candle completely engulfs previous red candle",
        "context": "downtrend",
        "signal": "Strong reversal - bulls took control"
    },
    "BEARISH_ENGULFING": {
        "name": "Bearish Engulfing",
        "type": "bearish_reversal",
        "candles": 2,
        "reliability": 4,
        "description": "Red candle completely engulfs previous green candle",
        "context": "uptrend",
        "signal": "Strong reversal - bears took control"
    },
    "BULLISH_HARAMI": {
        "name": "Bullish Harami",
        "type": "bullish_reversal",
        "candles": 2,
        "reliability": 2,
        "description": "Small green candle inside previous large red candle",
        "context": "downtrend",
        "signal": "Potential reversal - selling pressure reduced"
    },
    "BEARISH_HARAMI": {
        "name": "Bearish Harami",
        "type": "bearish_reversal",
        "candles": 2,
        "reliability": 2,
        "description": "Small red candle inside previous large green candle",
        "context": "uptrend",
        "signal": "Potential reversal - buying pressure reduced"
    },
    "PIERCING_LINE": {
        "name": "Piercing Line",
        "type": "bullish_reversal",
        "candles": 2,
        "reliability": 3,
        "description": "Green candle opens below prev low, closes above midpoint",
        "context": "downtrend",
        "signal": "Bullish reversal - strong close"
    },
    "DARK_CLOUD_COVER": {
        "name": "Dark Cloud Cover",
        "type": "bearish_reversal",
        "candles": 2,
        "reliability": 3,
        "description": "Red candle opens above prev high, closes below midpoint",
        "context": "uptrend",
        "signal": "Bearish reversal - strong selling"
    },
    "TWEEZER_BOTTOM": {
        "name": "Tweezer Bottom",
        "type": "bullish_reversal",
        "candles": 2,
        "reliability": 3,
        "description": "Two candles with same low - support confirmed",
        "context": "downtrend",
        "signal": "Double support test - potential bottom"
    },
    "TWEEZER_TOP": {
        "name": "Tweezer Top",
        "type": "bearish_reversal",
        "candles": 2,
        "reliability": 3,
        "description": "Two candles with same high - resistance confirmed",
        "context": "uptrend",
        "signal": "Double resistance test - potential top"
    },
    
    # ========== THREE CANDLE PATTERNS ==========
    "MORNING_STAR": {
        "name": "Morning Star",
        "type": "bullish_reversal",
        "candles": 3,
        "reliability": 5,
        "description": "Red candle, small body/doji, green candle",
        "context": "downtrend",
        "signal": "Very strong reversal - high reliability"
    },
    "EVENING_STAR": {
        "name": "Evening Star",
        "type": "bearish_reversal",
        "candles": 3,
        "reliability": 5,
        "description": "Green candle, small body/doji, red candle",
        "context": "uptrend",
        "signal": "Very strong reversal - high reliability"
    },
    "THREE_WHITE_SOLDIERS": {
        "name": "Three White Soldiers",
        "type": "bullish_continuation",
        "candles": 3,
        "reliability": 5,
        "description": "Three consecutive large green candles",
        "context": "any",
        "signal": "Strong bullish momentum - trend continuation"
    },
    "THREE_BLACK_CROWS": {
        "name": "Three Black Crows",
        "type": "bearish_continuation",
        "candles": 3,
        "reliability": 5,
        "description": "Three consecutive large red candles",
        "context": "any",
        "signal": "Strong bearish momentum - trend continuation"
    },
    "THREE_INSIDE_UP": {
        "name": "Three Inside Up",
        "type": "bullish_reversal",
        "candles": 3,
        "reliability": 4,
        "description": "Bullish harami followed by confirmation candle",
        "context": "downtrend",
        "signal": "Confirmed bullish reversal"
    },
    "THREE_INSIDE_DOWN": {
        "name": "Three Inside Down",
        "type": "bearish_reversal",
        "candles": 3,
        "reliability": 4,
        "description": "Bearish harami followed by confirmation candle",
        "context": "uptrend",
        "signal": "Confirmed bearish reversal"
    }
}


# ============================================================
# PATTERN DETECTION FUNCTIONS
# ============================================================

def _body_size(open_price, close_price):
    """Calculate candle body size"""
    return abs(close_price - open_price)

def _is_bullish(open_price, close_price):
    """Check if candle is bullish (green)"""
    return close_price > open_price

def _is_bearish(open_price, close_price):
    """Check if candle is bearish (red)"""
    return close_price < open_price

def _upper_shadow(high, open_price, close_price):
    """Calculate upper shadow length"""
    return high - max(open_price, close_price)

def _lower_shadow(low, open_price, close_price):
    """Calculate lower shadow length"""
    return min(open_price, close_price) - low

def _is_doji(open_price, close_price, high, low, threshold=0.1):
    """Check if candle is a doji (body < threshold of range)"""
    body = _body_size(open_price, close_price)
    range_size = high - low
    if range_size == 0:
        return True
    return body / range_size < threshold


def detect_hammer(row, prev_rows, trend):
    """Detect Hammer pattern"""
    o, h, l, c = row['Open'], row['High'], row['Low'], row['Close']
    body = _body_size(o, c)
    lower_shadow = _lower_shadow(l, o, c)
    upper_shadow = _upper_shadow(h, o, c)
    range_size = h - l
    
    if range_size == 0:
        return False
    
    # Hammer: small body at top, long lower shadow
    body_ratio = body / range_size
    lower_ratio = lower_shadow / range_size
    
    return (body_ratio < 0.3 and 
            lower_ratio > 0.6 and 
            upper_shadow < body and
            trend == 'down')


def detect_shooting_star(row, prev_rows, trend):
    """Detect Shooting Star pattern"""
    o, h, l, c = row['Open'], row['High'], row['Low'], row['Close']
    body = _body_size(o, c)
    upper_shadow = _upper_shadow(h, o, c)
    lower_shadow = _lower_shadow(l, o, c)
    range_size = h - l
    
    if range_size == 0:
        return False
    
    body_ratio = body / range_size
    upper_ratio = upper_shadow / range_size
    
    return (body_ratio < 0.3 and 
            upper_ratio > 0.6 and 
            lower_shadow < body and
            trend == 'up')


def detect_doji(row, prev_rows, trend):
    """Detect Doji pattern"""
    return _is_doji(row['Open'], row['Close'], row['High'], row['Low'])


def detect_bullish_engulfing(row, prev_row, trend):
    """Detect Bullish Engulfing pattern"""
    if prev_row is None:
        return False
    
    prev_o, prev_c = prev_row['Open'], prev_row['Close']
    curr_o, curr_c = row['Open'], row['Close']
    
    # Previous candle is bearish, current is bullish
    # Current body engulfs previous body
    return (_is_bearish(prev_o, prev_c) and 
            _is_bullish(curr_o, curr_c) and
            curr_o < prev_c and  # Opens below prev close
            curr_c > prev_o and  # Closes above prev open
            trend == 'down')


def detect_bearish_engulfing(row, prev_row, trend):
    """Detect Bearish Engulfing pattern"""
    if prev_row is None:
        return False
    
    prev_o, prev_c = prev_row['Open'], prev_row['Close']
    curr_o, curr_c = row['Open'], row['Close']
    
    return (_is_bullish(prev_o, prev_c) and 
            _is_bearish(curr_o, curr_c) and
            curr_o > prev_c and 
            curr_c < prev_o and
            trend == 'up')


def detect_morning_star(rows, trend):
    """Detect Morning Star pattern (3 candles)"""
    if len(rows) < 3:
        return False
    
    first, second, third = rows.iloc[-3], rows.iloc[-2], rows.iloc[-1]
    
    # First: large bearish
    first_bearish = _is_bearish(first['Open'], first['Close'])
    first_large = _body_size(first['Open'], first['Close']) > (first['High'] - first['Low']) * 0.5
    
    # Second: small body (star)
    second_small = _body_size(second['Open'], second['Close']) < (second['High'] - second['Low']) * 0.3
    
    # Third: large bullish, closes above midpoint of first
    third_bullish = _is_bullish(third['Open'], third['Close'])
    first_midpoint = (first['Open'] + first['Close']) / 2
    third_above_mid = third['Close'] > first_midpoint
    
    return (first_bearish and first_large and 
            second_small and 
            third_bullish and third_above_mid and
            trend == 'down')


def detect_evening_star(rows, trend):
    """Detect Evening Star pattern (3 candles)"""
    if len(rows) < 3:
        return False
    
    first, second, third = rows.iloc[-3], rows.iloc[-2], rows.iloc[-1]
    
    first_bullish = _is_bullish(first['Open'], first['Close'])
    first_large = _body_size(first['Open'], first['Close']) > (first['High'] - first['Low']) * 0.5
    second_small = _body_size(second['Open'], second['Close']) < (second['High'] - second['Low']) * 0.3
    third_bearish = _is_bearish(third['Open'], third['Close'])
    first_midpoint = (first['Open'] + first['Close']) / 2
    third_below_mid = third['Close'] < first_midpoint
    
    return (first_bullish and first_large and 
            second_small and 
            third_bearish and third_below_mid and
            trend == 'up')


def detect_three_white_soldiers(rows, trend):
    """Detect Three White Soldiers pattern"""
    if len(rows) < 3:
        return False
    
    candles = rows.iloc[-3:]
    
    for i in range(3):
        if not _is_bullish(candles.iloc[i]['Open'], candles.iloc[i]['Close']):
            return False
        # Each candle should be large
        body = _body_size(candles.iloc[i]['Open'], candles.iloc[i]['Close'])
        range_size = candles.iloc[i]['High'] - candles.iloc[i]['Low']
        if range_size > 0 and body / range_size < 0.6:
            return False
    
    # Each should close higher than previous
    return (candles.iloc[1]['Close'] > candles.iloc[0]['Close'] and
            candles.iloc[2]['Close'] > candles.iloc[1]['Close'])


def detect_three_black_crows(rows, trend):
    """Detect Three Black Crows pattern"""
    if len(rows) < 3:
        return False
    
    candles = rows.iloc[-3:]
    
    for i in range(3):
        if not _is_bearish(candles.iloc[i]['Open'], candles.iloc[i]['Close']):
            return False
        body = _body_size(candles.iloc[i]['Open'], candles.iloc[i]['Close'])
        range_size = candles.iloc[i]['High'] - candles.iloc[i]['Low']
        if range_size > 0 and body / range_size < 0.6:
            return False
    
    return (candles.iloc[1]['Close'] < candles.iloc[0]['Close'] and
            candles.iloc[2]['Close'] < candles.iloc[1]['Close'])


def detect_bullish_marubozu(row, prev_rows, trend):
    """Detect Bullish Marubozu"""
    o, h, l, c = row['Open'], row['High'], row['Low'], row['Close']
    body = _body_size(o, c)
    range_size = h - l
    
    if range_size == 0:
        return False
    
    # Large bullish candle with minimal shadows
    upper_shadow = _upper_shadow(h, o, c)
    lower_shadow = _lower_shadow(l, o, c)
    
    return (_is_bullish(o, c) and
            body / range_size > 0.9 and
            upper_shadow < body * 0.05 and
            lower_shadow < body * 0.05)


# ============================================================
# MAIN PATTERN SCANNER
# ============================================================

def determine_trend(closes: pd.Series, lookback: int = 10) -> str:
    """Determine short-term trend"""
    if len(closes) < lookback:
        return 'neutral'
    
    recent = closes.tail(lookback)
    sma = recent.mean()
    current = closes.iloc[-1]
    
    if current > sma * 1.02:
        return 'up'
    elif current < sma * 0.98:
        return 'down'
    return 'neutral'


def scan_patterns(df: pd.DataFrame) -> List[Dict]:
    """
    Scan for all candlestick patterns in the data
    
    Args:
        df: DataFrame with OHLC data
    
    Returns:
        List of detected patterns with details
    """
    if len(df) < 5:
        return []
    
    patterns = []
    trend = determine_trend(df['Close'])
    
    current = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else None
    
    # Single candle patterns
    if detect_hammer(current, None, trend):
        patterns.append({
            "pattern": "HAMMER",
            **CANDLESTICK_PATTERNS["HAMMER"]
        })
    
    if detect_shooting_star(current, None, trend):
        patterns.append({
            "pattern": "SHOOTING_STAR", 
            **CANDLESTICK_PATTERNS["SHOOTING_STAR"]
        })
    
    if detect_doji(current, None, trend):
        patterns.append({
            "pattern": "DOJI",
            **CANDLESTICK_PATTERNS["DOJI"]
        })
    
    if detect_bullish_marubozu(current, None, trend):
        patterns.append({
            "pattern": "BULLISH_MARUBOZU",
            **CANDLESTICK_PATTERNS["BULLISH_MARUBOZU"]
        })
    
    # Two candle patterns
    if prev is not None:
        if detect_bullish_engulfing(current, prev, trend):
            patterns.append({
                "pattern": "BULLISH_ENGULFING",
                **CANDLESTICK_PATTERNS["BULLISH_ENGULFING"]
            })
        
        if detect_bearish_engulfing(current, prev, trend):
            patterns.append({
                "pattern": "BEARISH_ENGULFING",
                **CANDLESTICK_PATTERNS["BEARISH_ENGULFING"]
            })
    
    # Three candle patterns
    if len(df) >= 3:
        if detect_morning_star(df, trend):
            patterns.append({
                "pattern": "MORNING_STAR",
                **CANDLESTICK_PATTERNS["MORNING_STAR"]
            })
        
        if detect_evening_star(df, trend):
            patterns.append({
                "pattern": "EVENING_STAR",
                **CANDLESTICK_PATTERNS["EVENING_STAR"]
            })
        
        if detect_three_white_soldiers(df, trend):
            patterns.append({
                "pattern": "THREE_WHITE_SOLDIERS",
                **CANDLESTICK_PATTERNS["THREE_WHITE_SOLDIERS"]
            })
        
        if detect_three_black_crows(df, trend):
            patterns.append({
                "pattern": "THREE_BLACK_CROWS",
                **CANDLESTICK_PATTERNS["THREE_BLACK_CROWS"]
            })
    
    return patterns


def get_bullish_patterns(patterns: List[Dict]) -> List[Dict]:
    """Filter for bullish patterns only"""
    return [p for p in patterns if 'bullish' in p.get('type', '')]


def get_bearish_patterns(patterns: List[Dict]) -> List[Dict]:
    """Filter for bearish patterns only"""
    return [p for p in patterns if 'bearish' in p.get('type', '')]


def get_pattern_score(patterns: List[Dict]) -> int:
    """Calculate combined pattern score (for signal strength)"""
    score = 0
    for p in patterns:
        reliability = p.get('reliability', 1)
        if 'bullish' in p.get('type', ''):
            score += reliability
        elif 'bearish' in p.get('type', ''):
            score -= reliability
    return score
