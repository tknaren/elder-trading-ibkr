"""
Elder Trading System - Enhanced Screener v2.0
Implements the Triple Screen methodology with CORRECT filter logic

FIXES APPLIED FROM VALIDATION:
1. Screen 1 (Weekly) is now a MANDATORY GATE - not just a scoring component
2. Impulse RED blocks trades entirely - not just a penalty
3. daily_ready uses proper AND/OR logic
4. is_a_trade includes weekly_bullish check
5. Stochastic threshold changed from 50 to 30 for oversold

NEW HIGH-SCORING RULES ADDED:
+3: Short term oversold (price near lower channel)
+3: MACD divergence (strongest signal per Elder)
+3: False downside breakout
+2: Kangaroo tails (long lower shadow)
+2: Force Index down spike
+3: Pullback to value in uptrend (Weekly EMA↑, Daily EMA↑, price < fast EMA)

ENTRY/STOP/TARGET CALCULATION (Elder's Method):
- ENTRY: Daily EMA-22 (buy at value)
- TARGET: Keltner Channel Upper Band
- STOP: Deepest historical EMA-22 penetration

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
    get_grading_criteria,
    calculate_keltner_channel,
    calculate_atr
)
from services.candlestick_patterns import scan_patterns, get_bullish_patterns, get_pattern_score
from services.indicator_config import (
    INDICATOR_CATALOG,
    DEFAULT_INDICATOR_CONFIG,
    get_indicator_info,
    get_config_summary
)
from services.ibkr_client import fetch_stock_data, check_connection, get_client, convert_to_native


# Default watchlists
# Full NASDAQ 100 stocks
NASDAQ_100 = [
    # Top 50 by market cap
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AMD', 'AVGO', 'NFLX',
    'COST', 'PEP', 'ADBE', 'CSCO', 'INTC', 'QCOM', 'TXN', 'INTU', 'AMAT', 'MU',
    'LRCX', 'KLAC', 'SNPS', 'CDNS', 'MRVL', 'ON', 'NXPI', 'ADI', 'MCHP', 'FTNT',
    'VRTX', 'CHTR', 'ASML', 'CRWD', 'PANW', 'MNST', 'TEAM', 'PAYX', 'AEP', 'REGN',
    'DXCM', 'CPRT', 'PCAR', 'ALGN', 'AMGN', 'MRNA', 'XEL', 'WDAY', 'ABNB', 'MDLZ',
    # Next 50
    'GILD', 'ISRG', 'BKNG', 'ADP', 'SBUX', 'PYPL', 'CME', 'ORLY', 'IDXX', 'CTAS',
    'MAR', 'CSX', 'ODFL', 'FAST', 'ROST', 'KDP', 'EXC', 'DLTR', 'BIIB', 'EA',
    'VRSK', 'ANSS', 'ILMN', 'SIRI', 'ZS', 'DDOG', 'CTSH', 'WBD', 'EBAY', 'FANG',
    'GFS', 'LCID', 'RIVN', 'CEG', 'TTD', 'GEHC', 'ZM', 'ROKU', 'OKTA', 'SPLK',
    'DOCU', 'BILL', 'ENPH', 'SEDG', 'DASH', 'COIN', 'HOOD', 'SOFI', 'PLTR', 'NET'
]

# Keep backward compatibility
NASDAQ_100_TOP = NASDAQ_100

# NIFTY 50 + NIFTY NEXT 50 = 100 stocks
NIFTY_100 = [
    # NIFTY 50
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
    'HINDUNILVR.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'ITC.NS', 'KOTAKBANK.NS',
    'LT.NS', 'AXISBANK.NS', 'ASIANPAINT.NS', 'MARUTI.NS', 'TITAN.NS',
    'SUNPHARMA.NS', 'ULTRACEMCO.NS', 'BAJFINANCE.NS', 'WIPRO.NS', 'HCLTECH.NS',
    'TATAMOTORS.NS', 'POWERGRID.NS', 'NTPC.NS', 'M&M.NS', 'JSWSTEEL.NS',
    'BAJAJFINSV.NS', 'ONGC.NS', 'TATASTEEL.NS', 'ADANIENT.NS', 'COALINDIA.NS',
    'GRASIM.NS', 'TECHM.NS', 'HINDALCO.NS', 'INDUSINDBK.NS', 'DRREDDY.NS',
    'APOLLOHOSP.NS', 'CIPLA.NS', 'EICHERMOT.NS', 'NESTLEIND.NS', 'DIVISLAB.NS',
    'BRITANNIA.NS', 'BPCL.NS', 'ADANIPORTS.NS', 'TATACONSUM.NS', 'HEROMOTOCO.NS',
    'SBILIFE.NS', 'HDFCLIFE.NS', 'BAJAJ-AUTO.NS', 'SHRIRAMFIN.NS', 'LTIM.NS',
    # NIFTY NEXT 50
    'ABB.NS', 'ACC.NS', 'ADANIGREEN.NS', 'ADANIPOWER.NS', 'AMBUJACEM.NS',
    'ATGL.NS', 'AUROPHARMA.NS', 'BANKBARODA.NS', 'BEL.NS', 'BERGEPAINT.NS',
    'BOSCHLTD.NS', 'CANBK.NS', 'CHOLAFIN.NS', 'COLPAL.NS', 'DLF.NS',
    'GAIL.NS', 'GODREJCP.NS', 'HAL.NS', 'HAVELLS.NS', 'ICICIPRULI.NS',
    'IDEA.NS', 'IGL.NS', 'INDHOTEL.NS', 'INDIGO.NS', 'IOC.NS',
    'IRCTC.NS', 'JINDALSTEL.NS', 'JSWENERGY.NS', 'LICI.NS', 'LUPIN.NS',
    'MARICO.NS', 'MAXHEALTH.NS', 'MPHASIS.NS', 'NAUKRI.NS', 'NHPC.NS',
    'OBEROIRLTY.NS', 'OFSS.NS', 'PAGEIND.NS', 'PFC.NS', 'PIDILITIND.NS',
    'PNB.NS', 'POLYCAB.NS', 'RECLTD.NS', 'SRF.NS', 'TATAPOWER.NS',
    'TORNTPHARM.NS', 'TRENT.NS', 'UNIONBANK.NS', 'VBL.NS', 'ZOMATO.NS'
]

# Keep backward compatibility
NIFTY_50 = NIFTY_100


def analyze_weekly_trend(hist: pd.DataFrame) -> Dict:
    """
    Screen 1: Weekly Trend Analysis

    SCORING:
    1. Weekly MACD-H Rising:
       - Rising from below 0 = "Spring" = +2
       - Rising above 0 = "Summer" = +1
       - Else = 0

    2. MACD Line vs Signal:
       - MACD Line < Signal AND both < 0 = +2
       - MACD Line < Signal = +1
       - Else = 0

    3. EMA Alignment (20 > 50 > 100 > 200):
       - Perfect alignment (20 > 50 > 100 > 200) = +2
       - Partial (20 < 50 but 50 > 100 > 200) = +1
       - Else = 0
    """
    weekly = hist.resample('W').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min',
        'Close': 'last', 'Volume': 'sum'
    }).dropna()

    if len(weekly) < 20:  # Need at least 20 weeks of weekly data for basic analysis
        return {
            'weekly_trend': 'INSUFFICIENT_DATA',
            'weekly_bullish': False,
            'screen1_score': 0,
            'macd_h_score': 0,
            'macd_line_score': 0,
            'ema_alignment_score': 0,
            'screen1_reason': 'Insufficient weekly data'
        }

    closes = weekly['Close']

    # Calculate MACD on weekly closes
    macd = calculate_macd(closes)
    current_macd_h = macd['histogram'].iloc[-1]
    prev_macd_h = macd['histogram'].iloc[-2] if len(
        macd['histogram']) > 1 else current_macd_h
    current_macd_line = macd['macd_line'].iloc[-1]
    current_signal = macd['signal_line'].iloc[-1]

    macd_h_rising = current_macd_h > prev_macd_h

    # 1. MACD-H Rising Score
    macd_h_score = 0
    macd_h_status = "Not rising"
    if macd_h_rising:
        if prev_macd_h < 0:  # Rising from below zero = Spring
            macd_h_score = 2
            macd_h_status = "Spring (rising from below 0)"
        else:  # Rising above zero = Summer
            macd_h_score = 1
            macd_h_status = "Summer (rising above 0)"

    # 2. MACD Line vs Signal Score
    macd_line_score = 0
    macd_line_status = "MACD above Signal"
    if current_macd_line < current_signal:
        if current_macd_line < 0 and current_signal < 0:  # Both below zero
            macd_line_score = 2
            macd_line_status = "MACD < Signal, both below 0"
        else:
            macd_line_score = 1
            macd_line_status = "MACD < Signal"

    # 3. EMA Alignment Score (20 > 50 > 100)
    # NEW RULE v2.3: ONLY check EMA 20 > EMA 50 > EMA 100. Ignore EMA 200 completely
    # Use min(len(closes), period) to handle cases with less historical data
    data_len = len(closes)

    ema_20_period = min(data_len, 20) if data_len >= 5 else data_len
    ema_50_period = min(data_len, 50) if data_len >= 10 else data_len
    ema_100_period = min(data_len, 100) if data_len >= 20 else data_len
    ema_200_period = min(data_len, 200) if data_len >= 40 else data_len

    ema_20 = calculate_ema(closes, ema_20_period).iloc[-1]
    ema_50 = calculate_ema(closes, ema_50_period).iloc[-1]
    ema_100 = calculate_ema(closes, ema_100_period).iloc[-1]
    ema_200 = calculate_ema(closes, ema_200_period).iloc[-1]

    ema_alignment_score = 0
    ema_status = "No alignment"

    # NEW RULE v2.3: If EMA 20 > EMA 50 AND EMA 50 > EMA 100, then +2 points
    # For all other scenarios (including when only 50>100), score is 0
    if ema_20 > ema_50 and ema_50 > ema_100:
        ema_alignment_score = 2
        ema_status = "Perfect: 20 > 50 > 100 (ignore 200)"
    else:
        # All other scenarios = 0 points (no partial credit)
        ema_alignment_score = 0
        ema_status = "No alignment"

    # Total Screen 1 Score
    screen1_score = macd_h_score + macd_line_score + ema_alignment_score

    # Weekly trend determination
    if screen1_score >= 5:
        weekly_trend = 'STRONG_BULLISH'
    elif screen1_score >= 3:
        weekly_trend = 'BULLISH'
    elif screen1_score >= 1:
        weekly_trend = 'WEAK_BULLISH'
    else:
        weekly_trend = 'NEUTRAL'

    # Build reason
    screen1_reason = f"MACD-H: {macd_h_status} (+{macd_h_score}), MACD Line: {macd_line_status} (+{macd_line_score}), EMA: {ema_status} (+{ema_alignment_score})"

    return {
        'weekly_trend': weekly_trend,
        'weekly_bullish': screen1_score >= 1,  # Any score = can proceed
        'screen1_score': int(screen1_score),

        # Component scores
        'macd_h_score': int(macd_h_score),
        'macd_h_status': macd_h_status,
        'macd_h_rising': bool(macd_h_rising),
        'macd_h_value': round(float(current_macd_h), 4),

        'macd_line_score': int(macd_line_score),
        'macd_line_status': macd_line_status,
        'macd_line': round(float(current_macd_line), 4),
        'macd_signal': round(float(current_signal), 4),

        'ema_alignment_score': int(ema_alignment_score),
        'ema_status': ema_status,
        'ema_20': round(float(ema_20), 2),
        'ema_50': round(float(ema_50), 2),
        'ema_100': round(float(ema_100), 2),
        'ema_200': round(float(ema_200), 2),

        'screen1_reason': screen1_reason
    }


def detect_kangaroo_tail(hist: pd.DataFrame, lookback: int = 5) -> Dict:
    """
    Detect Kangaroo Tail (Long Lower Shadow) - Bullish reversal pattern

    Criteria:
    - Lower shadow at least 2x the body size
    - Small or no upper shadow
    - Appears after a decline
    """
    if len(hist) < lookback:
        return {'detected': False, 'strength': 0}

    recent = hist.tail(lookback)
    last = recent.iloc[-1]

    body = abs(last['Close'] - last['Open'])
    lower_shadow = min(last['Open'], last['Close']) - last['Low']
    upper_shadow = last['High'] - max(last['Open'], last['Close'])
    total_range = last['High'] - last['Low']

    if total_range == 0:
        return {'detected': False, 'strength': 0}

    # Kangaroo tail: lower shadow > 2x body, upper shadow < body
    is_kangaroo = (
        lower_shadow > body * 2 and
        upper_shadow < body and
        lower_shadow / total_range > 0.6
    )

    # Check if it's after a decline
    price_change = (last['Close'] - recent.iloc[0]
                    ['Close']) / recent.iloc[0]['Close']
    after_decline = price_change < -0.02

    strength = 0
    if is_kangaroo:
        strength = 2 if after_decline else 1

    return {
        'detected': is_kangaroo,
        'strength': strength,
        'lower_shadow_ratio': round(lower_shadow / total_range, 2) if total_range > 0 else 0,
        'after_decline': after_decline
    }


def detect_false_breakout(hist: pd.DataFrame, lookback: int = 20) -> Dict:
    """
    Detect False Downside Breakout - Strong bullish signal

    Criteria:
    - Price breaks below support (recent low)
    - Quickly recovers with Bullish Pinbar (Hammer) or Bullish Engulfing candle
    - Closes above the breakdown level

    False Downside Breakout = Bullish Pinbar OR Bullish Engulfing at support
    """
    if len(hist) < lookback:
        return {'detected': False, 'strength': 0, 'pattern': None}

    recent = hist.tail(lookback)
    last_5 = hist.tail(5)

    # Find support level (lowest low in lookback excluding last 3 days)
    support_data = recent.head(lookback - 3)
    support_level = support_data['Low'].min()

    # Check for false breakout in last 3 days
    last_3 = hist.tail(3)
    broke_support = last_3['Low'].min() < support_level
    recovered = last_3.iloc[-1]['Close'] > support_level

    if not (broke_support and recovered):
        return {'detected': False, 'strength': 0, 'pattern': None}

    # Check for Bullish Pinbar (Hammer) in recovery candle
    last = last_3.iloc[-1]
    body = abs(last['Close'] - last['Open'])
    lower_shadow = min(last['Open'], last['Close']) - last['Low']
    upper_shadow = last['High'] - max(last['Open'], last['Close'])
    total_range = last['High'] - last['Low']

    is_bullish_pinbar = (
        total_range > 0 and
        lower_shadow > body * 2 and  # Long lower shadow
        upper_shadow < body * 0.5 and  # Small upper shadow
        last['Close'] >= last['Open']  # Bullish close
    )

    # Check for Bullish Engulfing in last 2 candles
    if len(last_3) >= 2:
        prev = last_3.iloc[-2]
        is_bullish_engulfing = (
            prev['Close'] < prev['Open'] and  # Previous candle bearish
            last['Close'] > last['Open'] and  # Current candle bullish
            last['Open'] <= prev['Close'] and  # Opens at or below prev close
            last['Close'] >= prev['Open']  # Closes at or above prev open
        )
    else:
        is_bullish_engulfing = False

    detected = is_bullish_pinbar or is_bullish_engulfing

    pattern = None
    if is_bullish_pinbar:
        pattern = 'Bullish Pinbar (Hammer)'
    elif is_bullish_engulfing:
        pattern = 'Bullish Engulfing'

    strength = 0
    if detected:
        strength = 2

    return {
        'detected': detected,
        'strength': strength,
        'pattern': pattern,
        'support_level': round(float(support_level), 2),
        'breakdown_low': round(float(last_3['Low'].min()), 2),
        'is_bullish_pinbar': is_bullish_pinbar,
        'is_bullish_engulfing': is_bullish_engulfing
    }


def detect_force_index_spike(indicators: Dict, hist: pd.DataFrame) -> Dict:
    """
    Detect Force Index Down Spike - Selling climax, potential reversal

    Criteria:
    - Force Index makes extreme negative reading
    - Significantly below recent average
    """
    force_index = indicators.get('force_index_2', 0)

    # Calculate recent Force Index average and std dev
    closes = hist['Close']
    volumes = hist['Volume']

    fi_raw = (closes - closes.shift(1)) * volumes
    fi_ema = fi_raw.ewm(span=2, adjust=False).mean()

    fi_mean = fi_ema.tail(20).mean()
    fi_std = fi_ema.tail(20).std()

    # Spike = current FI is more than 2 std devs below mean
    is_spike = force_index < (fi_mean - 2 * fi_std) and force_index < 0

    return {
        'detected': is_spike,
        'strength': 2 if is_spike else 0,
        'current_fi': round(float(force_index), 0),
        'fi_mean': round(float(fi_mean), 0),
        'fi_threshold': round(float(fi_mean - 2 * fi_std), 0)
    }


def detect_force_index_divergence(hist: pd.DataFrame, lookback: int = 10) -> Dict:
    """
    Detect bullish divergence of 2-day EMA of Force Index

    Bullish divergence: Price makes lower low, Force Index 2-EMA makes higher low
    This is a very important signal per Elder
    """
    if len(hist) < lookback + 5:
        return {'detected': False, 'strength': 0}

    recent = hist.tail(lookback)
    closes = recent['Close']
    lows = recent['Low']
    volumes = recent['Volume']

    # Calculate Force Index 2-EMA
    price_change = closes.diff()
    force_index = price_change * volumes
    fi_2ema = force_index.ewm(span=2, adjust=False).mean()

    # Find two lowest price points
    low_indices = lows.nsmallest(2).index.tolist()
    if len(low_indices) < 2:
        return {'detected': False, 'strength': 0}

    first_low_idx = min(low_indices)
    second_low_idx = max(low_indices)

    # Ensure indices are in fi_2ema
    if first_low_idx not in fi_2ema.index or second_low_idx not in fi_2ema.index:
        return {'detected': False, 'strength': 0}

    # Price makes lower low (or equal within 1%)
    price_lower_low = lows[second_low_idx] <= lows[first_low_idx] * 1.01

    # Force Index makes higher low (shallower bottom)
    fi_at_first = fi_2ema[first_low_idx]
    fi_at_second = fi_2ema[second_low_idx]
    fi_higher_low = fi_at_second > fi_at_first

    if price_lower_low and fi_higher_low:
        return {
            'detected': True,
            'strength': 2,
            'price_low_1': round(float(lows[first_low_idx]), 2),
            'price_low_2': round(float(lows[second_low_idx]), 2),
            'fi_low_1': round(float(fi_at_first), 0),
            'fi_low_2': round(float(fi_at_second), 0)
        }

    return {'detected': False, 'strength': 0}


def calculate_ema_penetration_history(hist: pd.DataFrame, ema_period: int = 22, lookback: int = 60) -> Dict:
    """
    Calculate historical EMA penetrations to determine optimal stop level

    Elder's Method: Put stop at or below the deepest penetration level
    """
    if len(hist) < lookback:
        lookback = len(hist)

    closes = hist['Close'].tail(lookback)
    lows = hist['Low'].tail(lookback)
    ema = calculate_ema(hist['Close'], ema_period).tail(lookback)

    # Find penetrations (lows below EMA)
    penetrations = []
    for i in range(len(lows)):
        if lows.iloc[i] < ema.iloc[i]:
            penetration_pct = (ema.iloc[i] - lows.iloc[i]) / ema.iloc[i] * 100
            penetrations.append(penetration_pct)

    if not penetrations:
        # No penetrations - use ATR-based stop
        atr = calculate_atr(hist['High'], hist['Low'], hist['Close']).iloc[-1]
        return {
            'deepest_penetration_pct': 0,
            'avg_penetration_pct': 0,
            'penetration_count': 0,
            'recommended_stop_pct': round(float(atr / hist['Close'].iloc[-1] * 100 * 2), 2)
        }

    deepest = max(penetrations)
    avg_penetration = sum(penetrations) / len(penetrations)

    # Recommended stop: slightly below deepest penetration
    recommended_stop_pct = deepest * 1.1  # Add 10% buffer

    return {
        'deepest_penetration_pct': round(deepest, 2),
        'avg_penetration_pct': round(avg_penetration, 2),
        'penetration_count': len(penetrations),
        'recommended_stop_pct': round(recommended_stop_pct, 2)
    }


def calculate_elder_trade_levels(hist: pd.DataFrame, indicators: Dict) -> Dict:
    """
    Calculate Entry/Stop/Target using Elder's methodology:

    - ENTRY: Daily EMA-22 (buy at value)
    - STOP: Below deepest historical EMA-22 penetration
    - TARGETS: KC Upper + ATR multiples
        * A-Target: KC Upper + 3×ATR
        * B-Target: KC Upper + 2×ATR  
        * C-Target: KC Upper + 1×ATR
    """
    ema_22 = indicators['ema_22']
    kc_upper = indicators.get('kc_upper', ema_22 * 1.03)
    kc_lower = indicators.get('kc_lower', ema_22 * 0.97)
    atr = indicators.get('atr', ema_22 * 0.02)  # Default ATR estimate
    current_price = indicators['price']

    # Calculate EMA penetration history for stop
    penetration = calculate_ema_penetration_history(hist)
    stop_pct = penetration['recommended_stop_pct']

    # Entry at EMA-22 (or current price if below EMA)
    entry = round(min(ema_22, current_price * 1.001), 2)

    # Stop below EMA using penetration history
    stop_loss = round(ema_22 * (1 - stop_pct / 100), 2)

    # Three targets based on KC Upper + ATR multiples
    target_a = round(float(kc_upper + 3 * atr), 2)  # A-Target: +3 ATR
    target_b = round(float(kc_upper + 2 * atr), 2)  # B-Target: +2 ATR
    target_c = round(float(kc_upper + 1 * atr), 2)  # C-Target: +1 ATR

    # Primary target is A
    target = target_a

    # Calculate risk/reward
    risk = entry - stop_loss
    reward_a = target_a - entry
    reward_b = target_b - entry
    reward_c = target_c - entry
    rr_ratio_a = reward_a / risk if risk > 0 else 0
    rr_ratio_b = reward_b / risk if risk > 0 else 0
    rr_ratio_c = reward_c / risk if risk > 0 else 0

    # Position sizing based on risk
    risk_pct = (risk / entry) * 100 if entry > 0 else 0

    return {
        'entry': entry,
        'entry_method': 'EMA-22 (Value Zone)',
        'stop_loss': stop_loss,
        'stop_method': f'Deepest EMA penetration ({stop_pct:.1f}%)',

        # Primary target (A)
        'target': target,
        'target_method': 'KC Upper + 3×ATR (A-Target)',

        # All three targets for reference
        'target_a': target_a,
        'target_b': target_b,
        'target_c': target_c,
        'target_a_method': 'KC Upper + 3×ATR',
        'target_b_method': 'KC Upper + 2×ATR',
        'target_c_method': 'KC Upper + 1×ATR',

        'kc_upper': round(float(kc_upper), 2),
        'kc_lower': round(float(kc_lower), 2),
        'atr': round(float(atr), 2),

        'risk_per_share': round(risk, 2),
        'reward_per_share': round(reward_a, 2),
        'risk_percent': round(risk_pct, 2),
        'reward_percent': round((reward_a / entry) * 100, 2) if entry > 0 else 0,
        'risk_reward_ratio': round(rr_ratio_a, 2),
        'rr_display': f'1:{rr_ratio_a:.2f}' if rr_ratio_a > 0 else '1:0',

        # Additional metrics for all targets
        'rr_ratio_b': round(rr_ratio_b, 2),
        'rr_ratio_c': round(rr_ratio_c, 2),
        'penetration_data': penetration
    }


def calculate_signal_strength_v2(indicators: Dict, weekly: Dict, hist: pd.DataFrame, patterns: list = None) -> Dict:
    """
    Calculate signal strength score based on REVISED Elder criteria (v2.3)

    SCREEN 1 (Weekly) - Max 6 points:
    1. MACD-H Rising:
       - Rising from below 0 (Spring) = +2
       - Rising above 0 (Summer) = +1
       - Else = 0

    2. MACD Line vs Signal:
       - MACD Line < Signal AND both < 0 = +2
       - MACD Line < Signal = +1
       - Else = 0

    3. EMA Alignment:
       - 20 > 50 > 100 > 200 = +2
       - 50 > 100 > 200 (20 < 50) = +1
       - Else = 0

    SCREEN 2 (Daily) - Max 5 points:
    1. Price vs Keltner Channel:
       - Between Lower(-1) and Lower(-3) = +2
       - Between Mid and Lower(-1) = +1
       - Else = 0

    2. Force Index EMA(2) < 0 = +1

    3. Stochastic < 50 = +1

    4. Bullish Pattern (Pinbar/Engulfing/False Breakout) = +1

    GRADES:
    ⭐ A: Score ≥ 7 → TRADE
    📊 B: Score 5-6 → PREPARE
    👀 C: Score 1-4 → WATCH
    🔴 AVOID: Score = 0
    """
    if patterns is None:
        patterns = []

    score = 0
    signals = []
    breakdown = []
    high_value_signals = []

    # ═══════════════════════════════════════════════════════════════
    # SCREEN 1: WEEKLY SCORING (from weekly dict)
    # ═══════════════════════════════════════════════════════════════
    screen1_score = weekly.get('screen1_score', 0)

    # Add Screen 1 component scores
    macd_h_score = weekly.get('macd_h_score', 0)
    if macd_h_score > 0:
        score += macd_h_score
        status = weekly.get('macd_h_status', 'Rising')
        breakdown.append(f'+{macd_h_score}: Weekly MACD-H {status}')
        if macd_h_score == 2:
            signals.append('⭐⭐ MACD-H Spring (rising from below 0)')
            high_value_signals.append('MACD_SPRING')
        else:
            signals.append('✅ MACD-H Summer (rising above 0)')

    macd_line_score = weekly.get('macd_line_score', 0)
    if macd_line_score > 0:
        score += macd_line_score
        status = weekly.get('macd_line_status', 'Below signal')
        breakdown.append(f'+{macd_line_score}: {status}')
        if macd_line_score == 2:
            signals.append('⭐⭐ MACD Line below Signal, both below 0')
            high_value_signals.append('MACD_DEEP_OVERSOLD')
        else:
            signals.append('✅ MACD Line below Signal')

    ema_score = weekly.get('ema_alignment_score', 0)
    if ema_score > 0:
        score += ema_score
        status = weekly.get('ema_status', 'Aligned')
        breakdown.append(f'+{ema_score}: EMA {status}')
        if ema_score == 2:
            signals.append('⭐⭐ Perfect EMA alignment (20>50>100>200)')
            high_value_signals.append('EMA_PERFECT_ALIGNMENT')
        else:
            signals.append('✅ Partial EMA alignment')

    # ═══════════════════════════════════════════════════════════════
    # SCREEN 2: DAILY SCORING
    # ═══════════════════════════════════════════════════════════════

    # 1. Price vs Keltner Channel positioning
    price = indicators.get('price', 0)
    kc_middle = indicators.get('kc_middle', price)
    kc_lower = indicators.get('kc_lower', price * 0.97)
    atr = indicators.get('atr', price * 0.02)

    # Calculate extended Keltner levels
    # Lower(-1) = standard lower = Middle - 1*ATR
    # Lower(-3) = extended lower = Middle - 3*ATR
    kc_lower_1 = kc_middle - atr  # Standard lower
    kc_lower_3 = kc_middle - 3 * atr  # Extended lower

    kc_score = 0
    kc_status = "Above mid-channel"

    if price >= kc_lower_3 and price < kc_lower_1:
        # Between Lower(-3) and Lower(-1) = Deep pullback
        kc_score = 2
        kc_status = f"Deep pullback zone (between KC-3 and KC-1)"
        signals.append('⭐⭐ Price in deep pullback zone')
        high_value_signals.append('KC_DEEP_PULLBACK')
    elif price >= kc_lower_1 and price < kc_middle:
        # Between Lower(-1) and Mid = Normal pullback
        kc_score = 1
        kc_status = f"Normal pullback zone (between KC-1 and Mid)"
        signals.append('✅ Price in pullback zone')

    if kc_score > 0:
        score += kc_score
        breakdown.append(f'+{kc_score}: Keltner Channel - {kc_status}')

    # 2. Force Index EMA(2) < 0
    force_index = indicators.get('force_index_2', 0)
    if force_index < 0:
        score += 1
        breakdown.append(f'+1: Force Index EMA(2) < 0 ({force_index:.0f})')
        signals.append('✅ Force Index negative')
        high_value_signals.append('FORCE_INDEX_NEGATIVE')

    # 3. Stochastic < 50
    stochastic = indicators.get('stochastic_k', 50)
    if stochastic < 50:
        score += 1
        breakdown.append(f'+1: Stochastic < 50 ({stochastic:.1f})')
        signals.append('✅ Stochastic in buy zone')
        high_value_signals.append('STOCHASTIC_LOW')

    # 4. Bullish Patterns (Finger to bottom / False breakout / Pinbar / Engulfing)
    # Check for false breakout
    false_breakout = detect_false_breakout(hist)
    has_pattern = false_breakout['detected']
    pattern_name = false_breakout.get('pattern', '')

    # Check candlestick patterns
    bullish_pattern_names = []
    target_patterns = ['hammer', 'bullish_pinbar',
                       'bullish_engulfing', 'piercing_line', 'morning_star']

    for p in patterns:
        pid = p.get('id', '').lower()
        if any(tp in pid for tp in target_patterns):
            has_pattern = True
            bullish_pattern_names.append(p.get('name', pid))

    if has_pattern:
        score += 1
        if pattern_name:
            breakdown.append(f'+1: {pattern_name}')
            signals.append(f'✅ {pattern_name}')
        elif bullish_pattern_names:
            breakdown.append(f'+1: {", ".join(bullish_pattern_names[:2])}')
            signals.append(
                f'✅ Pattern: {", ".join(bullish_pattern_names[:2])}')
        else:
            breakdown.append('+1: Bullish reversal pattern')
            signals.append('✅ Bullish pattern detected')
        high_value_signals.append('BULLISH_PATTERN')

    # ═══════════════════════════════════════════════════════════════
    # GRADE DETERMINATION (NEW v2.3 RULES)
    # IMPORTANT: A-Trades require ALL 3 weekly filters to have score > 0
    # ═══════════════════════════════════════════════════════════════

    # Check if all 3 weekly filters have positive scores
    macd_h_score = weekly.get('macd_h_score', 0)
    macd_line_score = weekly.get('macd_line_score', 0)
    ema_alignment_score = weekly.get('ema_alignment_score', 0)
    all_weekly_filters_pass = (
        macd_h_score > 0 and macd_line_score > 0 and ema_alignment_score > 0)

    # Grade determination with Screen 1 filter requirement
    if all_weekly_filters_pass and score >= 7:
        # A-Trade: ALL weekly filters active + total score >= 7
        grade = 'A'
        action = '⭐⭐ A-TRADE: All weekly filters active + strong setup - PLACE ORDER'
    elif all_weekly_filters_pass and score >= 5:
        # B-Trade: ALL weekly filters active + total score 5-6
        grade = 'B'
        action = '📊 B-TRADE: All weekly filters active + good setup - Prepare order'
    elif score >= 7:
        # Would be A-Trade by score but missing weekly filter requirement → downgrade to B
        grade = 'B'
        action = '📊 B-TRADE: Strong score but missing weekly filter - Monitor'
    elif score >= 5:
        # B-Trade by score alone (Screen 1 filters not all active)
        grade = 'B'
        action = '📊 PREPARE: Good setup developing - Set alerts, prepare trade plan'
    elif score >= 1:
        grade = 'C'
        action = '👀 WATCH: Early stage - Monitor for improving conditions'
    else:
        grade = 'AVOID'
        action = '🔴 AVOID: No bullish signals detected'

    # is_a_trade: Only Grade A qualifies
    is_a_trade = grade == 'A'

    # Calculate screen breakdown
    screen2_score = score - screen1_score

    return {
        'signal_strength': score,
        'screen1_score': screen1_score,
        'screen2_score': screen2_score,
        'grade': grade,
        'action': action,
        'is_a_trade': is_a_trade,
        'all_weekly_filters_active': all_weekly_filters_pass,
        'breakdown': breakdown,
        'signals': signals,
        'high_value_signals': high_value_signals,
        # Keltner levels for display
        'kc_lower_1': round(float(kc_lower_1), 2),
        'kc_lower_3': round(float(kc_lower_3), 2),
        'kc_position': kc_status
    }


def scan_stock_v2(symbol: str, config: Dict = None) -> Optional[Dict]:
    """
    Complete stock analysis v2 with Elder methodology corrections
    """
    if config is None:
        config = DEFAULT_INDICATOR_CONFIG

    # Fetch data
    data = fetch_stock_data(symbol)
    if not data:
        return None

    hist = data['history']

    # Screen 1: Weekly analysis (MANDATORY GATE)
    weekly = analyze_weekly_trend(hist)

    # Screen 2: Daily indicators
    indicators = calculate_all_indicators(
        hist['High'], hist['Low'], hist['Close'], hist['Volume']
    )

    # Candlestick patterns
    patterns = scan_patterns(hist)
    bullish_patterns = get_bullish_patterns(patterns)
    pattern_score = get_pattern_score(patterns)

    # Calculate signal strength with V2 logic
    scoring = calculate_signal_strength_v2(indicators, weekly, hist, patterns)

    # Calculate Elder trade levels (Entry at EMA-22, Target at KC Upper, Stop at deepest penetration)
    levels = calculate_elder_trade_levels(hist, indicators)

    # Price change
    current_price = hist['Close'].iloc[-1]
    prev_price = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
    change = current_price - prev_price
    change_pct = (change / prev_price) * 100

    result = {
        'symbol': symbol,
        'name': data['name'],
        'sector': data['sector'],
        'price': round(float(current_price), 2),
        'change': round(float(change), 2),
        'change_percent': round(float(change_pct), 2),

        # Screen 1 - Weekly Scoring
        'weekly_trend': weekly['weekly_trend'],
        'weekly_bullish': weekly['weekly_bullish'],
        'screen1_score': weekly.get('screen1_score', 0),
        'screen1_reason': weekly.get('screen1_reason', ''),

        # Screen 1 Components
        'macd_h_score': weekly.get('macd_h_score', 0),
        'macd_h_status': weekly.get('macd_h_status', ''),
        'macd_h_rising': weekly.get('macd_h_rising', False),
        'macd_h_value': weekly.get('macd_h_value', 0),

        'macd_line_score': weekly.get('macd_line_score', 0),
        'macd_line_status': weekly.get('macd_line_status', ''),
        'macd_line': weekly.get('macd_line', 0),
        'macd_signal': weekly.get('macd_signal', 0),

        'ema_alignment_score': weekly.get('ema_alignment_score', 0),
        'ema_status': weekly.get('ema_status', ''),
        'weekly_ema_20': weekly.get('ema_20', 0),
        'weekly_ema_50': weekly.get('ema_50', 0),
        'weekly_ema_100': weekly.get('ema_100', 0),
        'weekly_ema_200': weekly.get('ema_200', 0),

        # Screen 2 - Daily Indicators
        'ema_13': round(float(indicators['ema_13']), 2),
        'ema_22': round(float(indicators['ema_22']), 2),
        'macd_histogram': round(float(indicators['macd_histogram']), 4),
        'macd_rising': indicators['macd_rising'],
        'force_index': round(float(indicators['force_index_2']), 0),
        'stochastic': round(float(indicators['stochastic_k']), 1),
        'rsi': round(float(indicators['rsi']), 1),
        'atr': round(float(indicators['atr']), 2),
        'impulse_color': indicators['impulse_color'],
        'price_vs_ema': round(float(indicators['price_vs_ema']), 1),
        'channel_width': round(float(indicators['channel_width']), 1),

        # Keltner Channel
        'kc_upper': levels['kc_upper'],
        'kc_lower': levels['kc_lower'],
        'kc_middle': round(float(indicators.get('kc_middle', current_price)), 2),
        'kc_lower_1': scoring.get('kc_lower_1', 0),
        'kc_lower_3': scoring.get('kc_lower_3', 0),
        'kc_position': scoring.get('kc_position', ''),

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
        'screen1_total': scoring.get('screen1_score', 0),
        'screen2_total': scoring.get('screen2_score', 0),
        'grade': scoring['grade'],
        'action': scoring['action'],
        'is_a_trade': scoring['is_a_trade'],
        'score_breakdown': scoring['breakdown'],
        'signals': scoring['signals'],
        'high_value_signals': scoring.get('high_value_signals', []),

        # Elder Trade Levels (NEW)
        'entry': levels['entry'],
        'entry_method': levels['entry_method'],
        'stop_loss': levels['stop_loss'],
        'stop_method': levels['stop_method'],

        # Three targets based on ATR multiples
        'target': levels['target'],
        'target_method': levels['target_method'],
        'target_a': levels['target_a'],
        'target_b': levels['target_b'],
        'target_c': levels['target_c'],
        'target_a_method': levels['target_a_method'],
        'target_b_method': levels['target_b_method'],
        'target_c_method': levels['target_c_method'],

        'risk_percent': levels['risk_percent'],
        'reward_percent': levels['reward_percent'],
        'risk_reward_ratio': levels['risk_reward_ratio'],
        'rr_display': levels['rr_display'],
        'rr_ratio_b': levels['rr_ratio_b'],
        'rr_ratio_c': levels['rr_ratio_c'],
        'penetration_data': levels['penetration_data'],

        # Config
        'indicator_config': config.get('name', 'Custom'),
        'screener_version': '2.3'
    }

    # Save indicators to cache for next incremental calculation
    try:
        weekly_hist = hist.resample('W-FRI').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        save_indicators_to_cache(symbol, hist, indicators, weekly_hist)
    except Exception as e:
        print(f"⚠️ {symbol}: Warning - could not cache indicators: {e}")

    return convert_to_native(result)


def run_weekly_screen_v2(market: str = 'US', symbols: List[str] = None) -> Dict:
    """Run weekly screener v2 with corrected logic - Full 100 stocks"""
    if symbols is None:
        symbols = NASDAQ_100 if market == 'US' else NIFTY_100

    results = []
    passed = []
    failed_reasons = {}  # Track why stocks failed Screen 1

    for symbol in symbols:
        analysis = scan_stock_v2(symbol)
        if analysis:
            results.append(analysis)
            if analysis['weekly_bullish']:
                passed.append(analysis)
            else:
                # Track failure reason
                failed_reasons[symbol] = analysis.get(
                    'screen1_reason', 'Unknown')
        else:
            # Create placeholder entry for stocks with no data
            results.append({
                'symbol': symbol,
                'name': 'Data Unavailable',
                'price': 0,
                'change': 0,
                'change_percent': 0,
                'grade': 'AVOID',
                'signal_strength': 0,
                'is_a_trade': False,
                'weekly_bullish': False,
                'screen1_reason': 'No data available',
                'impulse_color': 'GRAY'
            })
            failed_reasons[symbol] = 'No data available'

    # Sort by signal strength
    results.sort(key=lambda x: x['signal_strength'], reverse=True)
    passed.sort(key=lambda x: x['signal_strength'], reverse=True)

    # Categorize
    a_trades = [r for r in results if r['is_a_trade']]
    b_trades = [r for r in results if r['grade'] == 'B']
    watch = [r for r in results if r['grade'] == 'C']
    avoid = [r for r in results if r['grade'] == 'AVOID']

    # Score distribution tracking
    score_distribution = {
        'screen1_scores': {
            'macd_h_spring': len([r for r in results if r.get('macd_h_score', 0) == 2]),
            'macd_h_summer': len([r for r in results if r.get('macd_h_score', 0) == 1]),
            'macd_line_deep': len([r for r in results if r.get('macd_line_score', 0) == 2]),
            'macd_line_below': len([r for r in results if r.get('macd_line_score', 0) == 1]),
            'ema_perfect': len([r for r in results if r.get('ema_alignment_score', 0) == 2]),
            'ema_partial': len([r for r in results if r.get('ema_alignment_score', 0) == 1])
        },
        'screen2_signals': {
            'kc_deep_pullback': len([r for r in results if 'KC_DEEP_PULLBACK' in r.get('high_value_signals', [])]),
            'force_index_neg': len([r for r in results if 'FORCE_INDEX_NEGATIVE' in r.get('high_value_signals', [])]),
            'stochastic_low': len([r for r in results if 'STOCHASTIC_LOW' in r.get('high_value_signals', [])]),
            'bullish_pattern': len([r for r in results if 'BULLISH_PATTERN' in r.get('high_value_signals', [])])
        }
    }

    return convert_to_native({
        'scan_date': datetime.now().isoformat(),
        'market': market,
        'total_scanned': len(symbols),
        'total_analyzed': len(results),
        'weekly_bullish_count': len([r for r in results if r.get('screen1_score', 0) >= 3]),
        'screener_version': '2.3',

        'summary': {
            'a_trades': len(a_trades),
            'b_trades': len(b_trades),
            'watch_list': len(watch),
            'avoid': len(avoid)
        },

        'score_distribution': score_distribution,

        'a_trades': a_trades,
        'b_trades': b_trades,
        'watch_list': watch,
        'avoid': avoid,
        'all_results': results,

        'grading_criteria': get_grading_criteria()
    })


def run_daily_screen_v2(weekly_results: List[Dict]) -> Dict:
    """
    Run daily screen v2.3

    Simply re-analyzes stocks and uses the unified scoring system.
    No mandatory gates - just pure scoring based on:

    SCREEN 1 (Weekly): 0-6 points
    SCREEN 2 (Daily): 0-5 points
    Total: 0-11 points

    Grades: A (≥7), B (5-6), C (1-4), AVOID (0)
    """
    if not weekly_results:
        return {
            'error': 'No weekly results provided',
            'message': 'Run weekly screen first'
        }

    symbols = [r['symbol'] for r in weekly_results]

    results = []
    for symbol in symbols:
        analysis = scan_stock_v2(symbol)
        if analysis:
            # Check pullback conditions
            pullback_confirmed = (
                analysis['force_index'] < 0 or
                analysis['stochastic'] < 50
            )

            # Ready = has some score and pullback confirmed
            daily_ready = analysis['signal_strength'] >= 5 and pullback_confirmed

            analysis['daily_ready'] = daily_ready
            analysis['pullback_confirmed'] = pullback_confirmed
            results.append(analysis)

    results.sort(key=lambda x: x['signal_strength'], reverse=True)
    a_trades = [r for r in results if r['is_a_trade']]

    return convert_to_native({
        'scan_date': datetime.now().isoformat(),
        'stocks_from_weekly': len(symbols),
        'daily_ready_count': len([r for r in results if r.get('daily_ready')]),
        'a_trades': a_trades,
        'all_results': results,
        'screener_version': '2.3',
        'scoring_summary': {
            'max_screen1': 6,
            'max_screen2': 5,
            'max_total': 11,
            'grade_a_threshold': 7,
            'grade_b_threshold': 5
        }
    })


def save_indicators_to_cache(symbol: str, hist: pd.DataFrame, indicators: Dict, weekly_hist: pd.DataFrame = None) -> bool:
    """
    Save calculated indicators to database cache for incremental calculation next time

    Args:
        symbol: Stock symbol
        hist: Daily OHLCV history with indicators
        indicators: Calculated indicators dictionary
        weekly_hist: Optional weekly resampled history

    Returns:
        True if saved successfully, False otherwise
    """
    from models.database import get_database
    from datetime import datetime

    try:
        db = get_database().get_connection()

        # Save daily indicators
        if 'ema_22' in indicators and len(hist) > 0:
            print(f"💾 {symbol}: Saving {len(hist)} daily indicators to cache...")

            for date, row in hist.iterrows():
                date_str = date.strftime('%Y-%m-%d')
                close = float(row['Close']) if isinstance(
                    row['Close'], (int, float)) else None

                db.execute('''
                    INSERT OR REPLACE INTO stock_indicators_daily
                    (symbol, date, close, ema_22, ema_50, ema_100, ema_200, 
                     macd_line, macd_signal, macd_histogram, rsi, stochastic, 
                     stoch_d, atr, force_index, kc_upper, kc_middle, kc_lower)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    symbol,
                    date_str,
                    close,
                    float(row.get('EMA_22', 0)) if pd.notna(
                        row.get('EMA_22')) else None,
                    float(row.get('EMA_50', 0)) if pd.notna(
                        row.get('EMA_50')) else None,
                    float(row.get('EMA_100', 0)) if pd.notna(
                        row.get('EMA_100')) else None,
                    float(row.get('EMA_200', 0)) if pd.notna(
                        row.get('EMA_200')) else None,
                    float(row.get('MACD_Line', 0)) if pd.notna(
                        row.get('MACD_Line')) else None,
                    float(row.get('MACD_Signal', 0)) if pd.notna(
                        row.get('MACD_Signal')) else None,
                    float(row.get('MACD_Histogram', 0)) if pd.notna(
                        row.get('MACD_Histogram')) else None,
                    float(row.get('RSI_14', 0)) if pd.notna(
                        row.get('RSI_14')) else None,
                    float(row.get('Stochastic', 0)) if pd.notna(
                        row.get('Stochastic')) else None,
                    float(row.get('Stochastic_D', 0)) if pd.notna(
                        row.get('Stochastic_D')) else None,
                    float(row.get('ATR', 0)) if pd.notna(
                        row.get('ATR')) else None,
                    float(row.get('Force_Index', 0)) if pd.notna(
                        row.get('Force_Index')) else None,
                    float(row.get('KC_Upper', 0)) if pd.notna(
                        row.get('KC_Upper')) else None,
                    float(row.get('KC_Middle', 0)) if pd.notna(
                        row.get('KC_Middle')) else None,
                    float(row.get('KC_Lower', 0)) if pd.notna(
                        row.get('KC_Lower')) else None
                ))

        # Update indicator sync record
        if len(hist) > 0:
            latest_date = hist.index.max().strftime('%Y-%m-%d')
            db.execute('''
                INSERT OR REPLACE INTO stock_indicator_sync
                (symbol, last_updated, last_daily_date, daily_record_count)
                VALUES (?, ?, ?, 
                    (SELECT COUNT(*) FROM stock_indicators_daily WHERE symbol = ?))
            ''', (symbol, datetime.now().isoformat(), latest_date, symbol))

        # Save weekly indicators if provided
        if weekly_hist is not None and len(weekly_hist) > 0:
            print(
                f"💾 {symbol}: Saving {len(weekly_hist)} weekly indicators to cache...")

            for date, row in weekly_hist.iterrows():
                date_str = date.strftime('%Y-%m-%d')
                close = float(row['Close']) if isinstance(
                    row['Close'], (int, float)) else None

                db.execute('''
                    INSERT OR REPLACE INTO stock_indicators_weekly
                    (symbol, week_end_date, close, ema_22, ema_50, ema_100, ema_200,
                     macd_line, macd_signal, macd_histogram)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    symbol,
                    date_str,
                    close,
                    float(row.get('EMA_22', 0)) if pd.notna(
                        row.get('EMA_22')) else None,
                    float(row.get('EMA_50', 0)) if pd.notna(
                        row.get('EMA_50')) else None,
                    float(row.get('EMA_100', 0)) if pd.notna(
                        row.get('EMA_100')) else None,
                    float(row.get('EMA_200', 0)) if pd.notna(
                        row.get('EMA_200')) else None,
                    float(row.get('MACD_Line', 0)) if pd.notna(
                        row.get('MACD_Line')) else None,
                    float(row.get('MACD_Signal', 0)) if pd.notna(
                        row.get('MACD_Signal')) else None,
                    float(row.get('MACD_Histogram', 0)) if pd.notna(
                        row.get('MACD_Histogram')) else None
                ))

        db.commit()
        db.close()
        print(f"✅ {symbol}: Indicators cached successfully")
        return True

    except Exception as e:
        print(f"❌ {symbol}: Error saving indicators to cache: {e}")
        return False
