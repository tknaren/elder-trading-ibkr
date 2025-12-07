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
NASDAQ_100_TOP = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AMD', 'AVGO', 'NFLX',
    'COST', 'PEP', 'ADBE', 'CSCO', 'INTC', 'QCOM', 'TXN', 'INTU', 'AMAT', 'MU',
    'LRCX', 'KLAC', 'SNPS', 'CDNS', 'MRVL', 'ON', 'NXPI', 'ADI', 'MCHP', 'FTNT',
    'VRTX', 'CHTR', 'ASML', 'CRWD', 'SPLK', 'ENPH', 'MNST', 'TEAM', 'PAYX', 'AEP',
    'DXCM', 'CPRT', 'PCAR', 'ALGN', 'AMGN', 'MRNA', 'XEL', 'WDAY', 'ABNB', 'MDLZ'
]

NIFTY_50 = [
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
    'HINDUNILVR.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'ITC.NS', 'KOTAKBANK.NS',
    'LT.NS', 'AXISBANK.NS', 'ASIANPAINT.NS', 'MARUTI.NS', 'TITAN.NS',
    'SUNPHARMA.NS', 'ULTRACEMCO.NS', 'BAJFINANCE.NS', 'WIPRO.NS', 'HCLTECH.NS',
    'TATAMOTORS.NS', 'POWERGRID.NS', 'NTPC.NS', 'M&M.NS', 'JSWSTEEL.NS'
]


def analyze_weekly_trend(hist: pd.DataFrame) -> Dict:
    """
    Screen 1: Analyze weekly trend using Elder's methodology
    THIS IS A MANDATORY GATE - If bearish, NO LONG TRADES
    """
    weekly = hist.resample('W').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 
        'Close': 'last', 'Volume': 'sum'
    }).dropna()

    if len(weekly) < 26:
        return {
            'weekly_ema_slope': 'INSUFFICIENT_DATA',
            'weekly_macd_rising': False,
            'weekly_trend': 'UNKNOWN',
            'weekly_bullish': False,
            'weekly_ema': 0,
            'weekly_ema_prev': 0
        }

    closes = weekly['Close']
    ema_22 = calculate_ema(closes, 22)
    macd = calculate_macd(closes)

    current_ema = ema_22.iloc[-1]
    prev_ema = ema_22.iloc[-2] if len(ema_22) > 1 else current_ema
    ema_slope = current_ema - prev_ema

    current_macd_h = macd['histogram'].iloc[-1]
    prev_macd_h = macd['histogram'].iloc[-2] if len(macd['histogram']) > 1 else current_macd_h
    macd_rising = current_macd_h > prev_macd_h

    # Weekly trend determination
    if ema_slope > 0 and macd_rising:
        weekly_trend = 'STRONG_BULLISH'
    elif ema_slope > 0 or macd_rising:
        weekly_trend = 'BULLISH'
    elif ema_slope < 0 and not macd_rising:
        weekly_trend = 'BEARISH'
    else:
        weekly_trend = 'NEUTRAL'

    return {
        'weekly_ema': round(float(current_ema), 2),
        'weekly_ema_prev': round(float(prev_ema), 2),
        'weekly_ema_slope': 'RISING' if ema_slope > 0 else 'FALLING' if ema_slope < 0 else 'FLAT',
        'weekly_ema_slope_value': round(float(ema_slope), 4),
        'weekly_macd_histogram': round(float(current_macd_h), 4),
        'weekly_macd_histogram_prev': round(float(prev_macd_h), 4),
        'weekly_macd_rising': bool(macd_rising),
        'weekly_trend': weekly_trend,
        'weekly_bullish': weekly_trend in ['STRONG_BULLISH', 'BULLISH']
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
    price_change = (last['Close'] - recent.iloc[0]['Close']) / recent.iloc[0]['Close']
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
    - Quickly recovers and closes above the breakdown level
    - Volume spike on recovery
    """
    if len(hist) < lookback:
        return {'detected': False, 'strength': 0}
    
    recent = hist.tail(lookback)
    last_5 = hist.tail(5)
    
    # Find support level (lowest low in lookback excluding last 3 days)
    support_data = recent.head(lookback - 3)
    support_level = support_data['Low'].min()
    
    # Check for false breakout in last 3 days
    last_3 = hist.tail(3)
    broke_support = last_3['Low'].min() < support_level
    recovered = last_3.iloc[-1]['Close'] > support_level
    
    # Volume confirmation
    avg_volume = recent['Volume'].mean()
    recovery_volume = last_3['Volume'].iloc[-1]
    volume_spike = recovery_volume > avg_volume * 1.3
    
    detected = broke_support and recovered
    strength = 0
    if detected:
        strength = 3 if volume_spike else 2
    
    return {
        'detected': detected,
        'strength': strength,
        'support_level': round(float(support_level), 2),
        'breakdown_low': round(float(last_3['Low'].min()), 2),
        'volume_spike': volume_spike
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
    - TARGET: Keltner Channel Upper Band
    - STOP: Below deepest historical EMA-22 penetration
    """
    ema_22 = indicators['ema_22']
    kc_upper = indicators.get('kc_upper', ema_22 * 1.03)
    kc_lower = indicators.get('kc_lower', ema_22 * 0.97)
    current_price = indicators['price']
    
    # Calculate EMA penetration history for stop
    penetration = calculate_ema_penetration_history(hist)
    stop_pct = penetration['recommended_stop_pct']
    
    # Entry at EMA-22 (or current price if below EMA)
    entry = round(min(ema_22, current_price * 1.001), 2)
    
    # Stop below EMA using penetration history
    stop_loss = round(ema_22 * (1 - stop_pct / 100), 2)
    
    # Target at KC upper
    target = round(float(kc_upper), 2)
    
    # Calculate risk/reward
    risk = entry - stop_loss
    reward = target - entry
    rr_ratio = reward / risk if risk > 0 else 0
    
    # Position sizing based on risk
    risk_pct = (risk / entry) * 100 if entry > 0 else 0
    
    return {
        'entry': entry,
        'entry_method': 'EMA-22 (Value Zone)',
        'stop_loss': stop_loss,
        'stop_method': f'Deepest EMA penetration ({stop_pct:.1f}%)',
        'target': target,
        'target_method': 'KC Upper Band',
        'kc_upper': round(float(kc_upper), 2),
        'kc_lower': round(float(kc_lower), 2),
        'risk_per_share': round(risk, 2),
        'reward_per_share': round(reward, 2),
        'risk_percent': round(risk_pct, 2),
        'reward_percent': round((reward / entry) * 100, 2) if entry > 0 else 0,
        'risk_reward_ratio': round(rr_ratio, 2),
        'rr_display': f'1:{rr_ratio:.2f}' if rr_ratio > 0 else '1:0',
        'penetration_data': penetration
    }


def calculate_signal_strength_v2(indicators: Dict, weekly: Dict, hist: pd.DataFrame, patterns: list = None) -> Dict:
    """
    Calculate signal strength score based on Elder criteria - V2 with FIXES
    
    CRITICAL FIX #1: Screen 1 is a MANDATORY GATE
    - If weekly_bullish is False, return AVOID immediately
    
    CRITICAL FIX #2: Impulse RED blocks trades entirely
    - Not just a penalty, but a disqualifier
    
    NEW HIGH-SCORING RULES:
    +3: Price near lower channel (short-term oversold)
    +3: MACD divergence (strongest signal)
    +3: False downside breakout
    +2: Kangaroo tail pattern
    +2: Force Index down spike
    +3: Pullback to value (Weekly EMA↑, Daily EMA↑, price < fast EMA)
    
    Standard Scoring:
    +2: Weekly EMA rising strongly
    +1: Weekly MACD-H rising
    +2: Force Index < 0 (pullback)
    +2: Stochastic < 30 (oversold) - FIXED from 50
    +1: Stochastic 30-50
    +1: Price at/below 22-EMA
    +1: Impulse GREEN
    +1/+2: Bullish candlestick patterns
    """
    if patterns is None:
        patterns = []

    # ═══════════════════════════════════════════════════════════════
    # CRITICAL FIX #1: Screen 1 MANDATORY GATE
    # ═══════════════════════════════════════════════════════════════
    if not weekly.get('weekly_bullish', False):
        return {
            'signal_strength': 0,
            'grade': 'AVOID',
            'action': '⛔ STAY OUT - Weekly trend not bullish (Screen 1 FAILED)',
            'is_a_trade': False,
            'breakdown': ['❌ Screen 1 FAILED: Weekly trend must be BULLISH for long trades'],
            'signals': ['⛔ Weekly trend is bearish or neutral - NO LONGS ALLOWED'],
            'high_value_signals': []
        }

    # ═══════════════════════════════════════════════════════════════
    # CRITICAL FIX #2: Impulse RED blocks trades
    # ═══════════════════════════════════════════════════════════════
    impulse = indicators.get('impulse_color', 'BLUE')
    if impulse == 'RED':
        return {
            'signal_strength': 0,
            'grade': 'AVOID',
            'action': '⛔ NO BUYING - Impulse RED (Bears in control)',
            'is_a_trade': False,
            'breakdown': ['❌ Impulse RED: Bears in control - DO NOT BUY'],
            'signals': ['⛔ Impulse System forbids buying when RED'],
            'high_value_signals': []
        }

    score = 0
    signals = []
    breakdown = []
    high_value_signals = []  # Track the most valuable setups

    # ═══════════════════════════════════════════════════════════════
    # WEEKLY TREND SCORING (Screen 1 already passed)
    # ═══════════════════════════════════════════════════════════════
    if weekly.get('weekly_ema_slope') == 'RISING':
        if weekly.get('weekly_trend') == 'STRONG_BULLISH':
            score += 2
            breakdown.append('+2: Weekly EMA strongly rising')
            signals.append('✅ Weekly uptrend confirmed')
        else:
            score += 1
            breakdown.append('+1: Weekly EMA rising')

    if weekly.get('weekly_macd_rising'):
        score += 1
        breakdown.append('+1: Weekly MACD-H rising')
        signals.append('✅ Weekly momentum bullish')

    # ═══════════════════════════════════════════════════════════════
    # NEW HIGH-VALUE SIGNALS (+3 points each)
    # ═══════════════════════════════════════════════════════════════
    
    # 1. MACD Divergence - "Strongest signal in technical analysis" - Elder
    if indicators.get('bullish_divergence_macd') or indicators.get('bullish_divergence_rsi'):
        score += 3
        breakdown.append('+3: ⭐ MACD/RSI Bullish Divergence (STRONGEST SIGNAL)')
        signals.append('⭐⭐⭐ MACD Divergence = Most powerful buy signal')
        high_value_signals.append('MACD_DIVERGENCE')

    # 2. Short-term Oversold (near lower channel)
    price = indicators.get('price', 0)
    kc_lower = indicators.get('kc_lower', price * 0.97)
    kc_upper = indicators.get('kc_upper', price * 1.03)
    channel_height = kc_upper - kc_lower
    
    if channel_height > 0:
        position_in_channel = (price - kc_lower) / channel_height
        if position_in_channel < 0.2:  # Near lower band
            score += 3
            breakdown.append(f'+3: ⭐ Price near lower channel ({position_in_channel:.0%})')
            signals.append('⭐⭐⭐ Short-term oversold at support')
            high_value_signals.append('LOWER_CHANNEL')

    # 3. False Downside Breakout
    false_breakout = detect_false_breakout(hist)
    if false_breakout['detected']:
        score += false_breakout['strength']
        breakdown.append(f'+{false_breakout["strength"]}: ⭐ False Downside Breakout detected')
        signals.append('⭐⭐⭐ False breakout = Strong reversal signal')
        high_value_signals.append('FALSE_BREAKOUT')

    # 4. Pullback to Value (Weekly EMA↑, Daily EMA↑, price < fast EMA)
    daily_ema_rising = indicators.get('ema_slope', 0) > 0
    price_below_fast_ema = price < indicators.get('ema_13', price)
    
    if weekly.get('weekly_ema_slope') == 'RISING' and daily_ema_rising and price_below_fast_ema:
        score += 3
        breakdown.append('+3: ⭐ Pullback to Value (Both EMAs rising, price dipped)')
        signals.append('⭐⭐⭐ Perfect pullback setup in uptrend')
        high_value_signals.append('PULLBACK_TO_VALUE')

    # ═══════════════════════════════════════════════════════════════
    # HIGH-VALUE SIGNALS (+2 points each)
    # ═══════════════════════════════════════════════════════════════
    
    # 5. Kangaroo Tail
    kangaroo = detect_kangaroo_tail(hist)
    if kangaroo['detected']:
        score += kangaroo['strength']
        breakdown.append(f'+{kangaroo["strength"]}: Kangaroo Tail (reversal pattern)')
        signals.append('⭐⭐ Kangaroo tail = Bullish reversal')
        high_value_signals.append('KANGAROO_TAIL')

    # 6. Force Index Down Spike
    fi_spike = detect_force_index_spike(indicators, hist)
    if fi_spike['detected']:
        score += fi_spike['strength']
        breakdown.append(f'+{fi_spike["strength"]}: Force Index Spike (selling climax)')
        signals.append('⭐⭐ Force Index spike = Capitulation')
        high_value_signals.append('FI_SPIKE')

    # ═══════════════════════════════════════════════════════════════
    # STANDARD ELDER SCORING
    # ═══════════════════════════════════════════════════════════════
    
    # Force Index scoring
    force_index = indicators.get('force_index_2', 0)
    if force_index < 0:
        score += 2
        breakdown.append(f'+2: Force Index < 0 ({force_index:.0f}) - Pullback zone')
        signals.append('✅ Force Index negative = buying opportunity')

    # Stochastic scoring - FIXED: threshold now 30, not 50
    stochastic = indicators.get('stochastic_k', 50)
    if stochastic < 30:
        score += 2
        breakdown.append(f'+2: Stochastic < 30 ({stochastic:.1f}) - OVERSOLD')
        signals.append('✅ Stochastic oversold = entry zone')
    elif stochastic < 50:
        score += 1
        breakdown.append(f'+1: Stochastic 30-50 ({stochastic:.1f})')

    # Price vs EMA scoring
    price_vs_ema = indicators.get('price_vs_ema', 0)
    if price_vs_ema <= 0:
        score += 1
        breakdown.append(f'+1: Price at/below EMA ({price_vs_ema:.1f}%) - Value zone')
        signals.append('✅ Buying value, not chasing')
    elif price_vs_ema > 5:
        breakdown.append(f'+0: Price far above EMA ({price_vs_ema:.1f}%) - Overpaying')
        signals.append('⚠️ Price extended above EMA')

    # Impulse system scoring
    if impulse == 'GREEN':
        score += 1
        breakdown.append('+1: Impulse GREEN - Bulls in control')
        signals.append('✅ Impulse permits buying')
    else:  # BLUE
        breakdown.append('+0: Impulse BLUE - Neutral')
        signals.append('⚠️ Impulse neutral - proceed with caution')

    # Candlestick pattern bonus
    bullish_patterns = [p for p in patterns if 'bullish' in p.get('type', '')]
    if bullish_patterns:
        pattern_names = [p['name'] for p in bullish_patterns]
        best_reliability = max(p.get('reliability', 1) for p in bullish_patterns)
        if best_reliability >= 4:
            score += 2
            breakdown.append(f'+2: Strong bullish pattern ({", ".join(pattern_names[:2])})')
        else:
            score += 1
            breakdown.append(f'+1: Bullish pattern ({", ".join(pattern_names[:2])})')
        signals.append(f'✅ Candlestick: {", ".join(pattern_names[:2])}')

    # ═══════════════════════════════════════════════════════════════
    # GRADE DETERMINATION - FIXED: includes weekly_bullish check
    # ═══════════════════════════════════════════════════════════════
    if score >= 5:
        grade = 'A'
        action = '⭐ TRADE: High probability setup - Place order'
    elif score >= 3:
        grade = 'B'
        action = '📊 PREPARE: Good setup developing - Set alerts'
    elif score >= 1:
        grade = 'C'
        action = '👀 WATCH: Early stage - Monitor for improvement'
    else:
        grade = 'AVOID'
        action = '🔴 AVOID: Conditions unfavorable'

    # CRITICAL FIX #3: is_a_trade includes weekly_bullish
    is_a_trade = (
        grade == 'A' and 
        impulse != 'RED' and 
        weekly.get('weekly_bullish', False)
    )

    return {
        'signal_strength': score,
        'grade': grade,
        'action': action,
        'is_a_trade': is_a_trade,
        'breakdown': breakdown,
        'signals': signals,
        'high_value_signals': high_value_signals
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

        # Screen 1 - Weekly (MANDATORY GATE)
        'weekly_trend': weekly['weekly_trend'],
        'weekly_ema': weekly['weekly_ema'],
        'weekly_ema_slope': weekly['weekly_ema_slope'],
        'weekly_macd_rising': weekly['weekly_macd_rising'],
        'weekly_bullish': weekly['weekly_bullish'],
        'screen1_passed': weekly['weekly_bullish'],

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
        'kc_channel_height': round(float(indicators.get('kc_channel_height', 0)), 2),

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
        'high_value_signals': scoring.get('high_value_signals', []),

        # Elder Trade Levels (NEW)
        'entry': levels['entry'],
        'entry_method': levels['entry_method'],
        'stop_loss': levels['stop_loss'],
        'stop_method': levels['stop_method'],
        'target': levels['target'],
        'target_method': levels['target_method'],
        'risk_percent': levels['risk_percent'],
        'reward_percent': levels['reward_percent'],
        'risk_reward_ratio': levels['risk_reward_ratio'],
        'rr_display': levels['rr_display'],
        'penetration_data': levels['penetration_data'],

        # Config
        'indicator_config': config.get('name', 'Custom'),
        'screener_version': '2.0'
    }

    return convert_to_native(result)


def run_weekly_screen_v2(market: str = 'US', symbols: List[str] = None) -> Dict:
    """Run weekly screener v2 with corrected logic"""
    if symbols is None:
        symbols = NASDAQ_100_TOP if market == 'US' else NIFTY_50

    results = []
    passed = []

    for symbol in symbols:
        analysis = scan_stock_v2(symbol)
        if analysis:
            results.append(analysis)
            if analysis['weekly_bullish']:
                passed.append(analysis)

    # Sort by signal strength
    results.sort(key=lambda x: x['signal_strength'], reverse=True)
    passed.sort(key=lambda x: x['signal_strength'], reverse=True)

    # Categorize
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
        'screener_version': '2.0',

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


def run_daily_screen_v2(weekly_results: List[Dict]) -> Dict:
    """
    Run daily screen with CORRECTED logic
    
    FIXED: daily_ready now uses proper AND/OR:
    - screen1_passed AND impulse_ok AND (force_index < 0 OR stochastic < 30)
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
            # FIXED: Correct Elder Triple Screen logic
            screen1_passed = analysis['weekly_bullish']
            impulse_ok = analysis['impulse_color'] != 'RED'
            pullback_signal = (
                analysis['force_index'] < 0 or 
                analysis['stochastic'] < 30  # FIXED: was 50
            )
            
            # FIXED: Proper AND/OR logic
            daily_ready = screen1_passed and impulse_ok and pullback_signal
            
            analysis['daily_ready'] = daily_ready
            analysis['screen1_passed'] = screen1_passed
            analysis['impulse_ok'] = impulse_ok
            analysis['pullback_signal'] = pullback_signal
            results.append(analysis)

    results.sort(key=lambda x: x['signal_strength'], reverse=True)
    a_trades = [r for r in results if r['is_a_trade']]

    return convert_to_native({
        'scan_date': datetime.now().isoformat(),
        'stocks_from_weekly': len(symbols),
        'daily_ready_count': len([r for r in results if r.get('daily_ready')]),
        'a_trades': a_trades,
        'all_results': results,
        'screener_version': '2.0'
    })
