"""
Elder Trading System - Historical Screener Service
===================================================

Scans historical data to find all signals with score >= threshold
Returns detailed indicator values for manual verification

This is NOT a backtest - it just finds historical signals for review
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

from services.indicators import (
    calculate_ema,
    calculate_macd,
    calculate_atr
)

# Stock lists
NASDAQ_100 = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AMD', 'AVGO', 'NFLX',
    'COST', 'PEP', 'ADBE', 'CSCO', 'INTC', 'QCOM', 'TXN', 'INTU', 'AMAT', 'MU',
    'LRCX', 'KLAC', 'SNPS', 'CDNS', 'MRVL', 'ON', 'NXPI', 'ADI', 'MCHP', 'FTNT',
    'VRTX', 'CHTR', 'ASML', 'CRWD', 'PANW', 'MNST', 'TEAM', 'PAYX', 'AEP', 'REGN',
    'DXCM', 'CPRT', 'PCAR', 'ALGN', 'AMGN', 'MRNA', 'XEL', 'WDAY', 'ABNB', 'MDLZ',
    'GILD', 'ISRG', 'BKNG', 'ADP', 'SBUX', 'PYPL', 'CME', 'ORLY', 'IDXX', 'CTAS',
    'MAR', 'CSX', 'ODFL', 'FAST', 'ROST', 'KDP', 'EXC', 'DLTR', 'BIIB', 'EA',
    'VRSK', 'ANSS', 'ILMN', 'SIRI', 'ZS', 'DDOG', 'CTSH', 'WBD', 'EBAY', 'FANG',
    'GFS', 'LCID', 'RIVN', 'CEG', 'TTD', 'GEHC', 'ZM', 'ROKU', 'OKTA', 'SPLK',
    'DOCU', 'BILL', 'ENPH', 'SEDG', 'DASH', 'COIN', 'HOOD', 'SOFI', 'PLTR', 'NET'
]


@dataclass
class HistoricalSignal:
    """A single historical signal found"""
    symbol: str
    date: str
    price: float
    score: int
    grade: str
    
    # Screen 1 scores
    screen1_score: int
    macd_h_score: int
    macd_line_score: int
    ema_alignment_score: int
    
    # Screen 2 scores
    screen2_score: int
    kc_score: int
    fi_score: int
    stoch_score: int
    pattern_score: int
    
    # All weekly filters pass
    all_weekly_filters: bool
    
    # Indicator values
    weekly_macd_h: float
    weekly_macd_line: float
    weekly_macd_signal: float
    weekly_ema_20: float
    weekly_ema_50: float
    weekly_ema_100: float
    
    daily_ema_22: float
    daily_atr: float
    daily_kc_upper: float
    daily_kc_middle: float
    daily_kc_lower: float
    daily_force_index_2: float
    daily_force_index_13: float
    daily_stochastic_k: float
    daily_stochastic_d: float
    daily_rsi: float
    
    pattern: str
    
    # Entry/Stop/Target (calculated)
    entry: float
    stop_loss: float
    target: float


def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
                         k_period: int = 14, d_period: int = 3) -> Dict:
    """Calculate Stochastic Oscillator"""
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    denom = highest_high - lowest_low
    denom = denom.replace(0, np.nan)
    k = 100 * (close - lowest_low) / denom
    d = k.rolling(window=d_period).mean()
    return {'k': k, 'd': d}


def calculate_force_index(close: pd.Series, volume: pd.Series, period: int = 2) -> pd.Series:
    """Calculate Force Index"""
    price_change = close.diff()
    force_index = price_change * volume
    return force_index.ewm(span=period, adjust=False).mean()


def calculate_keltner_channel(high: pd.Series, low: pd.Series, close: pd.Series,
                               ema_period: int = 20, atr_period: int = 10, multiplier: float = 2.0) -> Dict:
    """Calculate Keltner Channel"""
    middle = close.ewm(span=ema_period, adjust=False).mean()
    atr = calculate_atr(high, low, close, atr_period)
    upper = middle + multiplier * atr
    lower = middle - multiplier * atr
    return {'upper': upper, 'middle': middle, 'lower': lower, 'atr': atr}


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI"""
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    loss = loss.replace(0, np.nan)
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def analyze_weekly_at_date(hist: pd.DataFrame, analysis_date: pd.Timestamp) -> Dict:
    """
    Analyze weekly indicators at a specific date
    v2.3 scoring system
    """
    hist_slice = hist[hist.index <= analysis_date]
    
    # Need at least 50 daily bars for reliable weekly analysis
    if len(hist_slice) < 50:
        return {'screen1_score': 0, 'weekly_bullish': False}
    
    # Resample to weekly (use Friday as week end to match market weeks)
    weekly = hist_slice.resample('W-FRI').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min',
        'Close': 'last', 'Volume': 'sum'
    }).dropna()
    
    if len(weekly) < 10:  # Need at least 10 weeks for indicators
        return {'screen1_score': 0, 'weekly_bullish': False}
    
    closes = weekly['Close']
    
    # MACD
    macd = calculate_macd(closes)
    current_macd_h = float(macd['histogram'].iloc[-1])
    prev_macd_h = float(macd['histogram'].iloc[-2]) if len(macd['histogram']) > 1 else current_macd_h
    current_macd_line = float(macd['macd_line'].iloc[-1])
    current_signal = float(macd['signal_line'].iloc[-1])
    
    macd_h_rising = current_macd_h > prev_macd_h
    
    # 1. MACD-H Rising Score
    macd_h_score = 0
    if macd_h_rising:
        if prev_macd_h < 0:  # Spring - rising from below zero
            macd_h_score = 2
        else:  # Summer - rising from above zero
            macd_h_score = 1
    
    # 2. MACD Line vs Signal
    macd_line_score = 0
    if current_macd_line < current_signal:
        if current_macd_line < 0 and current_signal < 0:
            macd_line_score = 2
        else:
            macd_line_score = 1
    
    # 3. EMA Alignment (20 > 50 > 100)
    data_len = len(closes)
    ema_20 = float(calculate_ema(closes, min(data_len, 20)).iloc[-1])
    ema_50 = float(calculate_ema(closes, min(data_len, 50)).iloc[-1])
    ema_100 = float(calculate_ema(closes, min(data_len, 100)).iloc[-1])
    
    ema_alignment_score = 0
    if ema_20 > ema_50 and ema_50 > ema_100:
        ema_alignment_score = 2
    
    screen1_score = macd_h_score + macd_line_score + ema_alignment_score
    
    return {
        'screen1_score': screen1_score,
        'weekly_bullish': screen1_score >= 1,  # Any score = can proceed
        'macd_h_score': macd_h_score,
        'macd_line_score': macd_line_score,
        'ema_alignment_score': ema_alignment_score,
        'weekly_macd_h': round(current_macd_h, 4),
        'weekly_macd_line': round(current_macd_line, 4),
        'weekly_macd_signal': round(current_signal, 4),
        'weekly_ema_20': round(ema_20, 2),
        'weekly_ema_50': round(ema_50, 2),
        'weekly_ema_100': round(ema_100, 2),
        'macd_h_rising': macd_h_rising,
    }


def calculate_score_at_date(hist: pd.DataFrame, analysis_date: pd.Timestamp, weekly: Dict) -> Optional[Dict]:
    """
    Calculate full v2.3 score at a specific date
    """
    hist_slice = hist[hist.index <= analysis_date].copy()
    
    if len(hist_slice) < 50:
        return None
    
    current = hist_slice.iloc[-1]
    price = float(current['Close'])
    
    closes = hist_slice['Close']
    highs = hist_slice['High']
    lows = hist_slice['Low']
    volumes = hist_slice['Volume']
    
    # Daily indicators
    ema_22 = float(calculate_ema(closes, 22).iloc[-1])
    atr = float(calculate_atr(highs, lows, closes, 14).iloc[-1])
    
    kc = calculate_keltner_channel(highs, lows, closes, 20, 10, 2.0)
    kc_middle = float(kc['middle'].iloc[-1])
    kc_upper = float(kc['upper'].iloc[-1])
    kc_lower = float(kc['lower'].iloc[-1])
    
    force_index_2 = float(calculate_force_index(closes, volumes, 2).iloc[-1])
    force_index_13 = float(calculate_force_index(closes, volumes, 13).iloc[-1])
    
    stoch = calculate_stochastic(highs, lows, closes, 14, 3)
    stochastic_k = float(stoch['k'].iloc[-1]) if not pd.isna(stoch['k'].iloc[-1]) else 50
    stochastic_d = float(stoch['d'].iloc[-1]) if not pd.isna(stoch['d'].iloc[-1]) else 50
    
    rsi = calculate_rsi(closes, 14)
    rsi_value = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50
    
    # ═══════════════════════════════════════════════════════════════
    # SCORE CALCULATION (v2.3)
    # ═══════════════════════════════════════════════════════════════
    
    score = 0
    
    # SCREEN 1: Weekly scores
    macd_h_score = weekly.get('macd_h_score', 0)
    macd_line_score = weekly.get('macd_line_score', 0)
    ema_alignment_score = weekly.get('ema_alignment_score', 0)
    screen1_score = weekly.get('screen1_score', 0)
    
    score += screen1_score
    
    # SCREEN 2: Daily
    
    # 1. Keltner Channel position
    kc_lower_1 = kc_middle - atr
    kc_lower_3 = kc_middle - 3 * atr
    
    kc_score = 0
    if price >= kc_lower_3 and price < kc_lower_1:
        kc_score = 2  # Deep pullback
    elif price >= kc_lower_1 and price < kc_middle:
        kc_score = 1  # Normal pullback
    
    score += kc_score
    
    # 2. Force Index < 0
    fi_score = 1 if force_index_2 < 0 else 0
    score += fi_score
    
    # 3. Stochastic < 50
    stoch_score = 1 if stochastic_k < 50 else 0
    score += stoch_score
    
    # 4. Pattern detection (simplified)
    pattern_score = 0
    pattern_name = 'None'
    
    body = abs(current['Close'] - current['Open'])
    lower_shadow = min(current['Open'], current['Close']) - current['Low']
    upper_shadow = current['High'] - max(current['Open'], current['Close'])
    total_range = current['High'] - current['Low']
    
    if total_range > 0:
        is_hammer = (
            lower_shadow > body * 2 and
            upper_shadow < body * 0.5 and
            current['Close'] >= current['Open']
        )
        
        if len(hist_slice) >= 2:
            prev = hist_slice.iloc[-2]
            is_engulfing = (
                prev['Close'] < prev['Open'] and
                current['Close'] > current['Open'] and
                current['Open'] <= prev['Close'] and
                current['Close'] >= prev['Open']
            )
        else:
            is_engulfing = False
        
        if is_hammer:
            pattern_score = 1
            pattern_name = 'Hammer'
        elif is_engulfing:
            pattern_score = 1
            pattern_name = 'Bullish Engulfing'
    
    score += pattern_score
    
    # Grade determination
    all_weekly_filters = (macd_h_score > 0 and macd_line_score > 0 and ema_alignment_score > 0)
    
    if all_weekly_filters and score >= 7:
        grade = 'A'
    elif all_weekly_filters and score >= 5:
        grade = 'B'
    elif score >= 7:
        grade = 'B'
    elif score >= 5:
        grade = 'B'
    elif score >= 1:
        grade = 'C'
    else:
        grade = 'AVOID'
    
    screen2_score = score - screen1_score
    
    # Calculate entry/stop/target
    prev_high = float(hist_slice.iloc[-2]['High']) if len(hist_slice) > 1 else price
    entry = round(prev_high * 1.001, 2)  # Buy-stop above prev high
    
    # Stop: swing low or 2×ATR
    recent_low = float(hist_slice.tail(5)['Low'].min())
    stop_atr = entry - 2 * atr
    stop_loss = round(max(recent_low * 0.998, stop_atr), 2)
    
    # Target: 1.5 R:R
    risk = entry - stop_loss
    target = round(entry + risk * 1.5, 2) if risk > 0 else entry
    
    return {
        'price': round(price, 2),
        'score': score,
        'grade': grade,
        
        'screen1_score': screen1_score,
        'screen2_score': screen2_score,
        'macd_h_score': macd_h_score,
        'macd_line_score': macd_line_score,
        'ema_alignment_score': ema_alignment_score,
        
        'kc_score': kc_score,
        'fi_score': fi_score,
        'stoch_score': stoch_score,
        'pattern_score': pattern_score,
        
        'all_weekly_filters': all_weekly_filters,
        
        # Weekly indicators
        'weekly_macd_h': weekly.get('weekly_macd_h', 0),
        'weekly_macd_line': weekly.get('weekly_macd_line', 0),
        'weekly_macd_signal': weekly.get('weekly_macd_signal', 0),
        'weekly_ema_20': weekly.get('weekly_ema_20', 0),
        'weekly_ema_50': weekly.get('weekly_ema_50', 0),
        'weekly_ema_100': weekly.get('weekly_ema_100', 0),
        
        # Daily indicators
        'daily_ema_22': round(ema_22, 2),
        'daily_atr': round(atr, 2),
        'daily_kc_upper': round(kc_upper, 2),
        'daily_kc_middle': round(kc_middle, 2),
        'daily_kc_lower': round(kc_lower, 2),
        'daily_force_index_2': round(force_index_2, 0),
        'daily_force_index_13': round(force_index_13, 0),
        'daily_stochastic_k': round(stochastic_k, 1),
        'daily_stochastic_d': round(stochastic_d, 1),
        'daily_rsi': round(rsi_value, 1),
        
        'pattern': pattern_name,
        
        'entry': entry,
        'stop_loss': stop_loss,
        'target': target,
    }


def fetch_stock_data(symbol: str, lookback_days: int = 365) -> Optional[pd.DataFrame]:
    """
    Fetch historical data from IBKR or cache
    Uses the same data source as the live screener
    """
    try:
        from services.ibkr_client import fetch_stock_data as ibkr_fetch
        
        # Try IBKR fetch (this handles caching internally)
        data = ibkr_fetch(symbol)  # Returns dict with 'history' key
        
        if data is not None and 'history' in data:
            hist = data['history']
            if hist is not None and len(hist) > 100:
                print(f"✅ {symbol}: Got {len(hist)} bars from IBKR")
                return hist
            else:
                print(f"⚠️ {symbol}: IBKR returned insufficient data ({len(hist) if hist is not None else 0} bars)")
        else:
            print(f"⚠️ {symbol}: IBKR fetch returned None or no history")
        
        # Fall back to direct cache lookup
        from models.database import get_database
        db = get_database().get_connection()
        
        cached_rows = db.execute('''
            SELECT date, open, high, low, close, volume 
            FROM stock_historical_data 
            WHERE symbol = ? 
            ORDER BY date ASC
        ''', (symbol,)).fetchall()
        
        if cached_rows and len(cached_rows) >= 100:
            print(f"📦 {symbol}: Using {len(cached_rows)} cached rows")
            hist = pd.DataFrame([
                {
                    'Date': row['date'],
                    'Open': row['open'],
                    'High': row['high'],
                    'Low': row['low'],
                    'Close': row['close'],
                    'Volume': row['volume']
                }
                for row in cached_rows
            ])
            hist['Date'] = pd.to_datetime(hist['Date'])
            hist.set_index('Date', inplace=True)
            hist = hist.sort_index()
            return hist
        
        print(f"❌ {symbol}: No data available (IBKR down and no cache)")
        return None
        
    except Exception as e:
        print(f"❌ Error fetching {symbol}: {e}")
        
        # Last resort: try direct cache lookup
        try:
            from models.database import get_database
            db = get_database().get_connection()
            
            cached_rows = db.execute('''
                SELECT date, open, high, low, close, volume 
                FROM stock_historical_data 
                WHERE symbol = ? 
                ORDER BY date ASC
            ''', (symbol,)).fetchall()
            
            if cached_rows and len(cached_rows) >= 100:
                print(f"📦 {symbol}: Fallback to {len(cached_rows)} cached rows")
                hist = pd.DataFrame([
                    {
                        'Date': row['date'],
                        'Open': row['open'],
                        'High': row['high'],
                        'Low': row['low'],
                        'Close': row['close'],
                        'Volume': row['volume']
                    }
                    for row in cached_rows
                ])
                hist['Date'] = pd.to_datetime(hist['Date'])
                hist.set_index('Date', inplace=True)
                hist = hist.sort_index()
                return hist
        except Exception as e2:
            print(f"❌ Cache fallback also failed for {symbol}: {e2}")
        
        return None


def scan_stock_historical(
    symbol: str, 
    lookback_days: int = 180,
    min_score: int = 5
) -> Optional[List[Dict]]:
    """
    Scan a single stock's history for signals meeting minimum score
    Returns: List of signals, empty list if no signals found, None if no data
    """
    hist = fetch_stock_data(symbol, lookback_days + 365)  # Extra for indicators
    
    if hist is None or len(hist) < 200:
        print(f"⚠️ {symbol}: Insufficient data ({len(hist) if hist is not None else 0} bars)")
        return None
    
    # Ensure timezone-naive
    if hist.index.tz is not None:
        hist.index = hist.index.tz_localize(None)
    
    signals = []
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)
    
    # Get trading days in scan period
    scan_dates = hist[(hist.index >= pd.Timestamp(start_date)) & 
                      (hist.index <= pd.Timestamp(end_date))].index
    
    print(f"📊 {symbol}: Scanning {len(scan_dates)} trading days")
    
    dates_analyzed = 0
    weekly_bullish_count = 0
    
    for analysis_date in scan_dates:
        dates_analyzed += 1
        
        # Analyze weekly
        weekly = analyze_weekly_at_date(hist, analysis_date)
        if not weekly.get('weekly_bullish', False):
            continue
        
        weekly_bullish_count += 1
        
        # Calculate full score
        result = calculate_score_at_date(hist, analysis_date, weekly)
        if result is None:
            continue
        
        if result['score'] >= min_score:
            signal = {
                'symbol': symbol,
                'date': analysis_date.strftime('%Y-%m-%d'),
                **result
            }
            signals.append(signal)
    
    print(f"✅ {symbol}: Found {len(signals)} signals (weekly bullish on {weekly_bullish_count}/{dates_analyzed} days)")
    
    return signals


def run_historical_screener(
    symbols: List[str],
    lookback_days: int = 180,
    min_score: int = 5,
    progress_callback=None
) -> Dict:
    """
    Run historical screener across multiple symbols
    
    Returns:
        {
            'signals': [...],
            'summary': {...},
            'symbols_scanned': int,
            'symbols_with_signals': int,
            'diagnostics': {...}  # Debug info
        }
    """
    all_signals = []
    symbols_with_signals = 0
    symbols_with_data = 0
    symbols_failed = []
    
    for i, symbol in enumerate(symbols):
        if progress_callback:
            progress_callback(i + 1, len(symbols), symbol)
        
        try:
            signals = scan_stock_historical(symbol, lookback_days, min_score)
            if signals:
                all_signals.extend(signals)
                symbols_with_signals += 1
                symbols_with_data += 1
            elif signals is not None:  # Empty list means data was available but no signals
                symbols_with_data += 1
        except Exception as e:
            print(f"Error scanning {symbol}: {e}")
            symbols_failed.append({'symbol': symbol, 'error': str(e)})
    
    # Sort by date descending, then by score descending
    all_signals.sort(key=lambda x: (x['date'], x['score']), reverse=True)
    
    # Summary stats
    summary = {
        'total_signals': len(all_signals),
        'a_trades': len([s for s in all_signals if s['grade'] == 'A']),
        'b_trades': len([s for s in all_signals if s['grade'] == 'B']),
        'c_trades': len([s for s in all_signals if s['grade'] == 'C']),
        'avg_score': round(sum(s['score'] for s in all_signals) / len(all_signals), 1) if all_signals else 0,
    }
    
    # Diagnostics for debugging
    diagnostics = {
        'symbols_with_data': symbols_with_data,
        'symbols_no_data': len(symbols) - symbols_with_data - len(symbols_failed),
        'symbols_failed': symbols_failed[:10] if symbols_failed else [],  # First 10 failures
        'data_source': 'IBKR + Cache'
    }
    
    return {
        'signals': all_signals,
        'summary': summary,
        'symbols_scanned': len(symbols),
        'symbols_with_signals': symbols_with_signals,
        'lookback_days': lookback_days,
        'min_score': min_score,
        'diagnostics': diagnostics
    }


def get_stock_list(market: str = 'US') -> List[str]:
    """Get available stock list"""
    if market.upper() == 'US':
        return NASDAQ_100
    else:
        from services.screener_v2 import NIFTY_100
        return NIFTY_100
