"""
Elder Trading System - RSI + MACD Indicator Historical Screener
Scans for RSI oversold conditions with MACD confirmation

Filter Conditions (All must be TRUE):
1. RSI(14) < 30 (Oversold)
2. RSI is Increasing (today > yesterday)
3. MACD pointing up OR crossing up at the same time

"Pointing Up" = MACD Histogram increasing (today > yesterday)
"Crossing Up" = MACD Line crosses above Signal Line

Shows all indicator values in the results grid.
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
    """Calculate Keltner Channel"""
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


def calculate_force_index(close: pd.Series, volume: pd.Series, period: int = 2) -> pd.Series:
    """Calculate Force Index"""
    force = close.diff() * volume
    return calculate_ema(force, period)


def calculate_all_indicators(df: pd.DataFrame) -> Dict:
    """Calculate all required indicators for a dataframe"""
    if len(df) < 30:
        return {}

    # RSI
    rsi = calculate_rsi(df['Close'], 14)

    # MACD
    macd = calculate_macd(df['Close'], 12, 26, 9)

    # Stochastic
    stoch = calculate_stochastic(df['High'], df['Low'], df['Close'], 14, 3)

    # Keltner Channel (20, 10, 1)
    kc = calculate_keltner_channel(
        df['High'], df['Low'], df['Close'],
        ema_period=20, atr_period=10, atr_mult=1.0
    )

    # Force Index
    force_index = None
    if 'Volume' in df.columns:
        force_index = calculate_force_index(df['Close'], df['Volume'], 2)

    # EMA 20
    ema_20 = calculate_ema(df['Close'], 20)

    return {
        'rsi': rsi,
        'macd_line': macd['macd_line'],
        'signal_line': macd['signal_line'],
        'macd_hist': macd['histogram'],
        'stoch_k': stoch['stoch_k'],
        'stoch_d': stoch['stoch_d'],
        'kc_upper': kc['kc_upper'],
        'kc_middle': kc['kc_middle'],
        'kc_lower': kc['kc_lower'],
        'atr': kc['atr'],
        'force_index': force_index,
        'ema_20': ema_20
    }


def check_rsi_macd_conditions(indicators: Dict, idx: int) -> Dict:
    """
    Check RSI + MACD filter conditions for a specific index

    Conditions:
    1. RSI(14) < 30 (Oversold)
    2. RSI is Increasing (today > yesterday)
    3. MACD pointing up OR crossing up

    Returns:
        Dict with condition status and details
    """
    result = {
        'rsi_oversold': False,
        'rsi_increasing': False,
        'macd_pointing_up': False,
        'macd_crossing_up': False,
        'all_conditions_met': False,
        'details': []
    }

    if idx < 1:
        return result

    # Get current and previous values
    rsi_current = indicators['rsi'].iloc[idx]
    rsi_prev = indicators['rsi'].iloc[idx - 1]

    macd_current = indicators['macd_line'].iloc[idx]
    macd_prev = indicators['macd_line'].iloc[idx - 1]

    signal_current = indicators['signal_line'].iloc[idx]
    signal_prev = indicators['signal_line'].iloc[idx - 1]

    hist_current = indicators['macd_hist'].iloc[idx]
    hist_prev = indicators['macd_hist'].iloc[idx - 1]

    # Check NaN values
    if pd.isna(rsi_current) or pd.isna(rsi_prev):
        return result
    if pd.isna(macd_current) or pd.isna(hist_current):
        return result

    # Condition 1: RSI < 30
    result['rsi_oversold'] = rsi_current < 30
    if result['rsi_oversold']:
        result['details'].append(f'RSI oversold: {rsi_current:.1f} < 30')

    # Condition 2: RSI Increasing
    result['rsi_increasing'] = rsi_current > rsi_prev
    if result['rsi_increasing']:
        result['details'].append(
            f'RSI increasing: {rsi_prev:.1f} → {rsi_current:.1f}')

    # Condition 3a: MACD Histogram pointing up (increasing)
    result['macd_pointing_up'] = hist_current > hist_prev
    if result['macd_pointing_up']:
        result['details'].append(
            f'MACD histogram ↑: {hist_prev:.4f} → {hist_current:.4f}')

    # Condition 3b: MACD Line crossing above Signal Line
    macd_above_signal_now = macd_current > signal_current
    macd_below_signal_prev = macd_prev <= signal_prev
    result['macd_crossing_up'] = macd_above_signal_now and macd_below_signal_prev
    if result['macd_crossing_up']:
        result['details'].append(
            f'MACD crossover ↑: MACD({macd_current:.4f}) > Signal({signal_current:.4f})')

    # Combined: RSI < 30 AND RSI increasing AND (MACD pointing up OR crossing up)
    macd_condition = result['macd_pointing_up'] or result['macd_crossing_up']
    result['all_conditions_met'] = (
        result['rsi_oversold'] and
        result['rsi_increasing'] and
        macd_condition
    )

    return result


def scan_stock_rsi_macd_historical(
    symbol: str,
    hist: pd.DataFrame,
    lookback_days: int = 180,
    stock_info: Dict = None
) -> List[Dict]:
    """
    Scan a single stock's history for RSI + MACD signals

    Args:
        symbol: Stock ticker
        hist: Historical OHLCV dataframe
        lookback_days: Number of days to scan
        stock_info: Dict with company info (Name, Market Cap, Sector, Industry)

    Returns:
        List of signals matching all conditions with indicator values
    """
    if hist is None or len(hist) < 50:
        return []

    signals = []

    # Calculate all indicators
    indicators = calculate_all_indicators(hist)
    if not indicators:
        return []

    # Apply lookback period - only scan last N days
    latest_date = hist.index[-1]
    cutoff_date = latest_date - pd.Timedelta(days=lookback_days)

    # Scan each day (starting from day 35 to ensure indicator stability)
    for idx in range(35, len(hist)):
        date = hist.index[idx]

        # Skip dates outside lookback period
        if date < cutoff_date:
            continue
        current_row = hist.iloc[idx]
        date = hist.index[idx]

        # Check conditions
        conditions = check_rsi_macd_conditions(indicators, idx)

        # Only include if all conditions are met
        if not conditions['all_conditions_met']:
            continue

        # Get all indicator values
        rsi_val = float(indicators['rsi'].iloc[idx])
        rsi_prev = float(indicators['rsi'].iloc[idx - 1])
        macd_val = float(indicators['macd_line'].iloc[idx])
        signal_val = float(indicators['signal_line'].iloc[idx])
        macd_hist = float(indicators['macd_hist'].iloc[idx])
        macd_hist_prev = float(indicators['macd_hist'].iloc[idx - 1])
        stoch_k = float(indicators['stoch_k'].iloc[idx]) if not pd.isna(
            indicators['stoch_k'].iloc[idx]) else None
        stoch_d = float(indicators['stoch_d'].iloc[idx]) if not pd.isna(
            indicators['stoch_d'].iloc[idx]) else None
        kc_lower = float(indicators['kc_lower'].iloc[idx]) if not pd.isna(
            indicators['kc_lower'].iloc[idx]) else None
        kc_middle = float(indicators['kc_middle'].iloc[idx]) if not pd.isna(
            indicators['kc_middle'].iloc[idx]) else None
        kc_upper = float(indicators['kc_upper'].iloc[idx]) if not pd.isna(
            indicators['kc_upper'].iloc[idx]) else None
        ema_20 = float(indicators['ema_20'].iloc[idx]) if not pd.isna(
            indicators['ema_20'].iloc[idx]) else None

        force_idx_val = None
        if indicators.get('force_index') is not None:
            fi = indicators['force_index'].iloc[idx]
            if not pd.isna(fi):
                force_idx_val = float(fi)

        close_price = float(current_row['Close'])

        # Determine signal type
        signal_type = 'RSI + MACD Alignment'
        if conditions['macd_crossing_up']:
            signal_type = 'RSI Oversold + MACD Crossover'
        elif conditions['macd_pointing_up']:
            signal_type = 'RSI Oversold + MACD Rising'

        signal = {
            'symbol': symbol,
            'name': stock_info.get('Name', '') if stock_info else '',
            'market_cap': stock_info.get('Market Cap', '') if stock_info else '',
            'sector': stock_info.get('Sector', '') if stock_info else '',
            'industry': stock_info.get('Industry', '') if stock_info else '',
            'date': str(date)[:10] if hasattr(date, 'strftime') else str(date)[:10],
            'signal_type': signal_type,
            'close': round(float(close_price), 2),
            # RSI values
            'rsi': round(float(rsi_val), 1),
            'rsi_prev': round(float(rsi_prev), 1),
            'rsi_change': round(float(rsi_val - rsi_prev), 2),
            # MACD values
            'macd': round(float(macd_val), 4),
            'macd_signal': round(float(signal_val), 4),
            'macd_hist': round(float(macd_hist), 4),
            'macd_hist_prev': round(float(macd_hist_prev), 4),
            'macd_hist_change': round(float(macd_hist - macd_hist_prev), 4),
            # Other indicators
            'stoch_k': round(float(stoch_k), 1) if stoch_k else None,
            'stoch_d': round(float(stoch_d), 1) if stoch_d else None,
            'kc_lower': round(float(kc_lower), 2) if kc_lower else None,
            'kc_middle': round(float(kc_middle), 2) if kc_middle else None,
            'kc_upper': round(float(kc_upper), 2) if kc_upper else None,
            'ema_20': round(float(ema_20), 2) if ema_20 else None,
            'force_index': round(float(force_idx_val), 0) if force_idx_val else None,
            # Condition details
            'conditions': {
                'rsi_oversold': bool(conditions['rsi_oversold']),
                'rsi_increasing': bool(conditions['rsi_increasing']),
                'macd_pointing_up': bool(conditions['macd_pointing_up']),
                'macd_crossing_up': bool(conditions['macd_crossing_up'])
            },
            'condition_details': str(conditions['details'])
        }

        signals.append(signal)

    return signals


def run_rsi_macd_screener(
    symbols: List[str],
    hist_data: Dict[str, pd.DataFrame],
    lookback_days: int = 180,
    stock_info_map: Dict[str, Dict] = None
) -> Dict:
    """
    Run RSI + MACD screener across multiple symbols

    Args:
        symbols: List of stock tickers
        hist_data: Dict of symbol -> DataFrame with OHLCV data
        lookback_days: Number of days to scan
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
            signals = scan_stock_rsi_macd_historical(
                symbol, hist, lookback_days, stock_info)

            if signals:
                all_signals.extend(signals)
                symbols_with_signals += 1

        except Exception as e:
            print(f"Error scanning {symbol}: {e}")

    # Sort by date descending
    all_signals.sort(key=lambda x: x['date'], reverse=True)

    # Summary stats
    crossover_count = len(
        [s for s in all_signals if s['conditions']['macd_crossing_up']])
    rising_count = len([s for s in all_signals if s['conditions']
                       ['macd_pointing_up'] and not s['conditions']['macd_crossing_up']])

    avg_rsi = sum(s['rsi'] for s in all_signals) / \
        len(all_signals) if all_signals else 0

    summary = {
        'total_signals': len(all_signals),
        'crossover_signals': crossover_count,
        'rising_signals': rising_count,
        'avg_rsi_at_signal': round(avg_rsi, 1),
        'symbols_with_signals': symbols_with_signals
    }

    return {
        'signals': all_signals,
        'summary': summary,
        'symbols_scanned': len(symbols),
        'lookback_days': lookback_days
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
        # For non-US markets, return empty for now
        return [], {}
