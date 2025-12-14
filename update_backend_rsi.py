#!/usr/bin/env python3
"""
Script to add RSI Level parameter to backend
"""

import re

# File paths
screener_py = "C:/Naren/Working/Source/GitHubRepo/Claude_Trade/elder-trading-ibkr/backend/services/candlestick_screener.py"
api_py = "C:/Naren/Working/Source/GitHubRepo/Claude_Trade/elder-trading-ibkr/backend/routes/screener_api.py"

print("Step 1: Updating candlestick_screener.py...")

with open(screener_py, 'r', encoding='utf-8') as f:
    content = f.read()

# Add rsi_level parameter to scan_stock_candlestick_historical
content = re.sub(
    r'(def scan_stock_candlestick_historical\(\n    symbol: str,\n    hist: pd\.DataFrame,\n    lookback_days: int = 180,\n    kc_level: float = -1\.0,)\n(    selected_patterns: List\[str\] = None)',
    r'\1\n    rsi_level: int = 30,\n\2',
    content
)

# Update docstring
content = re.sub(
    r'(kc_level: KC channel level threshold.*?\n)(        selected_patterns:)',
    r'\1        rsi_level: RSI threshold level (e.g., 60, 50, 40, 30)\n\2',
    content,
    flags=re.DOTALL
)

# Update RSI filtering logic - change hardcoded 30 to use rsi_level parameter
content = re.sub(
    r'rsi_oversold = rsi_val < 30 if rsi_val else False',
    r'rsi_oversold = rsi_val < rsi_level if rsi_val else False',
    content
)

# Add rsi_level parameter to run_candlestick_screener
content = re.sub(
    r'(def run_candlestick_screener\(\n    symbols: List\[str\],\n    hist_data: Dict\[str, pd\.DataFrame\],\n    lookback_days: int = 180,\n    filter_mode: str = \'all\',.*?\n    kc_level: float = -1\.0,)\n(    selected_patterns: List\[str\] = None)',
    r'\1\n    rsi_level: int = 30,\n\2',
    content
)

# Update run_candlestick_screener docstring
content = re.sub(
    r'(kc_level: KC channel level threshold.*?\n)(        selected_patterns:)',
    r'\1        rsi_level: RSI threshold level (e.g., 60, 50, 40, 30)\n\2',
    content,
    flags=re.DOTALL
)

# Update the scan_stock_candlestick_historical call to include rsi_level
content = re.sub(
    r'signals = scan_stock_candlestick_historical\(\n(            symbol, hist, lookback_days, kc_level, selected_patterns)\)',
    r'signals = scan_stock_candlestick_historical(\n\1, rsi_level)',
    content
)

# Add rsi_level to return dict
content = re.sub(
    r"(    return \{\n        'signals': all_signals,\n        'summary': summary,\n        'symbols_scanned': len\(symbols\),\n        'lookback_days': lookback_days,\n        'filter_mode': filter_mode,\n        'kc_level': kc_level,)\n(        'selected_patterns': selected_patterns\n    \})",
    r"\1\n        'rsi_level': rsi_level,\n\2",
    content
)

with open(screener_py, 'w', encoding='utf-8') as f:
    f.write(content)

print("  - Added rsi_level parameter to scan_stock_candlestick_historical")
print("  - Updated RSI filtering to use rsi_level instead of hardcoded 30")
print("  - Added rsi_level parameter to run_candlestick_screener")
print("  - Updated return dict to include rsi_level")

print("\nStep 2: Updating screener_api.py...")

with open(api_py, 'r', encoding='utf-8') as f:
    content = f.read()

# Add rsi_level to endpoint documentation
content = re.sub(
    r'("kc_level": -1 \| 0 \| -2 \(KC channel level threshold\),)\n(        "selected_patterns":)',
    r'\1\n        "rsi_level": 30 | 40 | 50 | 60 (RSI threshold level),\n\2',
    content
)

# Add rsi_level description in docstring
content = re.sub(
    r'(kc_level:\n        - 0: Price < KC Middle\n        - -1: Price < KC Lower \(default\)\n        - -2: Price < KC Lower - ATR\n\n)(    selected_patterns:)',
    r'\1    rsi_level:\n        - 60: RSI < 60\n        - 50: RSI < 50\n        - 40: RSI < 40\n        - 30: RSI < 30 (default)\n\n\2',
    content
)

# Extract rsi_level from request data
content = re.sub(
    r'(    kc_level = data\.get\(\'kc_level\', -1\.0\))\n(    selected_patterns = data\.get\(\'selected_patterns\', None\))',
    r"\1\n    rsi_level = data.get('rsi_level', 30)\n\2",
    content
)

# Validate rsi_level
content = re.sub(
    r'(    # Validate selected_patterns\n    if selected_patterns and not isinstance\(selected_patterns, list\):\n        selected_patterns = None)',
    r'''\1

    # Validate rsi_level
    try:
        rsi_level = int(rsi_level)
        if rsi_level not in [60, 50, 40, 30]:
            rsi_level = 30
    except (ValueError, TypeError):
        rsi_level = 30''',
    content
)

# Add rsi_level to run_candlestick_screener call
content = re.sub(
    r'(result = run_candlestick_screener\(\n            symbols=list\(hist_data\.keys\(\)\),\n            hist_data=hist_data,\n            lookback_days=lookback_days,\n            filter_mode=filter_mode,\n            kc_level=kc_level,)\n(            selected_patterns=selected_patterns\n        \))',
    r'\1\n            rsi_level=rsi_level,\n\2',
    content
)

# Update single stock endpoint - add rsi_level query param
content = re.sub(
    r'(    kc_level = request\.args\.get\(\'kc_level\', -1\.0, type=float\))\n(    selected_patterns_param = request\.args\.get\(\'selected_patterns\', None\))',
    r"\1\n    rsi_level = request.args.get('rsi_level', 30, type=int)\n\2",
    content
)

# Add rsi_level to single stock scan call
content = re.sub(
    r'(signals = scan_stock_candlestick_historical\(\n            symbol=symbol\.upper\(\),\n            hist=hist,\n            lookback_days=lookback_days,\n            kc_level=kc_level,)\n(            selected_patterns=selected_patterns\n        \))',
    r'\1\n            rsi_level=rsi_level,\n\2',
    content
)

# Add rsi_level to single stock response
content = re.sub(
    r"(        return jsonify\(\{\n            'symbol': symbol\.upper\(\),\n            'signals': signals,\n            'count': len\(signals\),\n            'lookback_days': lookback_days,\n            'filter_mode': filter_mode,\n            'kc_level': kc_level,)\n(            'selected_patterns': selected_patterns,)",
    r"\1\n            'rsi_level': rsi_level,\n\2",
    content
)

with open(api_py, 'w', encoding='utf-8') as f:
    f.write(content)

print("  - Added rsi_level to endpoint documentation")
print("  - Added rsi_level extraction and validation")
print("  - Updated run_candlestick_screener call")
print("  - Updated single stock endpoint")

print("\n[OK] Backend Update Complete!")
print("\nChanges:")
print("1. Added rsi_level parameter (default: 30)")
print("2. RSI filtering now uses configurable rsi_level instead of hardcoded 30")
print("3. API endpoints accept and validate rsi_level")
print("\nRun with: python update_backend_rsi.py")
