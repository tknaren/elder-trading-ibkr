#!/usr/bin/env python3
"""
Fix API call to use 'all' instead of candlestickSelectedStocks
"""

template_file = "C:/Naren/Working/Source/GitHubRepo/Claude_Trade/elder-trading-ibkr/backend/templates/index.html"

print("Reading file...")
with open(template_file, 'r', encoding='utf-8') as f:
    content = f.read()

print("Updating runCandlestickScreener function...")
# Replace the entire function
content = content.replace(
    '''    async function runCandlestickScreener() {
        candlestickLoading = true;
        render();

        try {
            const market = document.getElementById('market')?.value || 'US';
            const symbols = candlestickSelectedStocks.map(s => market === 'IN' ? s + '.NS' : s);

            const res = await fetch('/api/v2/screener/candlestick/run', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    symbols: symbols,
                    lookback_days: candlestickLookbackDays,
                    market: market,
                    kc_level: candlestickKcLevel,
                    rsi_level: candlestickRsiLevel,
                    selected_patterns: candlestickSelectedPatterns.length > 0 ? candlestickSelectedPatterns : null
                })
            });''',
    '''    async function runCandlestickScreener() {
        candlestickLoading = true;
        render();

        try {
            const market = document.getElementById('market')?.value || 'US';

            const res = await fetch('/api/v2/screener/candlestick/run', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    symbols: 'all',
                    lookback_days: candlestickLookbackDays,
                    market: market,
                    kc_level: candlestickKcLevel,
                    rsi_level: candlestickRsiLevel,
                    selected_patterns: candlestickSelectedPatterns.length > 0 ? candlestickSelectedPatterns : null
                })
            });'''
)

print("Writing file...")
with open(template_file, 'w', encoding='utf-8') as f:
    f.write(content)

print("\n[OK] API call fixed - now uses 'all' for symbols!")
