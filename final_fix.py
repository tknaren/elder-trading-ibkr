#!/usr/bin/env python3
"""
Final fix - update the symbols to 'all'
"""

template_file = "C:/Naren/Working/Source/GitHubRepo/Claude_Trade/elder-trading-ibkr/backend/templates/index.html"

print("Reading file...")
with open(template_file, 'r', encoding='utf-8') as f:
    content = f.read()

print("Removing candlestickSelectedStocks variable...")
content = content.replace(
    '    let candlestickSelectedStocks = [];',
    ''
)

print("Updating runCandlestickScreener function...")
# Remove the symbols line and change symbols to 'all'
content = content.replace(
    '''        try {
            const market = document.getElementById('market')?.value || 'US';
            const symbols = candlestickSelectedStocks.map(s => market === 'IN' ? s + '.NS' : s);

            const res = await fetch('/api/v2/screener/candlestick/run', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    symbols: symbols,''',
    '''        try {
            const market = document.getElementById('market')?.value || 'US';

            const res = await fetch('/api/v2/screener/candlestick/run', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    symbols: 'all','''
)

# Remove the stocks array and stock selection initialization
content = content.replace(
    '''    function candlestickScreenerView() {
        const market = document.getElementById('market')?.value || 'US';
        const stocks = market === 'US' ? [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AMD', 'AVGO', 'NFLX',
            'COST', 'PEP', 'ADBE', 'CSCO', 'INTC', 'QCOM', 'TXN', 'INTU', 'AMAT', 'MU',
            'LRCX', 'KLAC', 'SNPS', 'CDNS', 'MRVL', 'ON', 'NXPI', 'ADI', 'MCHP', 'FTNT',
            'VRTX', 'CHTR', 'ASML', 'CRWD', 'MNST', 'TEAM', 'PAYX', 'AEP', 'CPRT', 'PCAR',
            'AMGN', 'MRNA', 'XEL', 'WDAY', 'ABNB', 'MDLZ', 'ANSS', 'DDOG', 'ODFL', 'GOOG'
        ] : [
            'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK', 'HINDUNILVR', 'SBIN',
            'BHARTIARTL', 'ITC', 'KOTAKBANK', 'LT', 'AXISBANK', 'ASIANPAINT', 'MARUTI',
            'TITAN', 'SUNPHARMA', 'ULTRACEMCO', 'BAJFINANCE', 'WIPRO', 'HCLTECH',
            'TATAMOTORS', 'POWERGRID', 'NTPC', 'M&M', 'JSWSTEEL'
        ];

        if (candlestickSelectedStocks.length === 0) candlestickSelectedStocks = stocks.slice(0, 20);
        ''',
    '''    function candlestickScreenerView() {
        const market = document.getElementById('market')?.value || 'US';
        '''
)

print("Writing file...")
with open(template_file, 'w', encoding='utf-8') as f:
    f.write(content)

print("\n[OK] Final fixes applied!")
print("\nAll changes complete:")
print("1. Variables: candlestickKcLevel, candlestickRsiLevel, candlestickSelectedPatterns")
print("2. UI: KC Level dropdown, RSI Level dropdown, Pattern multi-select listbox")
print("3. Scan All button scans all symbols")
print("4. API call sends kc_level, rsi_level, selected_patterns")
print("\nRestart Flask server and hard refresh browser!")
