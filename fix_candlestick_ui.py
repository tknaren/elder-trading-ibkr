#!/usr/bin/env python3
"""
Script to properly fix the Candlestick screener UI
"""

import re

template_file = "C:/Naren/Working/Source/GitHubRepo/Claude_Trade/elder-trading-ibkr/backend/templates/index.html"

print("Reading template file...")
with open(template_file, 'r', encoding='utf-8') as f:
    content = f.read()

print("Step 1: Updating state variables...")
# Replace the state variables section
content = re.sub(
    r'(// ========== CANDLESTICK PATTERN SCREENER ==========\n'
    r'    let candlestickResults = \[\];\n'
    r'    let candlestickLoading = false;\n'
    r'    let candlestickSelectedStocks = \[\];\n'
    r'    let candlestickLookbackDays = 180;\n)'
    r'    let candlestickFilterMode = \'all\';',
    r'\1    let candlestickKcLevel = -1;\n'
    r'    let candlestickRsiLevel = 30;\n'
    r'    let candlestickSelectedPatterns = [];',
    content
)

print("Step 2: Updating grid layout from 3 columns to 4 columns...")
# Update grid from grid-cols-3 to grid-cols-4
content = re.sub(
    r'<div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">',
    r'<div class="grid grid-cols-1 md:grid-cols-4 gap-4 mb-4">',
    content
)

print("Step 3: Replacing UI controls...")
# Find and replace the entire controls section (Lookback Days, Filter Mode, Selected buttons)
old_controls = r'''                    <div class="grid grid-cols-1 md:grid-cols-4 gap-4 mb-4">
                        <div>
                            <label class="block text-sm font-medium mb-1">Lookback Days</label>
                            <input type="number" id="candlestick-lookback" value="\$\{candlestickLookbackDays\}" min="30" max="365"
                                onchange="candlestickLookbackDays=parseInt\(this\.value\)"
                                class="w-full bg-\[#1f2937\] border border-gray-600 rounded px-3 py-2">
                        </div>
                        <div>
                            <label class="block text-sm font-medium mb-1">Filter Mode</label>
                            <select id="candlestick-filter" onchange="candlestickFilterMode=this\.value"
                                class="w-full bg-\[#1f2937\] border border-gray-600 rounded px-3 py-2">
                                <option value="all" \$\{candlestickFilterMode === 'all' \? 'selected' : ''\}>Show All Patterns</option>
                                <option value="filtered_only" \$\{candlestickFilterMode === 'filtered_only' \? 'selected' : ''\}>Filtered Only \(KC\+RSI\)</option>
                            </select>
                        </div>
                        <div>
                            <label class="block text-sm font-medium mb-1">Selected: \$\{candlestickSelectedStocks\.length\}</label>
                            <div class="flex gap-2">
                                <button onclick="candlestickSelectedStocks=\[\.\.\.document\.querySelectorAll\('#candlestick-stocks input:checked'\)\]\.map\(e=>e\.value\); toast\('Selected '\+candlestickSelectedStocks\.length\+' stocks'\)"
                                    class="px-3 py-2 bg-gray-700 rounded hover:bg-gray-600 text-sm">Update Selection</button>
                                <button onclick="runCandlestickScreener\(\)"
                                    class="px-4 py-2 bg-blue-600 rounded hover:bg-blue-500 font-medium \$\{candlestickLoading \? 'opacity-50' : ''\}">
                                    \$\{candlestickLoading \? '⏳ Scanning\.\.\.' : '🔍 Scan'\}
                                </button>
                            </div>
                        </div>
                    </div>'''

new_controls = '''                    <div class="grid grid-cols-1 md:grid-cols-4 gap-4 mb-4">
                        <div>
                            <label class="block text-sm font-medium mb-1">Lookback Days</label>
                            <input type="number" id="candlestick-lookback" value="${candlestickLookbackDays}" min="30" max="365"
                                onchange="candlestickLookbackDays=parseInt(this.value)"
                                class="w-full bg-[#1f2937] border border-gray-600 rounded px-3 py-2">
                        </div>
                        <div>
                            <label class="block text-sm font-medium mb-1">KC Channel Level</label>
                            <select id="candlestick-kc-level" onchange="candlestickKcLevel=parseFloat(this.value)"
                                class="w-full bg-[#1f2937] border border-gray-600 rounded px-3 py-2">
                                <option value="0" ${candlestickKcLevel === 0 ? 'selected' : ''}>KC &lt; 0 (Below Middle)</option>
                                <option value="-1" ${candlestickKcLevel === -1 ? 'selected' : ''}>KC &lt; -1 (Below Lower)</option>
                                <option value="-2" ${candlestickKcLevel === -2 ? 'selected' : ''}>KC &lt; -2 (Below Lower - ATR)</option>
                            </select>
                        </div>
                        <div>
                            <label class="block text-sm font-medium mb-1">RSI Level</label>
                            <select id="candlestick-rsi-level" onchange="candlestickRsiLevel=parseInt(this.value)"
                                class="w-full bg-[#1f2937] border border-gray-600 rounded px-3 py-2">
                                <option value="60" ${candlestickRsiLevel === 60 ? 'selected' : ''}>RSI &lt; 60</option>
                                <option value="50" ${candlestickRsiLevel === 50 ? 'selected' : ''}>RSI &lt; 50</option>
                                <option value="40" ${candlestickRsiLevel === 40 ? 'selected' : ''}>RSI &lt; 40</option>
                                <option value="30" ${candlestickRsiLevel === 30 ? 'selected' : ''}>RSI &lt; 30</option>
                            </select>
                        </div>
                        <div>
                            <label class="block text-sm font-medium mb-1">&nbsp;</label>
                            <button onclick="runCandlestickScreener()"
                                class="w-full px-4 py-2 bg-blue-600 rounded hover:bg-blue-500 font-medium ${candlestickLoading ? 'opacity-50' : ''}">
                                ${candlestickLoading ? '⏳ Scanning...' : '🔍 Scan All'}
                            </button>
                        </div>
                    </div>

                    <div class="mb-4">
                        <label class="block text-sm font-medium mb-2">Candlestick Patterns (Ctrl+Click to select multiple)</label>
                        <select id="candlestick-patterns" multiple
                            onchange="candlestickSelectedPatterns = Array.from(this.selectedOptions).map(opt => opt.value); render();"
                            class="w-full bg-[#1f2937] border border-gray-600 rounded px-3 py-2" style="height: 120px;">
                            <option value="Hammer" ${candlestickSelectedPatterns.includes('Hammer') ? 'selected' : ''}>Hammer - Small body at top, long lower shadow</option>
                            <option value="Bullish Engulfing" ${candlestickSelectedPatterns.includes('Bullish Engulfing') ? 'selected' : ''}>Bullish Engulfing - Green candle engulfs red candle</option>
                            <option value="Piercing Pattern" ${candlestickSelectedPatterns.includes('Piercing Pattern') ? 'selected' : ''}>Piercing Pattern - Closes above midpoint</option>
                            <option value="Tweezer Bottom" ${candlestickSelectedPatterns.includes('Tweezer Bottom') ? 'selected' : ''}>Tweezer Bottom - Two candles with same low</option>
                        </select>
                        <div class="text-xs text-gray-400 mt-1">Hold Ctrl/Cmd and click to select multiple patterns. Leave empty to scan all patterns.</div>
                    </div>'''

content = re.sub(old_controls, new_controls, content, flags=re.DOTALL)

print("Step 4: Updating API call...")
# Update the fetch body to include new parameters
old_fetch = r'''            const res = await fetch\('/api/v2/screener/candlestick/run', \{
                method: 'POST',
                headers: \{'Content-Type': 'application/json'\},
                body: JSON\.stringify\(\{
                    symbols: symbols,
                    lookback_days: candlestickLookbackDays,
                    market: market,
                    filter_mode: candlestickFilterMode'''

new_fetch = '''            const res = await fetch('/api/v2/screener/candlestick/run', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    symbols: symbols,
                    lookback_days: candlestickLookbackDays,
                    market: market,
                    kc_level: candlestickKcLevel,
                    rsi_level: candlestickRsiLevel,
                    selected_patterns: candlestickSelectedPatterns.length > 0 ? candlestickSelectedPatterns : null'''

content = re.sub(old_fetch, new_fetch, content, flags=re.DOTALL)

print("Writing updated template...")
with open(template_file, 'w', encoding='utf-8') as f:
    f.write(content)

print("\n[OK] UI Fix Complete!")
print("\nChanges:")
print("1. Removed candlestickFilterMode variable")
print("2. Added candlestickKcLevel (-1), candlestickRsiLevel (30), candlestickSelectedPatterns ([])")
print("3. Replaced FilterMode dropdown with KC Level dropdown")
print("4. Added RSI Level dropdown")
print("5. Added Pattern multi-select listbox")
print("6. Updated API call to send kc_level, rsi_level, selected_patterns")
print("\nRestart Flask and hard refresh browser!")
