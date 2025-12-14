#!/usr/bin/env python3
"""
Script to update Candlestick Screener UI
Updates:
1. Remove FilterMode dropdown
2. Add RSI Level dropdown (60/50/40/30)
3. Change pattern checkboxes to multi-select listbox
4. Update API call
"""

import re

# Path to the file
file_path = "C:/Naren/Working/Source/GitHubRepo/Claude_Trade/elder-trading-ibkr/backend/templates/index.html"

print(f"Reading {file_path}...")
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Step 1: Remove candlestickFilterMode variable
print("Step 1: Removing candlestickFilterMode variable...")
content = re.sub(
    r"    let candlestickFilterMode = 'all';\n",
    "",
    content
)

# Step 2: Add candlestickRsiLevel variable after candlestickKcLevel
print("Step 2: Adding candlestickRsiLevel variable...")
content = re.sub(
    r"(    let candlestickKcLevel = -1;)\n",
    r"\1\n    let candlestickRsiLevel = 30;\n",
    content
)

# Step 3: Update the info box
print("Step 3: Updating info box...")
old_info = '''                <div class="bg-blue-500/10 border border-blue-500/30 rounded-xl p-4 text-sm">
                    <strong>Patterns Available:</strong> Hammer, Bullish Engulfing, Piercing Pattern, Tweezer Bottom<br>
                    <strong>Filters:</strong> Configurable KC Level + RSI(14) &lt; 30
                </div>'''

new_info = '''                <div class="bg-blue-500/10 border border-blue-500/30 rounded-xl p-4 text-sm">
                    <strong>Patterns:</strong> Select from listbox below<br>
                    <strong>Filters:</strong> KC Level + RSI Level
                </div>'''

content = content.replace(old_info, new_info)

# Step 4: Replace the grid section (remove FilterMode, add RSI Level)
print("Step 4: Replacing filter controls grid...")
old_grid = '''                    <div class="grid grid-cols-1 md:grid-cols-4 gap-4 mb-4">
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
                            <label class="block text-sm font-medium mb-1">Filter Mode</label>
                            <select id="candlestick-filter" onchange="candlestickFilterMode=this.value"
                                class="w-full bg-[#1f2937] border border-gray-600 rounded px-3 py-2">
                                <option value="all" ${candlestickFilterMode === 'all' ? 'selected' : ''}>Show All Patterns</option>
                                <option value="filtered_only" ${candlestickFilterMode === 'filtered_only' ? 'selected' : ''}>Filtered Only (KC+RSI)</option>
                            </select>
                        </div>
                        <div>
                            <label class="block text-sm font-medium mb-1">Auto-Scan All Stocks</label>
                            <div class="flex gap-2 items-center">
                                <div class="text-sm text-gray-400">📊 Will scan all stocks from CSV</div>
                                <button onclick="runCandlestickScreener()"
                                    class="px-4 py-2 bg-blue-600 rounded hover:bg-blue-500 font-medium ${candlestickLoading ? 'opacity-50' : ''}">
                                    ${candlestickLoading ? '⏳ Scanning...' : '🔍 Scan All'}
                                </button>
                            </div>
                        </div>
                    </div>'''

new_grid = '''                    <div class="grid grid-cols-1 md:grid-cols-4 gap-4 mb-4">
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

content = content.replace(old_grid, new_grid)

# Step 5: Update API call - remove filter_mode, add rsi_level
print("Step 5: Updating API call...")
old_api_call = '''                body: JSON.stringify({
                    symbols: 'all',
                    lookback_days: candlestickLookbackDays,
                    market: market,
                    filter_mode: candlestickFilterMode,
                    kc_level: candlestickKcLevel,
                    selected_patterns: candlestickSelectedPatterns.length > 0 ? candlestickSelectedPatterns : null
                })'''

new_api_call = '''                body: JSON.stringify({
                    symbols: 'all',
                    lookback_days: candlestickLookbackDays,
                    market: market,
                    kc_level: candlestickKcLevel,
                    rsi_level: candlestickRsiLevel,
                    selected_patterns: candlestickSelectedPatterns.length > 0 ? candlestickSelectedPatterns : null
                })'''

content = content.replace(old_api_call, new_api_call)

# Write the updated content
print(f"Writing updated content to {file_path}...")
with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ UI Update Complete!")
print("\nChanges made:")
print("1. ✅ Removed FilterMode dropdown")
print("2. ✅ Added RSI Level dropdown (60/50/40/30)")
print("3. ✅ Converted pattern checkboxes to multi-select listbox")
print("4. ✅ Updated API call to include rsi_level")
print("\nNext: Run this script with: python update_candlestick_ui.py")
