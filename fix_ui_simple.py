#!/usr/bin/env python3
"""
Simple line-by-line replacement to fix UI
"""

template_file = "C:/Naren/Working/Source/GitHubRepo/Claude_Trade/elder-trading-ibkr/backend/templates/index.html"

print("Reading file...")
with open(template_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

print("Processing lines...")
output_lines = []
i = 0
while i < len(lines):
    line = lines[i]

    # Replace Filter Mode section (lines 1048-1055) with KC Level
    if i == 1047 and 'Filter Mode' in lines[1048]:
        print(f"  Replacing FilterMode at line {i+1}")
        output_lines.append(line)  # Keep line 1047
        # Replace the next 8 lines with KC Level dropdown
        output_lines.append('                        <div>\n')
        output_lines.append('                            <label class="block text-sm font-medium mb-1">KC Channel Level</label>\n')
        output_lines.append('                            <select id="candlestick-kc-level" onchange="candlestickKcLevel=parseFloat(this.value)"\n')
        output_lines.append('                                class="w-full bg-[#1f2937] border border-gray-600 rounded px-3 py-2">\n')
        output_lines.append('                                <option value="0" ${candlestickKcLevel === 0 ? \'selected\' : \'\'}>KC &lt; 0 (Below Middle)</option>\n')
        output_lines.append('                                <option value="-1" ${candlestickKcLevel === -1 ? \'selected\' : \'\'}>KC &lt; -1 (Below Lower)</option>\n')
        output_lines.append('                                <option value="-2" ${candlestickKcLevel === -2 ? \'selected\' : \'\'}>KC &lt; -2 (Below Lower - ATR)</option>\n')
        output_lines.append('                            </select>\n')
        output_lines.append('                        </div>\n')
        i += 8  # Skip the old Filter Mode lines
        continue

    # Replace Selected: section (lines 1056-1066) with RSI Level dropdown
    elif i == 1055 and 'Selected:' in lines[1056]:
        print(f"  Replacing Selected section at line {i+1}")
        output_lines.append(line)  # Keep line 1055
        # Replace with RSI Level dropdown
        output_lines.append('                        <div>\n')
        output_lines.append('                            <label class="block text-sm font-medium mb-1">RSI Level</label>\n')
        output_lines.append('                            <select id="candlestick-rsi-level" onchange="candlestickRsiLevel=parseInt(this.value)"\n')
        output_lines.append('                                class="w-full bg-[#1f2937] border border-gray-600 rounded px-3 py-2">\n')
        output_lines.append('                                <option value="60" ${candlestickRsiLevel === 60 ? \'selected\' : \'\'}>RSI &lt; 60</option>\n')
        output_lines.append('                                <option value="50" ${candlestickRsiLevel === 50 ? \'selected\' : \'\'}>RSI &lt; 50</option>\n')
        output_lines.append('                                <option value="40" ${candlestickRsiLevel === 40 ? \'selected\' : \'\'}>RSI &lt; 40</option>\n')
        output_lines.append('                                <option value="30" ${candlestickRsiLevel === 30 ? \'selected\' : \'\'}>RSI &lt; 30</option>\n')
        output_lines.append('                            </select>\n')
        output_lines.append('                        </div>\n')
        i += 11  # Skip old Selected section lines
        continue

    # Add pattern listbox after the grid section ends (after line 1067)
    elif i == 1066 and '                    </div>' in line:
        output_lines.append(line)  # Keep the closing div
        output_lines.append('                    <div>\n')
        output_lines.append('                            <label class="block text-sm font-medium mb-1">&nbsp;</label>\n')
        output_lines.append('                            <button onclick="runCandlestickScreener()"\n')
        output_lines.append('                                class="w-full px-4 py-2 bg-blue-600 rounded hover:bg-blue-500 font-medium ${candlestickLoading ? \'opacity-50\' : \'\'}">\n')
        output_lines.append('                                ${candlestickLoading ? \'⏳ Scanning...\' : \'🔍 Scan All\'}\n')
        output_lines.append('                            </button>\n')
        output_lines.append('                        </div>\n')
        output_lines.append('                    </div>\n')
        output_lines.append('\n')
        output_lines.append('                    <div class="mb-4">\n')
        output_lines.append('                        <label class="block text-sm font-medium mb-2">Candlestick Patterns (Ctrl+Click to select multiple)</label>\n')
        output_lines.append('                        <select id="candlestick-patterns" multiple\n')
        output_lines.append('                            onchange="candlestickSelectedPatterns = Array.from(this.selectedOptions).map(opt => opt.value); render();"\n')
        output_lines.append('                            class="w-full bg-[#1f2937] border border-gray-600 rounded px-3 py-2" style="height: 120px;">\n')
        output_lines.append('                            <option value="Hammer" ${candlestickSelectedPatterns.includes(\'Hammer\') ? \'selected\' : \'\'}>Hammer - Small body at top, long lower shadow</option>\n')
        output_lines.append('                            <option value="Bullish Engulfing" ${candlestickSelectedPatterns.includes(\'Bullish Engulfing\') ? \'selected\' : \'\'}>Bullish Engulfing - Green candle engulfs red candle</option>\n')
        output_lines.append('                            <option value="Piercing Pattern" ${candlestickSelectedPatterns.includes(\'Piercing Pattern\') ? \'selected\' : \'\'}>Piercing Pattern - Closes above midpoint</option>\n')
        output_lines.append('                            <option value="Tweezer Bottom" ${candlestickSelectedPatterns.includes(\'Tweezer Bottom\') ? \'selected\' : \'\'}>Tweezer Bottom - Two candles with same low</option>\n')
        output_lines.append('                        </select>\n')
        output_lines.append('                        <div class="text-xs text-gray-400 mt-1">Hold Ctrl/Cmd and click to select multiple patterns. Leave empty to scan all patterns.</div>\n')
        output_lines.append('                    </div>\n')
        i += 1
        continue

    # Skip the stock selection section (lines 1068-1078) completely
    elif i == 1067 and 'border border-gray-700 rounded p-2' in lines[1068]:
        print(f"  Removing stock selection section at line {i+1}")
        i += 12  # Skip the entire stock selection div
        continue

    # Update API call line
    elif 'filter_mode: candlestickFilterMode' in line:
        print(f"  Updating API call at line {i+1}")
        output_lines.append('                    kc_level: candlestickKcLevel,\n')
        output_lines.append('                    rsi_level: candlestickRsiLevel,\n')
        output_lines.append('                    selected_patterns: candlestickSelectedPatterns.length > 0 ? candlestickSelectedPatterns : null\n')
        i += 1
        continue

    else:
        output_lines.append(line)

    i += 1

print("Writing file...")
with open(template_file, 'w', encoding='utf-8') as f:
    f.writelines(output_lines)

print("\n[OK] Fixed!")
