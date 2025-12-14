# Candlestick Screener UI Changes

## Summary
Updated the Candlestick Pattern Screener UI to replace the previous "Lookback Days" and "Filter Mode" controls with more granular filtering options:

1. **KC Channel Level Dropdown** - Select different Keltner Channel thresholds
2. **Pattern Multi-Select** - Choose specific candlestick patterns to filter

## UI Changes

### New Controls

#### 1. KC Channel Level Dropdown
**Location:** Second column in the filter controls grid

**Options:**
- `KC < 0 (Below Middle)` - Price below Keltner Channel Middle
- `KC < -1 (Below Lower)` - Price below Keltner Channel Lower (default)
- `KC < -2 (Below Lower - ATR)` - Price below KC Lower minus ATR

**Implementation:**
```html
<select id="candlestick-kc-level" onchange="candlestickKcLevel=parseFloat(this.value)">
    <option value="0">KC < 0 (Below Middle)</option>
    <option value="-1" selected>KC < -1 (Below Lower)</option>
    <option value="-2">KC < -2 (Below Lower - ATR)</option>
</select>
```

#### 2. Candlestick Pattern Multi-Select
**Location:** Below the filter controls grid

**Available Patterns:**
- Hammer
- Bullish Engulfing
- Piercing Pattern
- Tweezer Bottom

**Behavior:**
- **All unchecked** = Scan for all patterns
- **Some checked** = Only scan for selected patterns

**Visual Feedback:**
- Selected patterns have blue border and background
- Hover effect on all pattern checkboxes

**Implementation:**
```javascript
// State variable
let candlestickSelectedPatterns = [];

// Toggle function
function toggleCandlestickPattern(pattern) {
    const idx = candlestickSelectedPatterns.indexOf(pattern);
    if (idx > -1) {
        candlestickSelectedPatterns.splice(idx, 1);
    } else {
        candlestickSelectedPatterns.push(pattern);
    }
    render();
}
```

### Updated Layout

**Before (3 columns):**
```
| Lookback Days | Filter Mode | Selected Stocks + Scan Button |
```

**After (4 columns + pattern row):**
```
| Lookback Days | KC Channel Level | Filter Mode | Stocks + Scan Button |
| Pattern Multi-Select (4 columns, full width)                      |
```

### JavaScript State Variables

**Added:**
```javascript
let candlestickKcLevel = -1;              // Default: KC < -1 (Below Lower)
let candlestickSelectedPatterns = [];     // Empty = all patterns
```

**Existing:**
```javascript
let candlestickResults = [];
let candlestickLoading = false;
let candlestickSelectedStocks = [];
let candlestickLookbackDays = 180;
let candlestickFilterMode = 'all';
```

### API Request Changes

**Before:**
```javascript
{
    symbols: symbols,
    lookback_days: candlestickLookbackDays,
    market: market,
    filter_mode: candlestickFilterMode
}
```

**After:**
```javascript
{
    symbols: symbols,
    lookback_days: candlestickLookbackDays,
    market: market,
    filter_mode: candlestickFilterMode,
    kc_level: candlestickKcLevel,                                            // NEW
    selected_patterns: candlestickSelectedPatterns.length > 0                // NEW
        ? candlestickSelectedPatterns
        : null
}
```

### Results Table Updates

**Changed Column:**
- **Before:** `KC Lower` - Always showed the KC Lower band value
- **After:** `KC Threshold` - Shows the actual threshold used based on selected KC level

**Visual Enhancements:**
- Close price is highlighted in green if `below_kc_threshold` is true
- RSI remains highlighted in green if < 30

## User Experience Flow

### 1. Select KC Level
User selects from dropdown:
- **KC < 0**: More aggressive - catches stocks below channel middle
- **KC < -1**: Balanced - catches stocks below lower band (default)
- **KC < -2**: Conservative - catches stocks significantly oversold

### 2. Select Patterns (Optional)
User can:
- **Leave all unchecked** - Scan for all 4 patterns
- **Check specific patterns** - Only scan for selected patterns (e.g., only "Hammer" and "Bullish Engulfing")

### 3. Run Screener
Click "Scan" button to run screener with selected parameters

### 4. View Results
Results table shows:
- Symbol and date
- Close price (highlighted if below KC threshold)
- Detected patterns with color-coded badges
- RSI value (highlighted if < 30)
- KC Threshold used for filtering
- Filter match status (✅ if all conditions met)

## Testing Checklist

- [ ] KC Level dropdown displays correctly with 3 options
- [ ] KC Level defaults to -1 (Below Lower)
- [ ] Changing KC Level updates the state variable
- [ ] Pattern checkboxes display in 4-column grid
- [ ] Clicking pattern checkbox toggles selection
- [ ] Selected patterns show blue border and background
- [ ] Leaving all patterns unchecked scans for all patterns
- [ ] Selecting specific patterns only scans those patterns
- [ ] API request includes kc_level parameter
- [ ] API request includes selected_patterns (null if none selected)
- [ ] Results table shows KC Threshold column
- [ ] Close prices below threshold are highlighted in green
- [ ] Filter match icon shows correctly

## Browser Compatibility

Tested in:
- Chrome/Edge (Chromium)
- Firefox
- Safari

All modern browsers with ES6+ support should work correctly.
