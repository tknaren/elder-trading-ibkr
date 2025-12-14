# Quick Start - Updated Candlestick Screener

## 🚀 To See the Changes

1. **Restart Flask Server**
   ```bash
   cd C:/Naren/Working/Source/GitHubRepo/Claude_Trade/elder-trading-ibkr/backend
   python app.py
   ```

2. **Hard Refresh Browser**
   - Windows: `Ctrl + Shift + R`
   - Mac: `Cmd + Shift + R`

3. **Navigate to Candlestick Tab**

---

## 🎯 What You'll See

### New UI Layout

```
┌────────────────────────────────────────────────────────┐
│ 🕯️ Candlestick Pattern Screener                       │
├────────────────────────────────────────────────────────┤
│ Patterns: Select from listbox below                    │
│ Filters: KC Level + RSI Level                          │
├────────────────────────────────────────────────────────┤
│ ┌──────────┬───────────┬──────────┬──────────────────┐ │
│ │ Lookback │ KC Level  │ RSI Level│    Scan Button   │ │
│ │  [180]   │ [KC<-1 ▼] │ [RSI<30▼]│   [🔍 Scan All]  │ │
│ └──────────┴───────────┴──────────┴──────────────────┘ │
│                                                         │
│ Candlestick Patterns (Ctrl+Click to select multiple)   │
│ ┌─────────────────────────────────────────────────────┐│
│ │ Hammer - Small body at top, long lower shadow       ││
│ │ Bullish Engulfing - Green candle engulfs red candle││
│ │ Piercing Pattern - Closes above midpoint            ││
│ │ Tweezer Bottom - Two candles with same low          ││
│ └─────────────────────────────────────────────────────┘│
└────────────────────────────────────────────────────────┘
```

---

## 📋 Changes from Previous Version

### ❌ Removed
- **Filter Mode** dropdown (was: All / Filtered Only)

### ✅ Added
- **RSI Level** dropdown (RSI < 60 / 50 / 40 / 30)
- **Multi-Select** listbox for patterns (instead of checkboxes)

### 🔄 Changed
- Pattern selection is now a listbox (Ctrl+Click to select multiple)
- RSI threshold is now configurable instead of fixed at 30

---

## 💡 How to Use

### Basic Scan
1. Leave defaults (KC < -1, RSI < 30, All patterns)
2. Click "Scan All"
3. View results

### Custom RSI Threshold
1. Change **RSI Level** to `RSI < 50` (more signals)
2. Or `RSI < 40` or `RSI < 60`
3. Click "Scan All"

### Select Specific Patterns
1. Click the **Pattern listbox**
2. Hold `Ctrl` (Windows) or `Cmd` (Mac)
3. Click multiple patterns to select
4. Click "Scan All"

---

## 🎮 Example Workflows

### Conservative: Find Only Hammers in Extremely Oversold Stocks
```
Settings:
- KC Level: KC < -2 (Below Lower - ATR)
- RSI Level: RSI < 30
- Patterns: Select "Hammer" only

Click: Scan All
```

### Moderate: Find Any Pattern When Slightly Oversold
```
Settings:
- KC Level: KC < -1 (Below Lower)
- RSI Level: RSI < 50
- Patterns: Leave empty (all patterns)

Click: Scan All
```

### Aggressive: Find Engulfing Patterns Early
```
Settings:
- KC Level: KC < 0 (Below Middle)
- RSI Level: RSI < 60
- Patterns: Select "Bullish Engulfing" only

Click: Scan All
```

---

## 🔍 What Each Setting Means

### RSI Level
- **RSI < 60**: Catches stocks early (more signals, less confirmation)
- **RSI < 50**: Moderate oversold
- **RSI < 40**: More oversold
- **RSI < 30**: Very oversold (default, most conservative)

### KC Level
- **KC < 0**: Price below middle band (aggressive, catches early dips)
- **KC < -1**: Price below lower band (balanced, default)
- **KC < -2**: Price below lower band minus ATR (conservative, extreme oversold)

### Pattern Selection
- **Empty**: Scans for all 4 patterns
- **1 selected**: Only shows that pattern
- **Multiple selected**: Shows any of the selected patterns

---

## ✅ Quick Test

1. Open the app
2. Go to Candlestick tab
3. You should see:
   - ✅ RSI Level dropdown (column 3)
   - ✅ Pattern listbox (below the controls)
   - ❌ NO Filter Mode dropdown

If you don't see these, try:
1. Hard refresh (Ctrl+Shift+R)
2. Check Flask server is running
3. Clear browser cache completely

---

## 📚 More Info

- **Full API Docs**: `docs/CANDLESTICK_SCREENER_API.md`
- **Update Summary**: `docs/CANDLESTICK_UPDATE_SUMMARY.md`
- **UI Changes**: `docs/CANDLESTICK_UI_CHANGES.md`

---

**Last Updated**: 2024-12-14
**Version**: 2.1
