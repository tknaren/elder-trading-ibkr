# Candlestick Screener API Documentation

## Overview
The Candlestick Screener API has been updated to support:
1. **KC Channel Level Selection** - Choose different Keltner Channel thresholds
2. **Pattern Multi-Select** - Filter by specific candlestick patterns

## Available Endpoints

### 1. Get Screener Options
**GET** `/api/v2/screener/candlestick/options`

Returns available options for the UI (patterns, KC levels, filter modes).

**Response:**
```json
{
  "patterns": [
    {
      "name": "Hammer",
      "description": "Small body at top, long lower shadow (2x+ body)"
    },
    {
      "name": "Bullish Engulfing",
      "description": "Green candle completely engulfs previous red candle"
    },
    {
      "name": "Piercing Pattern",
      "description": "Green candle opens below prev low, closes above midpoint"
    },
    {
      "name": "Tweezer Bottom",
      "description": "Two candles with same low - support confirmed"
    }
  ],
  "kc_levels": [
    {
      "value": 0,
      "label": "KC < 0 (Below Middle)",
      "description": "Price below Keltner Channel Middle"
    },
    {
      "value": -1,
      "label": "KC < -1 (Below Lower)",
      "description": "Price below Keltner Channel Lower"
    },
    {
      "value": -2,
      "label": "KC < -2 (Below Lower - ATR)",
      "description": "Price below KC Lower minus ATR"
    }
  ],
  "filter_modes": [
    {
      "value": "all",
      "label": "All Patterns",
      "description": "Show all patterns with indicator values"
    },
    {
      "value": "filtered_only",
      "label": "Filtered Only",
      "description": "Only show patterns matching KC and RSI filters"
    },
    {
      "value": "patterns_only",
      "label": "Patterns Only",
      "description": "Only show patterns, no filter requirement"
    }
  ]
}
```

### 2. Run Screener
**POST** `/api/v2/screener/candlestick/run`

**Request Body:**
```json
{
  "symbols": ["AAPL", "MSFT", "GOOGL"] or "all",
  "lookback_days": 180,
  "market": "US",
  "filter_mode": "filtered_only",
  "kc_level": -1,
  "selected_patterns": ["Hammer", "Bullish Engulfing"]
}
```

**Parameters:**
- `symbols` (array|string): List of symbols or "all" for full market scan
- `lookback_days` (number, 30-365): Number of days to scan (default: 180)
- `market` (string): "US" or "India" (default: "US")
- `filter_mode` (string): "all" | "filtered_only" | "patterns_only" (default: "all")
- `kc_level` (number): KC channel threshold (default: -1)
  - `0`: Price < KC Middle
  - `-1`: Price < KC Lower
  - `-2`: Price < KC Lower - ATR
- `selected_patterns` (array|null): Specific patterns to filter, or null for all patterns

**Response:**
```json
{
  "signals": [
    {
      "symbol": "AAPL",
      "date": "2024-01-15",
      "patterns": ["Hammer", "Bullish Engulfing"],
      "pattern_count": 2,
      "pattern_details": [...],
      "close": 185.50,
      "rsi": 28.5,
      "kc_lower": 180.25,
      "kc_middle": 185.00,
      "kc_upper": 189.75,
      "kc_threshold": 180.25,
      "macd": -0.45,
      "macd_signal": -0.52,
      "macd_hist": 0.07,
      "stoch_k": 25.3,
      "below_kc_threshold": false,
      "rsi_oversold": true,
      "filters_match": false
    }
  ],
  "summary": {
    "total_signals": 150,
    "filtered_signals": 45,
    "patterns_breakdown": {
      "Hammer": 50,
      "Bullish Engulfing": 60,
      "Piercing Pattern": 30,
      "Tweezer Bottom": 10
    },
    "symbols_with_patterns": 80
  },
  "symbols_scanned": 100,
  "lookback_days": 180,
  "filter_mode": "filtered_only",
  "kc_level": -1,
  "selected_patterns": ["Hammer", "Bullish Engulfing"]
}
```

### 3. Scan Single Stock
**GET** `/api/v2/screener/candlestick/single/<symbol>`

**Query Parameters:**
- `lookback_days` (number, 30-365): Number of days to scan (default: 180)
- `filter_mode` (string): "all" | "filtered_only" | "patterns_only" (default: "all")
- `kc_level` (number): KC channel threshold (default: -1)
- `selected_patterns` (string): Comma-separated pattern names (e.g., "Hammer,Bullish Engulfing")

**Example:**
```
GET /api/v2/screener/candlestick/single/AAPL?lookback_days=180&kc_level=-1&selected_patterns=Hammer,Bullish%20Engulfing
```

**Response:**
```json
{
  "symbol": "AAPL",
  "signals": [...],
  "count": 15,
  "lookback_days": 180,
  "filter_mode": "all",
  "kc_level": -1,
  "selected_patterns": ["Hammer", "Bullish Engulfing"],
  "patterns_found": ["Hammer", "Bullish Engulfing", "Piercing Pattern"]
}
```

## UI Implementation Guide

### 1. KC Channel Level Dropdown
Create a dropdown with the following options:

```javascript
const kcLevels = [
  { value: 0, label: 'KC < 0 (Below Middle)' },
  { value: -1, label: 'KC < -1 (Below Lower)' },  // Default
  { value: -2, label: 'KC < -2 (Below Lower - ATR)' }
];
```

### 2. Candlestick Pattern Multi-Select
Create a multi-select with checkboxes:

```javascript
const patterns = [
  { name: 'Hammer', description: 'Small body at top, long lower shadow' },
  { name: 'Bullish Engulfing', description: 'Green candle engulfs red candle' },
  { name: 'Piercing Pattern', description: 'Closes above midpoint of prev candle' },
  { name: 'Tweezer Bottom', description: 'Two candles with same low' }
];
```

### 3. Example UI State
```javascript
const [screenerParams, setScreenerParams] = useState({
  symbols: 'all',
  lookback_days: 180,
  market: 'US',
  filter_mode: 'filtered_only',
  kc_level: -1,
  selected_patterns: null  // null = all patterns
});

// When user selects specific patterns
const handlePatternChange = (selectedPatterns) => {
  setScreenerParams({
    ...screenerParams,
    selected_patterns: selectedPatterns.length > 0 ? selectedPatterns : null
  });
};

// When user changes KC level
const handleKcLevelChange = (level) => {
  setScreenerParams({
    ...screenerParams,
    kc_level: level
  });
};
```

### 4. Example API Call
```javascript
const runScreener = async () => {
  const response = await fetch('/api/v2/screener/candlestick/run', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(screenerParams)
  });

  const data = await response.json();
  return data;
};
```

## Migration Notes

### Before (Old Parameters)
```json
{
  "lookback_days": 180,
  "filter_mode": "all"
}
```

### After (New Parameters)
```json
{
  "lookback_days": 180,
  "filter_mode": "filtered_only",
  "kc_level": -1,
  "selected_patterns": ["Hammer", "Bullish Engulfing"]
}
```

**Backward Compatibility:**
- If `kc_level` is not provided, defaults to `-1` (KC Lower)
- If `selected_patterns` is not provided or is `null`, all patterns are scanned
- Old API calls will continue to work with default values

## Filter Logic

### KC Level Filtering
The KC threshold is calculated based on `kc_level`:

- **kc_level = 0**: Threshold = KC Middle
- **kc_level = -1**: Threshold = KC Lower (default)
- **kc_level = -2**: Threshold = KC Lower - ATR
- **Other values**: Threshold = KC Middle + (kc_level × ATR)

### Pattern Filtering
- If `selected_patterns` is `null` or empty: All patterns are detected
- If `selected_patterns` contains specific patterns: Only those patterns are detected

### Combined Filters
For a signal to have `filters_match: true`:
1. Price must be below the KC threshold
2. RSI must be < 30 (oversold)
3. At least one selected pattern must be detected

## Response Fields

### Signal Object
- `symbol`: Stock ticker
- `date`: Signal date (YYYY-MM-DD)
- `patterns`: Array of detected pattern names
- `pattern_count`: Number of patterns detected
- `pattern_details`: Detailed pattern information
- `close`: Closing price
- `rsi`: RSI(14) value
- `kc_lower`, `kc_middle`, `kc_upper`: Keltner Channel values
- `kc_threshold`: The actual threshold used based on kc_level
- `macd`, `macd_signal`, `macd_hist`: MACD indicator values
- `stoch_k`: Stochastic K value
- `below_kc_threshold`: Boolean - price is below the KC threshold
- `rsi_oversold`: Boolean - RSI < 30
- `filters_match`: Boolean - all filter conditions met
