# Elder Trading System - Additional Historical Screeners

## Overview

Two new historical screeners have been added to the Elder Trading System:

1. **Candlestick Pattern Screener** - Scans for bullish reversal patterns with indicator filters
2. **RSI + MACD Indicator Screener** - Finds oversold conditions with momentum confirmation

---

## 1. Candlestick Pattern Screener

### Patterns Detected (Long Trades Only)

Based on the rules from `Candle_Sticks.docx`:

| Pattern | Description | Reliability |
|---------|-------------|-------------|
| **Hammer** | Small body at top, long lower shadow (≥2x body), no/very short upper shadow | 3/5 |
| **Bullish Engulfing** | Current bullish candle engulfs previous bearish candle | 4/5 |
| **Piercing Pattern** | White candle pushes >50% into prior black body | 3/5 |
| **Tweezer Bottom** | Two successive candles with same lows | 2-3/5 |

### Filter Conditions

All filters must match for the stock to appear in "Filtered Only" mode:

- ✅ **Pattern detected** on daily candle
- ✅ **Price below Keltner Channel Lower** (KC-1 with params 20, 10, 1)
- ✅ **RSI(14) < 30** (Oversold)

### Filter Modes

| Mode | Description |
|------|-------------|
| `all` | Show all patterns with indicator values (default) |
| `filtered_only` | Only show patterns matching KC and RSI filters |
| `patterns_only` | Only show patterns, no filter requirement |

### API Endpoints

```
GET  /api/v2/screener/candlestick/stocks
     → Get list of available stocks

POST /api/v2/screener/candlestick/run
     → Run screener across multiple stocks

GET  /api/v2/screener/candlestick/single/<symbol>
     → Scan single stock history
```

### Example Request

```json
POST /api/v2/screener/candlestick/run
{
    "symbols": ["AAPL", "MSFT", "GOOGL"],
    "lookback_days": 180,
    "market": "US",
    "filter_mode": "filtered_only"
}
```

### Example Response

```json
{
    "signals": [
        {
            "symbol": "AAPL",
            "date": "2024-10-15",
            "patterns": ["Hammer", "Tweezer Bottom"],
            "close": 185.50,
            "rsi": 28.5,
            "kc_lower": 182.30,
            "kc_middle": 190.20,
            "kc_upper": 198.10,
            "macd": -0.52,
            "below_kc_lower": true,
            "rsi_oversold": true,
            "filters_match": true
        }
    ],
    "summary": {
        "total_signals": 45,
        "filtered_signals": 12,
        "patterns_breakdown": {
            "Hammer": 25,
            "Bullish Engulfing": 15,
            "Piercing Pattern": 3,
            "Tweezer Bottom": 2
        }
    }
}
```

---

## 2. RSI + MACD Indicator Screener

### Filter Conditions

ALL conditions must be TRUE:

1. ✅ **RSI(14) < 30** (Oversold)
2. ✅ **RSI is Increasing** (today > yesterday)
3. ✅ **MACD pointing up OR crossing up**
   - "Pointing Up" = MACD Histogram increasing (today > yesterday)
   - "Crossing Up" = MACD Line crosses above Signal Line

### Signal Types

| Type | Strength | Description |
|------|----------|-------------|
| **RSI Oversold + MACD Crossover** | Strong 🔥 | MACD line crossed above signal line |
| **RSI Oversold + MACD Rising** | Medium | MACD histogram is increasing |

### API Endpoints

```
GET  /api/v2/screener/rsi-macd/stocks
     → Get list of available stocks

POST /api/v2/screener/rsi-macd/run
     → Run screener across multiple stocks

GET  /api/v2/screener/rsi-macd/single/<symbol>
     → Scan single stock history
```

### Example Request

```json
POST /api/v2/screener/rsi-macd/run
{
    "symbols": ["AAPL", "MSFT", "GOOGL", "AMZN"],
    "lookback_days": 180,
    "market": "US"
}
```

### Example Response

```json
{
    "signals": [
        {
            "symbol": "NVDA",
            "date": "2024-10-22",
            "signal_type": "RSI Oversold + MACD Crossover",
            "close": 118.50,
            "rsi": 25.3,
            "rsi_prev": 23.8,
            "rsi_change": 1.5,
            "macd": -1.25,
            "macd_signal": -1.30,
            "macd_hist": 0.05,
            "macd_hist_prev": -0.10,
            "macd_hist_change": 0.15,
            "stoch_k": 18.5,
            "conditions": {
                "rsi_oversold": true,
                "rsi_increasing": true,
                "macd_pointing_up": true,
                "macd_crossing_up": true
            }
        }
    ],
    "summary": {
        "total_signals": 28,
        "crossover_signals": 8,
        "rising_signals": 20,
        "avg_rsi_at_signal": 26.4
    }
}
```

---

## Integration with Main App

### 1. Register Routes in app.py

Add the following to your `app.py`:

```python
from routes.screener_api import screener_routes

# Register the new screener routes
app.register_blueprint(screener_routes, url_prefix='/api/v2/screener')
```

### 2. Add Template Route (Optional)

To serve the standalone UI:

```python
@app.route('/screeners')
def additional_screeners():
    return render_template('additional_screeners.html')
```

---

## File Structure

```
backend/
├── services/
│   ├── candlestick_screener.py    # Candlestick pattern detection
│   └── rsi_macd_screener.py       # RSI + MACD indicator logic
├── routes/
│   └── screener_api.py            # API endpoints for both screeners
└── templates/
    └── additional_screeners.html  # React UI for screeners
```

---

## Indicator Values Displayed

Both screeners show the following indicators in the results grid:

| Indicator | Description | Parameters |
|-----------|-------------|------------|
| RSI | Relative Strength Index | Period: 14 |
| MACD Line | MACD main line | 12, 26, 9 |
| MACD Signal | MACD signal line | 12, 26, 9 |
| MACD Histogram | MACD histogram | 12, 26, 9 |
| Stochastic K | Stochastic oscillator | 14, 3 |
| KC Lower | Keltner Channel lower band | 20, 10, 1 |
| KC Middle | Keltner Channel middle (EMA) | 20, 10, 1 |
| KC Upper | Keltner Channel upper band | 20, 10, 1 |
| EMA 20 | 20-period Exponential MA | Period: 20 |
| Force Index | 2-EMA of Force Index | Period: 2 |

---

## Stock Universe

| Market | Count | Stocks |
|--------|-------|--------|
| US | 100 | NASDAQ 100 |
| India | 100 | NIFTY 50 + NIFTY NEXT 50 |

---

## Usage Notes

1. **Lookback Days**: Range from 30 to 365 days
2. **Data Source**: Uses yfinance for historical data
3. **Results Limit**: UI shows first 100 signals, API returns all
4. **Column Selection**: Toggle which indicators to display in the grid
5. **Sort Order**: Results sorted by date (newest first)

---

## Quick Start

```bash
# 1. Install dependencies
pip install flask yfinance pandas numpy

# 2. Run the app
cd backend
python app.py

# 3. Access the screeners
# API: http://localhost:5000/api/v2/screener/info
# UI:  http://localhost:5000/screeners
```

---

*Elder Trading System - Additional Historical Screeners v1.0*
