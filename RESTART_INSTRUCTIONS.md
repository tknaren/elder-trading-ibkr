# UI Not Updating? Follow These Steps

## The changes are complete, but you need to refresh:

### Option 1: Hard Refresh Browser (Quickest)
1. **Windows/Linux:** Press `Ctrl + Shift + R` or `Ctrl + F5`
2. **Mac:** Press `Cmd + Shift + R`

This clears the browser cache and reloads the page completely.

### Option 2: Restart Flask Server
1. Stop the Flask server (Ctrl+C in terminal)
2. Start it again:
   ```bash
   python backend/app.py
   ```
   or
   ```bash
   ./run.sh
   ```

### Option 3: Clear Browser Cache Manually
1. Open Developer Tools (F12)
2. Right-click the refresh button
3. Select "Empty Cache and Hard Reload"

### Option 4: Incognito/Private Window
Open the app in an incognito/private browser window to bypass cache entirely.

---

## What You Should See After Refresh:

In the **Candlestick** tab, you should now see:

```
┌─────────────────────────────────────────────────────────┐
│ Lookback Days | KC Channel Level | Filter Mode | Stocks │
├─────────────────────────────────────────────────────────┤
│ Candlestick Patterns                                    │
│ ☐ Hammer  ☐ Bullish Engulfing  ☐ Piercing  ☐ Tweezer  │
└─────────────────────────────────────────────────────────┘
```

The **KC Channel Level** dropdown should have 3 options:
- KC < 0 (Below Middle)
- KC < -1 (Below Lower)  ← default
- KC < -2 (Below Lower - ATR)

---

## Troubleshooting

**Still not seeing changes?**

1. Check the file location:
   ```
   backend/templates/index.html
   ```

2. Verify the Flask server is running from the correct directory

3. Check browser console (F12) for any JavaScript errors

4. Try a different browser

---

## Need to verify the changes are in the file?

Run this command to confirm:
```bash
grep -n "candlestickKcLevel" backend/templates/index.html
```

You should see multiple lines with the new variable.
