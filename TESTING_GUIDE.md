# 🧪 Testing Guide - All Fixed Features

## Overview

This guide walks through testing all the fixed issues end-to-end.

---

## Test 1: Screener Shows All 100 Stocks

### Steps

1. Go to **🔍 Screener** tab
2. Click **🔍 Run Weekly Scan** button
3. Wait for scan to complete

### Expected Results

- ✅ "Total" card shows **100** (not 30)
- ✅ Table displays all 100 NASDAQ stocks
- ✅ All stocks have their indicator values calculated
- ✅ Stocks are sorted by signal strength (highest first)

### What to Look For

```
┌─────────────────────────────────────────┐
│ Summary Cards at Top                    │
├─────────────────────────────────────────┤
│ ⭐ A-Trades: X    │ Scores: Y           │
│ B-Trades: Z       │ Watch: W            │
│ Avoid: V          │ Total: 100 ✓        │
└─────────────────────────────────────────┘
```

---

## Test 2: Add Watchlist Feature

### Steps

1. From Screener, click **Details** on any stock (e.g., AAPL)
2. Click **👀 Add to Watchlist** button
3. See toast confirmation message
4. Close modal (click X or outside)

### Expected Results

- ✅ Toast shows "Added AAPL to watchlist!" (green)
- ✅ No error messages
- ✅ Stock is now saved in database

### Verify Persistence

1. Refresh the page (F5)
2. Open Screener again and run scan
3. The stock you added is still marked in watchlist

---

## Test 3: Create Trade Bill from Screener

### Steps

1. From Screener, click **Details** on AAPL (or any stock)
2. Click **💼 Create Trade Bill** button
3. Should automatically go to Trade Bill tab

### Expected Results

- ✅ Automatically switches to 💼 Trade Bill tab
- ✅ Ticker field is pre-filled with **AAPL**
- ✅ CMP (Current Market Price) is filled with stock price
- ✅ Upper Channel: ~5% above price
- ✅ Lower Channel: ~5% below price

### Example Pre-Fill

```
If AAPL price is $150:
✓ Ticker: AAPL
✓ CMP: 150.25
✓ Upper Channel: 157.50  (approx)
✓ Lower Channel: 142.50  (approx)
```

---

## Test 4: Trade Bill Typeahead Search

### Steps

1. Go to 💼 Trade Bill tab
2. Click on **Ticker** field
3. Start typing: "APP" (for Apple)
4. See dropdown with suggestions

### Expected Results

- ✅ Dropdown shows up to 8 matching stocks
- ✅ Shows: Symbol, Company Name, Current Price
- ✅ Clicking any stock auto-fills ticker field

### Example Dropdown

```
AAPL        (highlighted on hover)
Apple Inc.  
$150.25

APLE        (if it exists)
...         (more matches)
```

---

## Test 5: Trade Bill Calculation & Save

### Complete Workflow

#### Step 1: Fill Required Fields

```
Ticker:        AAPL (via typeahead)
Entry:         149.50
Stop Loss:     145.00
Target:        160.00
Quantity:      10
```

#### Step 2: Click "🔢 Calculate Metrics"

```
Expected auto-filled fields:
✓ Risk/Share: 4.50 (|Entry - SL|)
✓ Position Size: 1,495.00 (Qty × Entry)
✓ Risk Amount: 45.00 (Qty × Risk/Share)
✓ Risk %: 0.75% (based on your account capital)
✓ Target Pips: 10.50 (|Target - Entry|)
✓ R:R Ratio: 2.33 (Reward/Risk)
✓ Break Even: 149.50 (= Entry)
```

#### Step 3: Add Optional Fields

```
Upper Channel: 157.50
Lower Channel: 142.50
Comments:      "Good setup, news at 2pm"
```

#### Step 4: Check Boxes (if applicable)

```
☐ Filled
☐ Stop Entered
☐ Target Entered
☐ Journal Entered
```

#### Step 5: Click "💾 Save Trade Bill"

```
Expected Results:
✓ Green toast: "Trade Bill for AAPL saved!"
✓ Auto-redirects to 📋 Trade Bills tab
✓ Your bill appears in the table
```

---

## Test 6: View & Edit Trade Bill

### From Trade Bills List

#### See All Bills

1. Go to 📋 Trade Bills tab
2. Your saved AAPL bill should appear in table

#### Table Columns Should Show

```
Ticker │ Entry  │ SL    │ Target │ Qty │ Risk  │ R:R │ Status │ Actions
AAPL   │ $149.5 │ $145  │ $160   │ 10  │ $45   │ 2.33│ active │ Edit Delete
```

#### Edit a Bill

1. Click **Edit** button on any row
2. Form should pre-populate with all saved data
3. Modify any field
4. Click "💾 Save Trade Bill" (now says Update)
5. See success toast "Trade Bill for AAPL updated!"

#### Delete a Bill

1. Click **Delete** button
2. See confirmation dialog
3. Confirm delete
4. Bill disappears from list
5. See success toast "Trade Bill deleted!"

---

## Test 7: Account Dashboard

### Step

1. Go to 💰 Account tab

### Expected Display

#### Top 4 Cards

```
┌────────────────────────────────────────┐
│ 📊 Account Size    │ 🔴 Risk per Trade │
│ £6,000             │ 2.0%              │
│                    │ £120              │
├────────────────────────────────────────┤
│ 📍 Open Positions  │ 💸 Money Locked   │
│ 2                  │ £3,000            │
└────────────────────────────────────────┘
```

#### Risk Management Section

```
Max Monthly Drawdown: 6%  (£360)
[████████░░░░░░░░░░░░] 38.3% Used

Money Remaining to Risk: £222 ✓ (in green)
60% of capital available
```

#### Account Details

```
Account Name:  ISA Account
Broker:        Trading212
Market:        US
Target R:R:    2.0:1
```

#### Position Summary

```
Total Positions: 2
Money Locked: £3,000
Slots Remaining: 3 (out of 5 max)
```

---

## Test 8: End-to-End Workflow

### Complete Trading Session

```
START: 🔍 Screener
  ├─ Run Weekly Scan
  ├─ See 100 stocks analyzed
  ├─ Find a good A-Trade stock
  └─ Click Details > Create Trade Bill

MOVE TO: 💼 Trade Bill
  ├─ Form pre-filled with stock data
  ├─ Fill Entry, SL, Target, Qty
  ├─ Click Calculate Metrics
  ├─ Verify R:R ≥ 2:1 ✓
  ├─ Add any comments
  └─ Click Save Trade Bill

MOVE TO: 📋 Trade Bills
  ├─ See your bill in the table
  ├─ Verify all numbers correct
  ├─ (Optional) Edit or Delete
  └─ Multiple bills can be created

CHECK: 💰 Account
  ├─ See updated money locked
  ├─ See open position count
  ├─ Verify money remaining to risk
  └─ Monitor risk metrics

SUCCESS: Everything saved to database!
  ├─ Refresh page (F5)
  ├─ All data persists
  └─ Workflow can be repeated
```

---

## Common Issues & Solutions

### Issue: Typeahead doesn't show

**Solution:**

1. Ticker field must be focused (clicked)
2. Start typing (minimum 1 character)
3. Refresh if no results appear
4. Try typing just the first letter

### Issue: "Save Trade Bill" gives error

**Solution:**

1. Check that **Ticker, Entry, SL, Target, Qty** are all filled
2. Numbers should be valid (not text)
3. Quantity can be fractional (e.g., 10.5)
4. Check browser console (F12 → Console) for error details

### Issue: Trade Bill doesn't appear in list

**Solution:**

1. Refresh the page (F5)
2. Check if you're on the correct user/tab
3. Check browser's Network tab (F12) for API errors
4. Toast message may show error from database

### Issue: Calculations show 0 or are empty

**Solution:**

1. Click "🔢 Calculate Metrics" button
2. Must have Entry, SL, Target, Qty filled first
3. Fields should be numbers, not text
4. Upper/Lower Channel are optional (for later calculation)

---

## Success Criteria

All tests should show ✅:

- [ ] Screener shows 100 stocks
- [ ] Add to watchlist works and persists
- [ ] Create Trade Bill from screener pre-fills form
- [ ] Typeahead shows stock suggestions
- [ ] Calculate Metrics auto-fills all fields
- [ ] Save Trade Bill stores in database
- [ ] Trade Bills list displays all bills
- [ ] Edit functionality updates bills
- [ ] Delete functionality removes bills
- [ ] Account dashboard shows all metrics
- [ ] Data persists after page refresh
- [ ] Multiple bills can be created and managed

---

## Browser Tools for Testing

### Open Developer Tools (F12)

#### Network Tab

- Watch API calls: `/api/screener/weekly`, `/api/trade-bills`, etc.
- Check response status (200 = success, 400/500 = error)
- View JSON response data

#### Console Tab

- Check for JavaScript errors
- See `console.log()` messages
- Test API calls manually

#### Application Tab

- Check localStorage (though we use database now)
- See cached data

#### Example Console Test

```javascript
// Manually call an API
fetch('/api/trade-bills')
  .then(r => r.json())
  .then(data => console.log(data))
```

---

## Performance Notes

- Initial scan (~100 stocks): 3-10 seconds
- Typeahead search: <100ms
- Save Trade Bill: <500ms  
- Load Trade Bills list: <200ms
- Calculate Metrics: instant

---

## What's NOT Fixed (Future Enhancements)

- ❌ Real-time stock prices (cached from last scan)
- ❌ IBKR live order execution (backend ready, GUI not connected)
- ❌ Trade journal with P&L tracking
- ❌ Multi-user support (system is single-user for now)
- ❌ Advanced analytics and performance reports

---

## Support

If you encounter issues:

1. **Check the logs:**
   - Terminal where app runs: Python/Flask errors
   - Browser console (F12): JavaScript errors

2. **Check the database:**
   - Bills should be in: `data/elder_trading.db`
   - Table: `trade_bills`
   - Fields: ticker, entry_price, stop_loss, target_price, etc.

3. **Reset if needed:**
   - Delete `data/elder_trading.db` to start fresh
   - App will recreate with default data on next start

---
