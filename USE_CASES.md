# Use Cases & Examples

## Use Case 1: Setting Up Your Trading Account

### Scenario

You're starting with a £6,000 ISA account and want to risk 2% per trade.

### Steps

1. **Go to Settings (⚙️ tab)**
   - Account Name: "ISA Account"
   - Trading Capital: 6000
   - Currency: GBP
   - Risk % per Trade: 2.0
   - Max Monthly Drawdown: 6.0
   - Target R:R: 2.0
   - Click Save

2. **Verify in Account Dashboard**
   - Capital shows: £6,000
   - Max risk per trade: £120 (2% of £6,000)
   - Monthly limit: £360 (6% of £6,000)

3. **You're now set up to create trades!**

---

## Use Case 2: Creating a Trade from Scanner Results

### Scenario

You find AAPL looks good in the screener with strong weekly trend and good daily setup.

### Steps

1. **Go to Screener (🔍 tab)**
2. **Run Weekly Scan** button
3. **Find AAPL in results**, click "Details" button
4. **In the modal, click "Create Trade Bill" button**
   - Ticker auto-fills: AAPL
   - Current Market Price auto-fills: 150.25 (example)
   - Channels auto-calculate:
     - Upper: 157.76
     - Lower: 142.74

5. **Form opens in Trade Bill tab with pre-filled data**

6. **You add your setup:**
   - Entry Price: 149.50
   - Stop Loss: 145.00
   - Target: 160.00
   - Quantity: 10

7. **Click "Calculate Metrics"**
   - Risk per Share: £4.50
   - Position Size: £1,495.00
   - Risk Amount: £45.00
   - Risk %: 0.75%
   - Potential Gain: £105.00
   - R:R Ratio: 2.33:1 ✓ (exceeds 2:1 target)

8. **Review everything looks good, click "Save Trade Bill"**
   - Success! Bill saved to database

9. **Go to Trade Bills (📋 tab)**
   - See your new AAPL trade in the table
   - Can edit or delete if needed

---

## Use Case 3: Manual Trade Bill Creation

### Scenario

You've been watching MSFT for a while and have a specific setup in mind.

### Steps

1. **Go to Trade Bill (💼 tab)**

2. **Fill in Trade Info Section:**

   ```
   Ticker: MSFT
   Current Market Price: 420.00
   Entry Price: 418.50
   Stop Loss: 410.00
   Target: 435.00
   Quantity: 5.5
   Upper Channel: 440.00
   Lower Channel: 400.00
   ```

3. **Auto-calculated fields show:**

   ```
   Target Pips: 16.50
   Stop Loss Pips: 8.50
   ```

4. **Click "Calculate Metrics"**
   - Risk per Share: £8.50
   - Max Qty for Risk: 14.1 units
   - Position Size: £2,301.75
   - Risk Amount: £46.75
   - Risk %: 0.78%
   - Potential Gain: £90.75
   - R:R Ratio: 1.94:1 ⚠️ (slightly below 2:1 target)

5. **Adjust Quantity to 6 for better R:R:**
   - Recalculate
   - New Risk Amount: £51.00
   - New R:R: 1.96:1 ✓ (closer to target)

6. **Set optional fields:**
   - Trailing Stop: 425.00
   - Comments: "Consolidation breakout setup, watch NASDAQ 100 correlation"

7. **Check the Action boxes:**
   - (Leave unchecked for now - will check as trade progresses)

8. **Click "Save Trade Bill"**

9. **Success! Trade saved**

---

## Use Case 4: Managing Your Trade Progress

### Scenario

You've placed your AAPL trade order and want to mark it as filled.

### Steps

1. **Go to Trade Bills (📋 tab)**

2. **Find AAPL in the table**

3. **Click "Edit" button**
   - Trade Bill form loads with your data
   - Current Market Price: 150.50 (updated from market)
   - All your previous entries still there

4. **Update status checkboxes:**
   - ✓ Check "Filled" (you bought the shares)
   - ✓ Check "Stop Entered" (stop order is placed)
   - (Leave others unchecked for now)

5. **Update Current Market Price:** 150.50 (current quote)

6. **Add comment:** "Filled at 149.50, stop at 145.00"

7. **Click "Save Trade Bill"**
   - Bill updates in database
   - You see ✓ Filled status in the table

---

## Use Case 5: Tracking Multiple Concurrent Trades

### Scenario

You now have 3 active trades and want to monitor your capital usage.

### Steps

1. **Go to Account (💰 tab)**
   - Account Size: £6,000
   - Open Positions: 3
   - Money Locked: £4,500
   - Money Remaining: £500 (for 6% monthly limit)
   - Position Slots Remaining: 2 (can open 2 more max)

2. **Your Risk Summary:**

   ```
   Trade 1: AAPL    £45   risk
   Trade 2: MSFT    £51   risk
   Trade 3: GOOGL   £42   risk
   ────────────────────
   Total:           £138  risk (2.3% of capital)
   ```

3. **Remaining Monthly Capacity:**
   - Started: £360 (6% of £6,000)
   - Used: £138
   - Remaining: £222
   - % Used: 38.3%

4. **Assessment:**
   - ✓ Still have plenty of room in monthly budget
   - ⚠️ Only 2 position slots left (max 5)
   - ✓ Individual risk (0.75%, 0.85%, 0.70%) all under 2% target

5. **You can open 2 more trades before hitting position limit**

---

## Use Case 6: Evaluating Risk:Reward Ratios

### Scenario

You have a setup but want to make sure it meets your requirements.

### Three Different Setups

**Setup A: Conservative**

```
Entry:   £100
Stop:    £95 (£5 risk)
Target:  £110 (£10 gain)
Qty:     10

Risk Amount: £50
Reward: £100
R:R: 2:1 ✓ Perfect!
Risk %: 0.83% ✓ Under 2%
```

**Setup B: Aggressive**

```
Entry:   £100
Stop:    £92 (£8 risk)
Target:  £110 (£10 gain)
Qty:     10

Risk Amount: £80
Reward: £100
R:R: 1.25:1 ✗ Too low (below 2:1)
Risk %: 1.33% ✓ Under 2%
Recommendation: Increase target to £116
```

**Setup C: High Conviction**

```
Entry:   £100
Stop:    £95 (£5 risk)
Target:  £125 (£25 gain)
Qty:     10

Risk Amount: £50
Reward: £250
R:R: 5:1 ✓ Excellent!
Risk %: 0.83% ✓ Under 2%
Recommendation: Perfect - highest conviction setup
```

---

## Use Case 7: End of Day Review

### Scenario

You're reviewing your trades at end of day.

### Steps

1. **Go to Trade Bills (📋 tab)**
2. **Review each trade:**
   - AAPL: Filled, Stop Entered, Target Entered, +1.2%
   - MSFT: Filled, Stop Entered, Target Pending
   - GOOGL: Filled, Stop Entered, Target Entered, -0.5%

3. **Click Edit on AAPL**
   - Check "Journal Entered" box
   - Add comment: "Hit first target at 152.50, trailing stop at 151.00"
   - Save

4. **Check your Account (💰 tab)**
   - Money Locked: Now shows updated position values
   - Money Remaining: Recalculated

5. **Evening Checklist (✅ tab)**
   - ✓ Check all completed items:
     - Check Positions & Orders
     - Weekly Chart Scan
     - Daily Chart Analysis
     - Calculate Entry/Stop/Target
     - Position Size
     - Place Orders
     - Update Trade Log

6. **You're done for the day!**

---

## Use Case 8: Modifying an Existing Trade

### Scenario

MSFT hit your first target at 435, and you want to adjust your trailing stop.

### Steps

1. **Go to Trade Bills (📋 tab)**
2. **Find MSFT, click "Edit"**
3. **Update fields:**
   - Current Market Price: 435.00 (new price)
   - Trailing Stop: 432.00 (protect profits)
   - Add comment: "Moved to trailing stop at breakeven +1.5%"
4. **Click "Save Trade Bill"**
4. **You now have a reduced risk on this trade**

---

## Use Case 9: Deleting a Trade

### Scenario

You realize you made a mistake in your setup and want to delete it before placing the order.

### Steps

1. **Go to Trade Bills (📋 tab)**
2. **Find the trade you want to delete**
3. **Click "Delete" button**
4. **Confirm deletion**
5. **Trade is removed from database**
6. **No impact on other trades**

---

## Use Case 10: Currency Conversion

### Scenario

You trade on both NYSE (USD) and NSE (INR) accounts.

### Account 1: US Market

```
Settings:
- Account Name: "NYSE Account"
- Currency: USD
- Trading Capital: $10,000
- Risk %: 2%

Shows: $ 10,000 capital, $200 max risk
```

### Account 2: India Market

```
Settings:
- Account Name: "NSE Account"  
- Currency: INR
- Trading Capital: ₹570,000
- Risk %: 2%

Shows: ₹ 570,000 capital, ₹11,400 max risk
```

Note: To switch accounts, update the currency in Settings.

---

## Common Mistakes to Avoid

### ❌ Mistake 1: Not Filling All Required Fields

- **Error**: Click Calculate without entry/SL/target/qty
- **Fix**: Fill all 4 required fields first

### ❌ Mistake 2: R:R Ratio Too Low

- **Example**: Risk £50, Reward £40 (0.8:1)
- **Fix**: Increase target or tighten stop loss

### ❌ Mistake 3: Exceeding Risk Limit

- **Check**: Risk % shows > 2%
- **Fix**: Reduce quantity or tighten risk

### ❌ Mistake 4: Forgetting to Save

- **Problem**: Form filled but not saved = no database entry
- **Fix**: Always click "Save Trade Bill" button

### ❌ Mistake 5: Wrong Currency Selected

- **Problem**: £100 shows as $100
- **Fix**: Update currency in Settings tab

---

## Performance Tips

1. **Use Calculate Metrics button** - Don't manually enter calculated fields
2. **Set up defaults in Settings** - Saves time for each new trade
3. **Use scanner** - Create from stock pre-fills your data
4. **Batch create bills** - Create multiple bills before market opens
5. **Review R:R first** - Check ratio before saving

---

## Best Practices

1. **Always set Risk:Reward ≥ 2:1** - Better long-term returns
2. **Keep Risk % under 2%** - Preserve capital
3. **Monitor Money Remaining** - Know your trading limit
4. **Update as you go** - Check boxes for trade progression
5. **Add comments** - Remember setup reasons later
6. **Review at end of day** - Update account and checklist
7. **Adjust trailing stops** - Protect profits as trades move in your favor
8. **Keep same currency per account** - Avoid conversion confusion

---

## Keyboard Shortcuts (Future)

- `Tab` - Move to next field
- `Shift+Tab` - Move to previous field
- `Enter` - Save (when in form)
- `Esc` - Cancel/Close modal
- `Ctrl+S` - Quick save

*Note: Currently entering manually, shortcuts coming in v2.1*

---

These use cases cover 80% of daily trading operations. The system adapts to your needs!
